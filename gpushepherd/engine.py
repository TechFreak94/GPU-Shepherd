"""Engine: turns plans into container changes. The only place that talks to the backend."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import budget, drain, gpu as gpu_mod, host, placement
from .backend import Backend, ContainerSpec, ContainerState
from .config import Catalog, ConfigError, Instance, Store
from .connectors import llamaswap
from .scheduler import Plan, plan as make_plan, start_order


@dataclass
class Snapshot:
    instances: dict[str, Instance]
    problems: list[str]
    states: dict[str, ContainerState]
    gpus: list[gpu_mod.Gpu]
    gpu_error: str
    allocations: dict[str, list[int]]       # running instance -> GPUs

    def gpu_users(self, index: int) -> list[str]:
        return [self.instances[i].name for i, g in self.allocations.items() if index in g and i in self.instances]


class Engine:
    def __init__(self, store: Store, backend: Backend, host_data_dir: str | None = None, activity=None):
        self.store = store
        self.backend = backend
        # how busy a running instance is (drain.probe); tests and demo mode pass their own
        self.activity = activity or self._probe
        self.drain_interval = 2.0          # seconds between "are you done?" checks
        # Where the data dir lives on the HOST (Docker needs host paths for bind mounts).
        # Default: same path inside and outside the container, like Dockge's stacks dir.
        self.host_data_dir = Path(host_data_dir or os.environ.get("SHEPHERD_HOST_DATA_DIR") or store.root)

    # ------------------------------------------------------------------ state
    def snapshot(self, states: dict[str, ContainerState] | None = None) -> Snapshot:
        """What is configured and running. `states` stands in for Docker's answer (the dashboard
        passes {} while Docker can't be reached)."""
        instances, problems = self.store.instances()
        states = self.backend.states() if states is None else states
        gpus, err = gpu_mod.query()
        alloc = {i: st.gpus for i, st in states.items() if st.running and i in instances}
        return Snapshot(instances, problems, states, gpus, err, alloc)

    def container_name(self, iid: str) -> str:
        return self.store.settings().container_prefix + iid

    def config_hash(self, inst: Instance) -> str:
        # the drain policy, the app it came from, what starts with it, skipped app updates and how
        # its image is updated only matter to GPU Shepherd: no restart for those
        data = inst.model_dump(mode="json", exclude={"drain", "app", "requires", "skipped_updates", "auto_update"})
        if inst._file_name is not None:            # renamed only on screen (see config._named)
            data["name"] = inst._file_name
        if inst.kind == "llama-swap":
            data["_catalog"] = self.store.catalog(inst.id).model_dump(mode="json")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ planning
    def plan(self, action: str, target: str | None, extra_wanted: set[str] | None = None) -> Plan:
        snap = self.snapshot()
        gpu_ids = [g.index for g in snap.gpus]
        if snap.gpu_error:
            # GPUs that can't be read are not gone: plan with the ones running containers hold, so a
            # passing nvidia-smi failure never stops or moves what is running
            gpu_ids = sorted({g for held in snap.allocations.values() for g in held})
        p = make_plan(gpu_ids, snap.instances, snap.allocations, action, target, extra_wanted,
                      self.usable(snap))
        if snap.gpu_error and not p.error and action != "stop" and (p.side_effects or p.to_stop):
            p.error = "it would stop or move other apps"
        if snap.gpu_error and p.error:
            p.error = f"The GPUs can't be read right now, so nothing was changed: {p.error}. ({snap.gpu_error})"
        return p

    # ------------------------------------------------------------------ model servers on the GPUs
    def model_needs(self, inst: Instance, cat: Catalog) -> dict[str, int]:
        return {m.id: budget.model_memory(inst, m) for m in cat.models}

    def usable(self, snap: Snapshot) -> dict[str, int]:
        """Elastic model servers -> the most GPUs their models can use (placement.wants)."""
        mem = [int(g.mem_total * 2**20) for g in sorted(snap.gpus, key=lambda g: g.index)]
        out = {}
        for iid, inst in snap.instances.items():
            if inst.kind != "llama-swap" or inst.gpu.mode != "elastic":
                continue
            try:
                cat = self.store.catalog(iid)
            except ConfigError:
                continue
            n = placement.wants(cat, self.model_needs(inst, cat), mem)
            if n is not None:
                out[iid] = n
        return out

    def placement(self, inst: Instance, cat: Catalog, gpus: list[int]) -> dict[str, list[int]]:
        """Model id -> positions of the server's GPUs it runs on (placement.place)."""
        mem_of = {g.index: int(g.mem_total * 2**20) for g in gpu_mod.query()[0]}
        mem = [mem_of.get(g, 0) for g in gpus]
        if not all(mem):
            return {}
        return placement.place(cat, self.model_needs(inst, cat), mem)

    def autostart_plan(self) -> Plan:
        instances, _ = self.store.instances()
        return self.plan("rebalance", None, {i for i, inst in instances.items() if inst.autostart})

    # ------------------------------------------------------------------ execution
    def _probe(self, inst: Instance) -> drain.Activity:
        return drain.probe(inst, self.store.catalog(inst.id) if inst.kind == "llama-swap" else None)

    def execute(self, p: Plan, log, job=None) -> None:
        """Carry out a plan. `job` (jobs.Job) lets the dashboard follow and steer the wait for
        running work before containers stop."""
        if p.error:
            raise ConfigError(p.error)
        instances, _ = self.store.instances()
        settings = self.store.settings()
        names = {i: inst.name for i, inst in instances.items()}
        if p.is_noop:
            log("Nothing to change.")
            return
        for line in p.describe(names):
            log("Plan: " + line)
        self.backend.ensure_network(settings.network)
        starts = start_order(instances, set(p.to_restart) | set(p.to_start), p.target)

        # download images BEFORE stopping anything, so a long first pull causes no downtime
        # and a failed pull leaves everything as it was
        for iid in starts:
            self.backend.ensure_image(instances[iid].image, log)

        # let running work (a chat answer, a queued image) finish before its container stops;
        # this can raise jobs.Cancelled, and then nothing has been changed
        stopping = p.to_stop + p.to_restart
        states = self.backend.states()
        busy_candidates = [i for i in stopping if i in instances and states.get(i) and states[i].running]
        drain.wait_until_idle(instances, busy_candidates, log, job, ask=self.activity,
                             interval=self.drain_interval)

        # stop first so GPUs are free before anything starts
        for iid in stopping:
            log(f"Stopping {names.get(iid, iid)}")
            self.backend.stop(self.container_name(iid))

        # start everything even if one fails, so e.g. a broken ComfyUI does not leave the
        # Llama Swap (which was moved to make room) stopped
        failures = []
        for iid in starts:
            inst = instances[iid]
            gpus = p.allocations[iid]
            try:
                spec = self.build_spec(inst, gpus, log)
                log(f"Starting {inst.name}" + (f" on GPU {','.join(map(str, gpus))}" if gpus else ""))
                self.backend.create_and_start(spec)
            except Exception as err:  # noqa: BLE001 - reported below
                log(f"Could not start {inst.name}: {err}")
                failures.append(f"{inst.name}: {err}")
        if failures:
            raise RuntimeError("; ".join(failures))
        log("Done.")

    def build_spec(self, inst: Instance, gpus: list[int], log=lambda _m: None) -> ContainerSpec:
        settings = self.store.settings()
        env = dict(inst.env)
        names = None
        if gpus:
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            # On WSL2 every GPU reaches Linux through one device (/dev/dxg), so Docker can't hide
            # the ones a container wasn't given: CUDA is told which ones are its own, by UUID
            # (right whether the container sees all the GPUs or only its own).
            if host.is_wsl():
                names = gpu_mod.cuda_names(gpus)
                if names:
                    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(names))
                else:
                    log("Could not read the GPUs' IDs, so on WSL2 this app may see every GPU")
        volumes = [(v.host, v.container, v.read_only) for v in inst.volumes]
        if inst.kind == "llama-swap":
            catalog = self._ensure_llama_server(inst, self.store.catalog(inst.id), log)
            threads = catalog.threads or gpu_mod.physical_cores()
            text = llamaswap.render(catalog, len(gpus), threads, self.placement(inst, catalog, gpus), names)
            out = self.store.generated_dir(inst.id) / "llama-swap.yaml"
            out.write_text(text, encoding="utf-8")
            host_file = str(self.host_data_dir / "generated" / inst.id / "llama-swap.yaml")
            volumes += [(host_file, path, True) for path in llamaswap.CONFIG_PATHS]
            env[llamaswap.API_KEY_ENV] = catalog.api_key
        return ContainerSpec(
            name=self.container_name(inst.id), image=inst.image, instance_id=inst.id,
            network=settings.network, alias=inst.id, command=list(inst.command), env=env,
            ports=[(settings.bind_address, p.host, p.container, p.protocol) for p in inst.ports],
            volumes=volumes, gpus=list(gpus), config_hash=self.config_hash(inst),
            shm_size=inst.shm_size, user=inst.user, mem_limit=inst.memory_limit)

    def _ensure_llama_server(self, inst: Instance, catalog: Catalog, log) -> Catalog:
        if catalog.llama_server:
            return catalog
        log("Looking for llama-server inside the image")
        out = self.backend.probe(inst.image, llamaswap.DETECT_SCRIPT)
        path = next((ln.strip() for ln in reversed(out.splitlines()) if ln.strip().startswith("/")), "")
        catalog.llama_server = path or "/app/llama-server"
        log(f"llama-server: {catalog.llama_server}" + ("" if path else " (guessed)"))
        self.store.save_catalog(catalog)
        return catalog

    def remove_containers(self, ids: list[str], log, job=None) -> None:
        """Stop the first instance the normal way (waiting for running work, like Stop; the
        instances it requires stop with it), then remove all their containers. Raises
        jobs.Cancelled, having changed nothing, if the wait is called off."""
        st = self.backend.states().get(ids[0])
        if st and st.running:
            self.execute(self.plan("stop", ids[0]), log, job)
        for iid in ids:
            log(f"Removing the container for {iid}")
            self.backend.remove(self.container_name(iid))

    # ------------------------------------------------------------------ llama-swap helpers
    def llama_client(self, iid: str) -> llamaswap.LlamaSwapClient:
        inst = self.store.instance(iid)
        return llamaswap.client_for(inst, self.store.catalog(iid))
