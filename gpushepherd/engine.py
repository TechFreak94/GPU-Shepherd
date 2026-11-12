"""Engine: turns plans into container changes. The only place that talks to the backend."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import budget, drain, gpu as gpu_mod, host, placement
from .backend import Backend, ContainerSpec, ContainerState
from .config import Catalog, ConfigError, Instance, Store, gpu_vendor
from .connectors import llamaswap
from .scheduler import Plan, gpu_text, plan as make_plan, start_order


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
        # (nor the CPU fallback, video memory budget and kind of GPU, which only steer the GPU planner)
        data = inst.model_dump(mode="json", exclude={"drain": True, "app": True, "requires": True,
                                                     "skipped_updates": True, "auto_update": True,
                                                     "gpu": {"cpu_fallback", "vram_gb", "vendor"}})
        if not data["devices"]:                    # added later: no restart for apps that pass none
            del data["devices"]
        if inst._file_name is not None:            # renamed only on screen (see config._named)
            data["name"] = inst._file_name
        if inst.kind == "llama-swap":
            data["_catalog"] = self.store.catalog(inst.id).model_dump(mode="json")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ planning
    def plan(self, action: str, target: str | None, extra_wanted: set[str] | None = None,
             first: set[str] | None = None) -> Plan:
        snap = self.snapshot()
        gpu_ids = [g.index for g in snap.gpus]
        gpu_mem = {g.index: g.mem_total / 1024 for g in snap.gpus}
        if snap.gpu_error:
            # GPUs that can't be read are not gone: plan with the ones running containers hold, so a
            # passing nvidia-smi failure never stops or moves what is running (and apps sharing
            # one keep sharing it: its size is unknown, so nothing limits them). AMD GPUs are
            # read without nvidia-smi, so they stay as they are.
            amd = {g.index: g.mem_total / 1024 for g in snap.gpus if g.vendor == "amd"}
            held = {g for gs in snap.allocations.values() for g in gs if g not in amd}
            gpu_ids = sorted(held | set(amd))
            gpu_mem = {g: 1e9 for g in held} | amd
        p = make_plan(gpu_ids, snap.instances, snap.allocations, action, target, extra_wanted,
                      self.usable(snap), gpu_mem, first)
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
        out = {}
        for iid, inst in snap.instances.items():
            if inst.kind != "llama-swap" or inst.gpu.mode != "elastic":
                continue
            mem = [int(g.mem_total * 2**20) for g in sorted(snap.gpus, key=lambda g: g.index)
                   if g.vendor == inst.gpu.vendor]
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
        if inst.gpu.shares:                 # a GPU shared with other apps: only its budget is its own
            mem = [min(m, int(inst.gpu.vram_gb * 2**30)) for m in mem]
        return placement.place(cat, self.model_needs(inst, cat), mem)

    def gpu_notes(self, snap: Snapshot) -> dict[str, str]:
        """Why an app isn't on a GPU, for its dashboard card: a running one that fell back to the
        CPU, or a stopped one whose start would stop or move others (the GPUs are taken, as on a
        machine with one GPU)."""
        names = {i: inst.name for i, inst in snap.instances.items()}
        holders = sorted({names[i] for i, g in snap.allocations.items() if g and i in names})

        def joined(items: list[str]) -> str:
            return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)
        out: dict[str, str] = {}
        if snap.gpu_error:
            return out
        gpu_ids = [g.index for g in snap.gpus]
        gpu_mem = {g.index: g.mem_total / 1024 for g in snap.gpus}
        usable = None
        for iid, inst in snap.instances.items():
            if inst.gpu.mode == "none":
                continue
            st = snap.states.get(iid)
            if st and st.running:
                if not snap.allocations.get(iid):
                    out[iid] = (f"No GPU was free ({joined(holders)} {'has' if len(holders) == 1 else 'have'} "
                                f"{'it' if len([g for g in gpu_ids if gpu_vendor(g) == inst.gpu.vendor]) == 1 else 'them'}), "
                                "so it runs on the CPU, slower. "
                                "It moves back onto a GPU when one is free; Restart takes one now."
                                if holders else "It runs on the CPU: this machine has no GPU it can use.")
                continue
            if usable is None:
                usable = self.usable(snap)
            p = make_plan(gpu_ids, snap.instances, snap.allocations, "start", iid, usable=usable, gpu_mem=gpu_mem)
            if p.error:
                out[iid] = f"Can't start: {p.error}."
                continue
            stops = [names[i] for i in p.preempted if i != iid]
            cpu = [names[i] for i in p.on_cpu if i != iid and i in p.to_restart]
            parts = ([f"stops {joined(stops)}"] if stops else []) + ([f"moves {joined(cpu)} to the CPU"] if cpu else [])
            if parts:
                kind = [g for g in gpu_ids if gpu_vendor(g) == inst.gpu.vendor]
                taken = "The GPU is" if len(kind) == 1 else "The GPUs are"
                out[iid] = f"{taken} in use. Starting it {' and '.join(parts)} (you are asked first)."
            elif iid in p.on_cpu:
                out[iid] = "No GPU is free for it, so it would start on the CPU, slower."
        return out

    def autostart_plan(self) -> Plan:
        """Start every autostart instance. When the GPUs can't hold them all (one GPU for Llama Swap
        and ComfyUI), model servers get theirs first, so chat works after a reboot; the others
        start when you press Start (which stops or moves what is in the way, after you confirm)."""
        instances, _ = self.store.instances()
        wanted = {i for i, inst in instances.items() if inst.autostart}
        return self.plan("rebalance", None, wanted,
                         first={i for i in wanted if instances[i].kind == "llama-swap"})

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
                log(f"Starting {inst.name}" + (f" on {gpu_text(gpus)}" if gpus else ""))
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
        devices = list(inst.devices)
        amd = [g for g in gpus if gpu_vendor(g) == "amd"]
        if amd:
            # an AMD GPU is its render node: Vulkan and ROCm find it there, and see no other GPU
            nodes = gpu_mod.render_nodes(amd)
            if nodes is None:
                raise RuntimeError(f"can't find the render node (/dev/dri/renderD...) of {gpu_text(amd)}")
            devices += [n for n in nodes if n not in devices]
        if gpus and not amd:
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
            text = llamaswap.render(catalog, len(gpus), threads, self.placement(inst, catalog, gpus), names,
                                    "GGML_VK_VISIBLE_DEVICES" if amd else "CUDA_VISIBLE_DEVICES")
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
            shm_size=inst.shm_size, user=inst.user, mem_limit=inst.memory_limit, devices=devices)

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
