"""Container backends: the real Docker Engine, and an in-memory fake for tests and demo mode.

Every container GPU Shepherd creates carries labels:
    gpushepherd.instance   the instance id
    gpushepherd.gpus       comma-separated GPU indices it was given ("" = none)
    gpushepherd.config     fingerprint of the instance config it was created from
Short-lived helper containers (model downloads) carry gpushepherd.helper instead, so they are
never mistaken for an instance.
GPU assignments are baked in at creation time, so moving an instance to other GPUs means
re-creating its container (its data lives in volumes, so nothing is lost).
"""
from __future__ import annotations

import os
import random
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

LABEL_INSTANCE = "gpushepherd.instance"
LABEL_GPUS = "gpushepherd.gpus"
LABEL_CONFIG = "gpushepherd.config"
LABEL_HELPER = "gpushepherd.helper"


@dataclass
class ContainerSpec:
    name: str
    image: str
    instance_id: str
    network: str
    alias: str
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    ports: list[tuple[str, int, int, str]] = field(default_factory=list)   # (bind ip, host, container, proto)
    volumes: list[tuple[str, str, bool]] = field(default_factory=list)     # (host, container, read only)
    gpus: list[int] = field(default_factory=list)
    config_hash: str = ""
    shm_size: str | None = None
    user: str | None = None
    mem_limit: str | None = None


@dataclass
class ContainerState:
    instance_id: str
    name: str
    running: bool
    status: str                 # running, exited, created, restarting ...
    gpus: list[int]
    config_hash: str
    image: str
    status_text: str = ""       # Docker's own summary, e.g. "Up 3 hours" or "Exited (1) 2 minutes ago"
    exit_code: int | None = None


@dataclass
class ContainerStats:
    cpu_pct: float | None = None   # share of the whole machine's CPU time, 0-100; None = no sample yet
    mem_used: int = 0           # bytes, without reclaimable file cache (like `docker stats`)
    mem_limit: int = 0          # bytes; the host's total memory when the container has no limit


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    def states(self) -> dict[str, ContainerState]: ...
    def ensure_network(self, name: str) -> None: ...
    def ensure_image(self, image: str, log) -> None: ...
    def create_and_start(self, spec: ContainerSpec) -> None: ...
    def stop(self, name: str, timeout: int = 30) -> None: ...
    def remove(self, name: str) -> None: ...
    def logs(self, name: str, tail: int = 300) -> str: ...
    def stats(self, names: list[str]) -> dict[str, ContainerStats]: ...
    def probe(self, image: str, script: str) -> str: ...
    def run_helper(self, name: str, image: str, script: str, volumes: list[tuple[str, str, bool]],
                   user: str | None, kind: str) -> None: ...
    def helpers(self, kind: str) -> dict[str, "HelperState"]: ...
    def ping(self) -> str: ...


@dataclass
class HelperState:
    running: bool
    exit_code: int | None = None        # once it has finished


def _gpus_from_label(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip().isdigit()]


class DockerBackend:
    """Talks to the Docker Engine API (via the docker SDK) over the mounted socket."""

    def __init__(self):
        import docker  # imported lazily so tests/demo mode do not need a daemon
        from docker import errors as derr
        self._errors = derr
        try:
            self.client = docker.from_env(timeout=120)
            self.api = self.client.api
        except Exception as err:  # noqa: BLE001
            raise BackendError(f"cannot connect to Docker: {err}. Is /var/run/docker.sock mounted?") from err
        self._cpu_prev: dict[str, tuple[int, int]] = {}

    def ping(self) -> str:
        try:
            return "Docker " + self.client.version().get("Version", "?")
        except Exception as err:  # noqa: BLE001
            return f"Docker not reachable: {err}"

    def states(self) -> dict[str, ContainerState]:
        out: dict[str, ContainerState] = {}
        for c in self.api.containers(all=True, filters={"label": LABEL_INSTANCE}):
            labels = c.get("Labels") or {}
            iid = labels.get(LABEL_INSTANCE, "")
            name = (c.get("Names") or ["/?"])[0].lstrip("/")
            state = c.get("State", "")
            out[iid] = ContainerState(
                instance_id=iid, name=name, running=state == "running", status=state,
                gpus=_gpus_from_label(labels.get(LABEL_GPUS, "")),
                config_hash=labels.get(LABEL_CONFIG, ""), image=c.get("Image", ""),
                status_text=c.get("Status", ""))
        return out

    def ensure_network(self, name: str) -> None:
        if not [n for n in self.api.networks(names=[name]) if n.get("Name") == name]:
            self.api.create_network(name, driver="bridge", labels={LABEL_INSTANCE: "network"})
        self._join(name)

    def _join(self, name: str) -> None:
        """When GPU Shepherd itself runs in a container, put it on the instances' network so it
        can reach model servers by name (e.g. http://llm:8080)."""
        if not os.path.exists("/.dockerenv"):
            return
        me = socket.gethostname()
        try:
            nets = self.api.inspect_container(me)["NetworkSettings"]["Networks"] or {}
            if name not in nets:
                self.api.connect_container_to_network(me, name)
        except self._errors.APIError:
            pass   # not fatal: model status/load buttons just won't work until it can reach them

    def ensure_image(self, image: str, log) -> None:
        try:
            self.api.inspect_image(image)
            return
        except self._errors.ImageNotFound:
            pass
        log(f"Pulling {image} (first time only - this can take a while)")
        repo, _, tag = image.rpartition(":")
        if not repo or "/" in tag:          # no tag given
            repo, tag = image, "latest"
        last = ""
        try:
            for event in self.api.pull(repo, tag=tag, stream=True, decode=True):
                status = event.get("status", "")
                if status and status != last and "Downloading" not in status and "Extracting" not in status:
                    log(f"  {status} {event.get('id', '')}".rstrip())
                    last = status
                if "error" in event:
                    raise BackendError(event["error"])
        except self._errors.APIError as err:
            raise BackendError(f"could not pull {image}: {err.explanation or err}") from err

    def create_and_start(self, spec: ContainerSpec) -> None:
        self.remove(spec.name)
        from docker.types import DeviceRequest
        device_requests = []
        if spec.gpus:
            device_requests = [DeviceRequest(device_ids=[str(g) for g in spec.gpus], capabilities=[["gpu"]])]
        port_bindings = {f"{c}/{proto}": (ip, h) for ip, h, c, proto in spec.ports}
        binds = [f"{h}:{c}:{'ro' if ro else 'rw'}" for h, c, ro in spec.volumes]
        host_config = self.api.create_host_config(
            binds=binds, port_bindings=port_bindings, device_requests=device_requests,
            restart_policy={"Name": "unless-stopped"}, shm_size=spec.shm_size, mem_limit=spec.mem_limit,
            init=True)
        networking = self.api.create_networking_config(
            {spec.network: self.api.create_endpoint_config(aliases=[spec.alias])})
        labels = {LABEL_INSTANCE: spec.instance_id, LABEL_GPUS: ",".join(map(str, spec.gpus)),
                  LABEL_CONFIG: spec.config_hash}
        try:
            created = self.api.create_container(
                spec.image, command=spec.command or None, name=spec.name, environment=spec.env,
                ports=[(c, proto) for _, _, c, proto in spec.ports], host_config=host_config,
                networking_config=networking, labels=labels, user=spec.user, hostname=spec.alias)
            self.api.start(created["Id"])
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

    def stop(self, name: str, timeout: int = 30) -> None:
        try:
            self.api.stop(name, timeout=timeout)
        except self._errors.NotFound:
            pass
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

    def remove(self, name: str) -> None:
        try:
            self.api.remove_container(name, force=True)
        except self._errors.NotFound:
            pass
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

    def logs(self, name: str, tail: int = 300) -> str:
        try:
            raw = self.api.logs(name, stdout=True, stderr=True, tail=tail, timestamps=False)
            return raw.decode("utf-8", "replace")
        except self._errors.NotFound:
            return "(no container yet)"
        except self._errors.APIError as err:
            return f"(could not read logs: {err.explanation or err})"

    def stats(self, names: list[str]) -> dict[str, ContainerStats]:
        """CPU and memory use of running containers. Docker's one-shot stats return at once but
        carry a single CPU sample, so CPU use is measured against the previous call's sample."""
        if not names:
            return {}
        with ThreadPoolExecutor(max_workers=min(len(names), 8)) as pool:
            raw = dict(zip(names, pool.map(self._raw_stats, names)))
        out = {}
        for name, data in raw.items():
            if not data:
                continue
            cpu = data.get("cpu_stats") or {}
            mem = data.get("memory_stats") or {}
            total = (cpu.get("cpu_usage") or {}).get("total_usage", 0)
            system = cpu.get("system_cpu_usage", 0)
            st = ContainerStats()
            prev = self._cpu_prev.get(name)
            if prev and system > prev[1] and total >= prev[0]:
                st.cpu_pct = round(100 * (total - prev[0]) / (system - prev[1]), 1)
            self._cpu_prev[name] = (total, system)
            detail = mem.get("stats") or {}
            cache = detail.get("inactive_file", detail.get("total_inactive_file", 0))
            st.mem_used = max(mem.get("usage", 0) - cache, 0)
            st.mem_limit = mem.get("limit", 0)
            out[name] = st
        return out

    def _raw_stats(self, name: str) -> dict:
        try:
            try:
                return self.api.stats(name, stream=False, one_shot=True)
            except self._errors.InvalidVersion:       # Docker older than API 1.41
                return self.api.stats(name, stream=False)
        except Exception:  # noqa: BLE001 - a missing number is not worth failing the page for
            return {}

    def probe(self, image: str, script: str) -> str:
        """Run a throwaway container from `image` with `sh -c script` and return its output."""
        try:
            out = self.client.containers.run(image, ["-c", script], entrypoint="sh", remove=True,
                                             stdout=True, stderr=False)
            return out.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""

    def run_helper(self, name: str, image: str, script: str, volumes: list[tuple[str, str, bool]],
                   user: str | None, kind: str) -> None:
        """Start a helper container that runs `sh -c script` once, in the background. It is kept
        after it exits, so its exit code and logs can be read; the next run of it replaces it."""
        self.remove(name)
        host_config = self.api.create_host_config(
            binds=[f"{h}:{c}:{'ro' if ro else 'rw'}" for h, c, ro in volumes], restart_policy={"Name": "no"},
            init=True)
        try:
            created = self.api.create_container(image, command=["-c", script], entrypoint="sh", name=name,
                                                host_config=host_config, labels={LABEL_HELPER: kind}, user=user)
            self.api.start(created["Id"])
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

    def helpers(self, kind: str) -> dict[str, HelperState]:
        out = {}
        for c in self.api.containers(all=True, filters={"label": f"{LABEL_HELPER}={kind}"}):
            name = (c.get("Names") or ["/?"])[0].lstrip("/")
            code = re.match(r"Exited \((-?\d+)\)", c.get("Status", ""))
            out[name] = HelperState(running=c.get("State") in ("running", "created", "restarting"),
                                    exit_code=int(code.group(1)) if code else None)
        return out


def _ago(since: float) -> str:
    secs = max(int(time.time() - since), 0)
    if secs < 60:
        return "less than a minute"
    if secs < 3600:
        return f"{secs // 60} minute{'s' if secs >= 120 else ''}"
    return f"{secs // 3600} hour{'s' if secs >= 7200 else ''}"


class FakeBackend:
    """In-memory backend: used by the test suite and by demo mode (SHEPHERD_DEMO=1)."""

    def __init__(self, slow: float = 0.0):
        self.containers: dict[str, ContainerSpec] = {}
        self.running: dict[str, bool] = {}
        self.calls: list[str] = []
        self.slow = slow
        self.started: dict[str, float] = {}
        self.helper_runs: dict[str, tuple[str, str, float]] = {}   # name -> (kind, script, started)
        self.helper_seconds = 0.0
        self.helper_exit = 0
        self._lock = threading.Lock()

    def ping(self) -> str:
        return "Demo mode (no Docker)"

    def states(self) -> dict[str, ContainerState]:
        with self._lock:
            return {spec.instance_id: ContainerState(
                instance_id=spec.instance_id, name=name, running=self.running[name],
                status="running" if self.running[name] else "exited", gpus=list(spec.gpus),
                config_hash=spec.config_hash, image=spec.image,
                status_text=(f"Up {_ago(self.started.get(name, time.time()))}" if self.running[name]
                             else "Exited (0)"))
                for name, spec in self.containers.items()}

    def ensure_network(self, name: str) -> None:
        self.calls.append(f"network {name}")

    def ensure_image(self, image: str, log) -> None:
        self.calls.append(f"image {image}")

    def create_and_start(self, spec: ContainerSpec) -> None:
        time.sleep(self.slow)
        with self._lock:
            self.containers[spec.name] = spec
            self.running[spec.name] = True
            self.started[spec.name] = time.time()
        self.calls.append(f"create {spec.name} gpus={','.join(map(str, spec.gpus))}")

    def stop(self, name: str, timeout: int = 30) -> None:
        time.sleep(self.slow)
        with self._lock:
            if name in self.running:
                self.running[name] = False
        self.calls.append(f"stop {name}")

    def remove(self, name: str) -> None:
        with self._lock:
            self.containers.pop(name, None)
            self.running.pop(name, None)
            self.helper_runs.pop(name, None)
        self.calls.append(f"remove {name}")

    def logs(self, name: str, tail: int = 300) -> str:
        return f"[demo] {name} is {'running' if self.running.get(name) else 'stopped'}\n"

    def stats(self, names: list[str]) -> dict[str, ContainerStats]:
        rnd = random.Random(int(time.time() / 4))      # steady between two dashboard refreshes
        return {n: ContainerStats(cpu_pct=round(rnd.uniform(0.2, 6.0), 1),
                                  mem_used=int(rnd.uniform(0.3, 3.5) * 2**30), mem_limit=64 * 2**30)
                for n in names if self.running.get(n)}

    def probe(self, image: str, script: str) -> str:
        return "/app/llama-server\n"

    def run_helper(self, name: str, image: str, script: str, volumes: list[tuple[str, str, bool]],
                   user: str | None, kind: str) -> None:
        with self._lock:
            self.helper_runs[name] = (kind, script, time.time())
        self.calls.append(f"helper {name}")

    def helpers(self, kind: str) -> dict[str, HelperState]:
        """Demo helpers run for helper_seconds, then exit 0 (or helper_exit)."""
        with self._lock:
            runs = dict(self.helper_runs)
        out = {}
        for name, (k, _, started) in runs.items():
            if k == kind:
                done = time.time() - started >= self.helper_seconds
                out[name] = HelperState(running=not done, exit_code=self.helper_exit if done else None)
        return out
