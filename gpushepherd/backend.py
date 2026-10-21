"""Container backends: the real Docker Engine, and an in-memory fake for tests and demo mode.

Every container GPU Shepherd creates carries labels:
    gpushepherd.instance   the instance id
    gpushepherd.gpus       comma-separated GPU indices it was given ("" = none)
    gpushepherd.config     fingerprint of the instance config it was created from
GPU assignments are baked in at creation time, so moving an instance to other GPUs means
re-creating its container (its data lives in volumes, so nothing is lost).
"""
from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

LABEL_INSTANCE = "gpushepherd.instance"
LABEL_GPUS = "gpushepherd.gpus"
LABEL_CONFIG = "gpushepherd.config"


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
    started_at: str = ""
    exit_code: int | None = None


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
    def probe(self, image: str, script: str) -> str: ...
    def ping(self) -> str: ...


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
                started_at=c.get("Status", ""))
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

    def probe(self, image: str, script: str) -> str:
        """Run a throwaway container from `image` with `sh -c script` and return its output."""
        try:
            out = self.client.containers.run(image, ["-c", script], entrypoint="sh", remove=True,
                                             stdout=True, stderr=False)
            return out.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""


class FakeBackend:
    """In-memory backend: used by the test suite and by demo mode (SHEPHERD_DEMO=1)."""

    def __init__(self, slow: float = 0.0):
        self.containers: dict[str, ContainerSpec] = {}
        self.running: dict[str, bool] = {}
        self.calls: list[str] = []
        self.slow = slow
        self._lock = threading.Lock()

    def ping(self) -> str:
        return "Demo mode (no Docker)"

    def states(self) -> dict[str, ContainerState]:
        with self._lock:
            return {spec.instance_id: ContainerState(
                instance_id=spec.instance_id, name=name, running=self.running[name],
                status="running" if self.running[name] else "exited", gpus=list(spec.gpus),
                config_hash=spec.config_hash, image=spec.image)
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
        self.calls.append(f"remove {name}")

    def logs(self, name: str, tail: int = 300) -> str:
        return f"[demo] {name} is {'running' if self.running.get(name) else 'stopped'}\n"

    def probe(self, image: str, script: str) -> str:
        return "/app/llama-server\n"
