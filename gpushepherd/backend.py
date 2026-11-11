"""Container backends: the real Docker Engine, and an in-memory fake for tests and demo mode.

Every container GPU Shepherd creates carries labels:
    gpushepherd.instance   the instance id
    gpushepherd.gpus       comma-separated GPU indices it was given ("" = none)
    gpushepherd.config     fingerprint of the instance config it was created from
Short-lived helper containers (model downloads) carry gpushepherd.helper instead, so they are
never mistaken for an instance.
Where Docker keeps logs in its default json-file format, which grows without limit, every
container gets a size limit (LOG_LIMIT), so a chatty model server can't fill the disk over months.
GPU assignments are baked in at creation time, so moving an instance to other GPUs means
re-creating its container (its data lives in volumes, so nothing is lost).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from .config import gpu_vendor

LABEL_INSTANCE = "gpushepherd.instance"
LABEL_GPUS = "gpushepherd.gpus"
LABEL_CONFIG = "gpushepherd.config"
LABEL_HELPER = "gpushepherd.helper"
LOG_LIMIT = {"max-size": "10m", "max-file": "3"}


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
    devices: list[str] = field(default_factory=list)   # host device paths passed in (an AMD GPU's render node)


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
    image_id: str = ""          # the image the container was created from (a newer one may be pulled since)


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
    def exec(self, name: str, cmd: list[str], env: dict[str, str] | None = None,
             workdir: str = "") -> tuple[int, str]: ...
    def stats(self, names: list[str]) -> dict[str, ContainerStats]: ...
    def probe(self, image: str, script: str) -> str: ...
    def run_helper(self, name: str, image: str, script: str, volumes: list[tuple[str, str, bool]],
                   user: str | None, kind: str) -> None: ...
    def helpers(self, kind: str) -> dict[str, "HelperState"]: ...
    def run_task(self, image: str, script: str, volumes: list[tuple[str, str, bool]],
                 user: str | None = None, host: bool = False) -> str: ...
    def images(self, refs: list[str]) -> dict[str, int]: ...
    def remove_image(self, ref: str) -> None: ...
    def image_info(self, refs: list[str]) -> dict[str, tuple[str, list[str]]]: ...
    def registry_digest(self, ref: str) -> str: ...
    def pull(self, image: str, log) -> None: ...
    def ping(self) -> str: ...


@dataclass
class HelperState:
    running: bool
    exit_code: int | None = None        # once it has finished


def split_image(image: str) -> tuple[str, str]:
    """An image reference as Docker's pull takes it: (repository, tag or digest). A reference pinned
    by digest (name@sha256:..., or name:tag@sha256:...) is pulled by that digest; without a tag it
    is "latest". A registry's port (host:5000/name) is not a tag."""
    name, at, digest = image.partition("@")
    if at:
        repo, _, tag = name.rpartition(":")
        return (name if not repo or "/" in tag else repo), digest
    repo, _, tag = image.rpartition(":")
    if not repo or "/" in tag:          # no tag given
        return image, "latest"
    return repo, tag


def _gpus_from_label(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip().isdigit()]


# Folders Docker refuses to relabel for SELinux (or that must keep their own labels): these are
# mounted without :z. Paths under the pseudo-filesystems are never relabeled either.
_NO_RELABEL = {"/", "/bin", "/boot", "/etc", "/home", "/lib", "/lib64", "/opt", "/root", "/sbin", "/srv",
               "/tmp", "/usr", "/var", "/mnt", "/media"}
_NO_RELABEL_UNDER = ("/dev/", "/proc/", "/run/", "/sys/", "/var/run/")


def bind_spec(host: str, container: str, read_only: bool, selinux: bool = False) -> str:
    """A Docker bind string. Where Docker labels containers for SELinux, :z lets every container
    share the folder (otherwise SELinux blocks it), except for system folders."""
    opts = "ro" if read_only else "rw"
    path = host.rstrip("/") or "/"
    if selinux and path not in _NO_RELABEL and not (path + "/").startswith(_NO_RELABEL_UNDER):
        opts += ",z"
    return f"{host}:{container}:{opts}"


class DockerBackend:
    """Talks to the Docker Engine API (via the docker SDK) over the mounted socket."""

    def __init__(self):
        from docker import errors as derr  # imported lazily so tests/demo mode do not need a daemon
        self._errors = derr
        self._cpu_prev: dict[str, tuple[int, int]] = {}
        try:
            self._connect()
        except BackendError:
            pass    # Docker is down or still starting: start anyway (the dashboard says so) and retry on use

    def _connect(self) -> None:
        import docker
        try:
            client = docker.from_env(timeout=120)
        except Exception as err:  # noqa: BLE001
            raise BackendError(f"cannot connect to Docker: {err}. Is /var/run/docker.sock mounted?") from err
        self.client, self.api = client, client.api

    def __getattr__(self, name: str):
        # only reached while not connected yet
        if name in ("client", "api"):
            self._connect()
            return self.__dict__[name]
        raise AttributeError(name)

    @property
    def selinux(self) -> bool:
        """Whether Docker labels containers for SELinux (asked once, once Docker answers)."""
        if "_selinux" not in self.__dict__:
            try:
                opts = self.client.info().get("SecurityOptions") or []
            except Exception:  # noqa: BLE001
                return False    # Docker is down: ask again next time
            self._selinux = any("name=selinux" in o for o in opts)
        return self._selinux

    @property
    def log_config(self) -> dict | None:
        """A size limit for the container's logs, when Docker's log driver is json-file (another
        driver someone chose, such as journald or local, already handles size and is left alone)."""
        if "_log_config" not in self.__dict__:
            try:
                driver = self.client.info().get("LoggingDriver") or "json-file"
            except Exception:  # noqa: BLE001
                return None     # Docker is down: ask again next time
            self._log_config = {"Type": "json-file", "Config": dict(LOG_LIMIT)} if driver == "json-file" else None
        return self._log_config

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
                # an app that keeps crashing (Docker restarts it) or is paused still holds its GPUs
                instance_id=iid, name=name, running=state in ("running", "restarting", "paused"), status=state,
                gpus=_gpus_from_label(labels.get(LABEL_GPUS, "")),
                config_hash=labels.get(LABEL_CONFIG, ""), image=c.get("Image", ""),
                status_text=c.get("Status", ""), image_id=c.get("ImageID", ""))
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
        self.pull(image, log)

    def pull(self, image: str, log) -> None:
        """Download the image the tag points at now, even when one is already here."""
        repo, tag = split_image(image)
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
        nvidia = [g for g in spec.gpus if gpu_vendor(g) == "nvidia"]   # AMD ones come as devices (engine)
        if nvidia:
            device_requests = [DeviceRequest(device_ids=[str(g) for g in nvidia], capabilities=[["gpu"]])]
        port_bindings = {f"{c}/{proto}": (ip, h) for ip, h, c, proto in spec.ports}
        binds = [bind_spec(h, c, ro, self.selinux) for h, c, ro in spec.volumes]
        host_config = self.api.create_host_config(
            binds=binds, port_bindings=port_bindings, device_requests=device_requests,
            devices=[f"{d}:{d}:rwm" for d in spec.devices] or None,
            restart_policy={"Name": "unless-stopped"}, shm_size=spec.shm_size, mem_limit=spec.mem_limit,
            init=True, log_config=self.log_config)
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

    def exec(self, name: str, cmd: list[str], env: dict[str, str] | None = None,
             workdir: str = "") -> tuple[int, str]:
        """Run a command inside a running container; its exit code and output."""
        try:
            ex = self.api.exec_create(name, cmd, environment=env or None, workdir=workdir or None)
            out = self.api.exec_start(ex["Id"])
            return self.api.exec_inspect(ex["Id"]).get("ExitCode") or 0, out.decode("utf-8", "replace")
        except self._errors.NotFound as err:
            raise BackendError(f"{name} is not running") from err
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

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
            binds=[bind_spec(h, c, ro, self.selinux) for h, c, ro in volumes], restart_policy={"Name": "no"},
            init=True, log_config=self.log_config)
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

    def run_task(self, image: str, script: str, volumes: list[tuple[str, str, bool]],
                 user: str | None = None, host: bool = False) -> str:
        """Run `sh -c script` in a throwaway container with `volumes` mounted and wait for it: for
        file work in folders GPU Shepherd can't write itself. With host, the container is
        privileged and sees the host's processes, so the script can step into the host's own
        namespaces (nsenter): for looking at and mounting the machine's drives. Raises BackendError
        if it fails."""
        extra = {"privileged": True, "pid_mode": "host"} if host else {}
        try:
            out = self.client.containers.run(
                image, ["-c", script], entrypoint="sh", remove=True, stdout=True, stderr=True, user=user,
                volumes=[bind_spec(h, c, ro, self.selinux) for h, c, ro in volumes], labels={LABEL_HELPER: "task"},
                **extra)
            return out.decode("utf-8", "replace")
        except self._errors.ContainerError as err:
            text = (err.stderr or b"").decode("utf-8", "replace").strip()
            raise BackendError(text.splitlines()[-1] if text else str(err)) from err
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err

    def images(self, refs: list[str]) -> dict[str, int]:
        """The images among `refs` that are on this machine, with their size in bytes."""
        out = {}
        for ref in refs:
            try:
                out[ref] = int(self.api.inspect_image(ref).get("Size") or 0)
            except self._errors.APIError:
                continue        # not here (or Docker is down)
        return out

    def remove_image(self, ref: str) -> None:
        try:
            self.api.remove_image(ref)
        except self._errors.NotFound:
            pass
        except self._errors.APIError as err:
            if err.status_code == 409:
                raise BackendError(f"{ref} is still used by a container") from err
            raise BackendError(err.explanation or str(err)) from err

    def image_info(self, refs: list[str]) -> dict[str, tuple[str, list[str]]]:
        """The images among `refs` that are on this machine: (image ID, registry digests)."""
        out = {}
        for ref in refs:
            try:
                img = self.api.inspect_image(ref)
            except self._errors.ImageNotFound:
                continue
            except self._errors.APIError as err:
                raise BackendError(err.explanation or str(err)) from err
            out[ref] = (img.get("Id", ""), list(img.get("RepoDigests") or []))
        return out

    def registry_digest(self, ref: str) -> str:
        """The digest the image's tag points at in its registry now (asked through Docker)."""
        try:
            return self.api.inspect_distribution(ref)["Descriptor"]["digest"]
        except self._errors.APIError as err:
            raise BackendError(err.explanation or str(err)) from err
        except (KeyError, TypeError) as err:
            raise BackendError(f"the registry gave no digest for {ref}") from err


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
        self.tasks: list[tuple[str, list[tuple[str, str, bool]]]] = []   # (script, volumes) of run_task
        self.present: dict[str, int] = {}     # images "on this machine" -> size
        # image versions: what the registry has (ref -> digest) and what is here (ref -> digest);
        # an image not listed is the same in both
        self.remote: dict[str, str] = {}
        self.local: dict[str, str] = {}
        self.registry_error: dict[str, str] = {}
        self.execs: list[tuple[str, list[str], dict[str, str], str]] = []   # (name, cmd, env, workdir)
        self.exec_reply = lambda name, cmd, env: (0, "[]")                  # what an exec answers
        self._ids: dict[str, str] = {}        # container -> image ID it was created from
        self._lock = threading.Lock()

    def ping(self) -> str:
        return "Demo mode (no Docker)"

    def states(self) -> dict[str, ContainerState]:
        with self._lock:
            return {spec.instance_id: ContainerState(
                instance_id=spec.instance_id, name=name, running=self.running[name],
                status="running" if self.running[name] else "exited", gpus=list(spec.gpus),
                config_hash=spec.config_hash, image=spec.image, image_id=self._ids.get(name, ""),
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
            self._ids[spec.name] = self._image_id(spec.image)
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

    def exec(self, name: str, cmd: list[str], env: dict[str, str] | None = None,
             workdir: str = "") -> tuple[int, str]:
        if not self.running.get(name):
            raise BackendError(f"{name} is not running")
        self.execs.append((name, cmd, dict(env or {}), workdir))
        return self.exec_reply(name, cmd, env or {})

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

    def run_task(self, image: str, script: str, volumes: list[tuple[str, str, bool]],
                 user: str | None = None, host: bool = False) -> str:
        self.tasks.append((script, volumes))
        self.calls.append("task")
        return ""

    def images(self, refs: list[str]) -> dict[str, int]:
        return {r: self.present[r] for r in refs if r in self.present}

    def remove_image(self, ref: str) -> None:
        if any(spec.image == ref for spec in self.containers.values()):
            raise BackendError(f"{ref} is still used by a container")
        self.present.pop(ref, None)
        self.calls.append(f"rmi {ref}")

    def _digest(self, ref: str, where: dict[str, str]) -> str:
        return where.get(ref) or "sha256:" + hashlib.sha256(ref.encode()).hexdigest()

    def _image_id(self, ref: str) -> str:
        return "sha256:id-" + self._digest(ref, self.local)[7:19]

    def image_info(self, refs: list[str]) -> dict[str, tuple[str, list[str]]]:
        return {r: (self._image_id(r), [f"{r.rpartition(':')[0] or r}@{self._digest(r, self.local)}"]) for r in refs}

    def registry_digest(self, ref: str) -> str:
        if ref in self.registry_error:
            raise BackendError(self.registry_error[ref])
        return self._digest(ref, self.remote)

    def pull(self, image: str, log) -> None:
        time.sleep(self.slow)
        self.local[image] = self._digest(image, self.remote)
        self.calls.append(f"pull {image}")

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
