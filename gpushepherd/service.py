"""What GPU Shepherd can do, in one place: the dashboard, the JSON API (api.py) and the
`gpu-shepherd` command all call these, so the three never drift apart.

Every method checks its input and raises `Problem` with a message a person can act on. The
dashboard shows it as a toast, the API returns it as {"error": ...} with `Problem.status`, and the
command prints it.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import secrets
import shlex
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from pydantic import ValidationError

from . import (app_updates, auth, host as host_mod, knowledge_skills, knowledge as kn, budget, downloads, drives, gpu as gpu_mod, host, image_updates,
               instance_templates as tpl, logins, model_library, moves, storage, tuning, updates)
from . import gpu_unlock, hf_models, tailscale
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, Instance, Settings, Store,
                     VolumeMap, friendly_errors)
from .connectors import comfyui, instance_url, llamaswap, openwebui
from .backend import BackendError
from .downloads import IMAGE as HELPER_IMAGE, KIND as DOWNLOAD_KIND, Downloads
from .engine import Engine
from .jobs import Busy, Cancelled, Job, JobRunner
from .scheduler import Plan
from .trash import FileOps, Trash

log = logging.getLogger("gpushepherd")
POWER_ACTIONS = ("restart", "reboot", "shutdown")   # the top menu's Power dialog
_LOADING = threading.Lock()   # guards Shepherd.loading (routes run in worker threads)
DRIVE_SCAN_SECONDS = 30       # how long a look at the machine's drives is reused while no drive came or went


def restart_process(delay: float = 1.0) -> None:
    """Start this process again in place, after `delay` seconds so the answer still gets out.
    Same PID and arguments, so it works in the container (PID 1) and outside it; the apps'
    containers are not touched."""
    def again():
        logging.shutdown()
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    threading.Timer(delay, again).start()
ACTIONS = ("start", "stop", "restart")
VERBS = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}


class Problem(Exception):
    """A request that can't be done as asked. kind is how the dashboard shows it (error / warn /
    info); status is the HTTP status the API answers with."""

    def __init__(self, msg: str, kind: str = "error", status: int = 400, data: dict | None = None):
        super().__init__(msg)
        self.msg, self.kind, self.status, self.data = msg, kind, status, data or {}


def message(err: Exception) -> str:
    return friendly_errors(err) if isinstance(err, ValidationError) else str(err)


def checked(fn):
    """Turn validation errors from the config models into a Problem."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ConfigError, ValueError) as err:
            missing = isinstance(err, ConfigError) and str(err).startswith(("no instance", "no template"))
            raise Problem(message(err), status=404 if missing else 400) from err
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


@dataclass
class Change:
    """The outcome of asking for start / stop / restart."""
    action: str
    iid: str
    name: str
    plan: Plan
    lines: list[str]                    # what the plan does, in words
    job: Job | None = None              # set once it runs
    noop: bool = False                  # already in that state
    warnings: list[str] = field(default_factory=list)   # memory that may not be enough (budget.py)

    @property
    def needs_confirm(self) -> bool:
        return self.job is None and not self.noop

    @property
    def text(self) -> str:
        if self.noop:
            return f"{self.name} is already {'stopped' if self.action == 'stop' else 'running'}."
        return f"{VERBS[self.action]} {self.name}…"


@dataclass
class Installed:
    inst: Instance
    models: list[CatalogModel]
    bundle: Instance | None = None      # the Llama Swap installed alongside it, if one was
    companions: list[Instance] = field(default_factory=list)   # helpers installed with it (its database)
    downloads: list[str] = field(default_factory=list)       # suggested models it started downloading


@dataclass
class Shepherd:
    store: Store
    engine: Engine
    jobs: JobRunner
    llama_client: object                # iid -> llama-swap client (a stand-in in demo mode)
    loading: dict[str, set[str]] = field(default_factory=dict)   # model server -> models loading
    downloads: Downloads | None = None
    trash: Trash | None = None
    peaks: budget.Peaks | None = None
    measurements: tuning.Measurements | None = None
    decisions: tuning.Decisions | None = None
    mover: moves.Mover | None = None
    move_log: moves.History | None = None
    move_interval: float = 5.0          # seconds between looks at a copy
    host_fs: drives.Host | None = None
    drive_log: drives.Memory | None = None
    drive_error: str = ""               # why the machine's drives can't be looked at, if they can't
    image_log: image_updates.Records | None = None
    _metrics_seen: dict[str, float] = field(default_factory=dict)   # server -> newest answer measured
    knowledge_task: kn.Task | None = None     # what is being done with the books now (or last)
    openwebui_client: object = openwebui.OpenWebUI   # (address, token) -> client; a stand-in in tests
    _library: kn.Library | None = None
    restarter: object = None            # () -> None: restarts this process (power "restart"); a stand-in in tests
    hf: hf_models.Library | None = None    # suggested models from Hugging Face

    def __post_init__(self):
        if self.downloads is None:
            self.downloads = Downloads(self.backend, self.store.settings().container_prefix)
        if self.trash is None:
            self.trash = Trash(self.store.root, HelperFileOps(self.backend))
        if self.peaks is None:
            self.peaks = budget.Peaks(self.store.root)
        if self.measurements is None:
            self.measurements = tuning.Measurements(self.store.root)
        if self.decisions is None:
            self.decisions = tuning.Decisions(self.store.root)
        if self.mover is None:
            self.mover = moves.HelperMover(self.backend, HELPER_IMAGE)
        if self.move_log is None:
            self.move_log = moves.History(self.store.root)
        if self.host_fs is None:
            self.host_fs = drives.HelperHost(self.backend, HELPER_IMAGE)
        if self.drive_log is None:
            self.drive_log = drives.Memory(self.store.root)
        if self.image_log is None:
            self.image_log = image_updates.Records(self.store.root)
        if self.hf is None:
            self.hf = hf_models.Library(self.store.root)
        if self.restarter is None:
            self.restarter = restart_process
        self._auto_tried: set[tuple[str, str]] = set()     # (instance, version) automatic updates tried
        self._drive_lock = threading.Lock()
        self._drives_seen: set[str] | None = None   # drive IDs at the last scan (None: no scan yet)
        # the last scan, kept while the kernel's list of drives is the same and it is fresh: a scan
        # runs a privileged helper on the host, and the Storage page asks every few seconds
        self._drive_scan: tuple[str, float, list[drives.Drive]] | None = None

    @property
    def backend(self):
        return self.engine.backend

    def names(self) -> dict[str, str]:
        return {i.id: i.name for i in self.store.instances()[0].values()}

    @checked
    def instance(self, iid: str) -> Instance:
        return self.store.instance(iid)

    # ------------------------------------------------------------------ start / stop / restart
    def change(self, action: str, iid: str, confirm: bool = False) -> Change:
        """Plan a start, stop or restart and run it. When it also stops or moves other instances,
        it only runs with confirm; otherwise the Change says what it would do."""
        if action not in ACTIONS:
            raise Problem(f"unknown action '{action}'")
        names = self.names()
        if iid not in names:
            raise Problem(f"no instance '{iid}'", status=404)
        if self.jobs.busy:
            raise Problem("Another change is still running - wait for it to finish.", "warn", 409)
        p = self.engine.plan(action, iid)
        if p.error:
            raise Problem(p.error, status=409)
        if action == "stop" and p.is_noop:
            # a helper (a database) that a running app starts with is kept running by the planner:
            # say so, rather than "already stopped" while it runs on
            states = self.backend.states()
            if states.get(iid) and states[iid].running:
                users = [i.name for i in self.store.instances()[0].values()
                         if iid in i.requires and states.get(i.id) and states[i.id].running]
                if users:
                    raise Problem(f"{names[iid]} is needed by {' and '.join(users)}, which {'is' if len(users) == 1 else 'are'} "
                                  f"running: stop {'it' if len(users) == 1 else 'them'} first.", "warn", 409)
        c = Change(action, iid, names[iid], p, p.describe(names), noop=p.is_noop)
        if not c.noop:
            c.warnings = self.memory_warnings(p)
        if c.noop or ((p.side_effects or c.warnings) and not confirm):
            return c
        try:
            c.job = self.jobs.submit(f"{VERBS[action]} {c.name}", lambda job: self.engine.execute(p, job, job))
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return c

    def memory_warnings(self, p: Plan) -> list[str]:
        """What may not fit once the plan runs: system RAM for what starts, and model server models
        bigger than the GPUs they get. Estimates (see budget.py), so only warnings."""
        try:
            snap = self.engine.snapshot()
            peaks = self.peaks.get()
            vram_of = {g.index: int(g.mem_total * 2**20) for g in snap.gpus}
            out, need, starting = [], 0, []
            for iid in dict.fromkeys(p.to_start + p.to_restart):
                inst = snap.instances.get(iid)
                if not inst:
                    continue
                gpus = p.allocations.get(iid, [])
                vram = sum(vram_of.get(g, 0) for g in gpus)
                if inst.gpu.shares:            # a shared GPU: only its budget is its own
                    vram = min(vram, int(inst.gpu.vram_gb * 2**30))
                cat = self.store.catalog(iid) if inst.kind == "llama-swap" else None
                n = budget.ram_needed(inst, cat, len(gpus), vram, peaks.get(iid, 0))
                if n:
                    need += n
                    starting.append(inst.name)
                if cat and gpus:
                    out += budget.too_big(inst, cat, len(gpus), vram)
            total, available = host.mem_available()
            if need and total:
                leaving = [self.engine.container_name(i) for i in p.to_stop + p.to_restart
                           if snap.states.get(i) and snap.states[i].running]
                freed = sum(s.mem_used for s in self.backend.stats(leaving).values()) if leaving else 0
                free = available + freed
                if need > free - budget.HEADROOM:
                    out.insert(0, f"{', '.join(starting)} may need about {budget.gb(need)} of system RAM, and about "
                                  f"{budget.gb(free)} is free. If the machine runs out, the system stops a "
                                  "container to make room.")
            return out
        except Exception as err:  # noqa: BLE001 - a check that fails must not block a start
            log.warning("memory check skipped: %s", err)
            return []

    def job(self, jid: int) -> Job:
        job = next((j for j in self.jobs.recent() if j.id == jid), None)
        if not job:
            raise Problem("job not found", status=404)
        return job

    def control_job(self, jid: int, op: str) -> str:
        """The buttons shown while a change waits for running work: stop now, or call it off."""
        if op not in ("skip", "cancel"):
            raise Problem("unknown operation")
        job = self.jobs.current
        if not job or job.id != jid:
            raise Problem("That change has already finished.", "info", 409)
        if op == "cancel":
            job.call_off()
            msg = "Calling the change off: nothing is stopped."
        else:
            job.skip.set()
            msg = "Stopping without waiting any longer."
        log.info("job %s: %s pressed", jid, op)
        return msg

    def logs(self, iid: str) -> str:
        self.instance(iid)
        return self.backend.logs(self.engine.container_name(iid))

    # ------------------------------------------------------------------ instances
    @checked
    def save_instance(self, inst: Instance, editing: bool) -> str:
        """Save an instance from the full form (or the API). Returns what to tell the user."""
        inst = Instance.model_validate(inst.model_dump())
        if editing:
            self.store.instance(inst.id)          # must exist
        elif self.store.instance_path(inst.id).exists():
            raise ConfigError(f"an instance called '{inst.id}' already exists")
        self.store.save_instance(inst)
        if inst.app and not editing:              # an App Center app installed by hand: its folders and helpers too
            try:
                t = tpl.get(inst.app)
            except ConfigError:
                t = None
            if t:
                self.add_companions(t, inst)
                self.make_app_folders(inst, t, tpl.seed_files(t, {}, self.machine()))
        if inst.kind == "llama-swap":
            self.store.catalog(inst.id)           # create an empty catalogue with an API key
        st = self.backend.states().get(inst.id)
        note = " Restart it to apply the changes." if st and st.running else ""
        return f"Saved {inst.name}.{note}"

    def delete_instance(self, iid: str) -> str:
        """Stop an instance the normal way and move it, with its app data folder and the helpers
        that only it uses (its database), to the trash. A running instance is deleted in the
        background, once its running work finishes."""
        inst = self.instance(iid)
        instances, _ = self.store.instances()
        others = [i for i in instances.values() if i.id != iid and iid in i.requires]
        if others:
            raise Problem(f"{others[0].name} needs {inst.name}: delete {others[0].name} first.", status=409)
        ids = [iid] + self.own_helpers(inst, instances)
        st = self.backend.states().get(iid)
        running = bool(st and st.running)
        try:
            job = self.jobs.submit(f"Deleting {inst.name}", lambda job: self._delete(ids, job), wait=not running)
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        if job.status == "failed":
            raise Problem(f"Could not delete {inst.name}: {job.error}", status=500)
        extra = f" (with {', '.join(instances[i].name for i in ids[1:])})" if ids[1:] else ""
        if running:
            return f"Deleting {inst.name}{extra}: it stops once its running work finishes, then goes to the trash."
        return f"Deleted {inst.name}{extra}. It is in the trash (Settings, Storage) if you need it back."

    @staticmethod
    def own_helpers(inst: Instance, instances: dict[str, Instance]) -> list[str]:
        """The helper instances an app installed with itself (its database) that nothing else uses."""
        out = []
        for r in inst.requires:
            c = instances.get(r)
            used = any(r in o.requires for o in instances.values() if o.id != inst.id)
            if c and not used and r.startswith(f"{inst.id}-") and c.app:
                out.append(r)
        return out

    def _delete(self, ids: list[str], job: Job) -> None:
        instances, _ = self.store.instances()
        self.engine.remove_containers(ids, job, job)
        paths = self.instance_files(ids, instances)
        entry = self.trash.put(", ".join(instances[i].name for i in ids if i in instances), ids, paths)
        kept = [i.original for i in entry.items if not i.stored]
        job(f"Moved {len(entry.items) - len(kept)} item(s) to the trash")
        for k in kept:
            job(f"Left {k} where it is (it could not be moved to the trash)")
        for iid in ids:               # whatever could not be moved must not stay behind as an instance
            self.store.delete_instance(iid)
            self.peaks.forget(iid)
        self.trash.purge_older(self.store.settings().trash_days)

    def instance_files(self, ids: list[str], instances: dict[str, Instance]) -> list[Path]:
        """What deleting these instances removes: their YAML and model list (with the .prev copies),
        generated files, and app data folders (appdata/<folder>) no other instance uses."""
        root = self.store.root
        paths = []
        for iid in ids:
            for p in (self.store.instance_path(iid), self.store.catalog_path(iid)):
                paths += [p, p.with_suffix(p.suffix + ".prev")]
            paths.append(root / "generated" / iid)
        appdata = root / "appdata"

        def tops(inst: Instance) -> set[Path]:
            out = set()
            for v in inst.volumes:
                host = Path(v.host)
                if host.is_relative_to(appdata) and host != appdata:
                    out.add(appdata / host.relative_to(appdata).parts[0])
            return out
        kept = set().union(*(tops(i) for i in instances.values() if i.id not in ids))
        mine = set().union(*(tops(instances[i]) for i in ids if i in instances))
        return paths + sorted(mine - kept)

    # ------------------------------------------------------------------ trash
    def trash_entries(self) -> list:
        self.trash.purge_older(self.store.settings().trash_days)
        return self.trash.entries()

    def restore_trash(self, eid: str) -> str:
        try:
            e = self.trash.restore(eid)
        except KeyError as err:
            raise Problem(str(err).strip("'\""), status=404) from err
        except FileExistsError as err:
            raise Problem(f"Can't put it back: {err}. Rename or delete that first.", status=409) from err
        except OSError as err:
            raise Problem(f"Could not put it back: {err}", status=500) from err
        return f"Put {e.title} back." + (" Start it from the dashboard when you want it." if e.instances else "")

    def purge_trash(self, eid: str = "") -> str:
        """Delete one trash entry for good, or all of them ("")."""
        ids = [eid] if eid else [e.id for e in self.trash.entries(sizes=False)]
        freed = 0
        for i in ids:
            try:
                freed += sum(e.size for e in self.trash.entries() if e.id == i)
                self.trash.purge(i)
            except KeyError as err:
                raise Problem(str(err).strip("'\""), status=404) from err
            except OSError as err:
                raise Problem(f"Could not delete everything: {err}", status=500) from err
        if not ids:
            return "The trash is already empty."
        return f"Deleted {'everything in the trash' if not eid else 'it'} for good, freeing {gb(freed)}."

    @checked
    def set_trash_days(self, days: int) -> str:
        s = self.store.settings()
        s.trash_days = days
        self.store.save_settings(Settings.model_validate(s.model_dump()))
        return f"The trash empties things after {days} days." if days else "The trash keeps things until you empty it."

    # ------------------------------------------------------------------ storage
    def storage(self) -> dict:
        """What takes space (see storage.py), the app images on this machine, containers left
        without an instance, and the trash."""
        instances, _ = self.store.instances()
        catalogs = {}
        for iid, inst in instances.items():
            if inst.kind == "llama-swap":
                cat = self.store.catalog(iid)
                catalogs[iid] = [f for m in cat.models for f in (m.file, m.mmproj, m.mtp_draft) if f]
        scan = storage.scan(self.store.root, instances, catalogs, tpl.MODELS_IN_CONTAINER, self._downloading(instances))
        states = self.backend.states()
        users: dict[str, list[str]] = {}
        for iid, st in states.items():
            users.setdefault(st.image, []).append(instances[iid].name if iid in instances else st.name)
        for inst in instances.values():
            users.setdefault(inst.image, [])
            if inst.name not in users[inst.image]:
                users[inst.image].append(inst.name)
        present = self.backend.images(sorted(self._image_candidates(instances)))
        images = [{"image": ref, "size": size, "used_by": sorted(users.get(ref, []))}
                  for ref, size in sorted(present.items())]
        return {"disks": [asdict(d) | {"low": d.low} for d in scan.disks],
                "folders": [asdict(f) for f in scan.folders],
                "hidden": scan.hidden,
                "files": [asdict(f) for f in scan.files],
                "leftover_folders": [asdict(f) for f in scan.leftover_folders],
                "images": images,
                "orphans": sorted(st.name for iid, st in states.items() if iid not in instances),
                "trash": [asdict(e) | {"age_days": e.age_days} for e in self.trash_entries()],
                "trash_days": self.store.settings().trash_days,
                "movable": [asdict(m) | {"size": next((f.size for f in scan.folders if f.path == m.path), None)}
                            for m in self.movable(instances)],
                "moves": self.move_history()}

    # ------------------------------------------------------------------ drives and folders (drives.py)
    def scan_drives(self, mount_new: bool = True) -> list[drives.Drive]:
        """The machine's drives. A drive attached since the last scan that has data on it but no mount
        point is mounted now when Settings say so (and it has not failed before): GPU Shepherd never
        formats or writes to one."""
        with self._drive_lock:
            sig, kept = self.host_fs.signature(), self._drive_scan
            if kept and kept[0] == sig and time.monotonic() - kept[1] < DRIVE_SCAN_SECONDS:
                return kept[2]
            try:
                found = self.host_fs.scan()
            except (BackendError, OSError, ValueError) as err:
                self.drive_error = f"The machine's drives can't be looked at from here: {err}"
                log.debug("drive scan failed: %s", err)
                self._drive_scan = None
                return []
            self.drive_error = ""
            memory = self.drive_log.read()
            for d in found:
                if d.uuid in memory["mounted"] and d.state == "mounted":
                    d.by_us = True
                if d.state == "ready" and d.uuid in memory["failed"]:
                    d.state, d.note = "failed", memory["failed"][d.uuid]
            # Only drives attached since the last scan are mounted by themselves: the ones already
            # there when GPU Shepherd starts are left as the owner set them up (Mount on the Storage
            # page does those).
            present = {d.uuid for d in found if d.uuid}
            attached = present - self._drives_seen if self._drives_seen is not None else set()
            self._drives_seen = present
            if mount_new and self.auto_mount():
                for d in found:
                    if d.state == "ready" and d.uuid in attached:
                        self._mount(d, found, persist=True)
            found = sorted(found, key=lambda d: (d.system, d.device))
            self._drive_scan = (sig, time.monotonic(), found)
            return found

    def _mount(self, d: drives.Drive, known: list[drives.Drive], persist: bool) -> None:
        taken = {x.mountpoint for x in known if x.mountpoint}
        try:
            where = self.host_fs.mount(d, drives.mount_name(d, taken), persist)
        except (BackendError, OSError, ValueError) as err:
            d.state, d.note = "failed", f"Could not mount it: {err}"
            self.drive_log.update(d.uuid, failed=d.note)
            log.warning("mounting %s failed: %s", d.device, err)
            return
        d.state, d.mountpoint, d.by_us, d.note = "mounted", where, True, getattr(self.host_fs, "note", "")
        self.drive_log.update(d.uuid, mounted=where)
        log.info("mounted %s (%s) at %s", d.device, d.name, where)

    def drives(self) -> dict:
        found = self.scan_drives()
        return {"drives": [asdict(d) for d in found], "error": self.drive_error,
                "auto_mount": self.auto_mount(), "wsl": host_mod.is_wsl()}

    def auto_mount(self) -> bool:
        """Whether a newly attached drive is mounted by itself. Never on WSL2: every WSL distro's
        disk (and Docker Desktop's) is attached to the same machine, so starting one looks like a
        new drive with data on it."""
        return self.store.settings().auto_mount_drives and not host_mod.is_wsl()

    def mount_drive(self, device: str) -> str:
        """Mount one drive now (also one that failed before, or when automatic mounting is off)."""
        with self._drive_lock:
            try:
                found = self.host_fs.scan()
            except (BackendError, OSError, ValueError) as err:
                raise Problem(f"The machine's drives can't be looked at from here: {err}", status=500) from err
            d = next((x for x in found if x.device == device), None)
            if not d:
                raise Problem(f"{device} is not there any more.", status=404)
            if d.state == "mounted":
                return f"{d.name} is already mounted at {d.mountpoint}."
            if d.state in ("empty", "unsupported"):
                raise Problem(f"{d.name} can't be mounted: {d.note or 'it has no filesystem'}", status=409)
            self._mount(d, found, persist=True)
            self._drive_scan = None               # the next look shows it mounted
            if d.state == "failed":
                raise Problem(d.note, status=500)
            return f"Mounted {d.name} at {d.mountpoint}." + (f" {d.note}" if d.note else "")

    def set_auto_mount(self, on: bool) -> str:
        if on and host_mod.is_wsl():
            raise Problem("On WSL2 new drives are never mounted by themselves: the other WSL distros' "
                             "disks would look like new drives. Use Mount on a drive.")
        s = self.store.settings()
        s.auto_mount_drives = on
        self.store.save_settings(Settings.model_validate(s.model_dump()))
        return ("New drives are mounted as soon as they are attached." if on
                else "New drives are listed, and mounted when you ask.")

    def watch_drives(self, interval: float = 5.0) -> None:
        """Look at the kernel's list of block devices every few seconds, and when it changes (a
        drive was attached) scan and mount it. The scan at start only notes the drives already there."""
        def loop():
            seen = None
            while True:
                sig = drives.proc_signature()
                if sig != seen:
                    seen = sig
                    try:
                        self.scan_drives()
                    except Exception as err:  # noqa: BLE001 - watching must never stop
                        log.debug("drive watch failed: %s", err)
                threading.Event().wait(interval)
        threading.Thread(target=loop, daemon=True, name="drives").start()

    def browse(self, path: str) -> drives.Listing:
        try:
            return self.host_fs.browse(path)
        except (BackendError, OSError, ValueError) as err:
            raise Problem(f"Could not look at {path}: {err}", status=500) from err

    def pick_place(self, source: str, folder: str, name: str) -> dict:
        """The new place for `source` if it goes in `folder` under `name`: the path, whether it
        would be allowed, and how much room the disk has."""
        m = self._movable(source)
        folder = drives.clean_path(folder)
        name = name.strip().strip("/")
        li = self.browse(folder)
        dest = folder.rstrip("/") + "/" + name if name else folder
        err = li.error or ("" if "/" not in name and name not in (".", "..") else "Write just a folder name, without /")
        err = err or moves.check_paths(m.path, dest, str(self.engine.host_data_dir))
        return {"folder": folder, "name": name, "dest": dest, "error": err, "free": li.free, "total": li.total,
                "disk": li.disk}

    # ------------------------------------------------------------------ moving folders (moves.py)
    def movable(self, instances: dict[str, Instance] | None = None) -> list[moves.Movable]:
        instances = instances if instances is not None else self.store.instances()[0]
        return moves.movable(instances, self.appdata_root(),
                             storage.model_folders(instances, tpl.MODELS_IN_CONTAINER), moves.homes(self.move_log.all()))

    def move_history(self) -> list[dict]:
        current = self.jobs.current
        out = []
        for r in self.move_log.all():
            row = asdict(r) | {"age_days": int((time.time() - r.started) // 86400)}
            if r.status == "copying" and not (current and current.id == r.job):
                row["status"] = "interrupted"            # GPU Shepherd restarted while it copied
            out.append(row)
        return out

    def _movable(self, source: str) -> moves.Movable:
        source = source.rstrip("/") or "/"
        m = next((m for m in self.movable() if m.path == source), None)
        if not m:
            raise Problem(f"{source} is not a folder that can be moved: pick one the Storage page lists.", status=404)
        return m

    def home_of(self, source: str) -> str:
        """Where a folder GPU Shepherd moved was first (usually on the drive it was installed on),
        for moving it back there."""
        m = self._movable(source)
        if not m.home:
            raise Problem(f"{m.path} is where it started: GPU Shepherd hasn't moved it.", status=409)
        old = next((r for r in self.move_log.all() if r.status == "done" and r.dest == m.path
                    and r.source == m.home and r.leftover == m.home), None)
        if old:
            raise Problem(f"The old copy is still in {m.home}. Use Move back on that move to point the apps at it "
                          f"again (anything they saved since stays only in {m.path}), or delete the old copy first "
                          "to copy the folder back as it is now.", status=409)
        return m.home

    def check_move(self, source: str, dest: str) -> dict:
        """What moving `source` to `dest` would do, and whether it can: nothing is changed."""
        m = self._movable(source)
        dest = dest.strip().rstrip("/") or dest.strip()
        err = moves.check_paths(m.path, dest, str(self.engine.host_data_dir))
        if err:
            raise Problem(err)
        instances, _ = self.store.instances()
        wider = [i.name for i in instances.values() for v in i.volumes
                 if moves.inside(m.path, v.host) and v.host != m.path and Path(v.host).is_dir()]
        if wider:
            raise Problem(f"{wider[0]} uses a folder that {m.path} is in, so it can't be moved on its own.", status=409)
        busy = [p for p in self._downloading(instances) if moves.inside(p, m.path)]
        if busy:
            raise Problem(f"{Path(busy[0]).name} is still downloading into {m.path}: wait for it, or cancel it.", status=409)
        try:
            c = self.mover.check(m.path, dest)
        except (BackendError, OSError, ValueError) as err:
            raise Problem(f"Could not look at {dest}: {err}", status=500) from err
        c.windows = moves.windows_drive(dest)
        c.hidden = not Path(dest).parent.is_dir()
        if c.error:
            hint = (f" Pick a new folder inside it, such as {dest}/{Path(m.path).name}."
                    if "not an empty folder" in c.error else "")
            raise Problem(f"Can't move there: {c.error}.{hint}", status=409)
        if c.size > c.free and not c.same_disk:
            raise Problem(f"{dest} has {gb(c.free)} free; {m.path} needs {gb(c.size)}.", status=409)
        if c.same_disk and c.size * 2 > c.free + c.size:
            raise Problem(f"{dest} is on the same disk, which has {gb(c.free)} free: not enough to copy "
                          f"{gb(c.size)} (is the other disk mounted?)", status=409)
        states = self.backend.states()
        running = [instances[i].name for i in m.used_by if states.get(i) and states[i].running]
        return {"source": m.path, "dest": dest, "label": m.label, "size": c.size, "free": c.free, "disk": c.disk,
                "same_disk": c.same_disk, "used_by": [instances[i].name for i in m.used_by],
                "stops": running, "warnings": c.warnings}

    def move_folder(self, source: str, dest: str) -> tuple[str, Job]:
        """Move a folder in the background (see moves.py): check, stop its apps, copy, compare,
        switch the apps over, start them again. The old copy is kept."""
        info = self.check_move(source, dest)
        rec = moves.Record(self.move_log.new_id(info["source"]), info["label"], info["source"], info["dest"],
                           self._movable(source).used_by, info["size"])
        try:
            job = self.jobs.submit(f"Moving {rec.source} to {rec.dest}", lambda job: self._move(rec, job))
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return (f"Moving {rec.source} to {rec.dest} ({gb(rec.size)})" +
                (f": {', '.join(info['stops'])} stop{'s' if len(info['stops']) == 1 else ''} until it is done."
                 if info["stops"] else ".")), job

    def _all_running(self) -> set[str]:
        return {i for i, st in self.backend.states().items() if st.running}

    def _stop_users(self, ids: list[str], job: Job) -> None:
        """Stop the instances using a folder the normal way (running work finishes first)."""
        for iid in ids:
            if iid in self._all_running():
                self.engine.execute(self.engine.plan("stop", iid), job, job)

    def _start_again(self, was: set[str], job: Job) -> None:
        instances, _ = self.store.instances()
        again = {i for i in was - self._all_running() if i in instances}
        if again:
            job(f"Starting again: {', '.join(instances[i].name for i in sorted(again))}")
            self.engine.execute(self.engine.plan("rebalance", None, again), job, job)

    def _switch(self, old: str, new: str, job: Job) -> None:
        instances, _ = self.store.instances()
        for inst in instances.values():
            vols = [v.model_copy(update={"host": moves.relocate(v.host, old, new)}) for v in inst.volumes]
            if vols != inst.volumes:
                inst.volumes = vols
                self.store.save_instance(inst)
                job(f"{inst.name} now uses {new}")
        if self.trash.relocate(old, new):
            job("Updated the trash's records of files in it")

    def _move(self, rec: moves.Record, job: Job) -> None:
        rec.job, rec.leftover = job.id, rec.dest       # until it is done, the new copy is the one to delete
        self.move_log.put(rec)
        was, copying = self._all_running(), False
        try:
            self._stop_users(rec.used_by, job)
            job(f"Copying {rec.source} to {rec.dest} ({gb(rec.size)})")
            copying = True
            self.mover.start(rec.id, rec.source, rec.dest)
            last = time.time()
            while True:
                running, code, copied = self.mover.poll(rec.id)
                if job.cancel.is_set():
                    raise Cancelled("the copy was stopped; the apps still use the old place")
                if not running:
                    break
                if copied and time.time() - last >= 30:
                    job(f"Copied {gb(copied)} of {gb(rec.size)}")
                    last = time.time()
                time.sleep(self.move_interval)
            self.mover.stop(rec.id)
            if code != 0:
                raise RuntimeError("the copy did not finish" if code is None else
                                   "the copy is not the same as the folder" if code == 5 else
                                   f"the copy failed (exit code {code}); is there room on the new disk?")
            job("Copied: every file is there, with the same size as in the old copy")
            self._switch(rec.source, rec.dest, job)
            rec.status, rec.leftover = "done", rec.source
            if not Path(rec.dest).is_dir():
                job(f"GPU Shepherd itself can't see {rec.dest}, so the Storage and Models pages can't list the "
                    f"files there. To fix that, add  - {rec.dest}:{rec.dest}  under volumes in its compose.yaml "
                    "and run  docker compose up -d  (the apps work either way).")
        except BaseException as err:
            rec.status = "cancelled" if isinstance(err, Cancelled) else "failed"
            rec.error = str(err) or err.__class__.__name__
            rec.leftover = ""
            if copying:
                try:
                    self.mover.stop(rec.id)
                    job(f"Removing the unfinished copy at {rec.dest}")
                    self.mover.remove(rec.dest)
                except Exception as rm_err:  # noqa: BLE001 - listed on the Storage page to delete later
                    job(f"Could not remove it: {rm_err}")
                    rec.leftover = rec.dest
            raise
        finally:
            rec.finished = time.time()
            self.move_log.put(rec)
            try:
                self._start_again(was, job)
            except Exception as err:  # noqa: BLE001
                job(f"Could not start everything again: {err}")
                if rec.status == "done":
                    raise RuntimeError(f"moved, but could not start everything again: {err}") from err
        job(f"Done. The old copy is still at {rec.source}: delete it from Settings, Storage once the apps work.")

    def _move_record(self, mid: str) -> moves.Record:
        try:
            return self.move_log.get(mid)
        except KeyError as err:
            raise Problem(str(err).strip("'\""), status=404) from err

    def move_back(self, mid: str) -> str:
        """Point the apps at the old copy again (it is still there). Changes made since the move
        stay in the new copy."""
        rec = self._move_record(mid)
        if rec.status != "done" or rec.leftover != rec.source:
            raise Problem("Only a finished move whose old copy is still there can be moved back.", status=409)

        def back(job: Job) -> None:
            was = self._all_running()
            self._stop_users(rec.used_by, job)
            try:
                self._switch(rec.dest, rec.source, job)
                rec.status, rec.leftover = "moved-back", rec.dest
                self.move_log.put(rec)
                job(f"Done. The copy at {rec.dest} stays until you delete it.")
            finally:
                self._start_again(was, job)
        try:
            self.jobs.submit(f"Moving {rec.dest} back to {rec.source}", back)
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return f"The apps go back to {rec.source}. The copy at {rec.dest} stays until you delete it."

    def delete_leftover(self, mid: str) -> str:
        """Delete the copy a move left behind (the old one, or the new one after moving back)."""
        rec = self._move_record(mid)
        path = rec.leftover
        if not path:
            raise Problem("Nothing is left over from that move.", status=409)
        instances, _ = self.store.instances()
        users = [i.name for i in instances.values() if any(moves.inside(v.host, path) for v in i.volumes)]
        if users:
            raise Problem(f"{users[0]} still uses {path}, so it is not deleted.", status=409)
        if (current := self.jobs.current) and current.id == rec.job:
            raise Problem("That move is still running.", "warn", 409)
        try:
            self.mover.stop(rec.id)
            self.mover.remove(path)
        except (BackendError, OSError, ValueError) as err:
            raise Problem(f"Could not delete {path}: {err}", status=500) from err
        rec.leftover = ""
        self.move_log.put(rec)
        return f"Deleted {path}."

    def low_space(self) -> list[dict]:
        """Disks GPU Shepherd uses that are nearly full, for the dashboard's banner. Cheap: no sizes."""
        instances, _ = self.store.instances()
        out, seen = [], set()
        for p in [str(self.store.root)] + storage.model_folders(instances, tpl.MODELS_IN_CONTAINER):
            try:
                dev = os.stat(p).st_dev
                if dev in seen:
                    continue
                seen.add(dev)
                free = shutil.disk_usage(p).free
            except OSError:
                continue
            if free < storage.LOW_GB * 1e9:
                out.append({"path": p, "free": free, "free_text": gb(free)})
        return out

    def _image_candidates(self, instances: dict[str, Instance]) -> set[str]:
        """Images GPU Shepherd may have pulled: the App Center's, the download helper, and the
        instances' own. Other images on the machine are none of its business."""
        return {t.image for t in tpl.TEMPLATES} | {HELPER_IMAGE} | {i.image for i in instances.values()}

    def _downloading(self, instances: dict[str, Instance]) -> set[str]:
        running = {n for n, h in self.backend.helpers(DOWNLOAD_KIND).items() if h.running}
        folders = storage.model_folders(instances, tpl.MODELS_IN_CONTAINER)
        return {str(Path(m) / s.path(f)) for s in model_library.SUGGESTIONS + self._extra()
                if self.downloads.name(s) in running
                for f in s.files for m in folders}

    def remove_unused_images(self) -> str:
        rows = [r for r in self.storage()["images"] if not r["used_by"]]
        freed, failed = 0, []
        for r in rows:
            try:
                self.backend.remove_image(r["image"])
                freed += r["size"]
            except BackendError as err:
                failed.append(str(err))
        if not rows:
            return "No unused app images to remove."
        msg = f"Removed {len(rows) - len(failed)} unused image(s), freeing {gb(freed)}."
        if failed:
            msg += " Kept: " + "; ".join(failed)
        return msg + " They download again if an app needs them."

    def trash_files(self, paths: list[str]) -> str:
        """Move model files the Storage page listed (unused, unfinished, duplicate) to the trash."""
        found = {f["path"]: f for f in self.storage()["files"]}
        unknown = [p for p in paths if p not in found]
        if unknown or not paths:
            raise Problem(f"Not a file the Storage page lists: {unknown[0]}" if unknown else "No files picked.",
                          status=404 if unknown else 400)
        by_folder: dict[str, list[str]] = {}
        for p in paths:
            by_folder.setdefault(found[p]["folder"], []).append(p)
        size = sum(found[p]["size"] for p in paths)
        for folder, files in by_folder.items():
            title = Path(files[0]).name if len(files) == 1 else f"{len(files)} model files from {folder}"
            try:
                self.trash.put_files(title, folder, files)
            except (OSError, BackendError) as err:
                raise Problem(f"Could not move the files: {err}", status=500) from err
        return f"Moved {len(paths)} file(s) ({gb(size)}) to the trash. The space is freed when the trash is emptied."

    def trash_leftover(self, name: str) -> str:
        folder = next((f for f in self.storage()["leftover_folders"] if f["label"] == name), None)
        if not folder:
            raise Problem(f"No unused app data folder called '{name}'", status=404)
        entry = self.trash.put(f"App data folder {name}", [], [Path(folder["path"])])
        if not entry.items or not entry.items[0].stored:
            raise Problem(f"Could not move {folder['path']} to the trash", status=500)
        return f"Moved the app data folder {name} to the trash."

    def remove_orphans(self) -> str:
        names = self.storage()["orphans"]
        for n in names:
            self.backend.remove(n)
        return f"Removed {len(names)} container(s) left without an instance." if names else "Nothing to remove."

    # ------------------------------------------------------------------ App Center
    def machine(self) -> tpl.Machine:
        instances, _ = self.store.instances()
        servers = {}
        for inst in instances.values():
            if inst.kind != "llama-swap":
                continue
            cat = self.store.catalog(inst.id)
            chat = [m for m in cat.models if m.role == "chat"]
            alias = next((a for a in ("chat", "coder") for m in chat if a in m.aliases or a == m.id), "")
            model = next((m for m in chat if alias in m.aliases or alias == m.id), chat[0] if chat else None)
            embed = next((m for m in cat.models if m.role == "embed"), None)
            servers[inst.id] = {"api_key": cat.api_key, "name": inst.name,
                                "default_model": alias or (model.id if model else ""),
                                "ctx": model.ctx_size if model else 0,
                                "embed": ("embed" if "embed" in embed.aliases else embed.id) if embed else "",
                                "embed_query_prefix": tpl.query_prefix(embed.file) if embed else "",
                                # the name each chat model answers to (its first alias), and its context
                                "models": {(x.aliases or [x.id])[0]: x.ctx_size for x in chat}}
        for inst in instances.values():           # vLLM and Ollama servers: apps ask them for the model chat
            info = tpl.vllm_server(inst) or tpl.ollama_server(inst)
            if info:
                servers[inst.id] = info
        # the GPUs only: a whole snapshot would ask Docker too, and pages call this several times
        gpus, ram = gpu_mod.query()[0], host.ram_gb()
        nvidia = [g.mem_total / 1024 for g in gpus if g.vendor == "nvidia"]
        m = tpl.Machine(gpu_mem_gb=nvidia, instances=instances, servers=servers, model_sizes={}, ram_gb=ram,
                        amd_gpu_mem_gb=[g.mem_total / 1024 for g in gpus if g.vendor == "amd"],
                        suggestions=self.hf_picks(nvidia, ram))
        for cid in tpl.comfyui_instances(m):        # the checkpoints Open WebUI can make pictures with
            v = next((v for v in instances[cid].volumes if v.container.rstrip("/").endswith("ComfyUI/models")), None)
            if v:
                m.checkpoints[cid] = comfyui.checkpoints(v.host)
        return m

    def default_models_dir(self, instances: dict[str, Instance]) -> str:
        """Where this machine keeps models: the folder another model server already uses, a common
        place that exists, or a folder inside the data folder."""
        for inst in instances.values():
            for v in inst.volumes:
                if inst.kind == "llama-swap" and v.container == tpl.MODELS_IN_CONTAINER:
                    return v.host
                if v.container.endswith("ComfyUI/models") and v.host.endswith("/comfyui"):
                    return v.host.removesuffix("/comfyui")
        for d in ("/srv/models", "/opt/models", "/data/models"):
            if Path(d).is_dir():
                return d
        return str(self.engine.host_data_dir / "models")

    def model_files(self, models_dir: str) -> tuple[list[dict], list[str]]:
        probe = Instance(id="probe", name="probe", image="probe",
                         volumes=[VolumeMap(host=models_dir, container=tpl.MODELS_IN_CONTAINER)])
        return llamaswap.local_model_files(probe)

    def has_model_server(self) -> bool:
        """Whether any model server an app can connect to is installed (Llama Swap, vLLM, Ollama)."""
        return any(i.kind == "llama-swap" or tpl.vllm_server(i) or tpl.ollama_server(i)
                   for i in self.store.instances()[0].values())

    def appdata_root(self) -> str:
        return str(self.engine.host_data_dir / "appdata")

    @checked
    def app_defaults(self, key: str) -> tuple[tpl.Template, dict, tpl.Machine]:
        """An app and the answers its install form starts with."""
        t = tpl.get(key)
        m = self.machine()
        return t, tpl.defaults(t, m, self.default_models_dir(m.instances), self.appdata_root()), m

    @checked
    def app_instance(self, key: str, answers: dict | None = None) -> Instance:
        """What a manual install starts from: the instance the answers (or the defaults) build,
        with a model server picked rather than made."""
        t, f, m = self.app_defaults(key)
        f |= answers or {}
        if f["server"] == tpl.NEW_SERVER:
            f["server"] = ""
        return tpl.build(t, f, m)[0]

    @checked
    def install(self, key: str, answers: dict, fill_defaults: bool = False) -> Installed:
        """Install an App Center app from its install form's answers. With fill_defaults, the
        answers only need what differs from the form's defaults (for the API and the command)."""
        t = tpl.get(key)
        m = self.machine()
        if fill_defaults:
            answers = tpl.defaults(t, m, self.default_models_dir(m.instances), self.appdata_root()) | answers
            answers["autostart"] = answers.get("autostart") in (True, "true", "on", "1", "yes")
        slot_fields = [f"{x.key}_file" for x in tpl.slots(t)]
        if slot_fields and answers.get("models_dir"):
            m.model_sizes = {x["container_path"]: x["size_gb"] for x in self.model_files(str(answers["models_dir"]))[0]}
        bundle = tpl.bundled_server(answers, m, self.appdata_root(), t.bundled_slot) if t.connects else None
        answers = answers | {"server": bundle[0].id} if bundle else answers
        inst, models = tpl.build(t, answers, m)
        seeds = tpl.seed_files(t, answers, m)
        if bundle:
            self.store.save_instance(bundle[0])
            self.store.save_catalog(bundle[1])
        self.store.save_instance(inst)
        helpers = self.add_companions(t, inst)
        self.make_app_folders(inst, t, seeds)
        if inst.kind == "llama-swap":
            cat = self.store.catalog(inst.id)
            if models:
                cat.models += models
                self.store.save_catalog(Catalog.model_validate(cat.model_dump()))
        server = inst if inst.kind == "llama-swap" else bundle[0] if bundle else None
        fetching = self.fetch_suggested(server, models if server is inst else bundle[1].models) if server else []
        return Installed(inst, models, bundle[0] if bundle else None, helpers, fetching)

    def fetch_suggested(self, server: Instance, models: list[CatalogModel]) -> list[str]:
        """Start downloading the suggested models among `models` that aren't in the models folder yet."""
        out, folder = [], self.models_dir(server)
        for m in models:
            s = model_library.for_file(m.file, tpl.MODELS_IN_CONTAINER, self._extra())
            if s and folder and self.downloads.status(s, folder).state != "ready":
                try:
                    self.downloads.start(s, folder)
                    self.hf.keep(s)
                    out.append(s.name)
                except ValueError as err:
                    log.warning("not downloading %s: %s", s.key, err)
        return out

    def add_companions(self, t: tpl.Template, inst: Instance) -> list[Instance]:
        """Save the helper instances an app requires (its database) that don't exist yet."""
        made = []
        for c in tpl.companions(t, inst):
            if self.store.instance_path(c.id).exists():
                continue
            self.store.save_instance(c)
            ct = tpl.get(c.app)
            self.make_app_folders(c, ct, tpl.seed_files(ct, {}, self.machine()))
            made.append(c)
        return made

    def make_app_folders(self, inst: Instance, t: tpl.Template, seeds: dict[str, str]) -> None:
        """Make a new app's own folders (so Docker doesn't make them as root) and its first settings
        files. Only folders inside the data folder, which GPU Shepherd can see."""
        appdata = {sub: v.host for (sub, inside) in t.appdata for v in inst.volumes if v.container == inside}
        files = {name: v.host for (name, inside) in t.files for v in inst.volumes if v.container == inside}
        for v in inst.volumes:
            host = Path(v.host)
            if v.read_only or not host.is_relative_to(self.store.root):
                continue
            if v.host in files.values():          # a settings file: its folder, then the file below
                host = host.parent
            try:
                host.mkdir(parents=True, exist_ok=True)
                if t.data_owner and os.geteuid() == 0:
                    os.chown(host, *t.data_owner)
            except OSError as err:
                log.warning("could not prepare %s for %s: %s", host, inst.id, err)
        for rel, text in seeds.items():
            sub, _, name = rel.partition("/")
            if rel in files:
                path = Path(files[rel])
            elif sub in appdata:
                path = Path(appdata[sub]) / name
            else:
                continue
            if path.exists() or not path.parent.is_dir() or not path.is_relative_to(self.store.root):
                continue
            try:
                path.write_text(text, encoding="utf-8")
                path.chmod(0o600)                 # it may hold the model server's API key
                if t.data_owner and os.geteuid() == 0:
                    os.chown(path, *t.data_owner)
            except OSError as err:
                log.warning("could not write %s for %s: %s", path, inst.id, err)

    # ------------------------------------------------------------------ app updates
    def _settings_file(self, inst: Instance) -> Path | None:
        """AnythingLLM's own settings file on this machine, when GPU Shepherd can see it."""
        inside = dict(tpl.get("anythingllm").files).get(app_updates.ANYTHINGLLM_FILE)
        v = next((v for v in inst.volumes if v.container == inside), None)
        path = Path(v.host) if v else None
        return path if path and path.is_relative_to(self.store.root) and path.is_file() else None

    def app_updates(self) -> list[app_updates.Update]:
        """What installed apps are missing of what their app sets up today, model servers first
        (an embeddings model there is what the chat apps' updates then point at)."""
        m = self.machine()
        out = []
        for inst in sorted(m.instances.values(), key=lambda i: (i.kind != "llama-swap", i.name.lower())):
            cat = self.store.catalog(inst.id) if inst.kind == "llama-swap" else None
            path = self._settings_file(inst) if inst.app == "anythingllm" else None
            try:
                text = path.read_text(encoding="utf-8") if path else None
            except OSError:
                text = None
            out += app_updates.for_instance(inst, m, cat, text)
        return out

    def _app_update(self, iid: str, key: str) -> app_updates.Update:
        self.instance(iid)
        u = next((u for u in self.app_updates() if u.iid == iid and u.key == key), None)
        if not u:
            raise Problem(f"no update '{key}' for {iid} (already applied or skipped?)", "info", 404)
        return u

    @checked
    def apply_app_update(self, iid: str, key: str) -> str:
        """Apply one app update. Nothing restarts: a running instance shows it needs a restart."""
        u = self._app_update(iid, key)
        inst = self.store.instance(iid)
        if u.kind == "embed-model":
            return self.add_suggestion(iid, app_updates.EMBED_MODEL)
        if u.kind == "model":
            msg = self.save_model(iid, u.model, check=False)
        elif u.kind == "file":
            path = self._settings_file(inst)
            if not path:
                raise Problem(f"can't find {inst.name}'s {app_updates.ANYTHINGLLM_FILE} any more", status=409)
            path.write_text(u.file_text, encoding="utf-8")
            msg = f"Updated {inst.name}'s {app_updates.ANYTHINGLLM_FILE}."
        else:
            inst.env = inst.env | u.env
            self.store.save_instance(inst)
            msg = f"Updated {inst.name}'s settings."
            if u.kind == "images":
                return msg + " " + self._send_images(inst) + (" " + " ".join(u.manual) if u.manual else "")
        st = self.backend.states().get(iid)
        if st and st.running and not msg.endswith("to use it."):
            msg += " Restart it to use them."
        return msg + (" " + " ".join(u.manual) if u.manual else "")

    def apply_all_app_updates(self) -> str:
        """Apply every app update. An embeddings model added to a model server brings the chat
        apps' updates that use it, so this looks again until nothing new turns up."""
        done: set[tuple[str, str]] = set()
        names, restart = self.names(), set()
        for _ in range(5):
            todo = [u for u in self.app_updates() if (u.iid, u.key) not in done]
            if not todo:
                break
            for u in todo:
                done.add((u.iid, u.key))
                msg = self.apply_app_update(u.iid, u.key)
                if "Restart" in msg:
                    restart.add(names.get(u.iid, u.iid))
        if not done:
            return "Every installed app is up to date with what its app sets up."
        apps = sorted({names.get(i, i) for i, _ in done})
        msg = f"Applied {len(done)} update{'s' if len(done) != 1 else ''} to {', '.join(apps)}."
        return msg + (f" Restart {', '.join(sorted(restart))} to use them." if restart else "")

    @checked
    def skip_app_update(self, iid: str, key: str) -> str:
        """Don't offer this update again for this instance."""
        u = self._app_update(iid, key)
        inst = self.store.instance(iid)
        inst.skipped_updates = sorted(set(inst.skipped_updates) | {u.key})
        self._save_setting(inst)
        return f"Skipped \"{u.title}\" for {inst.name}."

    def _save_setting(self, inst: Instance) -> None:
        """Save a setting that only matters to GPU Shepherd, keeping the name in the file as it was
        (config._named), so the card doesn't ask for a restart."""
        if inst._file_name is not None:
            inst.name = inst._file_name
        self.store.save_instance(inst)

    # ------------------------------------------------------------------ new app releases (image_updates.py)
    def check_images(self) -> str:
        """Ask the registries which image each instance's tag points at now."""
        refs = sorted({i.image for i in self.store.instances()[0].values()})
        try:
            here = self.backend.image_info(refs)
        except BackendError as err:
            raise Problem(f"Could not look at the images on this machine: {err}", status=503) from err
        found = {}
        for ref in refs:
            image_id, local = here.get(ref, ("", []))
            local = image_updates.digests(local)
            if not (image_id and local):      # not downloaded yet (its first start gets the newest), or built here
                continue
            rec = {"id": image_id, "local": local, "remote": "", "error": ""}
            try:
                rec["remote"] = self.backend.registry_digest(ref)
            except BackendError as err:
                rec["error"] = str(err)
            found[ref] = rec
        self.image_log.checked(found)
        names = self.names()
        new = [names[i] for i, st in self.image_status().items() if st.available]
        failed = sum(1 for r in found.values() if r["error"])
        msg = (f"New version available for {', '.join(sorted(new))}." if new
               else "Every app is on the newest version of its image.")
        return msg + (f" {failed} image{'s' if failed != 1 else ''} could not be checked." if failed else "")

    def image_status(self, states: dict | None = None) -> dict[str, image_updates.Status]:
        """Instance -> how up to date its image is, from the last check."""
        data = self.image_log.read()
        states = self.backend.states() if states is None else states
        out = {}
        for iid, inst in self.store.instances()[0].items():
            st = states.get(iid)
            out[iid] = image_updates.status(inst, st.image_id if st else "", bool(st and st.running),
                                            data["images"].get(inst.image),
                                            float((data["updated"].get(iid) or {}).get("at") or 0))
        return out

    def image_checked(self) -> float:
        return self.image_log.read()["checked"]

    @checked
    def set_auto_update(self, iid: str, on: bool) -> str:
        """Update an app's image by itself, or only when asked. What starts with it follows."""
        inst = self.store.instance(iid)
        instances = self.store.instances()[0]
        for x in [inst, *(instances[r] for r in inst.requires if r in instances)]:
            x.auto_update = on
            self._save_setting(x)
        return (f"{inst.name} now installs new versions by itself, after waiting for running work." if on
                else f"{inst.name} now waits for you to press Update when a new version is out.")

    def update_image(self, iid: str) -> tuple[str, Job]:
        """Download an app's new version and restart it on it (only it: a restart that would move
        other instances is left to the user)."""
        return self.update_images([iid])

    def image_updatable(self) -> list[str]:
        """Instances with a new version to download, or downloaded and waiting for a restart."""
        return [iid for iid, st in sorted(self.image_status().items()) if st.state in ("available", "restart")]

    def update_images(self, iids: list[str] | None = None) -> tuple[str, Job]:
        """Update several apps in one change (all that have a new version when `iids` is None).
        The downloads run side by side, since they don't touch the GPUs; the restarts then go one
        at a time, each waiting for its own running work, so no two of them swap GPUs at once."""
        status = self.image_status()
        if iids is None:
            iids = self.image_updatable()
            if not iids:
                raise Problem("Every app already runs the newest version GPU Shepherd knows of. "
                              "Check for updates first if a release just came out.", "info", 409)
        todo = {}
        for iid in dict.fromkeys(iids):
            inst = self.instance(iid)
            st = status.get(iid)
            if not st or st.state not in ("available", "restart"):
                raise Problem(f"{inst.name} already runs the newest version GPU Shepherd knows of. "
                              "Check for updates first if a release just came out.", "info", 409)
            todo[iid] = st.state == "available"
        names = self.names()
        title = (f"Updating {names[iids[0]]}" if len(todo) == 1
                 else f"Updating {len(todo)} apps")
        try:
            job = self.jobs.submit(title, lambda job: self._update_images(todo, job))
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return f"{title}…", job

    def _update_images(self, todo: dict[str, bool], job: Job) -> None:
        """todo: instance -> whether its image needs downloading first."""
        instances = self.store.instances()[0]
        states = self.backend.states()
        old = {}
        for iid in todo:
            st = states.get(iid)
            image = instances[iid].image
            old[iid] = (st.image_id if st else "") or self.backend.image_info([image]).get(image, ("", []))[0]
        failed: dict[str, str] = {}
        images = sorted({instances[iid].image for iid, pull in todo.items() if pull})

        def pull(image: str) -> None:
            job(f"Downloading the new version of {image}")
            try:
                self.backend.pull(image, job)
                new, local = self.backend.image_info([image]).get(image, ("", []))
                self.image_log.downloaded(image, new, image_updates.digests(local))
                job(f"Downloaded {image}")
            except BackendError as err:
                job(f"Could not download {image}: {err}")
                for iid, p in todo.items():
                    if p and instances[iid].image == image:
                        failed[iid] = f"could not download {image}"

        if len(images) > 1:
            with ThreadPoolExecutor(max_workers=min(3, len(images)), thread_name_prefix="pull") as pool:
                list(pool.map(pull, images))
        elif images:
            pull(images[0])
        names = self.names()
        for iid in todo:
            if iid in failed:
                continue
            inst = instances[iid]
            if len(todo) > 1:
                job(f"— {inst.name}")
            state = self.backend.states().get(iid)
            if not (state and state.running):
                job(f"{inst.name} is stopped: it runs the new version from its next start.")
                self.image_log.updated(iid, old[iid])
                continue
            p = self.engine.plan("restart", iid)
            if p.error or p.side_effects:
                job(f"Not restarted: {p.error or 'that would also stop or move other instances'}. "
                    f"Restart {inst.name} when it suits you to use the new version.")
                continue
            try:
                self.engine.execute(p, job, job)
            except Cancelled:
                raise
            except Exception as err:  # noqa: BLE001 - the other apps still get their update
                if len(todo) == 1:
                    raise
                job(f"Could not restart {inst.name}: {err}")
                failed[iid] = f"could not restart: {err}"
                continue
            self.image_log.updated(iid, old[iid])
            new = self.backend.states().get(iid)
            new_id = new.image_id if new else ""
            if old[iid] and new_id and old[iid] != new_id:   # the old version, unless something else still uses it
                try:
                    self.backend.remove_image(old[iid])
                    job("Removed the old version")
                except BackendError:
                    pass
        if failed:
            raise RuntimeError("; ".join(f"{names.get(i, i)} {why}" for i, why in failed.items()))

    def auto_update(self) -> None:
        """Check when it is time, then update every app that updates by itself in one change
        (downloads side by side, restarts one at a time)."""
        hours = self.store.settings().image_check_hours
        if hours and time.time() - self.image_checked() >= hours * 3600:
            log.info("checking for new app versions: %s", self.check_images())
        if self.jobs.busy:
            return
        instances = self.store.instances()[0]
        data = self.image_log.read()["images"]
        picked = []
        for iid, st in self.image_status().items():
            inst = instances.get(iid)
            if not (inst and inst.auto_update and st.state in ("available", "restart")):
                continue
            rec = data.get(inst.image) or {}
            tried = (iid, rec.get("remote", "") if st.available else rec.get("id", ""))
            if tried in self._auto_tried:     # tried this version already: the card says what is left
                continue
            if not st.available:
                p = self.engine.plan("restart", iid)
                if p.error or p.side_effects:
                    continue
            self._auto_tried.add(tried)
            picked.append(iid)
        if picked:
            try:
                self.update_images(picked)
            except Problem as err:
                log.info("automatic update of %s skipped: %s", ", ".join(picked), err.msg)

    def watch_images(self, interval: float = 300.0, first: float = 10.0) -> None:
        def loop():
            threading.Event().wait(first)
            while True:
                try:
                    self.auto_update()
                except Exception as err:  # noqa: BLE001 - watching must never stop
                    log.warning("app update check failed: %s", getattr(err, "msg", err))
                threading.Event().wait(interval)
        threading.Thread(target=loop, daemon=True, name="images").start()

    @checked
    def set_image_check_hours(self, hours: int) -> str:
        s = self.store.settings()
        s.image_check_hours = hours
        self.store.save_settings(Settings.model_validate(s.model_dump()))
        return (f"Looking for new app versions every {hours} hour{'s' if hours != 1 else ''}." if hours
                else "New app versions are only looked for when you press Check now.")

    # ------------------------------------------------------------------ app logins
    def _login_method(self, iid: str) -> tuple[Instance, logins.Method]:
        inst = self.instance(iid)
        m = logins.method(inst)
        if m is None:
            raise Problem(f"GPU Shepherd doesn't know how {inst.name} keeps its logins.", status=404)
        return inst, m

    def _running(self, iid: str) -> bool:
        st = self.backend.states().get(iid)
        return bool(st and st.running)

    def _login_script(self, inst: Instance, m: logins.Method, user: str = "", password: str = "") -> str:
        ex = logins.script(m, user, password)
        try:
            code, out = self.backend.exec(self.engine.container_name(inst.id), ex.cmd, ex.env, ex.workdir)
        except BackendError as err:
            raise Problem(f"couldn't reach {inst.name}: {err}", status=409) from err
        if code:
            raise Problem(f"{inst.name} couldn't {'set the password' if user else 'list its accounts'}: "
                          f"{logins.last_line(out)}", status=409)
        return out

    @checked
    def logins(self, iid: str) -> dict:
        """An app's logins: its accounts, with their passwords where GPU Shepherd can read them."""
        inst, m = self._login_method(iid)
        path = self._settings_file(inst) if m.app == "anythingllm" else None
        settings = path.read_text(encoding="utf-8") if path else None
        api_key = self.store.catalog(iid).api_key if m.app == "llm-server" else ""
        accounts, running, multi = logins.readable(m, inst, settings, api_key), self._running(iid), False
        if m.in_container and running:
            listed, multi = logins.parse_accounts(self._login_script(inst, m))
            if m.app != "anythingllm" or multi:
                accounts = listed
        status = ""
        if m.in_container and not running and m.app != "anythingllm":
            status = f"Start {inst.name} to see the accounts in {m.where}."
        elif not accounts:
            status = (f"{inst.name} has no password yet." if m.app in ("anythingllm", *logins.ENV_LOGINS)
                      else f"No accounts yet: the first one is made in {inst.name} itself.")
        return {"instance": iid, "name": inst.name, "where": m.where, "note": m.note, "running": running,
                "multi_user": multi, "status": status, "accounts": [asdict(a) for a in accounts],
                "can_reset": m.app != "llm-server" and (not m.in_container or running or m.app == "anythingllm")}

    @checked
    def reset_login(self, iid: str, user: str = "", password: str = "") -> tuple[str, str]:
        """Give an app account a new password (a made-up one when none is given). Returns what to
        tell the user and the password."""
        inst, m = self._login_method(iid)
        if m.app == "llm-server":
            raise Problem(f"{inst.name}'s password is its API key: give it a new one with New API key in its "
                          "Models page, Server settings.", status=409)
        password = password.strip() or secrets.token_urlsafe(12)
        logins.check_password(password)
        user = user.strip()
        running = self._running(iid)
        if m.app in logins.ENV_LOGINS:
            user_var, password_var = tpl.get(m.app).login
            inst.env = inst.env | ({user_var: user or inst.env.get(user_var) or "admin"} if user_var else {}) \
                | {password_var: password}
            self.store.save_instance(inst)
            return self._restart_for_login(inst, running), password
        multi = False
        if m.app == "anythingllm" and running:
            multi = logins.parse_accounts(self._login_script(inst, m))[1]
        if m.app == "anythingllm" and not multi:
            path = self._settings_file(inst)
            if not path:
                raise Problem(f"can't find {inst.name}'s {app_updates.ANYTHINGLLM_FILE}", status=409)
            text = logins.set_env_value(path.read_text(encoding="utf-8"), "AUTH_TOKEN", password)
            if not logins.env_value(text, "JWT_SECRET"):            # it signs logins with this
                text = logins.set_env_value(text, "JWT_SECRET", secrets.token_hex(32))
            path.write_text(text, encoding="utf-8")
            return self._restart_for_login(inst, running), password
        if not user:
            raise Problem("Which account? Give its user name or email.")
        if not running:
            raise Problem(f"Start {inst.name} first: its accounts are in {m.where}.", status=409)
        if not logins.parse_updated(self._login_script(inst, m, user, password)):
            raise Problem(f"{inst.name} has no account '{user}'.", status=404)
        return f"Set a new password for {user} on {inst.name}.", password

    def _restart_for_login(self, inst: Instance, running: bool) -> str:
        if not running:
            return f"Saved {inst.name}'s new password: it is used from its next start."
        try:
            self.change("restart", inst.id, confirm=True)
        except Problem:
            return f"Saved {inst.name}'s new password. Restart it to use it."
        return f"Saved {inst.name}'s new password and restarting it."

    # ------------------------------------------------------------------ model catalogue
    @checked
    def server(self, iid: str) -> tuple[Instance, Catalog]:
        inst = self.store.instance(iid)
        if inst.kind != "llama-swap":
            raise ConfigError(f"{inst.name} is not a model server")
        return inst, self.store.catalog(iid)

    def loaded(self, iid: str, cat: Catalog) -> tuple[dict[str, str], str]:
        """The models a running model server has loaded (id -> state), or why that's unknown."""
        st = self.backend.states().get(iid)
        if not (st and st.running):
            return {}, ""
        try:
            return self.llama_client(iid).running({m.id for m in cat.models}), ""
        except Exception:  # noqa: BLE001
            return {}, "The model server is not answering yet (it may still be starting)."

    # ------------------------------------------------------------------ tuning (tuning.py)
    def measure(self) -> None:
        """Note how much GPU memory and speed the running model servers' models use (called every
        minute or so). A model's memory is only counted while nothing else shares its GPUs."""
        states = self.backend.states()
        gpus = {g.index: g for g in gpu_mod.query()[0]}
        for iid, inst in self.store.instances()[0].items():
            st = states.get(iid)
            if inst.kind != "llama-swap" or not (st and st.running and st.gpus):
                continue
            try:
                cat = self.store.catalog(iid)
                client = self.llama_client(iid)
                loaded = {m for m, s in client.running({m.id for m in cat.models}).items() if s == "ready"}
                placed = self.engine.placement(inst, cat, st.gpus)
                on = {m.id: [st.gpus[p] for p in placed.get(m.id, range(len(st.gpus)))] for m in cat.models
                      if m.id in loaded}
                for mid, its in on.items():
                    if any(set(its) & set(o) for other, o in on.items() if other != mid):
                        continue
                    used = sum(int(gpus[g].mem_used * 2**20) for g in its if g in gpus)
                    if used:
                        self.measurements.record(iid, mid, len(its), vram=used)
                speeds, newest = tuning.metrics_speeds(client.metrics(), self._metrics_seen.get(iid, -1.0))
                self._metrics_seen[iid] = newest
                for mid, tps in speeds:
                    if mid in {m.id for m in cat.models}:
                        k = len(placed.get(mid, [])) or len(st.gpus)
                        self.measurements.record(iid, mid, k, tps=tps)
                if cat.auto_tune:
                    for s in self.tuning(iid):
                        if s["status"] == "new":
                            self.tuning_decide(iid, s["key"], "apply")
            except Exception as err:  # noqa: BLE001 - measuring must never get in the way
                log.debug("measuring %s failed: %s", iid, err)

    def start_measuring(self, interval: float = 60.0) -> None:
        def loop():
            while True:
                threading.Event().wait(interval)
                self.measure()
        threading.Thread(target=loop, daemon=True, name="measure").start()

    def _suggestions(self, iid: str) -> tuple[Instance, Catalog, list[tuning.Suggestion]]:
        inst, cat = self.server(iid)
        gpus = sorted((g for g in gpu_mod.query()[0] if g.vendor == inst.gpu.vendor), key=lambda g: g.index)
        if inst.gpu.mode == "fixed":
            gpus = [g for g in gpus if g.index in inst.gpu.gpus]
        elif inst.gpu.max_gpus:
            gpus = gpus[:inst.gpu.max_gpus]
        mem = [int(g.mem_total * 2**20) for g in gpus]
        current = tuning.suggest(cat, self.measurements.get(iid), self.engine.model_needs(inst, cat), mem)
        return inst, cat, tuning.listing(current, self.decisions.for_server(iid))

    @checked
    def tuning(self, iid: str) -> list[dict]:
        """Suggested adjustments for a model server: new ones, applied ones (with what they replaced,
        to undo) and dismissed ones (to apply after all)."""
        return [s.as_dict() for s in self._suggestions(iid)[2]]

    @checked
    def tuning_decide(self, iid: str, key: str, action: str) -> str:
        """apply, dismiss or undo a suggestion."""
        if action not in ("apply", "dismiss", "undo"):
            raise Problem(f"unknown action '{action}': use apply, dismiss or undo")
        inst, cat, listed = self._suggestions(iid)
        s = next((x for x in listed if x.key == key), None)
        if s is None:
            raise Problem(f"no suggestion '{key}' (it may no longer apply)", status=404)
        if action == "dismiss":
            self.decisions.put(s, "dismissed")
            return f"Dismissed: {s.title}. It stays listed, so you can apply it later."
        if action == "undo" and s.status != "applied":
            raise Problem("only an applied suggestion can be undone")
        value = s.old if action == "undo" else s.value
        if action == "apply":
            s.old = cat.fit_gpus if s.field == "fit_gpus" else next(
                (m.gpus for m in cat.models if m.id == s.model), s.old)
        if s.field == "fit_gpus":
            cat.fit_gpus = value
        else:
            model = next((m for m in cat.models if m.id == s.model), None)
            if model is None:
                raise Problem(f"{s.model} is no longer on {inst.name}", status=404)
            model.gpus = int(value or 0)
        self.store.save_catalog(Catalog.model_validate(cat.model_dump()))
        self.decisions.put(s, "dismissed" if action == "undo" else "applied")
        st = self.backend.states().get(iid)
        later = " It takes effect when the server restarts." if st and st.running else ""
        return (f"Undone: {s.title}. It stays listed, so you can apply it again." if action == "undo"
                else f"Applied: {s.title}.") + later

    def placement(self, iid: str, cat: Catalog) -> dict[str, list[int]]:
        """For a running model server: model id -> the GPUs (machine numbering) it runs on, for the
        models that use fewer than all of the server's GPUs."""
        st = self.backend.states().get(iid)
        if not (st and st.running and st.gpus):
            return {}
        try:
            inst = self.store.instance(iid)
            return {mid: [st.gpus[p] for p in pos]
                    for mid, pos in self.engine.placement(inst, cat, st.gpus).items()}
        except Exception:  # noqa: BLE001 - only shown
            return {}

    @checked
    def save_model(self, iid: str, model: CatalogModel, original: str = "", check: bool = True) -> str:
        """Add a model, or replace `original` (renaming it when the ID changed; auto names follow).
        A model set up by hand is checked against this machine first (see check_model)."""
        inst, cat = self.server(iid)
        model = CatalogModel.model_validate(model.model_dump())
        before = next((m for m in cat.models if m.id == (original or model.id)), None)
        notes = self.check_model(inst, model, new_file=not before or before.file != model.file) if check else []
        if not original and any(m.id == model.id for m in cat.models):
            original = model.id
        elif original and not any(m.id == original for m in cat.models):
            raise Problem(f"no model '{original}'", status=404)
        models = [m for m in cat.models if m.id != (original or model.id)]
        if original and original != model.id:
            for auto in cat.auto_aliases:        # follow a rename
                if auto.default_model == original:
                    auto.default_model = model.id
                for r in auto.rules:
                    if r.model == original:
                        r.model = model.id
        pos = next((n for n, m in enumerate(cat.models) if m.id == original), len(models))
        models.insert(min(pos, len(models)), model)
        cat = Catalog.model_validate({**cat.model_dump(), "models": [m.model_dump() for m in models],
                                      "auto_aliases": [a.model_dump() for a in cat.auto_aliases]})
        self.store.save_catalog(cat)
        return " ".join([f"Saved {model.id}.", *notes])

    def model_resources(self, iid: str) -> dict[str, dict]:
        """For each model in the catalogue: its size on disk, the memory it needs to run whole
        (weights plus its context's cache) and how that fits this machine's GPUs and RAM."""
        inst, cat = self.server(iid)
        m = self.machine()
        gpus = sorted(int(g * 2**30) for g in m.gpu_mem_gb)
        ram = int(m.ram_gb * 2**30)
        out = {}
        for x in cat.models:
            disk, need = budget.model_size(inst, x), budget.model_memory(inst, x)
            if not disk:
                out[x.id] = {"disk_gb": 0, "need_gb": 0, "ok": True, "text": "File not found, size unknown"}
                continue
            gb = lambda b: round(b / 2**30, 1)  # noqa: E731
            if not gpus:
                text, ok = f"Needs about {gb(need)} GB of GPU memory", True
            else:
                n = next((n for n in range(1, len(gpus) + 1) if sum(gpus[:n]) >= need), 0)
                if n:
                    text, ok = f"Fits on {'one GPU' if n == 1 else f'{n} GPUs'}", True
                else:
                    over = need - sum(gpus)
                    ok = not ram or over <= ram * 0.8
                    text = (f"All GPUs + about {gb(over)} GB of RAM" if ok
                            else f"Too big: needs about {gb(over)} GB of RAM beside your GPUs")
            out[x.id] = {"disk_gb": gb(disk), "need_gb": gb(need), "ok": ok, "text": text}
        return out

    def check_model(self, inst: Instance, model: CatalogModel, new_file: bool = True) -> list[str]:
        """Warnings for a model added or changed by hand: its file isn't there, it is bigger than the
        GPUs, or it has a setting that breaks llama.cpp's own GPU split. A new file too big for the
        GPUs and RAM together is refused, as it can never load."""
        m = self.machine()
        gpu_mem = [int(g * 2**30) for g in m.gpu_mem_gb]
        size = budget.model_size(inst, model)
        cpu_moe = any(r.cpu_moe and (r.min_gpus > 1 or len(gpu_mem) > 1) for r in model.offload)
        refusal, notes = budget.model_check(model.id, size, gpu_mem, int(m.ram_gb * 2**30), cpu_moe)
        if refusal and new_file:
            raise Problem(refusal, "warn", 422)
        if not budget.file_size(inst, model.file) and _volume_visible(inst, model.file) \
                and str(budget.host_path(inst, model.file)) not in self._downloading({inst.id: inst}):
            notes.insert(0, f"{model.file} isn't in the models folder yet, so {model.id} can't load until it is.")
        return notes

    def delete_model(self, iid: str, mid: str, files: bool = False) -> str:
        """Take a model out of the catalogue. With `files`, also delete its .gguf files (and vision
        projector / MTP draft) from the models folder, except any another model still uses."""
        inst, cat = self.server(iid)
        model = next((m for m in cat.models if m.id == mid), None)
        if not model:
            raise Problem(f"no model '{mid}'", status=404)
        if files and mid in self.loaded(iid, cat)[0]:
            raise Problem(f"{mid} is loaded. Unload it first, then delete its files.", "warn", 409)
        try:
            new = Catalog.model_validate({**cat.model_dump(), "models": [m.model_dump() for m in cat.models if m.id != mid]})
        except ValidationError as err:
            raise Problem(f"Cannot delete {mid}: {friendly_errors(err)}", status=409) from err
        doomed, kept = (self._model_files_to_delete(inst, model, mid) if files else ([], []))
        self.store.save_catalog(new)
        if not files:
            return f"Deleted {mid}."
        try:
            for host, rels in doomed:
                self.downloads.delete_files(host, rels)
        except BackendError as err:
            raise Problem(f"Removed {mid} from the catalogue, but could not delete its files: {err}", status=500) from err
        n = sum(len(r) for _, r in doomed)
        text = f"Deleted {mid} and its {n} file{'s' if n != 1 else ''}." if n else f"Deleted {mid}."
        return text + (f" Kept {', '.join(kept)}: another model uses it." if kept else "")

    def _model_files_to_delete(self, inst: Instance, model: CatalogModel, mid: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
        """([(models folder, [paths inside it])], files kept) for deleting `model`'s files: only
        files in a folder mounted into the server, and none another model (on any server) uses."""
        used: set[str] = set()
        for other, ocat in ((i, self.store.catalog(i.id)) for i in self.store.instances()[0].values() if i.kind == "llama-swap"):
            for m in ocat.models:
                if other.id == inst.id and m.id == mid:
                    continue
                used |= {str(p) for f in (m.file, m.mmproj, m.mtp_draft) if f and (p := budget.host_path(other, f))}
        by_host: dict[str, list[str]] = {}
        kept = []
        for f in dict.fromkeys(x for x in (model.file, model.mmproj, model.mtp_draft) if x):
            p = budget.host_path(inst, f)
            vol = next((v for v in inst.volumes if f.startswith(v.container.rstrip("/") + "/")), None)
            rel = f[len(vol.container.rstrip("/")) + 1:] if vol else ""
            if not p or not vol or ".." in Path(rel).parts or not rel.endswith(".gguf"):
                continue
            if str(p) in used:
                kept.append(Path(f).name)
            else:
                by_host.setdefault(vol.host, []).append(rel)
        return list(by_host.items()), kept

    # ------------------------------------------------------------------ suggested models
    @staticmethod
    def models_dir(inst: Instance) -> str:
        """The models folder on this machine that a model server sees at /models ("" if none)."""
        return next((v.host for v in inst.volumes if v.container == tpl.MODELS_IN_CONTAINER), "")

    def suggestions(self, iid: str) -> list[dict]:
        """The suggested models, what runs best on this machine first, with whether each is
        downloaded and already in this server's catalogue."""
        inst, cat = self.server(iid)
        m = self.machine()
        folder = self.models_dir(inst)
        files = {x.file for x in cat.models}
        out = []
        linked = self._linked()
        hf = self.hf_picks(m.gpu_mem_gb, m.ram_gb)
        helpers = self.downloads.helpers()        # one Docker listing for the whole list
        for s in model_library.SUGGESTIONS + linked + hf:
            f = model_library.fit(s, m.gpu_mem_gb, m.ram_gb)
            st = self.downloads.status(s, folder, helpers)
            out.append({"key": s.key, "name": s.name, "blurb": s.blurb, "uses": list(s.uses), "page": s.page,
                        "vision": s.vision, "moe": s.moe, "mtp": bool(s.mtp), "download_gb": s.download_gb,
                        "fits": f.ok, "fit": f.text, "rank": f.rank, "state": st.state, "status": st.text,
                        "pct": st.pct, "partial": st.partial, "linked": s in linked, "hf": s in hf, "license": s.license,
                        "in_catalog": f"{tpl.MODELS_IN_CONTAINER}/{s.path(s.model)}" in files,
                        "files": [{"path": f"{tpl.MODELS_IN_CONTAINER}/{s.path(x)}", "url": x.url, "gb": x.gb}
                                  for x in s.files]})
        return sorted(out, key=lambda x: x.pop("rank"))

    def _suggestion(self, key: str) -> model_library.Suggestion:
        s = model_library.get(key, self._extra())
        if not s:
            raise Problem(f"no suggested model '{key}'", status=404)
        return s

    def hf_picks(self, gpu_mem_gb: list[float], ram_gb: float) -> tuple[model_library.Suggestion, ...]:
        """Models from Hugging Face that run well on this machine (hf_models.py), and the ones kept
        because they were added or downloaded. With the setting off, only the kept ones."""
        models, kept = self.hf.state()
        if not self.store.settings().hf_suggestions:
            models = []
        return hf_models.picks(models, kept, gpu_mem_gb, ram_gb, model_library.SUGGESTIONS + self._linked())

    def _extra(self) -> tuple[model_library.Suggestion, ...]:
        """Every suggestion besides the built-in ones: added from a link, and from Hugging Face."""
        return self._linked() + self.hf_picks([g.mem_total / 1024 for g in gpu_mod.query()[0]], host.ram_gb())

    def hf_checked(self) -> float:
        return self.hf.checked

    def refresh_hf_models(self, wait: bool = False) -> str:
        """Ask Hugging Face for its trending models now (in the background unless `wait`)."""
        if not wait:
            threading.Thread(target=self.hf.refresh, daemon=True, name="hf-models-now").start()
            return "Asking Hugging Face; the suggested models update within a minute or two."
        n = self.hf.refresh()
        return (f"Found {n} models trending on Hugging Face." if n
                else "Could not reach Hugging Face; the last list (or the built-in picks) stays.")

    def watch_hf_models(self, interval: float = 600.0, first: float = 30.0) -> None:
        """Refresh the Hugging Face list about once a day, while the setting is on."""
        def loop():
            threading.Event().wait(first)
            while True:
                try:
                    if self.store.settings().hf_suggestions and self.hf.due():
                        self.hf.refresh()
                except Exception as err:  # noqa: BLE001 - watching must never stop
                    log.warning("Hugging Face model list: %s", err)
                threading.Event().wait(interval)
        threading.Thread(target=loop, daemon=True, name="hf-models").start()

    @checked
    def set_hf_suggestions(self, on: bool) -> str:
        s = self.store.settings()
        s.hf_suggestions = on
        self.store.save_settings(Settings.model_validate(s.model_dump()))
        if on and self.hf.due():
            threading.Thread(target=self.hf.refresh, daemon=True, name="hf-models-now").start()
        return ("Suggesting models trending on Hugging Face that fit this machine." if on
                else "Only the built-in model suggestions are shown now (and Hugging Face models you added).")

    def _linked(self) -> tuple[model_library.Suggestion, ...]:
        """The models added from a link, set up like suggested ones."""
        out = []
        for x in self.store.linked_models():
            try:
                out.append(model_library.from_link(str(x["url"]), str(x.get("name", "")), float(x.get("gb") or 0),
                                                   str(x.get("mmproj_url", "")), float(x.get("mmproj_gb") or 0),
                                                   key=str(x.get("key", ""))))
            except (ValueError, TypeError):
                log.warning("linked-models.yaml: skipping %s", x.get("url"))
        return tuple(out)

    def add_from_link(self, iid: str, url: str, name: str = "", mmproj_url: str = "") -> str:
        """Add a model by the download link of its .gguf file (and optionally its vision projector's):
        it is listed with the suggested models, added to this server's catalogue and downloaded."""
        try:
            s = model_library.from_link(url, name, mmproj_url=mmproj_url)
        except ValueError as err:
            raise Problem(str(err)) from err
        rows = self.store.linked_models()
        same = next((x for x in rows if x.get("url") == s.model.url), None)
        if same:
            key = str(same.get("key"))       # the same link again
        else:
            taken = {x.key for x in model_library.SUGGESTIONS + self._extra()}
            key = tpl.free_id(s.key, taken)
            sizes = (_remote_gb(s.model.url), _remote_gb(s.mmproj.url) if s.mmproj else 0)
            m = self.machine()
            refusal, _ = budget.model_check(s.name, int(sum(sizes) * 1e9), [int(g * 2**30) for g in m.gpu_mem_gb],
                                            int(m.ram_gb * 2**30))
            if refusal:
                raise Problem(refusal, "warn", 422)
            rows.append({"key": key, "name": s.name, "url": s.model.url, "gb": sizes[0],
                         "mmproj_url": s.mmproj.url if s.mmproj else "", "mmproj_gb": sizes[1]})
            self.store.save_linked_models(rows)
        return self.add_suggestion(iid, key)

    def forget_link(self, iid: str, key: str) -> str:
        """Take a model added from a link off the list (its files and catalogue entry stay)."""
        self.server(iid)
        rows = self.store.linked_models()
        left = [x for x in rows if x.get("key") != key]
        if len(left) == len(rows):
            raise Problem(f"no model added from a link as '{key}'", status=404)
        self.store.save_linked_models(left)
        return "Taken off the list. Its files and catalogue entry are kept."

    def download_suggestion(self, iid: str, key: str) -> str:
        inst, _ = self.server(iid)
        s = self._suggestion(key)
        folder = self.models_dir(inst)
        if not folder:
            raise Problem(f"{inst.name} has no models folder (a volume mounted at {tpl.MODELS_IN_CONTAINER}): "
                          "add one under Edit first.")
        if self.downloads.status(s, folder).state == "ready":
            return f"{s.name} is already downloaded."
        try:
            msg = self.downloads.start(s, folder)
            self.hf.keep(s)
            return msg
        except ValueError as err:
            raise Problem(str(err), "warn", 507) from err

    def cancel_download(self, iid: str, key: str) -> str:
        self.server(iid)
        return self.downloads.cancel(self._suggestion(key))

    def discard_download(self, iid: str, key: str) -> str:
        inst, _ = self.server(iid)
        try:
            return self.downloads.discard(self._suggestion(key), self.models_dir(inst))
        except BackendError as err:
            raise Problem(f"Could not delete the partial files: {err}", status=500) from err

    def delete_download(self, iid: str, key: str) -> str:
        """Delete a downloaded suggested model's files, when no model in a catalogue uses them."""
        inst, _ = self.server(iid)
        s = self._suggestion(key)
        folder = self.models_dir(inst)
        if self.downloads.status(s, folder).state != "ready":
            raise Problem(f"{s.name} is not downloaded.", "info", 409)
        for other in (i for i in self.store.instances()[0].values() if i.kind == "llama-swap"):
            for m in self.store.catalog(other.id).models:
                if {m.file, m.mmproj, m.mtp_draft} & {f"{tpl.MODELS_IN_CONTAINER}/{s.path(x)}" for x in s.files}:
                    raise Problem(f"{m.id} on {other.name} uses it: delete it from the catalogue (with its file) instead.", "warn", 409)
        try:
            self.downloads.delete_files(folder, [s.path(x) for x in s.files])
        except BackendError as err:
            raise Problem(f"Could not delete the files: {err}", status=500) from err
        return f"Deleted {s.name} from disk."

    @checked
    def add_suggestion(self, iid: str, key: str) -> str:
        """Add a suggested model to the catalogue with its makers' settings, and download it if it
        isn't there yet. An embeddings model also gets the name "embed" when that is free."""
        inst, cat = self.server(iid)
        s = self._suggestion(key)
        m = self.machine()
        model = model_library.catalog_model(s, m.gpu_mem_gb, tpl.MODELS_IN_CONTAINER)
        if any(x.file == model.file for x in cat.models):
            raise Problem(f"{s.name} is already on {inst.name}.", "info", 409)
        taken = {x.id for x in cat.models} | {a for x in cat.models for a in x.aliases} | \
            {a.alias for a in cat.auto_aliases}
        model.id = tpl.free_id(model.id, taken)
        if s.role == "embed" and "embed" not in taken:
            model.aliases = ["embed"]
        self.save_model(iid, model, check=False)
        self.hf.keep(s)
        fetching = self.fetch_suggested(inst, [model])
        st = self.backend.states().get(iid)
        note = " Restart the server to use it." if st and st.running else ""
        return f"Added {model.id}" + (f"; downloading it ({s.download_gb:.1f} GB)." if fetching else ".") + note

    def start_failures(self, iid: str) -> dict[str, str]:
        """Model id -> why it last failed to start on this running server (from its log), for
        loads asked for from the Models page and from chat apps alike."""
        inst, cat = self.server(iid)
        st = self.backend.states().get(iid)
        if not (st and st.running):
            return {}
        text = self.backend.logs(self.engine.container_name(iid), tail=3000)
        return llamaswap.start_failures(text, {m.id for m in cat.models})

    def missing_files(self, iid: str) -> dict[str, list[str]]:
        """Model id -> its files (paths inside the container) that aren't on disk. llama-server quits
        straight away without them, which chat apps only show as "upstream command exited
        prematurely". Files still downloading, and folders GPU Shepherd can't see, aren't counted."""
        inst, cat = self.server(iid)
        downloading = self._downloading({iid: inst})
        missing: dict[str, list[str]] = {}
        for m in cat.models:
            for f in (m.file, m.mmproj, m.mtp_draft):
                p = budget.host_path(inst, f) if f else None
                if p is None or str(p) in downloading or not _volume_visible(inst, f):   # both are paths on this machine
                    continue
                if not p.is_file():
                    missing.setdefault(m.id, []).append(f)
        return missing

    def load_model(self, iid: str, mid: str) -> str:
        """Load a model in the background (large ones take minutes from disk)."""
        _, cat = self.server(iid)
        if mid not in {m.id for m in cat.models}:
            raise Problem(f"unknown model {mid}", status=404)
        if gone := self.missing_files(iid).get(mid):
            raise Problem(f"{mid} can't load: {', '.join(gone)} is not in the models folder. Download it, "
                          "or fix the path with Edit.", "warn", 409)
        client = self.llama_client(iid)
        with _LOADING:                        # two clicks at once start one load, not two
            busy = self.loading.setdefault(iid, set())
            if mid in busy:
                raise Problem(f"{mid} is already loading.", "info", 409)
            busy.add(mid)

        def worker():
            try:
                client.load(mid)
            except Exception as err:  # noqa: BLE001
                log.warning("loading %s failed: %s", mid, err)
            finally:
                busy.discard(mid)
        threading.Thread(target=worker, daemon=True).start()
        return f"Loading {mid} - large models take a few minutes from disk."

    def unload_model(self, iid: str, mid: str = "") -> str:
        """Unload one model, or every model with mid "" (or "_all")."""
        _, cat = self.server(iid)
        mid = "" if mid == "_all" else mid
        if mid and mid not in {m.id for m in cat.models}:
            raise Problem(f"unknown model {mid}", status=404)
        # llama-swap cuts off the requests a model is answering when it is unloaded
        act = self.engine.activity(self.store.instance(iid))
        if act.busy:
            raise Problem(f"Still answering {act.detail}. Unloading now would cut that off: try again when it "
                          "finishes, or use Stop on the dashboard, which waits for it.", "warn", 409)
        try:
            self.llama_client(iid).unload(mid or None)
        except Exception as err:  # noqa: BLE001
            raise Problem(str(err), status=502) from err
        return "Model unloaded."

    @checked
    def server_settings(self, iid: str, threads: int | None = None, llama_server: str | None = None,
                        health_timeout: int | None = None, new_key: bool = False,
                        fit_gpus: bool | None = None, auto_tune: bool | None = None) -> str:
        """The model server's own settings; None leaves a setting as it is."""
        _, cat = self.server(iid)
        if threads is not None:
            cat.threads = int(threads)
        if llama_server is not None:
            cat.llama_server = str(llama_server).strip()
        if health_timeout is not None:
            cat.health_timeout = int(health_timeout)
        if fit_gpus is not None:
            cat.fit_gpus = bool(fit_gpus)
        if auto_tune is not None:
            cat.auto_tune = bool(auto_tune)
        if new_key:
            cat.api_key = "sk-" + secrets.token_hex(20)
        self.store.save_catalog(Catalog.model_validate(cat.model_dump()))
        return "Settings saved."

    @checked
    def set_auto_aliases(self, iid: str, autos: list[AutoAlias]) -> str:
        """Replace all of a model server's auto names."""
        _, cat = self.server(iid)
        cat = Catalog.model_validate({**cat.model_dump(), "auto_aliases": [a.model_dump() for a in autos]})
        self.store.save_catalog(cat)
        return "Auto names saved."

    # ------------------------------------------------------------------ settings
    def set_containers(self, bind_address: str | None = None, network: str | None = None) -> str:
        s = self.store.settings()
        bind = s.bind_address if bind_address is None else str(bind_address).strip()
        net = s.network if network is None else str(network).strip()
        try:
            ipaddress.ip_address(bind)
        except ValueError as err:
            raise Problem("Bind address must be an IP address, e.g. 0.0.0.0") from err
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$", net):
            raise Problem("Invalid Docker network name.")
        s.bind_address, s.network = bind, net
        self.store.save_settings(s)
        return "Saved. Restart instances to apply."

    def change_password(self, current: str, new: str, again: str | None = None) -> str:
        s = self.store.settings()
        if not auth.verify_password(current, s.password_hash):
            raise Problem("The current password is wrong.", status=403)
        if len(new) < 10:
            raise Problem("Use at least 10 characters.")
        if again is not None and new != again:
            raise Problem("The new passwords do not match.")
        s.password_hash = auth.hash_password(new)
        self.store.save_settings(s)
        auth.initial_password_file(self.store).unlink(missing_ok=True)
        return "Password changed."

    # ------------------------------------------------------------------ updates (done by update.sh on the host)
    def request_update(self, action: str) -> str:
        if action not in updates.ACTIONS:
            raise Problem("unknown action")
        u = updates.status(self.store.root)
        if not u.dashboard_updates:
            raise Problem(f"Updating from the dashboard is off. On the server run: {u.command}", "warn", 409)
        if u.busy:
            raise Problem("An update or check is already running.", "warn", 409)
        if action == "update" and self.jobs.busy:
            raise Problem("Wait for the current change to finish, then update.", "warn", 409)
        if action == "update" and u.local_changes:
            raise Problem(f"Files in the clone were changed by hand. Run {u.command} --stash on the server.", status=409)
        updates.request(self.store.root, action)
        log.info("update: %s requested", action)
        return "Checking for updates…" if action == "check" else "Updating…"

    def switch_branch(self, name: str) -> str:
        """Follow another branch of the repository (to try changes before they are released)."""
        name = name.strip()
        u = updates.status(self.store.root)
        if not u.dashboard_updates:
            raise Problem(f"Switching branches from the dashboard needs the update service. On the server run: {u.command} --branch {name or 'NAME'}",
                          "warn", 409)
        if u.busy:
            raise Problem("An update or check is already running.", "warn", 409)
        if self.jobs.busy:
            raise Problem("Wait for the current change to finish, then switch.", "warn", 409)
        if u.local_changes:
            raise Problem(f"Files in the clone were changed by hand. Run {u.command} --stash on the server.", status=409)
        if name == u.branch:
            raise Problem(f"Already following {name}.", "info", 409)
        if name not in u.branches:
            raise Problem(f"There is no branch '{name}' on the remote (check for updates to refresh the list).", status=404)
        try:
            updates.request_branch(self.store.root, name)
        except ValueError as err:
            raise Problem(str(err)) from err
        log.info("update: switch to branch %s requested", name)
        return f"Switching to {name} and rebuilding…"

    def update_credentials(self, user: str = "", token: str = "", clear: bool = False) -> str:
        u = updates.status(self.store.root)
        if not u.dashboard_updates:
            raise Problem(f"Saving a token needs the update service. On the server run: {u.command} --set-token <user>",
                          "warn", 409)
        if u.busy:
            raise Problem("An update or check is running; try again when it is done.", "warn", 409)
        try:
            updates.request_credentials(self.store.root, user.strip(), token.strip(), clear=clear)
        except ValueError as err:
            raise Problem(str(err)) from err
        log.info("update: GitHub token %s", "removal requested" if clear else "handed to the host")
        return "Token removed." if clear else "Token saved; checking for updates with it."

    # ------------------------------------------------------------------ power (the top menu)
    def power_status(self) -> dict:
        """What the Power dialog can do: restart GPU Shepherd itself (always), and restart or shut
        down the server (through update.sh's service on the host, like updates)."""
        u = updates.status(self.store.root)
        instances, _ = self.store.instances()
        running = sorted(instances[i].name for i in self._all_running() if i in instances)
        downloading = sorted(n for n, h in self.backend.helpers(DOWNLOAD_KIND).items() if h.running)
        if u.image:
            hint = "This install has no update service, so the server can't be restarted from here. On the server run: sudo reboot"
        elif not u.dashboard_updates:
            hint = (f"Restarting the server from here needs the update service. On the server run: "
                    f"{u.command} --enable-dashboard-updates (or just sudo reboot)")
        else:
            hint = ""
        return {"host": not hint, "hint": hint, "busy": u.busy, "change": self.jobs.busy,
                "knowledge": bool(self.knowledge_task and self.knowledge_task.running),
                "running": running, "downloads": len(downloading)}

    def power(self, action: str) -> str:
        """restart: GPU Shepherd only (the apps keep running); reboot / shutdown: the whole server."""
        if action not in POWER_ACTIONS:
            raise Problem("unknown action")
        st = self.power_status()
        if st["change"]:
            raise Problem("Wait for the current change to finish (or cancel it) first.", "warn", 409)
        if st["knowledge"]:
            raise Problem(f"Wait for the current knowledge work to finish ({self.knowledge_task.title}), or stop it.", "warn", 409)
        if action == "restart":
            log.info("power: restarting GPU Shepherd")
            self.restarter()
            return "Restarting GPU Shepherd: the dashboard is back in a few seconds."
        if not st["host"]:
            raise Problem(st["hint"], "warn", 409)
        if st["busy"]:
            raise Problem("An update or check is running; wait for it to finish.", "warn", 409)
        updates.request(self.store.root, "reboot" if action == "reboot" else "poweroff")
        log.info("power: server %s requested", action)
        return ("Restarting the server: the dashboard is back once it has started again." if action == "reboot"
                else "Shutting down the server. Turn it back on to use GPU Shepherd again.")

    # ------------------------------------------------------------------ tensor unlock (gpu-unlock.sh on the host)
    def unlock_status(self) -> gpu_unlock.UnlockStatus:
        st = gpu_unlock.status(self.store.root)
        if not st.repo:
            st.repo = updates.status(self.store.root).repo
        return st

    def request_unlock(self, action: str, cards: list[str] | None = None) -> str:
        """Experimental: install gpu-unlock, choose the cards it unlocks from the next boot, or
        remove it. The host does the work (gpu-unlock.sh); every change needs a reboot."""
        if action not in gpu_unlock.ACTIONS:
            raise Problem("unknown action")
        st = self.unlock_status()
        if not st.service:
            raise Problem(f"The dashboard can't reach gpu-unlock.sh on the server yet. Turn on dashboard updates, or on the "
                          f"server run: {st.command} --install-service", "warn", 409)
        if st.busy:
            raise Problem("gpu-unlock is already being changed; wait for it to finish.", "warn", 409)
        if action in ("install", "set") and st.blocker:
            raise Problem(st.blocker, "warn", 409)
        if action == "install" and not st.supported and st.source:
            raise Problem("None of this machine's NVIDIA cards is supported by gpu-unlock (RTX 20, 30 and 40 are; RTX 50 isn't).",
                          "warn", 409)
        if action == "set":
            cards = [gpu_unlock.normalize_pci(c) for c in cards or [] if c.strip()]
            if cards and not st.installed:
                raise Problem("Install gpu-unlock first.", "warn", 409)
            known = {c.pci for c in st.supported}
            for c in cards:
                if c not in known:
                    raise Problem(f"{c} is not a card gpu-unlock supports on this machine.", status=404)
            if sorted(cards) == sorted(st.configured):
                raise Problem("That is already what the next boot does.", "info", 409)
        if action == "uninstall" and not st.installed and not st.configured:
            raise Problem("gpu-unlock is not installed.", "info", 409)
        try:
            gpu_unlock.request(self.store.root, action, cards)
        except ValueError as err:
            raise Problem(str(err)) from err
        log.info("gpu-unlock: %s %s requested", action, ",".join(cards or []))
        return {"check": "Checking this machine…",
                "install": "Installing gpu-unlock (this takes a few minutes)…",
                "set": (f"Saving: {len(cards or [])} card(s) unlocked from the next boot. Reboot to apply it." if cards
                        else "Saving: no card unlocked from the next boot. Reboot to put the cards back to normal."),
                "uninstall": "Removing gpu-unlock. Reboot afterwards to put the cards back to normal."}[action]

    # ------------------------------------------------------------------ Tailscale (on the host, through update.sh)
    def tailscale_status(self, refresh: bool = False) -> tailscale.TailscaleStatus:
        """This machine's Tailscale state as the host last reported it; with refresh, ask the host
        for a new report when the last one is missing or old (and nothing else is waiting)."""
        st = tailscale.status(self.store.root)
        if refresh and tailscale.needs_refresh(st) and not updates.status(self.store.root).pending:
            tailscale.request(self.store.root, "refresh")
            st = tailscale.status(self.store.root)
        return st

    def request_tailscale(self, action: str) -> str:
        """Install Tailscale and log this machine in to a tailnet, serve the dashboard over HTTPS on
        its Tailscale name, disconnect, or log out. update.sh's service on the host does the work."""
        if action not in tailscale.ACTIONS:
            raise Problem("unknown action")
        st = tailscale.status(self.store.root)
        if st.image:
            raise Problem("This install has no update service, so Tailscale can't be set up from here. Install Tailscale "
                          "on the server (https://tailscale.com/download) and run: sudo tailscale up", "warn", 409)
        if not st.service:
            raise Problem(f"Setting up Tailscale from here needs dashboard updates. On the server run: "
                          f"{st.command} --enable-dashboard-updates (or install it with: install.sh --tailscale)", "warn", 409)
        if st.wsl and action in ("install", "up"):
            raise Problem("In WSL2, install Tailscale for Windows instead: the apps answer on the Windows machine's "
                          "Tailscale address.", "warn", 409)
        if st.busy:
            raise Problem("Tailscale is already being changed; wait for it to finish.", "warn", 409)
        u = updates.status(self.store.root)
        if u.busy:
            raise Problem("An update or check is running; wait for it to finish.", "warn", 409)
        if action != "install" and action != "refresh" and not st.installed:
            raise Problem("Install Tailscale first.", "warn", 409)
        if action == "https-on" and st.checked and st.connected and not st.cert_domains:
            raise Problem("Turn on HTTPS certificates in Tailscale's admin console (DNS page) first, then try again.",
                          "warn", 409)
        tailscale.request(self.store.root, action)
        log.info("tailscale: %s requested", action)
        return {"refresh": "Asking the server…",
                "install": "Installing Tailscale; a login link appears here in a minute.",
                "up": "Connecting; if a login is needed, a link appears here.",
                "down": "Disconnecting from your tailnet…",
                "https-on": "Serving the dashboard over HTTPS…",
                "https-off": "Turning HTTPS off…",
                "logout": "Logging out of Tailscale…"}[action]

    # ------------------------------------------------------------------ knowledge
    @property
    def library(self) -> kn.Library:
        if self._library is None:
            self._library = kn.Library(self.store.root)
            for line in self._library.moved:
                log.info("knowledge: moved %s", line)
        return self._library

    def _book_apps(self, instances: dict[str, Instance]) -> tuple[list[Instance], list[Instance]]:
        """Open WebUI instances (documents go into their knowledge) and Hermes Agent instances (skills)."""
        apps = {t: [i for i in instances.values() if tpl.matches(tpl.get(t), i)] for t in ("open-webui", "hermes")}
        return apps["open-webui"], apps["hermes"]

    def knowledge(self, sizes: bool = True) -> dict:
        """The documents folder, the documents found in it, what is being done with them, where they
        can go and where it is all kept (with sizes, unless `sizes` is False)."""
        data = self.library.load()
        folder = self.library.folder(data)
        instances, _ = self.store.instances()
        webuis, hermes = self._book_apps(instances)
        m = self.machine()
        counts: dict[str, int] = {}
        rows = []
        reasons: dict[str, dict[str, int]] = {}
        for b in sorted(data["books"].values(), key=lambda b: (b.group.lower(), b.name.lower())):
            counts[b.status] = counts.get(b.status, 0) + 1
            rows.append(asdict(b) | {"name": b.name})
            if b.status in ("failed", "skipped") and b.reason:
                why = reasons.setdefault(b.status, {})
                why[b.reason] = why.get(b.reason, 0) + 1
        skills = []
        for slug, s in sorted((data.get("skills") or {}).items()):
            skills.append(s | {"slug": slug, "path": str(self._skill_dir(slug)),
                               "exists": (self._skill_dir(slug) / "SKILL.md").exists()})
        storage = []
        for what, path in (("Documents", folder), ("Converted text", self.library.text_dir),
                           ("Skill drafts", self.library.skills_dir)):
            files, size = kn.size(path) if sizes else (None, None)
            storage.append({"what": what, "path": str(path), "exists": path.is_dir(), "files": files, "bytes": size,
                            "size": gb(size) if size is not None else ""})
        try:
            free = gb(shutil.disk_usage(folder if folder.is_dir() else self.store.root).free)
        except OSError:
            free = ""
        return {"folder": str(folder), "folder_ok": folder.is_dir(), "default_folder": str(self.library.default_folder),
                "own_folder": folder == self.library.default_folder, "home": str(self.library.home),
                "storage": storage, "free": free, "ocr": kn.ocr_available(),
                "reasons": {k: sorted(v.items(), key=lambda x: (-x[1], x[0])) for k, v in reasons.items()},
                "scanned": data.get("scanned"), "counts": counts, "documents": rows,
                "words": sum(b.words for b in data["books"].values() if b.status == "ready"),
                "task": self.knowledge_task.as_dict() if self.knowledge_task else None,
                "webuis": [{"id": i.id, "name": i.name, "running": self._running(i.id)} for i in webuis],
                "hermes": [{"id": i.id, "name": i.name} for i in hermes],
                "servers": [{"id": k, "name": v["name"], "model": v.get("default_model", ""), "embed": v.get("embed", "")}
                            for k, v in m.servers.items()],
                "sends": data.get("sends") or [], "sync": data.get("sync"), "skills": skills, "upload_mb": kn.UPLOAD_MB,
                "zip_mb": kn.ZIP_MB}

    def _knowledge_busy(self) -> None:
        if self.knowledge_task and self.knowledge_task.running:
            raise Problem(f"Wait for the current knowledge work to finish ({self.knowledge_task.title}), or stop it.", "warn", 409)

    def set_documents_folder(self, path: str) -> str:
        self._knowledge_busy()
        path = path.strip()
        if path and not path.startswith("/"):
            raise Problem("Give the documents folder as a full path, like /srv/documents.")
        target = Path(path) if path else self.library.default_folder
        if path and not target.is_dir():
            raise Problem(f"GPU Shepherd can't see {path}. Mount it into GPU Shepherd's container at the same path "
                          "(like a models folder), or put the documents in its own documents folder.", status=404)
        self.library.update(lambda d: d.__setitem__("folder", path))
        return f"Documents folder: {target}. Scan it to find the documents."

    def document_files(self) -> dict:
        """The document files already in the documents folder, {path: size}, so an upload can skip them."""
        folder = self.library.folder()
        return {"folder": str(folder), "files": kn.files(folder) if folder.is_dir() else {}}

    def _book_dest(self, path: str, size: int | None) -> tuple[str, Path, int]:
        """Where an uploaded file goes, and the most bytes it may have."""
        try:
            rel = kn.safe_rel(path)
        except kn.BookError as err:
            raise Problem(str(err)) from err
        limit = (kn.ZIP_MB if rel.lower().endswith(".zip") else kn.UPLOAD_MB) << 20
        if size is not None and size > limit:
            raise self._too_big(rel, limit)
        folder = self.library.folder()
        if not folder.is_dir():
            if folder != self.library.default_folder:
                raise Problem(f"GPU Shepherd can't see the documents folder {folder}.", status=404)
            folder.mkdir(parents=True, exist_ok=True)
        if size and size > shutil.disk_usage(folder).free:
            raise Problem(f"Not enough free space in {folder} for {rel}.", status=507)
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        return rel, dest, limit

    @staticmethod
    def _too_big(rel: str, limit: int) -> Problem:
        return Problem(f"{rel} is over {limit >> 20} MB: copy it into the documents folder another way.", status=413)

    def _place_book(self, rel: str, dest: Path, tmp: Path) -> dict:
        """Put a received file in place; a zip is unpacked into a folder named after it, then removed."""
        if not rel.lower().endswith(".zip"):
            os.replace(tmp, dest)
            return {"added": [rel], "left_out": 0}
        into = dest.with_name(dest.name[:-4])
        try:
            saved, other = kn.unpack(tmp, into, Path(rel).name)
        except kn.BookError as err:
            raise Problem(str(err)) from err
        finally:
            tmp.unlink(missing_ok=True)
        base = rel[:-4]
        if not saved:
            raise Problem(f"{Path(rel).name} has no documents in it (EPUB, PDF, DOCX, TXT, Markdown, AZW3, MOBI or PRC).")
        return {"added": [f"{base}/{r}" for r in saved], "left_out": other}

    def save_document(self, path: str, f, size: int | None = None) -> dict:
        """Save one uploaded document or zip of documents (a file-like) at `path` in the documents folder,
        folders kept. Returns {"added": [paths], "left_out": n}. Doesn't scan."""
        rel, dest, limit = self._book_dest(path, size)
        tmp = dest.with_name(dest.name + ".part")
        try:
            n = 0
            with open(tmp, "wb") as out:
                while chunk := f.read(1 << 20):
                    n += len(chunk)
                    if n > limit:
                        raise self._too_big(rel, limit)
                    out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self._place_book(rel, dest, tmp)

    async def receive_document(self, path: str, stream, size: int | None = None,
                               offset: int | None = None, total: int | None = None) -> dict:
        """save_document for a request body streamed in (async chunks), so big files never sit in memory.
        With `offset` and `total` the body is one piece of a file sent in pieces, from that byte on:
        pieces are added to <name>.part in order, and the last one puts the file in place. Until then
        the answer is {"received": bytes so far}. A piece that doesn't start where the file stops
        is refused with "have" (where to carry on), and a piece that fails is cut off again, so it can
        be sent again. (Browsers send a file read into memory piece by piece even where they can't
        send the file itself, like a Windows path over 260 characters.)"""
        if offset is not None:
            return await self._receive_piece(path, stream, offset, total)
        rel, dest, limit = self._book_dest(path, size)
        tmp = dest.with_name(dest.name + ".part")
        try:
            n = 0
            with open(tmp, "wb") as out:
                async for chunk in stream:
                    n += len(chunk)
                    if n > limit:
                        raise self._too_big(rel, limit)
                    out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return await asyncio.to_thread(self._place_book, rel, dest, tmp)

    async def _receive_piece(self, path: str, stream, offset: int, total: int | None) -> dict:
        if total is None or total < 0 or offset < 0 or offset > total:
            raise Problem("A piece needs offset and total (the whole file's size), with offset at most total.")
        rel, dest, limit = self._book_dest(path, total if offset == 0 else None)
        if total > limit:
            raise self._too_big(rel, limit)
        tmp = dest.with_name(dest.name + ".part")
        have = tmp.stat().st_size if tmp.exists() else 0
        if offset and offset != have:
            raise Problem(f"{rel}: this piece starts at byte {offset}, but {have} are here; carry on from there.",
                          status=409, data={"have": have})
        n = offset
        try:
            with open(tmp, "r+b" if offset else "wb") as out:
                out.seek(offset)
                out.truncate()
                async for chunk in stream:
                    n += len(chunk)
                    if n > total:
                        raise Problem(f"{rel}: more bytes than its size ({total}).")
                    out.write(chunk)
        except BaseException:
            try:
                if offset:
                    os.truncate(tmp, offset)
                else:
                    tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        if n < total:
            return {"received": n}
        return await asyncio.to_thread(self._place_book, rel, dest, tmp)

    def add_documents(self, files: list[tuple[str, object]]) -> str:
        """Save uploaded documents or zips of them (name, file-like) into the documents folder, then scan."""
        saved, refused = [], []
        for name, f in files:
            try:
                r = self.save_document(str(name), f)
            except Problem as err:
                refused.append(Path(str(name).replace("\\", "/")).name or "?")
                log.info("document upload refused: %s", err.msg)
                continue
            saved += r["added"]
        if not saved:
            raise Problem("No documents added: " + (", ".join(refused) + " " if refused else "") +
                          "(EPUB, PDF, DOCX, TXT, Markdown, AZW3, MOBI or PRC files, or zips of them).")
        msg = f"Added {len(saved)} document{'s' if len(saved) != 1 else ''}"
        msg += f"; left out {', '.join(refused)}." if refused else "."
        try:
            return msg + " " + self.scan_documents()
        except Problem as err:
            return msg + " " + err.msg + " Then scan the folder again."

    def _knowledge_work(self, title: str, fn) -> str:
        self._knowledge_busy()
        task = kn.Task(title)
        self.knowledge_task = task

        def run():
            try:
                task.finish(fn(task) or "Done.")
            except InterruptedError:
                task.finish("Stopped. Start it again to carry on where it stopped.", stopped=True)
            except (Problem, kn.BookError, openwebui.OpenWebUIError, BackendError, OSError, ValueError) as err:
                task.fail(err.msg if isinstance(err, Problem) else str(err))
            except Exception as err:  # noqa: BLE001 - shown on the Knowledge page
                log.exception("knowledge: %s failed", title)
                task.fail(f"{err.__class__.__name__}: {err}")
        threading.Thread(target=run, name="knowledge", daemon=True).start()
        return title + "…"

    def stop_knowledge(self) -> str:
        if not (self.knowledge_task and self.knowledge_task.running):
            raise Problem("Nothing is running.", "warn", 409)
        self.knowledge_task.stop = True
        return "Stopping after the current document…"

    def scan_documents(self) -> str:
        """Find and read the documents, in the background. With sync on (see send_documents), the
        Open WebUI it syncs with is brought up to date afterwards: new documents go in, documents
        gone from the folder (or changed) come out."""
        folder = self.library.folder()
        if not folder.is_dir():
            if folder == self.library.default_folder:
                folder.mkdir(parents=True, exist_ok=True)
            else:
                raise Problem(f"GPU Shepherd can't see the documents folder {folder}.", status=404)

        def work(task: kn.Task) -> str:
            def progress(done, total, name):
                task.progress(done, total, f"Reading {name}")
            before = {k: (b.stamp, dict(b.sent)) for k, b in self.library.load()["books"].items() if b.sent}
            counts = self.library.scan(progress, lambda: task.stop)
            self._note_stale(before)
            if task.stop:
                raise InterruptedError
            parts = [f"{counts.get('ready', 0)} ready"] + [f"{n} {s}" for s, n in sorted(counts.items()) if s != "ready"]
            msg = "Found the documents: " + ", ".join(parts) + "."
            sync = self.library.load().get("sync")
            if not sync:
                return msg
            instances, _ = self.store.instances()
            inst = instances.get(sync.get("instance", ""))
            if not inst:
                return msg
            if not self._running(inst.id):
                return msg + f" Start {inst.name} and scan again to sync it."
            return msg + " " + self._sync_openwebui(task, inst, sync["name"], bool(sync.get("split")),
                                                    bool(sync.get("embeddings")))
        return self._knowledge_work("Finding and reading documents", work)

    def _note_stale(self, before: dict[str, tuple[str, dict]]) -> None:
        """After a scan: copies in Open WebUI of documents that are gone from the folder, or were
        changed (and read again), are noted as stale, to be taken out by the next sync or clear.
        A changed document is sent again."""
        def note(d):
            stale = d.setdefault("stale", [])
            for key, (stamp, sent) in before.items():
                b = d["books"].get(key)
                if b and b.stamp == stamp:
                    continue
                stale += [{"target": t, "fid": f} for t, f in sent.items()]
                if b:
                    b.sent = {}
        if before:
            self.library.update(note)

    # Open WebUI
    def _openwebui(self, inst: Instance) -> openwebui.OpenWebUI:
        try:
            code, out = self.backend.exec(self.engine.container_name(inst.id), ["python3", "-c", openwebui.TOKEN_SCRIPT])
        except BackendError as err:
            raise Problem(f"couldn't reach {inst.name}: {err}", status=409) from err
        data = logins._last_json(out) or {}
        if code or not data.get("token"):
            raise Problem(f"couldn't sign in to {inst.name}: {data.get('error') or logins.last_line(out)}", status=409)
        port = inst.ports[0].container if inst.ports else openwebui.CONTAINER_PORT
        return self.openwebui_client(instance_url(inst, port), data["token"])

    def _send_images(self, inst: Instance) -> str:
        """Hand a running Open WebUI the image settings in its environment through its API: it reads
        the variables only until its image settings are first saved."""
        if not self._running(inst.id):
            return (f"Start {inst.name} to use them; if its image settings were changed in Open WebUI before, "
                    "apply this again while it runs, or set them in Open WebUI: Admin Panel, Settings, Images.")
        try:
            self._openwebui(inst).use_images(comfyui.from_env(inst.env))
        except (Problem, openwebui.OpenWebUIError) as err:
            return (f"Restart {inst.name} to use them. GPU Shepherd couldn't also set them in Open WebUI "
                    f"({err}), so if pictures don't work after the restart, set them in Open WebUI: "
                    "Admin Panel, Settings, Images.")
        return f"{inst.name} makes and edits pictures now: turn on Image in a chat's message box."

    def _embed_server(self, inst: Instance, m: tpl.Machine) -> str:
        """The model server whose embeddings model Open WebUI should use: the one it is connected to."""
        server = tpl.connected_server(inst, m)
        if server and m.servers[server].get("embed"):
            return server
        return next((k for k, v in m.servers.items() if v.get("embed") and k in m.instances), "")

    @checked
    def send_documents(self, iid: str, name: str = "Documents", split: bool = False,
                       use_server_embeddings: bool | None = None, sync: bool | None = None) -> str:
        """Put the ready documents into an Open WebUI knowledge base (one per first-level folder with
        `split`), in the background. Documents already sent there are skipped, so it carries on after a stop.
        use_server_embeddings: True switches Open WebUI to the model server's embeddings model, False
        keeps whatever it uses, None stops with a question when they differ.
        sync: True keeps that knowledge base in sync from now on (every scan sends new documents and
        takes out the ones gone from the folder), False stops syncing, None leaves it as it is."""
        inst = self.instance(iid)
        if not tpl.matches(tpl.get("open-webui"), inst):
            raise Problem(f"{inst.name} isn't Open WebUI: documents go into Open WebUI's knowledge bases.")
        if not self._running(iid):
            raise Problem(f"Start {inst.name} first.", status=409)
        name = re.sub(r"\s+", " ", name or "").strip()[:80] or "Documents"
        if not any(b.status == "ready" for b in self.library.books()):
            raise Problem("No documents ready to send: add documents and scan the documents folder first.", status=409)
        self._knowledge_busy()
        if sync is not None:
            self.library.update(lambda d: d.__setitem__("sync", {
                "instance": iid, "name": name, "split": split,
                "embeddings": use_server_embeddings is not False} if sync else None))

        def work(task: kn.Task) -> str:
            return self._sync_openwebui(task, inst, name, split, use_server_embeddings)
        return self._knowledge_work(f"Sending documents to {inst.name}", work)

    def stop_sync(self) -> str:
        """Stop keeping an Open WebUI knowledge base in sync (what is in it stays)."""
        sync = self.library.load().get("sync")
        if not sync:
            raise Problem("Sync isn't on.", "info", 409)
        self.library.update(lambda d: d.__setitem__("sync", None))
        return (f"Stopped syncing {sync['name']}. What is in it stays; send again to add new documents, "
                "or turn sync back on.")

    def _sync_openwebui(self, task: kn.Task, inst: Instance, name: str, split: bool,
                        use_server_embeddings: bool | None) -> str:
        """Take stale copies out of an Open WebUI, then send it the ready documents it doesn't have."""
        iid = inst.id
        m = self.machine()
        server = self._embed_server(inst, m)
        task.text = f"Signing in to {inst.name}"
        ow = self._openwebui(inst)
        if server:
            info = m.servers[server]
            emb = ow.embedding()
            uses = emb["engine"] == "openai" and emb["model"] == info["embed"] and server in emb["url"]
            if not uses and use_server_embeddings:
                ow.use_embedding(tpl._server_url(server, m), info["api_key"], info["embed"],
                                 info.get("embed_query_prefix", ""))
            elif not uses and use_server_embeddings is None:
                raise Problem(f"{inst.name} embeds documents with {emb['model'] or 'its own model'}"
                              f"{' on its CPU' if not emb['engine'] else ''}, not {info['name']}'s embeddings "
                              "model. Choose which to use, then send again.", status=409)
        removed = self._take_out_stale(task, ow, iid)
        there = ow.knowledge_bases()

        def forget_gone(d):                      # knowledge bases deleted in Open WebUI: send to them again
            for b in d["books"].values():
                b.sent = {t: f for t, f in b.sent.items() if t.partition("/")[0] != iid or t.partition("/")[2] in there}
        self.library.update(forget_gone)
        todo = [b for b in self.library.books() if b.status == "ready"]
        kbs: dict[str, str] = {}
        sent = skipped = failed = in_a_row = 0
        for i, b in enumerate(todo):
            if task.stop:
                raise InterruptedError
            kb = f"{name} - {b.group}" if split and b.group else name
            target = f"{iid}/{kb}"
            if target in b.sent:
                skipped += 1
                continue
            task.progress(i, len(todo), f"Sending {b.name}")
            try:
                if kb not in kbs:
                    kbs[kb] = ow.knowledge_id(kb, "Documents added by GPU Shepherd")
                fid = ow.upload(b.file_name, self.library.text(b.key))
                ow.add(kbs[kb], fid)
            except (openwebui.OpenWebUIError, OSError) as err:
                if "answered 401" in str(err):          # the sign-in ran out: sign in again, then retry it
                    ow = self._openwebui(inst)
                    kbs.clear()
                    try:
                        kbs[kb] = ow.knowledge_id(kb, "Documents added by GPU Shepherd")
                        fid = ow.upload(b.file_name, self.library.text(b.key))
                        ow.add(kbs[kb], fid)
                    except (openwebui.OpenWebUIError, OSError) as err2:
                        err = err2
                    else:
                        err = None
                if err is not None:
                    failed += 1
                    in_a_row += 1
                    task.errors.append(f"{b.name}: {err}")
                    if in_a_row >= 5:
                        raise openwebui.OpenWebUIError(f"stopped after 5 documents in a row failed; the last: {err}")
                    continue
            in_a_row = 0
            sent += 1

            def mark(d, key=b.key, fid=fid):
                if key in d["books"]:
                    d["books"][key].sent[target] = fid
            self.library.update(mark)
        self.library.update(lambda d: d.__setitem__("sends", [
            s for s in (d.get("sends") or []) if (s["instance"], s["name"]) != (iid, name)] + [
            {"instance": iid, "name": name, "split": split, "when": time.time()}]))
        msg = f"Sent {sent} document{'s' if sent != 1 else ''} to {inst.name}"
        msg += f" ({skipped} were there already)" if skipped else ""
        msg += f", took out {removed} old cop{'ies' if removed != 1 else 'y'}" if removed else ""
        msg += f"; {failed} failed." if failed else "."
        return msg + " In a chat, type # and pick the knowledge base."

    def _take_out_stale(self, task: kn.Task, ow, iid: str) -> int:
        """Take the stale copies (see _note_stale) out of one Open WebUI. Returns how many."""
        stale = [s for s in self.library.load().get("stale") or [] if s["target"].partition("/")[0] == iid]
        if not stale:
            return 0
        task.text = "Taking out documents that are gone or changed"
        ids = ow.knowledge_bases()
        done = []
        for s in stale:
            kb = s["target"].partition("/")[2]
            try:
                if ids.get(kb):
                    ow.remove(ids[kb], s["fid"])
                ow.delete_file(s["fid"])
            except openwebui.OpenWebUIError as err:
                task.errors.append(f"taking out an old copy in {kb}: {err}")
                continue
            done.append(s)
        self.library.update(lambda d: d.__setitem__("stale", [s for s in d.get("stale") or [] if s not in done]))
        return len(done)

    def openwebui_knowledge(self, iid: str) -> dict:
        """What is in an Open WebUI's knowledge: its knowledge bases, how many files each has, and
        for the ones GPU Shepherd sends to, how many documents are in it and how many aren't yet."""
        inst = self.instance(iid)
        if not tpl.matches(tpl.get("open-webui"), inst):
            raise Problem(f"{inst.name} isn't Open WebUI.")
        if not self._running(iid):
            raise Problem(f"Start {inst.name} to see its knowledge.", status=409)
        data = self.library.load()
        ready = [b for b in data["books"].values() if b.status == "ready"]
        sends = {s["name"]: s for s in data.get("sends") or [] if s.get("instance") == iid}
        sync = data.get("sync") if (data.get("sync") or {}).get("instance") == iid else None
        ours: dict[str, dict] = {}               # knowledge base -> {"sent", "missing", "send", "sync"}
        for s in sends.values():
            for b in ready:
                kb = f"{s['name']} - {b.group}" if s.get("split") and b.group else s["name"]
                row = ours.setdefault(kb, {"sent": 0, "missing": 0, "send": s["name"],
                                           "sync": bool(sync and sync["name"] == s["name"])})
                row["sent" if f"{iid}/{kb}" in b.sent else "missing"] += 1
        try:
            ow = self._openwebui(inst)
            bases = ow.knowledge_list()
            for k in bases[:50]:
                if k["files"] is None:
                    k["files"] = ow.file_count(k["id"])
        except openwebui.OpenWebUIError as err:
            raise Problem(f"couldn't read {inst.name}'s knowledge: {err}", status=502) from err
        rows = []
        for k in sorted(bases, key=lambda k: (k["name"] not in ours, k["name"].lower())):
            mine = ours.pop(k["name"], None)
            rows.append(k | {"ours": bool(mine), "sent": mine["sent"] if mine else None,
                             "missing": mine["missing"] if mine else 0, "sync": bool(mine and mine["sync"]),
                             "path": f"/workspace/knowledge/{k['id']}"})
        for kb, mine in sorted(ours.items()):    # sent there before, but the knowledge base is gone
            if mine["sent"]:
                rows.append({"id": "", "name": kb, "description": "", "files": 0, "ours": True, "gone": True,
                             "sent": mine["sent"], "missing": mine["missing"], "sync": mine["sync"], "path": ""})
        return {"instance": iid, "name": inst.name, "bases": rows, "sync": sync,
                "stale": sum(1 for s in data.get("stale") or [] if s["target"].partition("/")[0] == iid)}

    # skills
    def _skill_dir(self, slug: str) -> Path:
        return self.library.skills_dir / slug

    def _hermes_data(self, inst: Instance) -> Path | None:
        """A Hermes Agent's data folder, if GPU Shepherd can see it."""
        inside = dict(tpl.get("hermes").appdata)["data"]
        v = next((v for v in inst.volumes if v.container == inside), None)
        return Path(v.host) if v and Path(v.host).is_relative_to(self.store.root) else None

    def _skill(self, slug: str) -> dict:
        s = (self.library.load().get("skills") or {}).get(slug)
        if not s:
            raise Problem(f"No skill '{slug}'.", status=404)
        return s

    def _chat(self, server: str, m: tpl.Machine):
        """prompt -> answer from a model server's default chat model."""
        inst, info = m.instances[server], m.servers[server]
        port = inst.ports[0].container if inst.ports else 8080
        url = instance_url(inst, port) + "/v1/chat/completions"

        def chat(prompt: str) -> str:
            body = {"model": info.get("default_model") or "chat", "max_tokens": 4096, "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            try:
                r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {info['api_key']}"}, timeout=1800)
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as err:
                raise kn.BookError(f"{info['name']} didn't answer: {err.__class__.__name__}: {err}") from err
        return chat

    def make_skill(self, key: str, server: str = "") -> str:
        """Draft an agent skill from a document with a model server, in the background."""
        b = self.library.load()["books"].get(key)
        if not b or b.status not in ("ready", "duplicate"):
            raise Problem("That document isn't ready: scan the documents folder first.", status=404)
        m = self.machine()
        server = server or next(iter(m.servers), "")
        if server not in m.servers or server not in m.instances:
            raise Problem("Making a skill needs a model server (install the Llama Swap from the App Center)."
                          if not server else f"No model server '{server}'.", status=404)
        if not self._running(server):
            raise Problem(f"Start {m.servers[server]['name']} first: its chat model writes the skill.", status=409)
        slug = knowledge_skills.slug(b.name)

        def work(task: kn.Task) -> str:
            text = self.library.text(b.key)
            md = knowledge_skills.make(b.name, text, self._chat(server, m), m.servers[server].get("ctx") or 0,
                                  lambda s: setattr(task, "text", f"{b.name}: {s}"), lambda: task.stop)
            folder = self._skill_dir(slug)
            knowledge_skills.write(folder, md, text)
            self.library.update(lambda d: d.setdefault("skills", {}).__setitem__(slug, {
                "book": b.key, "name": b.name, "server": server, "made": time.time(),
                "installed": ((d.get("skills") or {}).get(slug) or {}).get("installed", {})}))
            return f"Drafted the skill {slug}. Check it, then install it in an agent."
        return self._knowledge_work(f"Making a skill from {b.name}", work)

    def skill_text(self, slug: str) -> str:
        self._skill(slug)
        try:
            return (self._skill_dir(slug) / "SKILL.md").read_text(encoding="utf-8")
        except OSError as err:
            raise Problem(f"The skill's SKILL.md is missing: {err}", status=404) from err

    def save_skill(self, slug: str, text: str) -> str:
        self._skill(slug)
        if not re.match(r"^---\n.*?\bname:.*?\bdescription:.*?\n---", text.strip().replace("\r\n", "\n"), re.S):
            raise Problem("Keep the --- lines at the top with name: and description: (agents need them).")
        (self._skill_dir(slug) / "SKILL.md").write_text(text.strip().replace("\r\n", "\n") + "\n", encoding="utf-8")
        return f"Saved {slug}. Install it again where it is installed, so agents get the change."

    def install_skill(self, slug: str, iid: str) -> str:
        """Copy a skill into a Hermes Agent's skills folder."""
        self._skill(slug)
        inst = self.instance(iid)
        if not tpl.matches(tpl.get("hermes"), inst):
            raise Problem(f"{inst.name} isn't Hermes Agent: skills from documents go to Hermes Agent.")
        home = self._hermes_data(inst)
        if home is None:
            raise Problem(f"GPU Shepherd can't see {inst.name}'s data folder, so it can't add skills there.", status=409)
        owner = tpl.get("hermes").data_owner
        dest = knowledge_skills.install(self._skill_dir(slug), home)
        if owner and os.geteuid() == 0:
            for p in [dest.parent, dest, *dest.rglob("*")]:
                os.chown(p, *owner)
        self.library.update(lambda d: d["skills"][slug].setdefault("installed", {}).__setitem__(iid, time.time()))
        return f"Installed {slug} in {inst.name}. It sees the skill in a new conversation (or after a restart)."

    def delete_skill(self, slug: str) -> str:
        """Forget a draft (installed copies stay in the agents)."""
        self._skill(slug)
        shutil.rmtree(self._skill_dir(slug), ignore_errors=True)
        self.library.update(lambda d: d["skills"].pop(slug, None))
        return f"Deleted the draft {slug}. Copies installed in agents stay there."

    CLEAR = ("documents", "converted", "skills", "apps")

    def clear_knowledge(self, what: tuple[str, ...] | list[str] = CLEAR) -> str:
        """Start over, in the background. `what` picks the parts:
            documents   the files in the documents folder (all of it when it is GPU Shepherd's own
                        folder; only document files and zips in a folder of your own)
            converted   the converted text and the list of documents (and where they were sent)
            skills      the skill drafts, and the copies installed in Hermes Agents
            apps        the documents sent to Open WebUI: taken out of its knowledge bases and deleted
                        there, and a knowledge base GPU Shepherd made is deleted once it is empty"""
        what = tuple(w for w in self.CLEAR if w in what)
        if not what:
            raise Problem(f"Say what to clear: {', '.join(self.CLEAR)}.")
        self._knowledge_busy()
        data = self.library.load()
        instances, _ = self.store.instances()
        targets: dict[str, dict[str, list[str]]] = {}         # instance -> knowledge base -> file ids
        if "apps" in what:
            sent = [(t, f) for b in data["books"].values() for t, f in b.sent.items()]
            for target, fid in sent + [(s["target"], s["fid"]) for s in data.get("stale") or []]:
                iid, _, kb = target.partition("/")
                targets.setdefault(iid, {}).setdefault(kb, []).append(fid)
            stopped = [instances[i].name for i in targets if i in instances and not self._running(i)]
            if stopped:
                raise Problem(f"Start {', '.join(stopped)} first, so the documents can be taken out of it, or clear "
                              "without Open WebUI.", status=409)

        def work(task: kn.Task) -> str:
            done = []
            if "apps" in what and targets:
                removed = 0
                for iid, kbs in targets.items():
                    if iid not in instances:
                        continue
                    task.text = f"Taking the documents out of {instances[iid].name}"
                    ow = self._openwebui(instances[iid])
                    ids = ow.knowledge_bases()
                    total = sum(len(f) for f in kbs.values())
                    for kb, fids in kbs.items():
                        kid = ids.get(kb)
                        for fid in fids:
                            if task.stop:
                                raise InterruptedError
                            task.progress(removed, total, f"Taking the documents out of {instances[iid].name}: {kb}")
                            try:
                                if kid:
                                    ow.remove(kid, fid)
                                ow.delete_file(fid)
                                removed += 1
                            except openwebui.OpenWebUIError as err:
                                task.errors.append(f"{kb}: {err}")
                        if kid and ow.knowledge_empty(kid):
                            ow.delete_knowledge(kid)

                    def unsent(d, iid=iid):
                        for b in d["books"].values():
                            b.sent = {t: f for t, f in b.sent.items() if not t.startswith(iid + "/")}
                        d["sends"] = [x for x in (d.get("sends") or []) if x.get("instance") != iid]
                        d["stale"] = [x for x in (d.get("stale") or []) if not x["target"].startswith(iid + "/")]
                        if (d.get("sync") or {}).get("instance") == iid:
                            d["sync"] = None          # or the next scan would send it all again
                    self.library.update(unsent)
                done.append(f"took {removed} document{'s' if removed != 1 else ''} out of Open WebUI")
            if "skills" in what:
                task.text = "Removing the skills"
                n = 0
                for slug, sk in (self.library.load().get("skills") or {}).items():
                    for iid in (sk.get("installed") or {}):
                        home = self._hermes_data(instances[iid]) if iid in instances else None
                        if home is not None:
                            knowledge_skills.uninstall(slug, home)
                    n += 1
                shutil.rmtree(self.library.skills_dir, ignore_errors=True)
                self.library.update(lambda d: d.__setitem__("skills", {}))
                done.append(f"deleted {n} skill{'s' if n != 1 else ''}")
            if "documents" in what:
                folder = self.library.folder()
                task.text = f"Deleting the documents in {folder}"
                n = self._empty_documents(folder, own=folder == self.library.default_folder)
                done.append(f"deleted {n} file{'s' if n != 1 else ''} from {folder}")
            if "converted" in what:
                task.text = "Deleting the converted text"
                shutil.rmtree(self.library.text_dir, ignore_errors=True)
                self.library.update(lambda d: d.update({"books": {}, "sends": [], "stale": [], "scanned": None}))
                done.append("deleted the converted text and the list of documents")
            return "Cleared: " + ", ".join(done) + "."
        return self._knowledge_work("Clearing out knowledge", work)

    def ocr_documents(self) -> str:
        """Read the scanned PDFs with OCR (Tesseract), in the background."""
        n = sum(b.status == "scanned" for b in self.library.books())
        if not n:
            raise Problem("No scanned PDFs waiting for OCR.", "warn", 409)
        if not kn.ocr_available():
            raise Problem("OCR needs Tesseract: update GPU Shepherd (its container comes with it), or install "
                          "tesseract-ocr where it runs.", status=409)

        def work(task: kn.Task) -> str:
            done, errors = self.library.ocr(task.progress, lambda: task.stop)
            task.errors += errors
            if task.stop:
                raise InterruptedError
            return (f"Read {done} scanned PDF{'s' if done != 1 else ''} with OCR"
                    + (f"; {len(errors)} couldn't be read." if errors else ".")
                    + (" Send them to Open WebUI to add them to the knowledge base." if done else ""))
        return self._knowledge_work(f"Reading {n} scanned PDF{'s' if n != 1 else ''} with OCR", work)

    BULK_DELETE = {"duplicate": "duplicates", "failed": "documents that can't be read"}

    def delete_documents(self, status: str) -> str:
        """Delete every document with a status, in the background: "duplicate" (the copies with the
        same text as another document; that one stays) or "failed" (can't be read). Their files,
        every copy, and their text go. Documents waiting for OCR or skipped (MOBI/AZW3) stay."""
        if status not in self.BULK_DELETE:
            raise Problem(f"Delete which: {', '.join(self.BULK_DELETE)}.")
        self._knowledge_busy()
        keys = [b.key for b in self.library.books() if b.status == status]
        if not keys:
            raise Problem(f"No {self.BULK_DELETE[status]} to delete.", "warn", 409)

        def work(task: kn.Task) -> str:
            done = 0
            for i, key in enumerate(keys):
                if task.stop:
                    raise InterruptedError
                task.progress(i, len(keys), f"Deleting {i + 1} of {len(keys)}")
                try:
                    b = self.library.delete(key)
                except kn.BookError:
                    continue                       # gone already
                except OSError as err:
                    task.errors.append(f"{key}: {err}")
                    continue
                if b.status == status:
                    done += 1
            what = self.BULK_DELETE[status] if done != 1 else {"duplicate": "duplicate", "failed": "document that can't be read"}[status]
            return f"Deleted {done} {what}" + (f"; {len(task.errors)} couldn't be." if task.errors else ".")
        return self._knowledge_work(f"Deleting {len(keys)} {self.BULK_DELETE[status]}", work)

    def delete_document(self, key: str) -> str:
        """Delete one document: its files (every copy), its text, and the copies sent to Open WebUI
        (taken out when that Open WebUI is running; otherwise they stay there)."""
        self._knowledge_busy()
        try:
            b = self.library.delete(key)
        except kn.BookError as err:
            raise Problem(f"No document '{key}'.", status=404) from err
        instances, _ = self.store.instances()
        left = []
        by_app: dict[str, list[tuple[str, str]]] = {}
        for target, fid in b.sent.items():
            iid, _, kb = target.partition("/")
            by_app.setdefault(iid, []).append((kb, fid))
        for iid, files in by_app.items():
            inst = instances.get(iid)
            if not inst:
                continue
            try:
                if not self._running(iid):
                    raise Problem(f"{inst.name} isn't running")
                ow = self._openwebui(inst)
                ids = ow.knowledge_bases()
                for kb, fid in files:
                    if ids.get(kb):
                        ow.remove(ids[kb], fid)
                    ow.delete_file(fid)
            except (Problem, openwebui.OpenWebUIError) as err:
                log.warning("knowledge: couldn't take %s out of %s: %s", b.name, inst.name, err)
                left.append(inst.name)
                self.library.update(lambda d, iid=iid, files=files: d.setdefault("stale", []).extend(
                    {"target": f"{iid}/{kb}", "fid": fid} for kb, fid in files))
        return f"Deleted {b.name}." + (f" It is still in {', '.join(left)} (it wasn't running or answering): the next "
                                       "send or sync takes it out, or remove it there by hand." if left else "")

    @staticmethod
    def _empty_documents(folder: Path, own: bool) -> int:
        """Delete the documents in a folder: everything in GPU Shepherd's own one; in a folder of
        the user's, only document files and zips (other files and folders with them stay)."""
        if not folder.is_dir():
            return 0
        n = 0
        for root, dirs, names in os.walk(folder, topdown=False):
            for f in names:
                p = Path(root) / f
                if own or p.suffix.lower() in kn.RANK or p.suffix.lower() in (".zip", ".part"):
                    try:
                        p.unlink()
                        n += 1
                    except OSError:
                        pass
            if Path(root) != folder:
                try:
                    Path(root).rmdir()                  # only when empty
                except OSError:
                    pass
        return n


class HelperFileOps(FileOps):
    """Trash moves in a models folder: directly when GPU Shepherd can write it, otherwise in a
    helper container with the folder mounted (models folders are often mounted read-only here)."""

    def __init__(self, backend):
        self.backend = backend

    def _inside(self, folder: str, path: str) -> str:
        return "/f/" + str(Path(path).relative_to(folder))

    def _run(self, folder: str, lines: list[str]) -> None:
        self.backend.run_task(HELPER_IMAGE, "set -e\n" + "\n".join(lines), [(folder, "/f", False)],
                              downloads._owner(folder))

    def move(self, folder: str, pairs: list[tuple[str, str]]) -> None:
        if os.access(folder, os.W_OK):
            return super().move(folder, pairs)
        self._run(folder, [f"mkdir -p {shlex.quote(os.path.dirname(self._inside(folder, d)))} && "
                           f"mv -n {shlex.quote(self._inside(folder, s))} {shlex.quote(self._inside(folder, d))}"
                           for s, d in pairs])

    def remove(self, folder: str, paths: list[str]) -> None:
        if os.access(folder, os.W_OK):
            return super().remove(folder, paths)
        self._run(folder, ["rm -rf " + " ".join(shlex.quote(self._inside(folder, p)) for p in paths)])


def gb(n: int | float) -> str:
    """Bytes as a short size, e.g. 3.2 GB or 140 MB."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def parse_auto_rules(text: str) -> list[AutoRule]:
    """'2=big-model, 3=bigger' (or one per line): with N or more GPUs, use that model."""
    parsed = []
    for part in re.split(r"[,\n]", str(text)):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*\+?\s*(?:gpus?)?\s*[=:]\s*(\S+)$", part, re.I)
        if not m:
            raise Problem(f"rule '{part}': write it as 2=model-id (2 or more GPUs use model-id)")
        parsed.append(AutoRule(min_gpus=int(m.group(1)), model=m.group(2)))
    return parsed



def _volume_visible(inst: Instance, inside: str) -> bool:
    """Whether GPU Shepherd can see the host folder of the volume a container path is in."""
    for v in inst.volumes:
        if inside.startswith(v.container.rstrip("/") + "/"):
            return Path(v.host).is_dir()
    return False


def _remote_gb(url: str) -> float:
    """A file's size from its server, in GB (0 when it doesn't say)."""
    try:
        r = httpx.head(url, follow_redirects=True, timeout=10)
        size = int(r.headers.get("x-linked-size") or r.headers.get("content-length") or 0) if r.status_code < 400 else 0
    except (httpx.HTTPError, ValueError):
        size = 0
    return round(size / 1e9, 2)
