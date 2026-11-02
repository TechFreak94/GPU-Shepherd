"""What GPU Shepherd can do, in one place: the dashboard, the JSON API (api.py) and the
`gpu-shepherd` command all call these, so the three never drift apart.

Every method checks its input and raises `Problem` with a message a person can act on. The
dashboard shows it as a toast, the API returns it as {"error": ...} with `Problem.status`, and the
command prints it.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import secrets
import shlex
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from pydantic import ValidationError

from . import (app_updates, auth, budget, downloads, drives, gpu as gpu_mod, host, image_updates,
               instance_templates as tpl, logins, model_library, moves, storage, tuning, updates)
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, Instance, Settings, Store,
                     VolumeMap, friendly_errors)
from .connectors import llamaswap
from .backend import BackendError
from .downloads import IMAGE as HELPER_IMAGE, KIND as DOWNLOAD_KIND, Downloads
from .engine import Engine
from .jobs import Busy, Cancelled, Job, JobRunner
from .scheduler import Plan
from .trash import FileOps, Trash

log = logging.getLogger("gpushepherd")
ACTIONS = ("start", "stop", "restart")
VERBS = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}


class Problem(Exception):
    """A request that can't be done as asked. kind is how the dashboard shows it (error / warn /
    info); status is the HTTP status the API answers with."""

    def __init__(self, msg: str, kind: str = "error", status: int = 400):
        super().__init__(msg)
        self.msg, self.kind, self.status = msg, kind, status


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
    bundle: Instance | None = None      # the LLM server installed alongside it, if one was
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
        self._auto_tried: set[tuple[str, str]] = set()     # (instance, version) automatic updates tried
        self._drive_lock = threading.Lock()
        self._drives_seen: set[str] | None = None   # drive IDs at the last scan (None: no scan yet)

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
        if inst.app and not editing:              # an App Store app installed by hand: its folders and helpers too
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
            try:
                found = self.host_fs.scan()
            except (BackendError, OSError, ValueError) as err:
                self.drive_error = f"The machine's drives can't be looked at from here: {err}"
                log.debug("drive scan failed: %s", err)
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
            if mount_new and self.store.settings().auto_mount_drives:
                for d in found:
                    if d.state == "ready" and d.uuid in attached:
                        self._mount(d, found, persist=True)
            return sorted(found, key=lambda d: (d.system, d.device))

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
                "auto_mount": self.store.settings().auto_mount_drives}

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
            if d.state == "failed":
                raise Problem(d.note, status=500)
            return f"Mounted {d.name} at {d.mountpoint}." + (f" {d.note}" if d.note else "")

    def set_auto_mount(self, on: bool) -> str:
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
                             storage.model_folders(instances, tpl.MODELS_IN_CONTAINER))

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
            job("Copied, and every file matches the old copy")
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
        """Images GPU Shepherd may have pulled: the App Store's, the download helper, and the
        instances' own. Other images on the machine are none of its business."""
        return {t.image for t in tpl.TEMPLATES} | {HELPER_IMAGE} | {i.image for i in instances.values()}

    def _downloading(self, instances: dict[str, Instance]) -> set[str]:
        running = {n for n, h in self.backend.helpers(DOWNLOAD_KIND).items() if h.running}
        folders = storage.model_folders(instances, tpl.MODELS_IN_CONTAINER)
        return {str(Path(m) / s.path(f)) for s in model_library.SUGGESTIONS + self._linked()
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

    # ------------------------------------------------------------------ App Store
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
                                "embed": ("embed" if "embed" in embed.aliases else embed.id) if embed else ""}
        return tpl.Machine(gpu_mem_gb=[g.mem_total / 1024 for g in self.engine.snapshot().gpus],
                           instances=instances, servers=servers, model_sizes={}, ram_gb=host.ram_gb())

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

    def has_llm_server(self) -> bool:
        return any(i.kind == "llama-swap" for i in self.store.instances()[0].values())

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
        """Install an App Store app from its install form's answers. With fill_defaults, the
        answers only need what differs from the form's defaults (for the API and the command)."""
        t = tpl.get(key)
        m = self.machine()
        if fill_defaults:
            answers = tpl.defaults(t, m, self.default_models_dir(m.instances), self.appdata_root()) | answers
            answers["autostart"] = answers.get("autostart") in (True, "true", "on", "1", "yes")
        slot_fields = [f"{x.key}_file" for x in tpl.slots(t)]
        if slot_fields and answers.get("models_dir"):
            m.model_sizes = {x["container_path"]: x["size_gb"] for x in self.model_files(str(answers["models_dir"]))[0]}
        bundle = tpl.bundled_server(answers, m, self.appdata_root()) if t.connects else None
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
            s = model_library.for_file(m.file, tpl.MODELS_IN_CONTAINER, self._linked())
            if s and folder and self.downloads.status(s, folder).state != "ready":
                try:
                    self.downloads.start(s, folder)
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
        st = self.backend.states().get(iid)
        if st and st.running and not msg.endswith("to use it."):
            msg += " Restart it to use them."
        return msg + (" " + " ".join(u.manual) if u.manual else "")

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
        inst = self.instance(iid)
        st = self.image_status().get(iid)
        if not st or st.state not in ("available", "restart"):
            raise Problem(f"{inst.name} already runs the newest version GPU Shepherd knows of. "
                          "Check for updates first if a release just came out.", "info", 409)
        try:
            job = self.jobs.submit(f"Updating {inst.name}",
                                   lambda job: self._update_image(iid, st.state == "available", job))
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return f"Updating {inst.name}…", job

    def _update_image(self, iid: str, pull: bool, job: Job) -> None:
        inst = self.store.instance(iid)
        state = self.backend.states().get(iid)
        old = (state.image_id if state else "") or self.backend.image_info([inst.image]).get(inst.image, ("", []))[0]
        new = old
        if pull:
            job(f"Downloading the new version of {inst.image}")
            self.backend.pull(inst.image, job)
            new, local = self.backend.image_info([inst.image]).get(inst.image, ("", []))
            self.image_log.downloaded(inst.image, new, image_updates.digests(local))
        if not (state and state.running):
            job(f"{inst.name} is stopped: it runs the new version from its next start.")
            self.image_log.updated(iid, old)
            return
        p = self.engine.plan("restart", iid)
        if p.error or p.side_effects:
            job(f"Not restarted: {p.error or 'that would also stop or move other instances'}. "
                f"Restart {inst.name} when it suits you to use the new version.")
            return
        self.engine.execute(p, job, job)
        self.image_log.updated(iid, old)
        if old and new and old != new:        # the old version, unless something else still uses it
            try:
                self.backend.remove_image(old)
                job("Removed the old version")
            except BackendError:
                pass

    def auto_update(self) -> None:
        """Check when it is time, then update one app that updates by itself (the next look takes
        the next one: changes run one at a time)."""
        hours = self.store.settings().image_check_hours
        if hours and time.time() - self.image_checked() >= hours * 3600:
            log.info("checking for new app versions: %s", self.check_images())
        if self.jobs.busy:
            return
        instances = self.store.instances()[0]
        data = self.image_log.read()["images"]
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
            try:
                self.update_image(iid)
            except Problem as err:
                log.info("automatic update of %s skipped: %s", iid, err.msg)
            return

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
            status = (f"{inst.name} has no password yet." if m.app in ("anythingllm", "hermes")
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
        if m.app == "hermes":
            inst.env = inst.env | {"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": user or "admin",
                                   "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": password}
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
        gpus = sorted(gpu_mod.query()[0], key=lambda g: g.index)       # the most it may get
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
                and model.file not in self._downloading({inst.id: inst}):
            notes.insert(0, f"{model.file} isn't in the models folder yet, so {model.id} can't load until it is.")
        return notes

    def delete_model(self, iid: str, mid: str) -> str:
        _, cat = self.server(iid)
        if mid not in {m.id for m in cat.models}:
            raise Problem(f"no model '{mid}'", status=404)
        try:
            cat = Catalog.model_validate({**cat.model_dump(), "models": [m.model_dump() for m in cat.models if m.id != mid]})
        except ValidationError as err:
            raise Problem(f"Cannot delete {mid}: {friendly_errors(err)}", status=409) from err
        self.store.save_catalog(cat)
        return f"Deleted {mid}."

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
        for s in model_library.SUGGESTIONS + linked:
            f = model_library.fit(s, m.gpu_mem_gb, m.ram_gb)
            st = self.downloads.status(s, folder)
            out.append({"key": s.key, "name": s.name, "blurb": s.blurb, "uses": list(s.uses), "page": s.page,
                        "vision": s.vision, "moe": s.moe, "mtp": bool(s.mtp), "download_gb": s.download_gb,
                        "fits": f.ok, "fit": f.text, "rank": f.rank, "state": st.state, "status": st.text,
                        "pct": st.pct, "partial": st.partial, "linked": s in linked, "in_catalog": f"{tpl.MODELS_IN_CONTAINER}/{s.path(s.model)}" in files,
                        "files": [{"path": f"{tpl.MODELS_IN_CONTAINER}/{s.path(x)}", "url": x.url, "gb": x.gb}
                                  for x in s.files]})
        return sorted(out, key=lambda x: x.pop("rank"))

    def _suggestion(self, key: str) -> model_library.Suggestion:
        s = model_library.get(key, self._linked())
        if not s:
            raise Problem(f"no suggested model '{key}'", status=404)
        return s

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
            taken = {x.key for x in model_library.SUGGESTIONS + self._linked()}
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
            return self.downloads.start(s, folder)
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
                if p is None or f in downloading or not _volume_visible(inst, f):
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
        busy = self.loading.setdefault(iid, set())
        if mid in busy:
            raise Problem(f"{mid} is already loading.", "info", 409)

        def worker():
            busy.add(mid)
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
