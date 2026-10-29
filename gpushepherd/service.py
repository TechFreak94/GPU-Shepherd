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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from pydantic import ValidationError

from . import (app_updates, auth, budget, downloads, host, instance_templates as tpl, logins, model_library, storage,
               updates)
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, Instance, Settings, Store,
                     VolumeMap, friendly_errors)
from .connectors import llamaswap
from .backend import BackendError
from .downloads import IMAGE as HELPER_IMAGE, KIND as DOWNLOAD_KIND, Downloads
from .engine import Engine
from .jobs import Busy, Job, JobRunner
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

    def __post_init__(self):
        if self.downloads is None:
            self.downloads = Downloads(self.backend, self.store.settings().container_prefix)
        if self.trash is None:
            self.trash = Trash(self.store.root, HelperFileOps(self.backend))
        if self.peaks is None:
            self.peaks = budget.Peaks(self.store.root)

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
                "trash_days": self.store.settings().trash_days}

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
        self.store.save_instance(inst)
        return f"Skipped \"{u.title}\" for {inst.name}."

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
                        health_timeout: int | None = None, new_key: bool = False) -> str:
        """The model server's own settings; None leaves a setting as it is."""
        _, cat = self.server(iid)
        if threads is not None:
            cat.threads = int(threads)
        if llama_server is not None:
            cat.llama_server = str(llama_server).strip()
        if health_timeout is not None:
            cat.health_timeout = int(health_timeout)
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
