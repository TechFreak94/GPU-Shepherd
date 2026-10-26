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
import threading
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import auth, host, instance_templates as tpl, model_library, updates
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, Instance, Store, VolumeMap,
                     friendly_errors)
from .connectors import llamaswap
from .downloads import Downloads
from .engine import Engine
from .jobs import Busy, Job, JobRunner
from .scheduler import Plan

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

    def __post_init__(self):
        if self.downloads is None:
            self.downloads = Downloads(self.backend, self.store.settings().container_prefix)

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
        if c.noop or (p.side_effects and not confirm):
            return c
        try:
            c.job = self.jobs.submit(f"{VERBS[action]} {c.name}", lambda job: self.engine.execute(p, job, job))
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return c

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
        inst = self.instance(iid)
        try:
            self.jobs.submit(f"Deleting {inst.name}", lambda logf: self.engine.remove_instance(iid, logf), wait=True)
        except Busy as err:
            raise Problem(str(err), "warn", 409) from err
        return f"Deleted {inst.name}."

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
            s = model_library.for_file(m.file, tpl.MODELS_IN_CONTAINER)
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
    def save_model(self, iid: str, model: CatalogModel, original: str = "") -> str:
        """Add a model, or replace `original` (renaming it when the ID changed; auto names follow)."""
        _, cat = self.server(iid)
        model = CatalogModel.model_validate(model.model_dump())
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
        return f"Saved {model.id}."

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
        for s in model_library.SUGGESTIONS:
            f = model_library.fit(s, m.gpu_mem_gb, m.ram_gb)
            st = self.downloads.status(s, folder)
            out.append({"key": s.key, "name": s.name, "blurb": s.blurb, "uses": list(s.uses), "page": s.page,
                        "vision": s.vision, "moe": s.moe, "mtp": bool(s.mtp), "download_gb": s.download_gb,
                        "fits": f.ok, "fit": f.text, "rank": f.rank, "state": st.state, "status": st.text,
                        "pct": st.pct, "in_catalog": f"{tpl.MODELS_IN_CONTAINER}/{s.path(s.model)}" in files,
                        "files": [{"path": f"{tpl.MODELS_IN_CONTAINER}/{s.path(x)}", "url": x.url, "gb": x.gb}
                                  for x in s.files]})
        return sorted(out, key=lambda x: x.pop("rank"))

    def _suggestion(self, key: str) -> model_library.Suggestion:
        s = model_library.get(key)
        if not s:
            raise Problem(f"no suggested model '{key}'", status=404)
        return s

    def download_suggestion(self, iid: str, key: str) -> str:
        inst, _ = self.server(iid)
        s = self._suggestion(key)
        folder = self.models_dir(inst)
        if not folder:
            raise Problem(f"{inst.name} has no models folder (a volume mounted at {tpl.MODELS_IN_CONTAINER}): "
                          "add one under Edit first.")
        if self.downloads.status(s, folder).state == "ready":
            return f"{s.name} is already downloaded."
        return self.downloads.start(s, folder)

    def cancel_download(self, iid: str, key: str) -> str:
        self.server(iid)
        return self.downloads.cancel(self._suggestion(key))

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
        self.save_model(iid, model)
        fetching = self.fetch_suggested(inst, [model])
        st = self.backend.states().get(iid)
        note = " Restart the server to use it." if st and st.running else ""
        return f"Added {model.id}" + (f"; downloading it ({s.download_gb:.1f} GB)." if fetching else ".") + note

    def load_model(self, iid: str, mid: str) -> str:
        """Load a model in the background (large ones take minutes from disk)."""
        _, cat = self.server(iid)
        if mid not in {m.id for m in cat.models}:
            raise Problem(f"unknown model {mid}", status=404)
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
