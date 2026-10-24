"""Web UI (FastAPI + Jinja2 + htmx). Server-rendered pages; htmx swaps in fragments."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import __version__, auth, buildinfo, changelog, drain, host as host_mod, instance_templates as tpl, updates
from .backend import Backend, DockerBackend, FakeBackend
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, DrainPolicy, GpuPolicy, Instance,
                     OffloadRule,
                     Store, VolumeMap, format_command, format_env, format_ports, format_volumes, friendly_errors,
                     parse_command, parse_env, parse_ports, parse_volumes)
from .connectors import llamaswap
from .engine import Engine
from .jobs import Busy, JobRunner
from .scheduler import gpu_text

log = logging.getLogger("gpushepherd")
HERE = Path(__file__).parent
PUBLIC = ("/login", "/static/", "/healthz")


class DemoLlamaClient:
    """Stands in for llama-swap in demo mode and tests."""
    loaded: dict[str, str] = {}

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def models(self):
        return [{"id": m.id, "name": m.name or m.id} for m in self.catalog.models]

    def running(self, known):
        return {m: s for m, s in DemoLlamaClient.loaded.items() if m in known}

    def load(self, model, timeout=0):
        DemoLlamaClient.loaded.clear()
        DemoLlamaClient.loaded[model] = "ready"

    def unload(self, model=None):
        DemoLlamaClient.loaded.clear()


def gib(n: float | int | None) -> str:
    """Bytes as GB the way people read them: 0.4 GB, 12.3 GB, 125 GB."""
    v = (n or 0) / 2**30
    return f"{v:.0f} GB" if v >= 100 else f"{v:.1f} GB"


def meter_level(pct: float | None) -> str:
    """CSS class for a usage meter, so a nearly full one stands out."""
    pct = pct or 0
    return "full" if pct >= 90 else "high" if pct >= 75 else ""


def create_app(data_dir: str | None = None, backend: Backend | None = None, demo: bool | None = None,
               autostart: bool | None = None) -> FastAPI:
    demo = demo if demo is not None else os.environ.get("SHEPHERD_DEMO") == "1"
    store = Store(data_dir or os.environ.get("SHEPHERD_DATA_DIR", "/opt/gpu-shepherd"))
    if backend is None:
        backend = FakeBackend(slow=0.5) if demo else DockerBackend()
    # demo mode has no real servers to ask, so everything counts as idle
    engine = Engine(store, backend, activity=(lambda _inst: drain.Activity(0)) if demo else None)
    jobs = JobRunner()
    try:
        backend.ensure_network(store.settings().network)
    except Exception as err:  # noqa: BLE001
        log.warning("could not prepare the Docker network: %s", err)
    generated = auth.ensure_admin(store)
    if generated:
        log.warning("First start: log in as '%s' with password '%s' (also in %s). Change it under Settings.",
                    store.settings().admin_user, generated, auth.initial_password_file(store))
    settings = store.settings()

    app = FastAPI(title="GPU Shepherd", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.engine, app.state.jobs, app.state.demo = store, engine, jobs, demo
    app.state.loading: dict[str, set[str]] = {}
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    build = buildinfo.load()
    # static files are cached per build, so an update never runs with the old CSS or JS
    templates.env.globals.update(version=__version__, asset_version=build.get("commit", __version__)[:12],
                                 gpu_text=gpu_text, demo=demo, level=meter_level)
    templates.env.filters["gib"] = gib
    templates.env.filters["md"] = changelog.inline
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ------------------------------------------------------------------ security
    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if not path.startswith(PUBLIC):
                if not request.session.get("user"):
                    if path == "/" and request.query_params.get("login"):
                        # just logged in, yet the session cookie did not come back
                        return RedirectResponse("/login?nocookie=1", status_code=303)
                    if request.headers.get("HX-Request"):
                        return Response(status_code=401, headers={"HX-Redirect": "/login"})
                    return RedirectResponse("/login", status_code=303)
                # every state-changing request must come from htmx (custom header = no cross-site forms)
                if request.method != "GET" and request.headers.get("HX-Request") != "true":
                    return PlainTextResponse("missing HX-Request header", status_code=403)
            response = await call_next(request)
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "same-origin")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "frame-ancestors 'none'; form-action 'self'")
            return response

    app.add_middleware(Guard)
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, session_cookie="gpushepherd",
                       same_site="strict", https_only=False, max_age=7 * 24 * 3600)

    # ------------------------------------------------------------------ helpers
    def page(request: Request, template: str, status: int = 200, **ctx):
        ctx.setdefault("user", request.session.get("user"))
        ctx.setdefault("flash", request.query_params.get("msg", ""))
        return templates.TemplateResponse(request, template, ctx, status_code=status)

    def toast(resp: Response, msg: str, kind: str = "info") -> Response:
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": kind}})
        return resp

    def llama_client(iid: str):
        if demo:
            return DemoLlamaClient(store.catalog(iid))
        return engine.llama_client(iid)

    def base_url(request: Request, port: int, path: str = "/") -> str:
        """The address an instance's port is reached at, from the browser's point of view."""
        name = request.url.hostname or "localhost"
        if ":" in name:                   # IPv6 literal
            name = f"[{name}]"
        return f"http://{name}:{port}{path if path.startswith('/') else '/' + path}"

    def dashboard_ctx(request: Request) -> dict:
        snap = engine.snapshot()
        machine = host_mod.query()
        running = [i for i, st in snap.states.items() if st.running and i in snap.instances]
        try:
            usage = backend.stats([engine.container_name(i) for i in running])
        except Exception:  # noqa: BLE001
            usage = {}
        gpu_by_index = {g.index: g for g in snap.gpus}
        cards, cpu_rows, mem_rows = [], [], []
        order = sorted(snap.instances.values(), key=lambda i: (-i.gpu.priority if i.gpu.mode != "none" else 1, i.name.lower()))
        for inst in order:
            st = snap.states.get(inst.id)
            gpus = snap.allocations.get(inst.id, [])
            assigned = [gpu_by_index[g] for g in gpus if g in gpu_by_index]
            card = {"inst": inst, "state": st, "running": bool(st and st.running),
                    "status": (st.status if st else "not created"),
                    "container": engine.container_name(inst.id),
                    "gpus": gpus, "assigned": assigned,
                    "vram_used": sum(g.mem_used for g in assigned), "vram_total": sum(g.mem_total for g in assigned),
                    "usage": usage.get(engine.container_name(inst.id)),
                    "ports": [{"port": p, "url": (u := base_url(request, p.host)), "label": u[7:].rstrip("/")}
                              for p in inst.ports],
                    "open_url": base_url(request, inst.open_url_port, inst.open_path) if inst.open_url_port else "",
                    "pending": bool(st and st.running and st.config_hash and st.config_hash != engine.config_hash(inst)),
                    "loaded": [], "models_error": "", "model_count": 0, "aliases": {}, "offload": {},
                    "loading": sorted(app.state.loading.get(inst.id, set())),
                    "waiting": (jobs.current.waiting.get(inst.id) if jobs.current else None)}
            if inst.kind == "llama-swap":
                cat = store.catalog(inst.id)
                card["model_count"] = len(cat.models)
                if card["running"]:
                    card["aliases"] = llamaswap.resolved_aliases(cat, len(gpus))
                    try:
                        client = llama_client(inst.id)
                        card["loaded"] = sorted(client.running({m.id for m in cat.models}))
                    except Exception:  # noqa: BLE001
                        card["models_error"] = "model server not answering yet"
                    by_id = {m.id: m for m in cat.models}
                    for mid in card["loaded"]:
                        rule = by_id[mid].offload_for(len(gpus)) if mid in by_id else None
                        if rule:
                            card["offload"][mid] = rule
            if card["usage"]:
                cpu_rows.append({"name": inst.name, "id": inst.id, "value": card["usage"].cpu_pct})
                mem_rows.append({"name": inst.name, "id": inst.id, "value": card["usage"].mem_used})
            cards.append(card)
        gpus = [{"gpu": g, "users": [(i, snap.instances[i].name) for i, a in snap.allocations.items()
                                     if g.index in a and i in snap.instances]} for g in snap.gpus]
        mem_rows.sort(key=lambda r: -r["value"])
        cpu_rows.sort(key=lambda r: -(r["value"] or 0))
        return {"cards": cards, "gpus": gpus, "gpu_error": snap.gpu_error, "problems": snap.problems,
                "host": machine, "cpu_rows": cpu_rows, "mem_rows": mem_rows,
                "mem_other": max(machine.mem_used - sum(r["value"] for r in mem_rows), 0),
                "running_count": len(running), "instance_count": len(snap.instances),
                "job": jobs.current, "last_job": (jobs.history[0] if jobs.history else None),
                "now": time.time(), "u": updates.status(store.root)}

    def run_plan(request: Request, action: str, iid: str, confirm: bool, back: str = "") -> Response:
        names = {i.id: i.name for i in store.instances()[0].values()}
        if jobs.busy:
            return toast(Response(status_code=204), "Another change is still running - wait for it to finish.", "warn")
        p = engine.plan(action, iid)
        if p.error:
            return toast(Response(status_code=204), p.error, "error")
        if p.is_noop:
            return toast(Response(status_code=204), f"{names.get(iid, iid)} is already {'stopped' if action == 'stop' else 'running'}.")
        if p.side_effects and not confirm:
            resp = page(request, "partials/confirm.html", plan=p, lines=p.describe(names), iid=iid, action=action,
                        name=names.get(iid, iid), back=back)
            resp.headers["HX-Retarget"] = "#modal-body"
            resp.headers["HX-Reswap"] = "innerHTML"
            resp.headers["HX-Trigger-After-Swap"] = "open-modal"
            return resp
        verb = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}[action]
        try:
            jobs.submit(f"{verb} {names.get(iid, iid)}", lambda job: engine.execute(p, job, job))
        except Busy as err:
            return toast(Response(status_code=204), str(err), "warn")
        if back:
            return Response(status_code=204, headers={"HX-Redirect": f"{back}?msg={verb} {names.get(iid, iid)}…"})
        resp = page(request, "partials/dashboard.html", **dashboard_ctx(request))
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": f"{verb} {names.get(iid, iid)}…", "kind": "info"},
                                                 "close-modal": True})
        return resp

    # ------------------------------------------------------------------ login
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__, "commit": build.get("commit", "")}

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        error = ("The password was right, but your browser did not keep the login cookie. Allow cookies for this "
                 "address (or try a private window) and log in again.") if request.query_params.get("nocookie") else ""
        return page(request, "login.html", error=error, cookie_problem=bool(error), first_run=auth.initial_password_file(store).exists())

    @app.post("/login")
    async def login(request: Request):
        form = await request.form()
        s = store.settings()
        user, password = str(form.get("user", "")), str(form.get("password", ""))
        # Copying the password out of a terminal easily picks up a space or line break at either end.
        ok = auth.verify_password(password, s.password_hash) or (
            password != password.strip() and auth.verify_password(password.strip(), s.password_hash))
        client = request.client.host if request.client else "?"
        if secrets.compare_digest(user.strip(), s.admin_user) and ok:
            log.info("login: '%s' logged in from %s", s.admin_user, client)
            request.session.clear()
            request.session["user"] = s.admin_user
            return RedirectResponse("/?login=1", status_code=303)
        log.warning("login: rejected from %s (%s)", client,
                    "wrong password" if secrets.compare_digest(user.strip(), s.admin_user) else f"unknown user {user!r}")
        return page(request, "login.html", status=401, error="Wrong user name or password.",
                    first_run=auth.initial_password_file(store).exists())

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return Response(status_code=204, headers={"HX-Redirect": "/login"})

    # ------------------------------------------------------------------ dashboard
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard.html", **dashboard_ctx(request), docker=backend.ping())

    @app.get("/ui/dashboard", response_class=HTMLResponse)
    def dashboard_partial(request: Request):
        return page(request, "partials/dashboard.html", **dashboard_ctx(request))

    @app.post("/ui/instances/{iid}/{action}")
    async def instance_action(request: Request, iid: str, action: str):
        if action not in ("start", "stop", "restart"):
            return PlainTextResponse("unknown action", status_code=400)
        form = await request.form()
        back = str(form.get("back", ""))
        if back and not re.match(r"^/[A-Za-z0-9/_-]*$", back):   # local paths only
            back = ""
        return run_plan(request, action, iid, confirm=form.get("confirm") == "1", back=back)

    @app.get("/ui/instances/{iid}/logs", response_class=HTMLResponse)
    def instance_logs(request: Request, iid: str):
        inst = store.instance(iid)
        text = backend.logs(engine.container_name(iid))
        return page(request, "partials/logs.html", title=f"Logs: {inst.name}", text=text,
                    refresh=f"/ui/instances/{iid}/logs")

    @app.get("/ui/jobs/{jid}", response_class=HTMLResponse)
    def job_detail(request: Request, jid: int):
        job = next((j for j in jobs.recent() if j.id == jid), None)
        if not job:
            return PlainTextResponse("job not found", status_code=404)
        return page(request, "partials/logs.html", title=f"{job.title} ({job.status})", text="\n".join(job.lines),
                    refresh=f"/ui/jobs/{jid}" if job.status == "running" else "")

    @app.post("/ui/jobs/{jid}/{op}")
    def job_control(jid: int, op: str):
        """The buttons shown while a change waits for running work: stop now, or call it off."""
        job = jobs.current
        if op not in ("skip", "cancel"):
            return PlainTextResponse("unknown operation", status_code=400)
        if not job or job.id != jid:
            return toast(Response(status_code=204, headers={"HX-Trigger-After-Settle": "refresh"}),
                         "That change has already finished.")
        if op == "cancel":
            job.call_off()
            msg = "Calling the change off: nothing is stopped."
        else:
            job.skip.set()
            msg = "Stopping without waiting any longer."
        log.info("job %s: %s pressed", jid, op)
        resp = Response(status_code=204)
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": "info"}, "refresh": True})
        return resp

    # ------------------------------------------------------------------ instance editor
    def blank_instance(kind: str) -> dict:
        if kind == "llama-swap":
            return {"kind": "llama-swap", "image": "ghcr.io/mostlygeek/llama-swap:cuda", "ports": "8080:8080",
                    "gpu_mode": "elastic", "min_gpus": 1, "max_gpus": 0, "priority": 50, "open_path": "/ui",
                    "volumes": "/srv/models:/models:ro", "name": "LLM server", "id": "llm",
                    "drain_check": "auto", "drain_timeout": 600, "drain_on_timeout": "stop"}
        return {"kind": "container", "gpu_mode": "none", "min_gpus": 1, "max_gpus": 0, "priority": 50, "open_path": "/",
                "drain_check": "auto", "drain_timeout": 600, "drain_on_timeout": "stop"}

    def form_from_instance(inst: Instance) -> dict:
        return {"id": inst.id, "name": inst.name, "description": inst.description, "kind": inst.kind,
                "image": inst.image, "command": format_command(inst.command), "env": format_env(inst.env),
                "ports": format_ports(inst.ports), "volumes": format_volumes(inst.volumes),
                "gpu_mode": inst.gpu.mode, "gpu_list": inst.gpu.gpus, "min_gpus": inst.gpu.min_gpus,
                "max_gpus": inst.gpu.max_gpus, "priority": inst.gpu.priority, "autostart": inst.autostart,
                "open_port": inst.open_port or "", "open_path": inst.open_path, "shm_size": inst.shm_size or "",
                "user_spec": inst.user or "", "memory_limit": inst.memory_limit or "",
                "drain_check": inst.drain.check, "drain_timeout": inst.drain.timeout,
                "drain_on_timeout": inst.drain.on_timeout, "drain_port": inst.drain.port or "",
                "drain_path": inst.drain.path, "drain_key": inst.drain.key}

    def editor(request: Request, form: dict, editing: bool, error: str = "", status: int = 200):
        gpus, _ = engine.snapshot().gpus, None
        gpu_count = max(len(gpus), 1)
        return page(request, "instance_form.html", status=status, f=form, editing=editing, error=error,
                    gpu_indices=list(range(max(gpu_count, max([0] + list(form.get("gpu_list") or [])) + 1))))

    # ------------------------------------------------------------------ guided setup (templates)
    def machine() -> tpl.Machine:
        instances, _ = store.instances()
        servers = {}
        for inst in instances.values():
            if inst.kind != "llama-swap":
                continue
            cat = store.catalog(inst.id)
            chat = next((m.id for m in cat.models if "chat" in m.aliases), "")
            servers[inst.id] = {"api_key": cat.api_key, "name": inst.name,
                                "default_model": "chat" if chat else (cat.models[0].id if cat.models else "")}
        return tpl.Machine(gpu_mem_gb=[g.mem_total / 1024 for g in engine.snapshot().gpus],
                           instances=instances, servers=servers, model_sizes={})

    def default_models_dir(instances: dict[str, Instance]) -> str:
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
        return str(engine.host_data_dir / "models")

    def model_files(models_dir: str) -> tuple[list[dict], list[str]]:
        probe = Instance(id="probe", name="probe", image="probe",
                         volumes=[VolumeMap(host=models_dir, container=tpl.MODELS_IN_CONTAINER)])
        return llamaswap.local_model_files(probe)

    def guided(request: Request, t: tpl.Template, f: dict, m: tpl.Machine, error: str = "", status: int = 200):
        files, notes = model_files(f["models_dir"]) if t.first_model and f.get("models_dir") else ([], [])
        gpus = engine.snapshot().gpus
        choices = [(k, v) for k, v in tpl.GPU_CHOICES.items()]
        choices[-1:-1] = [(f"pin:{g.index}", f"Always GPU {g.index} ({g.name}, {g.mem_total / 1024:.0f} GB)")
                          for g in gpus] if len(gpus) > 1 else []
        comfy = [(i.id, i.name) for i in m.instances.values() if i.kind == "container" and i.gpu.mode != "none"]
        return page(request, "instance_guided.html", status=status, t=t, f=f, error=error, files=files,
                    file_notes=notes, gpu_choices=choices, offload_choices=tpl.OFFLOAD_CHOICES,
                    servers=[(k, v["name"]) for k, v in m.servers.items()], comfy=comfy, gpus=gpus)

    @app.get("/instances/new", response_class=HTMLResponse)
    def new_instance(request: Request, kind: str = ""):
        if kind:                              # old links went straight to the full form
            return RedirectResponse(f"/instances/new/custom?kind={kind}", status_code=303)
        return page(request, "instance_new.html", groups=tpl.groups(), gpu_choices=tpl.GPU_CHOICES)

    @app.get("/instances/new/custom", response_class=HTMLResponse)
    def new_custom(request: Request, kind: str = "container"):
        return editor(request, blank_instance(kind), editing=False)

    @app.get("/instances/new/{key}", response_class=HTMLResponse)
    def new_from_template(request: Request, key: str):
        try:
            t = tpl.get(key)
        except ConfigError as err:
            return RedirectResponse(f"/instances/new?msg={err}", status_code=303)
        m = machine()
        f = tpl.defaults(t, m, default_models_dir(m.instances), str(engine.host_data_dir / "appdata"))
        return guided(request, t, f, m)

    @app.post("/instances/guided")
    async def save_guided(request: Request):
        raw = await request.form()
        try:
            t = tpl.get(str(raw.get("template", "")))
        except ConfigError as err:
            return toast(Response(status_code=204), str(err), "error")
        f = {k: str(raw.get(k, "")) for k in ("name", "id", "port", "gpu", "models_dir", "appdata_dir", "server",
                                               "comfyui", "file", "ctx_size", "offload")}
        f["autostart"] = raw.get("autostart") == "on"
        m = machine()
        if t.first_model and f["models_dir"]:
            m.model_sizes = {x["container_path"]: x["size_gb"] for x in model_files(f["models_dir"])[0]}
        try:
            inst, model = tpl.build(t, f, m)
            if raw.get("action") == "advanced":
                form = form_from_instance(inst)
                resp = editor(request, form, editing=False)
                resp.headers["HX-Retarget"] = "#editor"
                resp.headers["HX-Reselect"] = "#editor"
                return resp
            store.save_instance(inst)
            for v in inst.volumes:            # make the app's own folders, so they are not owned by root
                if not v.read_only and Path(v.host).is_relative_to(store.root):
                    try:
                        Path(v.host).mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
            if inst.kind == "llama-swap":
                cat = store.catalog(inst.id)
                if model:
                    cat.models.append(model)
                    store.save_catalog(Catalog.model_validate(cat.model_dump()))
        except (ConfigError, ValueError) as err:
            msg = friendly_errors(err) if isinstance(err, ValidationError) else str(err)
            resp = guided(request, t, f, m, error=msg)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        if inst.kind == "llama-swap":
            what = f"with {model.id}" if model else "- add a model below"
            return Response(status_code=204, headers={
                "HX-Redirect": f"/instances/{inst.id}/models?msg=Created {inst.name} {what}. Press Start on the dashboard to run it."})
        return Response(status_code=204, headers={
            "HX-Redirect": f"/?msg=Created {inst.name}. Press Start to run it.#inst-{inst.id}"})

    @app.get("/instances/{iid}/edit", response_class=HTMLResponse)
    def edit_instance(request: Request, iid: str):
        try:
            return editor(request, form_from_instance(store.instance(iid)), editing=True)
        except ConfigError as err:
            return RedirectResponse(f"/?msg={err}", status_code=303)

    @app.post("/instances/save")
    async def save_instance(request: Request):
        raw = await request.form()
        f = {k: raw.get(k, "") for k in ("id", "name", "description", "kind", "image", "command", "env", "ports",
                                          "volumes", "gpu_mode", "min_gpus", "max_gpus", "priority", "open_port",
                                          "open_path", "shm_size", "user_spec", "memory_limit", "editing",
                                          "drain_check", "drain_timeout", "drain_on_timeout", "drain_port",
                                          "drain_path", "drain_key")}
        f["gpu_list"] = [int(x) for x in raw.getlist("gpu_list") if str(x).isdigit()]
        f["autostart"] = raw.get("autostart") == "on"
        editing = f["editing"] == "1"
        try:
            if editing:
                store.instance(f["id"])       # must exist
            elif store.instance_path(f["id"]).exists():
                raise ConfigError(f"an instance called '{f['id']}' already exists")
            inst = Instance(
                id=f["id"].strip(), name=f["name"].strip() or f["id"], description=f["description"].strip(),
                kind=f["kind"] or "container", image=f["image"], command=parse_command(f["command"]),
                env=parse_env(f["env"]), ports=parse_ports(f["ports"]), volumes=parse_volumes(f["volumes"]),
                gpu=GpuPolicy(mode=f["gpu_mode"] or "none", gpus=f["gpu_list"],
                              min_gpus=int(f["min_gpus"] or 1), max_gpus=int(f["max_gpus"] or 0),
                              priority=int(f["priority"] or 50)),
                autostart=f["autostart"], open_port=int(f["open_port"]) if str(f["open_port"]).strip() else None,
                open_path=f["open_path"] or "/", shm_size=f["shm_size"] or None, user=f["user_spec"] or None,
                memory_limit=f["memory_limit"] or None,
                drain=DrainPolicy(check=f["drain_check"] or "auto",
                                  timeout=int(f["drain_timeout"]) if str(f["drain_timeout"]).strip() else 600,
                                  on_timeout=f["drain_on_timeout"] or "stop",
                                  port=int(f["drain_port"]) if str(f["drain_port"]).strip() else None,
                                  path=f["drain_path"], key=f["drain_key"]))
            store.save_instance(inst)
            if inst.kind == "llama-swap":
                store.catalog(inst.id)        # create an empty catalogue with an API key
        except (ConfigError, ValueError) as err:
            msg = friendly_errors(err) if isinstance(err, ValidationError) else str(err)
            resp = editor(request, f, editing, error=msg)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        st = backend.states().get(inst.id)
        note = " Restart it to apply the changes." if st and st.running else ""
        return Response(status_code=204, headers={"HX-Redirect": f"/?msg=Saved {inst.name}.{note}"})

    @app.post("/instances/{iid}/delete")
    def delete_instance(iid: str):
        try:
            inst = store.instance(iid)
            jobs.submit(f"Deleting {inst.name}", lambda logf: engine.remove_instance(iid, logf), wait=True)
        except (ConfigError, Busy) as err:
            return toast(Response(status_code=204), str(err), "error")
        return Response(status_code=204, headers={"HX-Redirect": f"/?msg=Deleted {inst.name}."})

    # ------------------------------------------------------------------ model catalogue
    def models_ctx(iid: str) -> dict:
        inst = store.instance(iid)
        if inst.kind != "llama-swap":
            raise ConfigError(f"{inst.name} is not a model server")
        cat = store.catalog(iid)
        st = backend.states().get(iid)
        running = bool(st and st.running)
        gpu_count = len(st.gpus) if running else 0
        loaded, err = {}, ""
        if running:
            try:
                loaded = llama_client(iid).running({m.id for m in cat.models})
            except Exception:  # noqa: BLE001
                err = "The model server is not answering yet (it may still be starting)."
        files, notes = llamaswap.local_model_files(inst)
        preview_gpus = gpu_count or max(len(engine.snapshot().gpus), 1)
        threads = cat.threads or 8
        return {"inst": inst, "cat": cat, "running": running, "gpu_count": gpu_count, "loaded": loaded,
                "models_error": err, "files": files, "file_notes": notes,
                "loading": sorted(app.state.loading.get(iid, set())),
                "resolved": llamaswap.resolved_aliases(cat, preview_gpus), "preview_gpus": preview_gpus,
                "preview": llamaswap.render(cat, preview_gpus, threads),
                "pending": bool(running and st.config_hash != engine.config_hash(inst))}

    @app.get("/instances/{iid}/models", response_class=HTMLResponse)
    def models_page(request: Request, iid: str):
        try:
            return page(request, "models.html", **models_ctx(iid))
        except ConfigError as err:
            return RedirectResponse(f"/?msg={err}", status_code=303)

    @app.get("/instances/{iid}/models/edit", response_class=HTMLResponse)
    def model_form(request: Request, iid: str, model: str = ""):
        inst, cat = store.instance(iid), store.catalog(iid)
        m = next((x for x in cat.models if x.id == model), None)
        files, notes = llamaswap.local_model_files(inst)
        f = m.model_dump() if m else {"id": "", "name": "", "description": "", "file": "", "ctx_size": 32768,
                                     "parallel": 1, "sampling": "", "extra_args": "", "kv_cache": "f16",
                                     "mmproj": "", "aliases": []}
        f["aliases_text"] = ", ".join(f.get("aliases") or [])
        f["offload_rows"] = offload_rows({r.min_gpus: r for r in (m.offload if m else [])})
        return page(request, "model_form.html", inst=inst, f=f, editing=bool(m), files=files, file_notes=notes,
                    error="")

    def offload_rows(rules: dict) -> list[dict]:
        """One form row per GPU count this machine can give a model server (1..number of GPUs),
        plus any count a saved rule mentions. Values are strings, as the form posts them."""
        top = max([len(engine.snapshot().gpus), 1] + [int(k) for k in rules])
        rows = []
        for n in range(1, top + 1):
            r = rules.get(n)
            if isinstance(r, OffloadRule):
                r = {"gpu_layers": "" if r.gpu_layers is None else str(r.gpu_layers),
                     "cpu_moe": str(r.cpu_moe) if r.cpu_moe else ""}
            rows.append({"gpus": n, **(r or {"gpu_layers": "", "cpu_moe": ""})})
        return rows

    def posted_offload(raw) -> dict[int, dict]:
        """The GPU and RAM rows as posted: GPU count -> {"gpu_layers": str, "cpu_moe": str}."""
        rows = {}
        for n, layers, moe in zip(raw.getlist("offload_gpus"), raw.getlist("offload_gpu_layers"),
                                  raw.getlist("offload_cpu_moe")):
            if str(n).isdigit() and 1 <= int(n) <= 64:
                rows[int(n)] = {"gpu_layers": str(layers).strip(), "cpu_moe": str(moe).strip()}
        return rows

    def offload_rules(rows: dict[int, dict]) -> list[OffloadRule]:
        """Rows with both boxes empty mean "no setting for this many GPUs"."""
        rules = []
        for n, row in rows.items():
            if not row["gpu_layers"] and not row["cpu_moe"]:
                continue
            if not all(v.isdigit() for v in (row["gpu_layers"], row["cpu_moe"]) if v):
                raise ConfigError(f"GPU and RAM settings for {n} GPU(s): use whole numbers")
            rules.append(OffloadRule(min_gpus=n, gpu_layers=int(row["gpu_layers"]) if row["gpu_layers"] else None,
                                     cpu_moe=int(row["cpu_moe"] or 0)))
        return rules

    @app.post("/instances/{iid}/models/save")
    async def save_model(request: Request, iid: str):
        raw = await request.form()
        inst, cat = store.instance(iid), store.catalog(iid)
        original = str(raw.get("original_id", ""))
        f = {k: str(raw.get(k, "")) for k in ("id", "name", "description", "file", "ctx_size", "parallel",
                                               "sampling", "extra_args", "kv_cache", "mmproj", "aliases_text")}
        posted = posted_offload(raw)
        f["offload_rows"] = offload_rows(posted)
        try:
            model = CatalogModel(id=f["id"].strip(), name=f["name"].strip(), description=f["description"].strip(),
                                 file=f["file"], ctx_size=int(f["ctx_size"] or 32768),
                                 parallel=int(f["parallel"] or 1), sampling=f["sampling"],
                                 extra_args=f["extra_args"], kv_cache=f["kv_cache"] or "f16",
                                 mmproj=f["mmproj"].strip(),
                                 aliases=[a.strip() for a in f["aliases_text"].split(",") if a.strip()],
                                 offload=offload_rules(posted))
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
            store.save_catalog(cat)
        except (ValueError, ConfigError) as err:
            msg = friendly_errors(err) if isinstance(err, ValidationError) else str(err)
            files, notes = llamaswap.local_model_files(inst)
            resp = page(request, "model_form.html", inst=inst, f=f, editing=bool(original), files=files,
                        file_notes=notes, error=msg, original=original)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg=Saved {model.id}."})

    @app.post("/instances/{iid}/models/{mid}/delete")
    def delete_model(iid: str, mid: str):
        cat = store.catalog(iid)
        try:
            cat = Catalog.model_validate({**cat.model_dump(), "models": [m.model_dump() for m in cat.models if m.id != mid]})
            store.save_catalog(cat)
        except ValidationError as err:
            return toast(Response(status_code=204), f"Cannot delete {mid}: {friendly_errors(err)}", "error")
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg=Deleted {mid}."})

    @app.post("/instances/{iid}/models/{mid}/{op}")
    def model_op(iid: str, mid: str, op: str):
        if op not in ("load", "unload"):
            return PlainTextResponse("unknown operation", status_code=400)
        cat = store.catalog(iid)
        if mid != "_all" and mid not in {m.id for m in cat.models}:
            return toast(Response(status_code=204), f"unknown model {mid}", "error")
        client = llama_client(iid)
        if op == "unload":
            # llama-swap cuts off the requests a model is answering when it is unloaded
            act = engine.activity(store.instance(iid))
            if act.busy:
                return toast(Response(status_code=204),
                             f"Still answering {act.detail}. Unloading now would cut that off: try again when it "
                             "finishes, or use Stop on the dashboard, which waits for it.", "warn")
            try:
                client.unload(None if mid == "_all" else mid)
            except Exception as err:  # noqa: BLE001
                return toast(Response(status_code=204), str(err), "error")
            return toast(Response(status_code=204, headers={"HX-Trigger-After-Settle": "refresh"}), "Model unloaded.")
        busy = app.state.loading.setdefault(iid, set())
        if mid in busy:
            return toast(Response(status_code=204), f"{mid} is already loading.")

        def worker():
            busy.add(mid)
            try:
                client.load(mid)
            except Exception as err:  # noqa: BLE001
                log.warning("loading %s failed: %s", mid, err)
            finally:
                busy.discard(mid)
        threading.Thread(target=worker, daemon=True).start()
        return toast(Response(status_code=204), f"Loading {mid} - large models take a few minutes from disk.")

    @app.post("/instances/{iid}/catalog/settings")
    async def catalog_settings(request: Request, iid: str):
        raw = await request.form()
        cat = store.catalog(iid)
        try:
            cat.threads = int(raw.get("threads") or 0)
            cat.llama_server = str(raw.get("llama_server", "")).strip()
            cat.health_timeout = int(raw.get("health_timeout") or 900)
            if raw.get("new_key") == "on":
                cat.api_key = "sk-" + secrets.token_hex(20)
            cat = Catalog.model_validate(cat.model_dump())
            store.save_catalog(cat)
        except (ValueError, ValidationError) as err:
            msg = friendly_errors(err) if isinstance(err, ValidationError) else str(err)
            return toast(Response(status_code=204), msg, "error")
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg=Settings saved."})

    @app.post("/instances/{iid}/catalog/auto")
    async def catalog_auto(request: Request, iid: str):
        raw = await request.form()
        cat = store.catalog(iid)
        autos = []
        try:
            for alias, default, rules in zip(raw.getlist("alias"), raw.getlist("default_model"), raw.getlist("rules")):
                alias = str(alias).strip()
                if not alias:
                    continue
                parsed = []
                for part in re.split(r"[,\n]", str(rules)):
                    part = part.strip()
                    if not part:
                        continue
                    m = re.match(r"^(\d+)\s*\+?\s*(?:gpus?)?\s*[=:]\s*(\S+)$", part, re.I)
                    if not m:
                        raise ValueError(f"rule '{part}': write it as 2=model-id (2 or more GPUs use model-id)")
                    parsed.append(AutoRule(min_gpus=int(m.group(1)), model=m.group(2)))
                autos.append(AutoAlias(alias=alias, default_model=str(default), rules=parsed))
            cat = Catalog.model_validate({**cat.model_dump(), "auto_aliases": [a.model_dump() for a in autos]})
            store.save_catalog(cat)
        except (ValueError, ValidationError) as err:
            msg = friendly_errors(err) if isinstance(err, ValidationError) else str(err)
            return toast(Response(status_code=204), msg, "error")
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg=Auto names saved."})

    # ------------------------------------------------------------------ updates (done by update.sh on the host)
    def update_page(request: Request, confirm: bool = False) -> Response:
        u = updates.status(store.root)
        return page(request, "partials/update.html", u=u, confirm=confirm,
                    log_tail=updates.live_log(store.root) if u.busy else "")

    @app.get("/ui/update", response_class=HTMLResponse)
    def update_dialog(request: Request, confirm: str = ""):
        return update_page(request, confirm == "1")

    @app.get("/ui/changelog", response_class=HTMLResponse)
    def changelog_dialog(request: Request):
        ours = changelog.read(changelog.find())
        # the newest changelog (saved by the last check) also lists versions not installed yet
        latest = changelog.read(updates.folder(store.root) / "changelog.md")
        shown = latest if latest and (not ours or changelog.key(latest[0].version) >= changelog.key(ours[0].version)) else ours
        mark = lambda v: "running" if v == __version__ else ("new" if changelog.key(v) > changelog.key(__version__) else "")  # noqa: E731
        releases = [(r, mark(r.version)) for r in shown]
        return page(request, "partials/changelog.html", releases=releases,
                    newer=sum(1 for _, m in releases if m == "new"))

    @app.get("/ui/update/card", response_class=HTMLResponse)
    def update_card(request: Request):
        return page(request, "partials/update_card.html", u=updates.status(store.root))

    @app.get("/ui/update/log", response_class=HTMLResponse)
    def update_log(request: Request):
        u = updates.status(store.root)
        return page(request, "partials/logs.html", title="Update log", refresh="/ui/update/log" if u.busy else "",
                    text=updates.log_text(store.root) or "Nothing yet: the updater writes here once it starts.")

    @app.post("/ui/update/credentials")
    async def update_credentials(request: Request):
        raw = await request.form()
        u = updates.status(store.root)
        if not u.dashboard_updates:
            return toast(Response(status_code=204), f"Saving a token needs the update service. On the server run: {u.command} --set-token <user>", "warn")
        if u.busy:
            return toast(Response(status_code=204), "An update or check is running; try again when it is done.", "warn")
        try:
            updates.request_credentials(store.root, str(raw.get("user", "")).strip(), str(raw.get("token", "")).strip(),
                                        clear=raw.get("clear") == "1")
        except ValueError as err:
            return toast(Response(status_code=204), str(err), "error")
        log.info("update: GitHub token %s from the dashboard", "removal requested" if raw.get("clear") == "1" else "handed to the host")
        resp = update_page(request)          # shows the check that follows, which proves the token works
        resp.headers["HX-Trigger"] = json.dumps({"refresh": True, "credentials-sent": True})
        return resp

    @app.post("/ui/update/{action}")
    def update_request(request: Request, action: str):
        if action not in updates.ACTIONS:
            return PlainTextResponse("unknown action", status_code=400)
        u = updates.status(store.root)
        if not u.dashboard_updates:
            return toast(Response(status_code=204), f"Updating from the dashboard is off. On the server run: {u.command}", "warn")
        if u.busy:
            return toast(Response(status_code=204), "An update or check is already running.", "warn")
        if action == "update" and jobs.busy:
            return toast(Response(status_code=204), "Wait for the current change to finish, then update.", "warn")
        if action == "update" and u.local_changes:
            return toast(Response(status_code=204), f"Files in the clone were changed by hand. Run {u.command} --stash on the server.", "error")
        updates.request(store.root, action)
        log.info("update: %s requested from the dashboard", action)
        resp = update_page(request)          # the dialog, now showing the progress
        resp.headers["HX-Trigger"] = json.dumps({"refresh": True})
        return resp

    # ------------------------------------------------------------------ settings
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return page(request, "settings.html", s=store.settings(), docker=backend.ping(), data_dir=str(store.root),
                    host_data_dir=str(engine.host_data_dir), initial=auth.initial_password_file(store).exists(),
                    error="", u=updates.status(store.root))

    @app.post("/settings/password")
    async def change_password(request: Request):
        raw = await request.form()
        s = store.settings()
        current, new, again = (str(raw.get(k, "")) for k in ("current", "new", "again"))
        if not auth.verify_password(current, s.password_hash):
            return toast(Response(status_code=204), "The current password is wrong.", "error")
        if len(new) < 10:
            return toast(Response(status_code=204), "Use at least 10 characters.", "error")
        if new != again:
            return toast(Response(status_code=204), "The new passwords do not match.", "error")
        s.password_hash = auth.hash_password(new)
        store.save_settings(s)
        auth.initial_password_file(store).unlink(missing_ok=True)
        return Response(status_code=204, headers={"HX-Redirect": "/settings?msg=Password changed."})

    @app.post("/settings/general")
    async def general_settings(request: Request):
        raw = await request.form()
        s = store.settings()
        bind, network = str(raw.get("bind_address", "")).strip(), str(raw.get("network", "")).strip()
        try:
            ipaddress.ip_address(bind)
        except ValueError:
            return toast(Response(status_code=204), "Bind address must be an IP address, e.g. 0.0.0.0", "error")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$", network):
            return toast(Response(status_code=204), "Invalid Docker network name.", "error")
        s.bind_address, s.network = bind, network
        store.save_settings(s)
        return Response(status_code=204, headers={"HX-Redirect": "/settings?msg=Saved. Restart instances to apply."})

    # ------------------------------------------------------------------ JSON for scripts
    @app.get("/api/state")
    def api_state():
        snap = engine.snapshot()
        machine = host_mod.query()
        return JSONResponse({
            "version": __version__,
            "commit": build.get("commit", ""),
            "update_available": updates.status(store.root).behind,
            "gpus": [g.__dict__ | {"users": snap.gpu_users(g.index)} for g in snap.gpus],
            "instances": {i: {"name": inst.name, "running": bool(snap.states.get(i) and snap.states[i].running),
                              "gpus": snap.allocations.get(i, []), "policy": inst.gpu.model_dump()}
                          for i, inst in snap.instances.items()},
            "host": {"cpu_pct": machine.cpu_pct, "cpu_threads": machine.cpu_threads, "load": machine.load,
                     "mem_total": machine.mem_total, "mem_used": machine.mem_used,
                     "mem_available": machine.mem_available},
            "job": jobs.current.title if jobs.current else None,
            "waiting_for": dict(jobs.current.waiting) if jobs.current else {},
        })

    # ------------------------------------------------------------------ autostart
    if autostart if autostart is not None else os.environ.get("SHEPHERD_AUTOSTART", "1") == "1":
        try:
            p = engine.autostart_plan()
            if not p.error and not p.is_noop:
                jobs.submit("Autostart", lambda job: engine.execute(p, job, job))
        except Exception as err:  # noqa: BLE001
            log.warning("autostart skipped: %s", err)

    return app
