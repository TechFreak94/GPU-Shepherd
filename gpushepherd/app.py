"""Web UI (FastAPI + Jinja2 + htmx). Server-rendered pages; htmx swaps in fragments."""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import __version__, api, auth, buildinfo, changelog, drain, host as host_mod, instance_templates as tpl, model_library, updates
from .backend import Backend, DockerBackend, FakeBackend
from .config import (AutoAlias, Catalog, CatalogModel, ConfigError, DrainPolicy, GpuPolicy, Instance, OffloadRule, Store,
                     format_command, format_env, format_ports, format_volumes, parse_command, parse_env, parse_ports,
                     parse_volumes)
from .connectors import llamaswap
from .engine import Engine
from .jobs import JobRunner
from .scheduler import gpu_text
from .service import Problem, Shepherd, message, parse_auto_rules

log = logging.getLogger("gpushepherd")
HERE = Path(__file__).parent
PUBLIC = ("/login", "/static/", "/healthz")
API = "/api/"
THEMES = ("light", "dark")  # no cookie = follow the system


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
        if demo:
            backend.helper_seconds = 120          # demo downloads take a while, like real ones
    # demo mode has no real servers to ask, so everything counts as idle
    engine = Engine(store, backend, activity=(lambda _inst: drain.Activity(0)) if demo else None)
    jobs = JobRunner()
    notices: list[str] = []                # shown with the dashboard's problems until a restart
    recovered = store.recover_settings()
    if recovered:
        log.error("%s", recovered)
        notices.append(recovered)
    try:
        backend.ensure_network(store.settings().network)
    except Exception as err:  # noqa: BLE001
        log.warning("could not prepare the Docker network: %s", err)
    generated = auth.ensure_admin(store)
    if generated:
        log.warning("First start: log in as '%s' with password '%s' (also in %s). Change it under Settings.",
                    store.settings().admin_user, generated, auth.initial_password_file(store))
    settings = store.settings()
    api_token = auth.api_token(store)      # made now, so `gpu-shepherd` commands find it in the data folder
    if demo:
        log.warning("Demo mode: commands need --token %s (or SHEPHERD_TOKEN)", api_token)

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
            if path.startswith(API):
                # scripts and the gpu-shepherd command: the API token; a logged-in browser may read
                bearer = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if not auth.check_api_token(store, bearer) and not (
                        request.method == "GET" and request.session.get("user")):
                    return JSONResponse({"error": "log in, or send the API token as 'Authorization: Bearer <token>'"},
                                        status_code=401)
            elif not path.startswith(PUBLIC):
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
        # picked in the browser (app.js), so the page is drawn in that theme from the start
        theme = request.cookies.get("theme", "")
        ctx.setdefault("theme", theme if theme in THEMES else "")
        return templates.TemplateResponse(request, template, ctx, status_code=status)

    def toast(resp: Response, msg: str, kind: str = "info") -> Response:
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": kind}})
        return resp

    def llama_client(iid: str):
        if demo:
            return DemoLlamaClient(store.catalog(iid))
        return engine.llama_client(iid)

    # everything the dashboard changes goes through the same calls as the JSON API and the command
    svc = Shepherd(store, engine, jobs, llama_client, loading=app.state.loading)
    app.state.svc = svc

    def base_url(request: Request, port: int, path: str = "/") -> str:
        """The address an instance's port is reached at, from the browser's point of view."""
        name = request.url.hostname or "localhost"
        if ":" in name:                   # IPv6 literal
            name = f"[{name}]"
        return f"http://{name}:{port}{path if path.startswith('/') else '/' + path}"

    def dashboard_ctx(request: Request) -> dict:
        problems, unknown = list(notices), False
        try:
            snap = engine.snapshot()
        except Exception as err:  # noqa: BLE001 - Docker restarting or gone: say so instead of an error page
            log.warning("dashboard: cannot read the containers: %s", err)
            snap, unknown = engine.snapshot(states={}), True
            problems.append(f"Docker is not reachable, so what is running is unknown: {err}")
        problems += snap.problems
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
            cat, cat_error = None, ""
            if inst.kind == "llama-swap":
                try:
                    cat = store.catalog(inst.id)
                except ConfigError as err:     # one bad model list must not take the whole dashboard down
                    cat_error = f"catalogs/{err}"
                    problems.append(cat_error)
            card = {"inst": inst, "state": st, "running": bool(st and st.running), "tag": tpl.tag_for(inst),
                    "status": (st.status if st else "unknown" if unknown else "not created"),
                    "container": engine.container_name(inst.id),
                    "gpus": gpus, "assigned": assigned,
                    "vram_used": sum(g.mem_used for g in assigned), "vram_total": sum(g.mem_total for g in assigned),
                    "usage": usage.get(engine.container_name(inst.id)),
                    "ports": [{"port": p, "url": (u := base_url(request, p.host)), "label": u[7:].rstrip("/")}
                              for p in inst.ports],
                    "open_url": base_url(request, inst.open_url_port, inst.open_path) if inst.open_url_port else "",
                    "pending": bool(st and st.running and st.config_hash and not cat_error
                                    and st.config_hash != engine.config_hash(inst)),
                    "loaded": [], "models_error": cat_error, "model_count": 0, "aliases": {}, "offload": {},
                    "loading": sorted(app.state.loading.get(inst.id, set())),
                    "waiting": (jobs.current.waiting.get(inst.id) if jobs.current else None)}
            if cat is not None:
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
        return {"cards": cards, "gpus": gpus, "gpu_error": snap.gpu_error, "problems": problems,
                "host": machine, "cpu_rows": cpu_rows, "mem_rows": mem_rows,
                "mem_other": max(machine.mem_used - sum(r["value"] for r in mem_rows), 0),
                "running_count": len(running), "instance_count": len(snap.instances),
                "job": jobs.current, "last_job": (jobs.history[0] if jobs.history else None),
                "now": time.time(), "u": updates.status(store.root), "app_updates": app_update_count()}

    def app_update_count() -> int:
        try:
            return len(svc.app_updates())
        except Exception:  # noqa: BLE001 - the dashboard shows without the notice
            log.exception("could not work out app updates")
            return 0

    def problem(err: Problem) -> Response:
        return toast(Response(status_code=204), err.msg, err.kind)

    def run_plan(request: Request, action: str, iid: str, confirm: bool, back: str = "") -> Response:
        try:
            c = svc.change(action, iid, confirm)
        except Problem as err:
            return problem(err)
        if c.noop:
            return toast(Response(status_code=204), c.text)
        if c.needs_confirm:
            resp = page(request, "partials/confirm.html", plan=c.plan, lines=c.lines, iid=iid, action=action,
                        name=c.name, back=back)
            resp.headers["HX-Retarget"] = "#modal-body"
            resp.headers["HX-Reswap"] = "innerHTML"
            resp.headers["HX-Trigger-After-Swap"] = "open-modal"
            return resp
        if back:
            return Response(status_code=204, headers={"HX-Redirect": f"{back}?msg={c.text}"})
        resp = page(request, "partials/dashboard.html", **dashboard_ctx(request))
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": c.text, "kind": "info"}, "close-modal": True})
        return resp

    # ------------------------------------------------------------------ login
    @app.get("/healthz")
    def healthz(request: Request, full: bool = False):
        body = {"ok": True, "version": __version__, "commit": build.get("commit", "")}
        if full:
            # update.sh asks for this after an update: the dashboard must draw too, not just the server
            # answer (a Docker outage is shown on the dashboard, so it doesn't fail this)
            try:
                page(request, "dashboard.html", **dashboard_ctx(request), docker=backend.ping(),
                     user=store.settings().admin_user)
            except Exception:  # noqa: BLE001
                log.exception("health check: the dashboard does not draw")
                return JSONResponse({**body, "ok": False}, status_code=500)
        return body

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
        text = svc.logs(iid)
        return page(request, "partials/logs.html", title=f"Logs: {inst.name}", text=text,
                    refresh=f"/ui/instances/{iid}/logs")

    @app.get("/ui/jobs/{jid}", response_class=HTMLResponse)
    def job_detail(request: Request, jid: int):
        try:
            job = svc.job(jid)
        except Problem as err:
            return PlainTextResponse(err.msg, status_code=err.status)
        return page(request, "partials/logs.html", title=f"{job.title} ({job.status})", text="\n".join(job.lines),
                    refresh=f"/ui/jobs/{jid}" if job.status == "running" else "")

    @app.post("/ui/jobs/{jid}/{op}")
    def job_control(jid: int, op: str):
        """The buttons shown while a change waits for running work: stop now, or call it off."""
        try:
            msg = svc.control_job(jid, op)
        except Problem as err:
            return toast(Response(status_code=204, headers={"HX-Trigger-After-Settle": "refresh"}), err.msg, err.kind)
        resp = Response(status_code=204)
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": "info"}})
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
                "drain_path": inst.drain.path, "drain_key": inst.drain.key, "app": inst.app,
                "requires": ", ".join(inst.requires)}

    def editor(request: Request, form: dict, editing: bool, error: str = "", status: int = 200):
        gpus, _ = engine.snapshot().gpus, None
        gpu_count = max(len(gpus), 1)
        try:
            app_t = tpl.get(form.get("app") or "") if not editing else None
        except ConfigError:
            app_t = None
        no_server = bool(app_t and app_t.connects) and not svc.has_llm_server()
        return page(request, "instance_form.html", status=status, f=form, editing=editing, error=error, app_t=app_t,
                    no_server=no_server,
                    gpu_indices=list(range(max(gpu_count, max([0] + list(form.get("gpu_list") or [])) + 1))))

    # ------------------------------------------------------------------ automatic install (templates)
    machine, model_files = svc.machine, svc.model_files

    def next_app(key: str) -> tpl.Template | None:
        """The app to go on to after installing an LLM server for it (?then=)."""
        try:
            t = tpl.get(key) if key else None
        except ConfigError:
            return None
        return t if t and t.connects else None

    def guided(request: Request, t: tpl.Template, f: dict, m: tpl.Machine, error: str = "", status: int = 200,
               then: tpl.Template | None = None):
        wants_files = bool(tpl.slots(t))
        files, notes = model_files(f["models_dir"]) if wants_files and f.get("models_dir") else ([], [])
        have = {x["container_path"] for x in files}
        files = [x for x in files if not x["name"].startswith(("mmproj", "mtp-"))]   # these go with a model
        gpus = engine.snapshot().gpus
        choices = [(k, v) for k, v in tpl.GPU_CHOICES.items()]
        choices[-1:-1] = [(f"pin:{g.index}", f"Always GPU {g.index} ({g.name}, {g.mem_total / 1024:.0f} GB)")
                          for g in gpus] if len(gpus) > 1 else []
        comfy = [(i.id, i.name) for i in m.instances.values() if i.kind == "container" and i.gpu.mode != "none"]
        keys = {sl.key for sl in tpl.slots(t)}
        suggested = [(s, fit, [sl for sl in tpl.slots(t) if sl.key in s.uses],
                      f"{tpl.MODELS_IN_CONTAINER}/{s.path(s.model)}" in have)
                     for s, fit in sorted(((s, model_library.fit(s, m.gpu_mem_gb, m.ram_gb))
                                           for s in model_library.SUGGESTIONS if keys & set(s.uses)),
                                          key=lambda x: x[1].rank)] if wants_files else []
        return page(request, "instance_guided.html", status=status, t=t, f=f, error=error, files=files,
                    suggested=suggested, models_root=tpl.MODELS_IN_CONTAINER,
                    file_notes=notes, gpu_choices=choices, offload_choices=tpl.OFFLOAD_CHOICES,
                    servers=[(k, v["name"]) for k, v in m.servers.items()], comfy=comfy, gpus=gpus,
                    new_server=tpl.NEW_SERVER, bundled=tpl.get(tpl.BUNDLED_SERVER), slots=tpl.slots(t),
                    companions=[tpl.get(c) for c in t.companions],
                    no_server=bool(t.connects) and not m.servers, then_t=then if t.kind == "llama-swap" else None)

    # ------------------------------------------------------------------ App Store
    @app.get("/apps", response_class=HTMLResponse)
    def app_store(request: Request):
        instances, _ = store.instances()
        return page(request, "apps.html", groups=tpl.groups(), gpu_choices=tpl.GPU_CHOICES,
                    installed=tpl.installed(instances),
                    has_server=any(i.kind == "llama-swap" for i in instances.values()), server_app=tpl.BUNDLED_SERVER)

    @app.get("/apps/{key}", response_class=HTMLResponse)
    def install_app(request: Request, key: str, mode: str = "auto", then: str = ""):
        try:
            t = tpl.get(key)
        except ConfigError as err:
            return RedirectResponse(f"/apps?msg={err}", status_code=303)
        if t.key != key:                      # renamed app (the chat and coding servers are one LLM server now)
            return RedirectResponse(f"/apps/{t.key}" + ("?mode=manual" if mode == "manual" else ""), status_code=303)
        if mode != "manual":
            _, f, m = svc.app_defaults(key)
            return guided(request, t, f, m, then=next_app(then))
        # manual: the full form, filled in from the same defaults (a model server is picked, not made)
        return editor(request, form_from_instance(svc.app_instance(key)), editing=False)

    @app.get("/instances/new")
    def new_instance(kind: str = ""):
        if kind:                              # old links went straight to the full form
            return RedirectResponse(f"/instances/new/custom?kind={kind}", status_code=303)
        return RedirectResponse("/apps", status_code=303)

    @app.get("/instances/new/custom", response_class=HTMLResponse)
    def new_custom(request: Request, kind: str = "container"):
        return editor(request, blank_instance(kind), editing=False)

    @app.get("/instances/new/{key}")
    def new_from_template(key: str):          # links from before the App Store
        return RedirectResponse(f"/apps/{key}", status_code=303)

    @app.post("/instances/guided")
    async def save_guided(request: Request):
        raw = await request.form()
        try:
            t = tpl.get(str(raw.get("template", "")))
        except ConfigError as err:
            return toast(Response(status_code=204), str(err), "error")
        slot_fields = [f"{x.key}_{part}" for x in tpl.slots(t) for part in ("file", "ctx")]
        f = {k: str(raw.get(k, "")) for k in ["name", "id", "port", "gpu", "models_dir", "appdata_dir", "server",
                                              "comfyui", "offload", "login_user", "password", "hass_url",
                                              "hass_token", *slot_fields]}
        f["autostart"] = raw.get("autostart") == "on"
        then = next_app(str(raw.get("then", ""))) if t.kind == "llama-swap" else None
        if raw.get("action") == "advanced":
            try:
                inst = svc.app_instance(t.key, f)
            except Problem as err:
                resp = guided(request, t, f, machine(), error=err.msg, then=then)
            else:
                resp = editor(request, form_from_instance(inst), editing=False)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        try:
            done = svc.install(t.key, f)
        except Problem as err:
            resp = guided(request, t, f, machine(), error=err.msg, then=then)   # without a model server made on the way
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        inst, models, bundle = done.inst, done.models, done.bundle
        fetching = f" Downloading {', '.join(done.downloads)}: progress is on its Models page." if done.downloads else ""
        if then:                              # installed for another app: go on to that one, now connected to it
            what = f" with {', '.join(x.id for x in models)}" if models else ""
            return Response(status_code=204, headers={
                "HX-Redirect": f"/apps/{then.key}?msg=Installed {inst.name}{what}.{fetching} Now set up {then.title}: it will use {inst.name}."})
        if inst.kind == "llama-swap":
            what = f"with {', '.join(x.id for x in models)}" if models else "- add a model below"
            return Response(status_code=204, headers={
                "HX-Redirect": f"/instances/{inst.id}/models?msg=Installed {inst.name} {what}.{fetching} Press Start on the dashboard to run it."})
        also = "".join(f" and {x.name}" for x in [bundle, *done.companions] if x)
        after = f" {t.after_install}" if t.after_install else ""
        return Response(status_code=204, headers={
            "HX-Redirect": f"/?msg=Installed {inst.name}{also}. Press Start to run it.{after}{fetching}#inst-{inst.id}"})

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
                                          "drain_path", "drain_key", "app", "requires")}
        f["gpu_list"] = [int(x) for x in raw.getlist("gpu_list") if str(x).isdigit()]
        f["autostart"] = raw.get("autostart") == "on"
        editing = f["editing"] == "1"
        try:
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
                                  path=f["drain_path"], key=f["drain_key"]),
                app=str(f["app"]).strip(), requires=re.split(r"[\s,]+", str(f["requires"]).strip()))
            msg = svc.save_instance(inst, editing)
        except (Problem, ConfigError, ValueError) as err:
            resp = editor(request, f, editing, error=err.msg if isinstance(err, Problem) else message(err))
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        return Response(status_code=204, headers={"HX-Redirect": f"/?msg={msg}"})

    @app.post("/instances/{iid}/delete")
    def delete_instance(iid: str):
        try:
            msg = svc.delete_instance(iid)
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/?msg={msg}"})

    # ------------------------------------------------------------------ model catalogue
    def models_ctx(iid: str) -> dict:
        inst, cat = svc.server(iid)
        st = backend.states().get(iid)
        running = bool(st and st.running)
        gpu_count = len(st.gpus) if running else 0
        loaded, err = svc.loaded(iid, cat)
        files, notes = llamaswap.local_model_files(inst)
        preview_gpus = gpu_count or max(len(engine.snapshot().gpus), 1)
        threads = cat.threads or 8
        return {"inst": inst, "cat": cat, "running": running, "gpu_count": gpu_count, "loaded": loaded,
                "models_error": err, "files": files, "file_notes": notes,
                "loading": sorted(app.state.loading.get(iid, set())),
                "resolved": llamaswap.resolved_aliases(cat, preview_gpus), "preview_gpus": preview_gpus,
                "preview": llamaswap.render(cat, preview_gpus, threads),
                "pending": bool(running and st.config_hash != engine.config_hash(inst)),
                "suggestions": svc.suggestions(iid), "download_dir": svc.models_dir(inst)}

    @app.get("/instances/{iid}/models", response_class=HTMLResponse)
    def models_page(request: Request, iid: str):
        try:
            return page(request, "models.html", **models_ctx(iid))
        except Problem as err:
            return RedirectResponse(f"/?msg={err.msg}", status_code=303)

    @app.get("/instances/{iid}/suggestions", response_class=HTMLResponse)
    def suggestions_part(request: Request, iid: str):
        try:
            inst, _ = svc.server(iid)
            return page(request, "partials/suggestions.html", inst=inst, suggestions=svc.suggestions(iid),
                        download_dir=svc.models_dir(inst))
        except Problem as err:
            return problem(err)

    @app.post("/instances/{iid}/suggestions/{key}/{op}")
    def suggestion_op(request: Request, iid: str, key: str, op: str):
        try:
            if op == "add":
                msg = svc.add_suggestion(iid, key)
                return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg={msg}"})
            if op == "download":
                msg = svc.download_suggestion(iid, key)
            elif op == "cancel":
                msg = svc.cancel_download(iid, key)
            else:
                return PlainTextResponse("unknown operation", status_code=400)
        except Problem as err:
            return problem(err)
        return toast(suggestions_part(request, iid), msg)

    @app.get("/instances/{iid}/models/edit", response_class=HTMLResponse)
    def model_form(request: Request, iid: str, model: str = ""):
        inst, cat = store.instance(iid), store.catalog(iid)
        m = next((x for x in cat.models if x.id == model), None)
        files, notes = llamaswap.local_model_files(inst)
        f = m.model_dump() if m else CatalogModel(id="new", file="/models/new.gguf").model_dump() | {"id": "", "file": ""}
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
        inst = store.instance(iid)
        original = str(raw.get("original_id", ""))
        f = {k: str(raw.get(k, "")) for k in ("id", "name", "description", "file", "ctx_size", "parallel",
                                               "sampling", "extra_args", "kv_cache", "mmproj", "aliases_text",
                                               "role", "thinking", "mtp", "mtp_draft", "split_mode", "tensor_split",
                                               "ttl")}
        f |= {k: raw.get(k) == "on" for k in ("keep_loaded", "preload")}
        posted = posted_offload(raw)
        f["offload_rows"] = offload_rows(posted)
        try:
            model = CatalogModel(id=f["id"].strip(), name=f["name"].strip(), description=f["description"].strip(),
                                 file=f["file"], ctx_size=int(f["ctx_size"] or 32768),
                                 parallel=int(f["parallel"] or 1), sampling=f["sampling"],
                                 extra_args=f["extra_args"], kv_cache=f["kv_cache"] or "f16",
                                 mmproj=f["mmproj"].strip(),
                                 aliases=[a.strip() for a in f["aliases_text"].split(",") if a.strip()],
                                 offload=offload_rules(posted), role=f["role"] or "chat",
                                 thinking=f["thinking"] or "auto", mtp=int(f["mtp"] or 0), mtp_draft=f["mtp_draft"],
                                 split_mode=f["split_mode"] or "layer", tensor_split=f["tensor_split"],
                                 ttl=int(f["ttl"] or 0), keep_loaded=f["keep_loaded"], preload=f["preload"])
            msg = svc.save_model(iid, model, original)
        except (Problem, ValueError, ConfigError) as err:
            files, notes = llamaswap.local_model_files(inst)
            resp = page(request, "model_form.html", inst=inst, f=f, editing=bool(original), files=files,
                        file_notes=notes, error=err.msg if isinstance(err, Problem) else message(err), original=original)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg={msg}"})

    @app.post("/instances/{iid}/models/{mid}/delete")
    def delete_model(iid: str, mid: str):
        try:
            msg = svc.delete_model(iid, mid)
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg={msg}"})

    @app.post("/instances/{iid}/models/{mid}/{op}")
    def model_op(iid: str, mid: str, op: str):
        if op not in ("load", "unload"):
            return PlainTextResponse("unknown operation", status_code=400)
        try:
            if op == "unload":
                return toast(Response(status_code=204, headers={"HX-Trigger-After-Settle": "refresh"}),
                             svc.unload_model(iid, mid))
            return toast(Response(status_code=204), svc.load_model(iid, mid))
        except Problem as err:
            return problem(err)

    @app.post("/instances/{iid}/catalog/settings")
    async def catalog_settings(request: Request, iid: str):
        raw = await request.form()
        try:
            msg = svc.server_settings(iid, threads=int(raw.get("threads") or 0),
                                      llama_server=str(raw.get("llama_server", "")),
                                      health_timeout=int(raw.get("health_timeout") or 900),
                                      new_key=raw.get("new_key") == "on")
        except ValueError:
            return toast(Response(status_code=204), "Threads and the start-up wait are whole numbers.", "error")
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg={msg}"})

    @app.post("/instances/{iid}/catalog/auto")
    async def catalog_auto(request: Request, iid: str):
        raw = await request.form()
        try:
            autos = [AutoAlias(alias=str(alias).strip(), default_model=str(default), rules=parse_auto_rules(rules))
                     for alias, default, rules in zip(raw.getlist("alias"), raw.getlist("default_model"),
                                                      raw.getlist("rules")) if str(alias).strip()]
            msg = svc.set_auto_aliases(iid, autos)
        except ValueError as err:
            return toast(Response(status_code=204), message(err), "error")
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/instances/{iid}/models?msg={msg}"})

    # ------------------------------------------------------------------ updates (done by update.sh on the host)
    def update_page(request: Request, confirm: bool = False) -> Response:
        u = updates.status(store.root)
        return page(request, "partials/update.html", u=u, confirm=confirm,
                    log_tail=updates.live_log(store.root) if u.busy else "")

    @app.get("/ui/update", response_class=HTMLResponse)
    def update_dialog(request: Request, confirm: str = ""):
        return update_page(request, confirm == "1")

    @app.get("/ui/app-updates", response_class=HTMLResponse)
    def app_updates_dialog(request: Request):
        return page(request, "partials/app_updates.html", updates=svc.app_updates(), names=svc.names())

    @app.post("/ui/app-updates/{iid}/{key}/{op}")
    def app_update_op(request: Request, iid: str, key: str, op: str):
        if op not in ("apply", "skip"):
            return PlainTextResponse("unknown operation", status_code=400)
        try:
            msg = svc.apply_app_update(iid, key) if op == "apply" else svc.skip_app_update(iid, key)
        except Problem as err:
            return toast(app_updates_dialog(request), err.msg, err.kind)
        resp = app_updates_dialog(request)
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": "info"}})
        return resp

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
        try:
            svc.update_credentials(str(raw.get("user", "")), str(raw.get("token", "")), clear=raw.get("clear") == "1")
        except Problem as err:
            return problem(err)
        resp = update_page(request)          # shows the check that follows, which proves the token works
        resp.headers["HX-Trigger"] = json.dumps({"refresh": True, "credentials-sent": True})
        return resp

    @app.post("/ui/update/{action}")
    def update_request(request: Request, action: str):
        if action not in updates.ACTIONS:
            return PlainTextResponse("unknown action", status_code=400)
        try:
            svc.request_update(action)
        except Problem as err:
            return problem(err)
        resp = update_page(request)          # the dialog, now showing the progress
        resp.headers["HX-Trigger"] = json.dumps({"refresh": True})
        return resp

    # ------------------------------------------------------------------ settings
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return page(request, "settings.html", s=store.settings(), docker=backend.ping(), data_dir=str(store.root),
                    host_data_dir=str(engine.host_data_dir), initial=auth.initial_password_file(store).exists(),
                    error="", u=updates.status(store.root), api_token=auth.api_token(store))

    @app.post("/settings/password")
    async def change_password(request: Request):
        raw = await request.form()
        try:
            msg = svc.change_password(*(str(raw.get(k, "")) for k in ("current", "new", "again")))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/settings?msg={msg}"})

    @app.post("/settings/general")
    async def general_settings(request: Request):
        raw = await request.form()
        try:
            msg = svc.set_containers(str(raw.get("bind_address", "")), str(raw.get("network", "")))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": f"/settings?msg={msg}"})

    @app.post("/settings/api-token")
    def api_token_new():
        auth.new_api_token(store)
        log.info("API token replaced from the dashboard")
        return Response(status_code=204, headers={"HX-Redirect": "/settings?msg=New API token made. The old one no longer works."})

    # ------------------------------------------------------------------ JSON API (api.py)
    app.include_router(api.router(svc, build))

    # ------------------------------------------------------------------ autostart
    if autostart if autostart is not None else os.environ.get("SHEPHERD_AUTOSTART", "1") == "1":
        try:
            p = engine.autostart_plan()
            if not p.error and not p.is_noop:
                jobs.submit("Autostart", lambda job: engine.execute(p, job, job))
        except Exception as err:  # noqa: BLE001
            log.warning("autostart skipped: %s", err)

    return app
