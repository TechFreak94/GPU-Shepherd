"""Web UI (FastAPI + Jinja2 + htmx). Server-rendered pages; htmx swaps in fragments."""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import __version__, api, auth, buildinfo, changelog, credits, drain, drives, first_run, gpu_unlock, host as host_mod, image_updates, instance_templates as tpl, logins, model_library, moves, updates
from .backend import Backend, DockerBackend, FakeBackend
from .config import (AutoAlias, Catalog, CatalogModel, ConfigError, DrainPolicy, GpuPolicy, Instance, OffloadRule, Store,
                     format_command, format_env, format_ports, format_volumes, parse_command, parse_env, parse_ports,
                     parse_volumes)
from .connectors import llamaswap
from .engine import Engine
from .jobs import JobRunner
from .scheduler import gpu_text
from .service import POWER_ACTIONS, Problem, Shepherd, gb, message, parse_auto_rules

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

    def metrics(self, wait=0):
        return []


def with_msg(path: str, msg: str) -> str:
    """A page address that shows `msg` as a notice. Encoded: headers only take latin-1, and the
    messages have "…", app names and "&" in them."""
    return f"{path}?msg={quote(msg, safe='')}"


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
            backend.remote["ghcr.io/open-webui/open-webui:main"] = "sha256:" + "d" * 64   # a release to update to
    # demo mode has no real servers to ask, so everything counts as idle
    engine = Engine(store, backend, activity=(lambda _inst: drain.Activity(0)) if demo else None)
    jobs = JobRunner()
    notices: list[str] = []                # shown with the dashboard's problems until a restart
    recovered = store.recover_settings()
    if recovered:
        log.error("%s", recovered)
        notices.append(recovered)
    store.tighten()                        # instance files from older versions: owner-only, like new ones
    try:
        backend.ensure_network(store.settings().network)
    except Exception as err:  # noqa: BLE001
        log.warning("could not prepare the Docker network: %s", err)
    generated = auth.ensure_admin(store)
    if generated:
        log.warning("First start: log in as '%s' with password '%s' (also in %s). Change it under Settings.",
                    store.settings().admin_user, generated, auth.initial_password_file(store))
    preset = None
    if not demo:
        try:                               # the published image's first start (first_run.py)
            preset = first_run.create_preset(store)
        except Exception as err:  # noqa: BLE001 - starting empty is better than not starting
            log.warning("could not create the preset: %s", err)
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
    templates.env.filters["gb"] = gb
    templates.env.filters["by_disk"] = drives.group_by_disk
    templates.env.filters["md"] = changelog.inline
    templates.env.filters["ago"] = image_updates.ago
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ------------------------------------------------------------------ security
    limiter = auth.LoginLimiter()

    def session_user(request: Request) -> str:
        """Who is logged in ("" = nobody). A cookie from before the last password change or logout
        no longer counts (auth.session_stamp)."""
        user = request.session.get("user")
        if not user:
            return ""
        s = store.settings()
        if user != s.admin_user or not auth.same(str(request.session.get("stamp", "")), auth.session_stamp(s)):
            return ""
        return user

    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if path.startswith(API):
                # scripts and the gpu-shepherd command: the API token; a logged-in browser may read
                bearer = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if not auth.check_api_token(store, bearer) and not (
                        request.method == "GET" and session_user(request)):
                    return JSONResponse({"error": "log in, or send the API token as 'Authorization: Bearer <token>'"},
                                        status_code=401)
            elif not path.startswith(PUBLIC):
                if not session_user(request):
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
        ctx.setdefault("user", session_user(request))
        ctx.setdefault("flash", request.query_params.get("msg", ""))
        # picked in the browser (app.js), so the page is drawn in that theme from the start
        theme = request.cookies.get("theme", "")
        ctx.setdefault("theme", theme if theme in THEMES else "")
        return templates.TemplateResponse(request, template, ctx, status_code=status)

    def threaded(fn):
        """For a route that takes a form: read it first (request.state.form), then run the route in a
        worker thread like the plain `def` routes, so one that waits for Docker, a model server or a
        job never holds up the dashboard for everyone else."""
        @functools.wraps(fn)
        async def wrapper(request: Request, **kw):
            request.state.form = await request.form()
            return await run_in_threadpool(fn, request, **kw)
        return wrapper

    def toast(resp: Response, msg: str, kind: str = "info") -> Response:
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": kind}})
        return resp

    def llama_client(iid: str):
        if demo:
            return DemoLlamaClient(store.catalog(iid))
        return engine.llama_client(iid)

    # everything the dashboard changes goes through the same calls as the JSON API and the command
    svc = Shepherd(store, engine, jobs, llama_client, loading=app.state.loading,
                   mover=moves.LocalMover() if demo else None,
                   host_fs=drives.LocalHost(drives.demo_drives()) if demo else None)
    app.state.svc = svc
    try:
        svc.trash.purge_older(store.settings().trash_days)
    except Exception as err:  # noqa: BLE001 - never worth not starting for
        log.warning("could not empty old trash entries: %s", err)
    if preset:
        try:
            first_run.download(preset, svc.downloads)
        except Exception as err:  # noqa: BLE001
            log.warning("could not start downloading the preset's models: %s", err)

    def base_url(request: Request, port: int, path: str = "/") -> str:
        """The address an instance's port is reached at, from the browser's point of view."""
        name = request.url.hostname or "localhost"
        if ":" in name:                   # IPv6 literal
            name = f"[{name}]"
        return f"http://{name}:{port}{path if path.startswith('/') else '/' + path}"

    def dashboard_ctx(request: Request) -> dict:
        problems, unknown = list(notices), False
        job = jobs.current                  # read once: the job can finish while the page is built
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
            svc.peaks.record({i: usage[engine.container_name(i)].mem_used for i in running
                              if engine.container_name(i) in usage})
        except Exception:  # noqa: BLE001
            usage = {}
        gpu_by_index = {g.index: g for g in snap.gpus}
        try:
            images = svc.image_status(snap.states)
        except Exception:  # noqa: BLE001 - the cards show without update notices
            log.exception("could not read the app update checks")
            images = {}
        try:
            gpu_notes = engine.gpu_notes(snap)
        except Exception:  # noqa: BLE001 - only a hint
            log.exception("could not work out the GPU notes")
            gpu_notes = {}
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
                    "container": engine.container_name(inst.id), "has_logins": logins.method(inst) is not None,
                    "gpus": gpus, "assigned": assigned, "image": images.get(inst.id),
                    "vram_used": sum(g.mem_used for g in assigned), "vram_total": sum(g.mem_total for g in assigned),
                    "usage": usage.get(engine.container_name(inst.id)),
                    "ports": [{"port": p, "url": (u := base_url(request, p.host)), "label": u[7:].rstrip("/")}
                              for p in inst.ports],
                    "open_url": base_url(request, inst.open_url_port, inst.open_path) if inst.open_url_port else "",
                    "pending": bool(st and st.running and st.config_hash and not cat_error
                                    and st.config_hash != engine.config_hash(inst)),
                    "loaded": [], "models_error": cat_error, "model_count": 0, "aliases": {}, "offload": {},
                    "loading": sorted(app.state.loading.get(inst.id, set())), "gpu_note": gpu_notes.get(inst.id, ""),
                    "waiting": (job.waiting.get(inst.id) if job else None)}
            if cat is not None:
                card["model_count"] = len(cat.models)
                try:
                    card["tuning_new"] = sum(1 for s in svc.tuning(inst.id) if s["status"] == "new")
                except Exception:  # noqa: BLE001 - only a hint
                    card["tuning_new"] = 0
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
        try:
            unlock = gpu_unlock.badges(gpu_unlock.status(store.root), {g.index: g.bus_id for g in snap.gpus})
        except Exception:  # noqa: BLE001 - the GPU tiles show without the badge
            log.exception("could not read the gpu-unlock status")
            unlock = {}
        gpus = [{"gpu": g, "unlock": unlock.get(g.index),
                 "users": [(i, snap.instances[i].name) for i, a in snap.allocations.items()
                           if g.index in a and i in snap.instances]} for g in snap.gpus]
        mem_rows.sort(key=lambda r: -r["value"])
        cpu_rows.sort(key=lambda r: -(r["value"] or 0))
        return {"cards": cards, "gpus": gpus, "gpu_error": snap.gpu_error, "problems": problems,
                "host": machine, "cpu_rows": cpu_rows, "mem_rows": mem_rows,
                "mem_other": max(machine.mem_used - sum(r["value"] for r in mem_rows), 0),
                "running_count": len(running), "instance_count": len(snap.instances),
                "job": job, "last_job": (jobs.history[0] if jobs.history else None),
                "now": time.time(), "u": updates.status(store.root), "app_updates": app_update_count(),
                "image_updates": [c["inst"].name for c in cards
                                  if c.get("image") and c["image"].state == "available"],
                "low_space": low_space()}

    def app_update_count() -> int:
        try:
            return len(svc.app_updates())
        except Exception:  # noqa: BLE001 - the dashboard shows without the notice
            log.exception("could not work out app updates")
            return 0

    def low_space() -> list:
        try:
            return svc.low_space()
        except Exception:  # noqa: BLE001 - the dashboard shows without the banner
            log.exception("could not check free disk space")
            return []

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
                        name=c.name, back=back, warnings=c.warnings)
            resp.headers["HX-Retarget"] = "#modal-body"
            resp.headers["HX-Reswap"] = "innerHTML"
            resp.headers["HX-Trigger-After-Swap"] = "open-modal"
            return resp
        if back:
            return Response(status_code=204, headers={"HX-Redirect": with_msg(back, c.text)})
        resp = page(request, "partials/dashboard.html", **dashboard_ctx(request))
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": c.text, "kind": "info"}, "close-modal": True})
        return resp

    # ------------------------------------------------------------------ login
    @app.get("/healthz")
    def healthz(request: Request, full: bool = False):
        body = {"ok": True, "version": __version__, "commit": build.get("commit", "")}
        # drawing the dashboard asks Docker and nvidia-smi: only for update.sh (which sends the API
        # token) or a logged-in browser, not for anyone who can reach the port
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if full and (auth.check_api_token(store, bearer) or session_user(request)):
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
        client = request.client.host if request.client else "?"
        first_run = auth.initial_password_file(store).exists()
        wait = limiter.wait(client)
        if wait:
            log.warning("login: refused from %s (too many wrong logins)", client)
            mins = -(-wait // 60)
            return page(request, "login.html", status=429, first_run=first_run,
                        error=f"Too many wrong logins. Try again in {mins} minute{'s' if mins != 1 else ''}.")

        def check() -> bool:      # scrypt takes a moment: off the event loop, so the dashboard keeps answering
            # Copying the password out of a terminal easily picks up a space or line break at either end.
            return auth.verify_password(password, s.password_hash) or (
                password != password.strip() and auth.verify_password(password.strip(), s.password_hash))
        ok = await run_in_threadpool(check)
        known = auth.same(user.strip(), s.admin_user)
        if known and ok:
            log.info("login: '%s' logged in from %s", s.admin_user, client)
            limiter.passed(client)
            request.session.clear()
            request.session.update(user=s.admin_user, stamp=auth.session_stamp(s))
            return RedirectResponse("/?login=1", status_code=303)
        limiter.failed(client)
        log.warning("login: rejected from %s (%s)", client, "wrong password" if known else f"unknown user {user[:80]!r}")
        return page(request, "login.html", status=401, error="Wrong user name or password.", first_run=first_run)

    @app.post("/logout")
    def logout(request: Request):
        # a copy of the cookie would stay valid until it expires: end every login made so far
        s = store.settings()
        s.session_epoch += 1
        store.save_settings(s)
        request.session.clear()
        return Response(status_code=204, headers={"HX-Redirect": "/login"})

    # ------------------------------------------------------------------ dashboard
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard.html", **dashboard_ctx(request), docker=backend.ping())

    @app.get("/ui/dashboard", response_class=HTMLResponse)
    def dashboard_partial(request: Request):
        return page(request, "partials/dashboard.html", **dashboard_ctx(request))

    def logins_dialog(request: Request, iid: str, result: tuple[str, str] | None = None, error: str = ""):
        info = svc.logins(iid)
        return page(request, "partials/logins.html", title=f"Logins: {info['name']}", info=info, result=result,
                    error=error)

    @app.get("/ui/instances/{iid}/logins", response_class=HTMLResponse)
    def instance_logins(request: Request, iid: str):
        try:
            return logins_dialog(request, iid)
        except Problem as err:
            return problem(err)

    @app.post("/ui/instances/{iid}/logins", response_class=HTMLResponse)
    @threaded
    def instance_logins_reset(request: Request, iid: str):
        form = request.state.form
        try:
            result = svc.reset_login(iid, str(form.get("user", "")), str(form.get("password", "")))
        except Problem as err:
            try:
                return logins_dialog(request, iid, error=err.msg)
            except Problem:
                return problem(err)
        return logins_dialog(request, iid, result=result)

    @app.post("/ui/instances/{iid}/update-image")
    def instance_update_image(request: Request, iid: str):
        try:
            msg, _ = svc.update_image(iid)
        except Problem as err:
            return problem(err)
        return toast(page(request, "partials/dashboard.html", **dashboard_ctx(request)), msg)

    @app.post("/ui/image-updates/update-all")
    def update_all_images(request: Request):
        try:
            msg, _ = svc.update_images()
        except Problem as err:
            return problem(err)
        return toast(page(request, "partials/dashboard.html", **dashboard_ctx(request)), msg)

    @app.post("/ui/instances/{iid}/auto-update")
    @threaded
    def instance_auto_update(request: Request, iid: str):
        form = request.state.form
        try:
            msg = svc.set_auto_update(iid, form.get("on") == "1")
        except Problem as err:
            return problem(err)
        return toast(page(request, "partials/dashboard.html", **dashboard_ctx(request)), msg)

    @app.post("/ui/image-updates/check")
    def check_images(request: Request):
        try:
            msg = svc.check_images()
        except Problem as err:
            return problem(err)
        resp = toast(Response(status_code=204), msg)
        resp.headers["HX-Refresh"] = "true"
        return resp

    @app.post("/settings/app-updates")
    @threaded
    def settings_app_updates(request: Request):
        raw = request.state.form
        hours = str(raw.get("hours", "")).strip()
        if not hours.isdigit():
            return problem(Problem("Hours must be a whole number (0 checks only when you press Check now)."))
        try:
            return toast(Response(status_code=204), svc.set_image_check_hours(int(hours)))
        except Problem as err:
            return problem(err)

    @app.post("/ui/instances/{iid}/{action}")
    @threaded
    def instance_action(request: Request, iid: str, action: str):
        if action not in ("start", "stop", "restart"):
            return PlainTextResponse("unknown action", status_code=400)
        form = request.state.form
        back = str(form.get("back", ""))
        if back and not re.match(r"^/(?!/)[A-Za-z0-9/_-]*$", back):   # local paths only (//host is another site)
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
                    "cpu_fallback": True,
                    "volumes": "/srv/models:/models:ro", "name": "Llama Swap", "id": "llm",
                    "drain_check": "auto", "drain_timeout": 600, "drain_on_timeout": "stop"}
        return {"kind": "container", "gpu_mode": "none", "min_gpus": 1, "max_gpus": 0, "priority": 50, "open_path": "/",
                "drain_check": "auto", "drain_timeout": 600, "drain_on_timeout": "stop"}

    def form_from_instance(inst: Instance) -> dict:
        return {"id": inst.id, "name": inst.name, "description": inst.description, "kind": inst.kind,
                "image": inst.image, "command": format_command(inst.command), "env": format_env(inst.env),
                "ports": format_ports(inst.ports), "volumes": format_volumes(inst.volumes),
                "gpu_mode": inst.gpu.mode, "gpu_list": inst.gpu.gpus, "min_gpus": inst.gpu.min_gpus,
                "max_gpus": inst.gpu.max_gpus, "priority": inst.gpu.priority, "autostart": inst.autostart,
                "vram_gb": "%g" % inst.gpu.vram_gb if inst.gpu.vram_gb else "", "cpu_fallback": inst.gpu.cpu_fallback,
                "open_port": inst.open_port or "", "open_path": inst.open_path, "shm_size": inst.shm_size or "",
                "user_spec": inst.user or "", "memory_limit": inst.memory_limit or "",
                "drain_check": inst.drain.check, "drain_timeout": inst.drain.timeout,
                "drain_on_timeout": inst.drain.on_timeout, "drain_port": inst.drain.port or "",
                "drain_path": inst.drain.path, "drain_key": inst.drain.key, "app": inst.app,
                "requires": ", ".join(inst.requires), "auto_update": inst.auto_update}

    def editor(request: Request, form: dict, editing: bool, error: str = "", status: int = 200):
        gpus, _ = engine.snapshot().gpus, None
        gpu_count = max(len(gpus), 1)
        try:
            app_t = tpl.get(form.get("app") or "") if not editing else None
        except ConfigError:
            app_t = None
        no_server = bool(app_t and app_t.connects) and not svc.has_model_server()
        return page(request, "instance_form.html", status=status, f=form, editing=editing, error=error, app_t=app_t,
                    no_server=no_server, model_servers=[x for x in tpl.TEMPLATES if tpl.is_model_server(x)],
                    gpu_indices=list(range(max(gpu_count, max([0] + list(form.get("gpu_list") or [])) + 1))))

    # ------------------------------------------------------------------ automatic install (templates)
    machine, model_files = svc.machine, svc.model_files

    def next_app(key: str) -> tpl.Template | None:
        """The app to go on to after installing a model server for it (?then=)."""
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
        choices = list(tpl.gpu_choices(gpus, t).items())
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
                    speech=list(tpl.speech_servers(m).items()),
                    new_server=tpl.NEW_SERVER, bundled=tpl.get(tpl.BUNDLED_SERVER), slots=tpl.slots(t),
                    companions=[tpl.get(c) for c in t.companions],
                    no_server=bool(t.connects) and not m.servers, then_t=then if tpl.is_model_server(t) else None,
                    model_servers=[x for x in tpl.TEMPLATES if tpl.is_model_server(x)])

    # ------------------------------------------------------------------ App Center
    @app.get("/apps", response_class=HTMLResponse)
    def app_store(request: Request):
        instances, _ = store.instances()
        return page(request, "apps.html", groups=tpl.groups(), gpu_choices=tpl.GPU_CHOICES,
                    installed=tpl.installed(instances),
                    has_server=svc.has_model_server(), server_app=tpl.BUNDLED_SERVER)

    @app.get("/apps/{key}", response_class=HTMLResponse)
    def install_app(request: Request, key: str, mode: str = "auto", then: str = ""):
        try:
            t = tpl.get(key)
        except ConfigError as err:
            return RedirectResponse(with_msg("/apps", str(err)), status_code=303)
        if t.key != key:                      # renamed app (the chat and coding servers are one Llama Swap now)
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
    def new_from_template(key: str):          # links from before the App Center
        return RedirectResponse(f"/apps/{key}", status_code=303)

    @app.post("/instances/guided")
    @threaded
    def save_guided(request: Request):
        raw = request.state.form
        try:
            t = tpl.get(str(raw.get("template", "")))
        except ConfigError as err:
            return toast(Response(status_code=204), str(err), "error")
        slot_fields = [f"{x.key}_{part}" for x in tpl.slots(t) for part in ("file", "ctx")]
        f = {k: str(raw.get(k, "")) for k in ["name", "id", "port", "gpu", "models_dir", "appdata_dir", "server",
                                              "comfyui", "speech", "offload", "login_user", "password", "hass_url",
                                              "hass_token", "hf_model", "hf_token", "max_len", *slot_fields]}
        f["autostart"] = raw.get("autostart") == "on"
        then = next_app(str(raw.get("then", ""))) if tpl.is_model_server(t) else None
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
                "HX-Redirect": with_msg(f"/apps/{then.key}", f"Installed {inst.name}{what}.{fetching} Now set up {then.title}: it will use {inst.name}.")})
        if inst.kind == "llama-swap":
            what = f"with {', '.join(x.id for x in models)}" if models else "- add a model below"
            return Response(status_code=204, headers={
                "HX-Redirect": with_msg(f"/instances/{inst.id}/models", f"Installed {inst.name} {what}.{fetching} Press Start on the dashboard to run it.")})
        also = "".join(f" and {x.name}" for x in [bundle, *done.companions] if x)
        after = f" {t.after_install}" if t.after_install else ""
        return Response(status_code=204, headers={
            "HX-Redirect": with_msg("/", f"Installed {inst.name}{also}. Press Start to run it.{after}{fetching}") + f"#inst-{inst.id}"})

    @app.get("/instances/{iid}/edit", response_class=HTMLResponse)
    def edit_instance(request: Request, iid: str):
        try:
            return editor(request, form_from_instance(store.instance(iid)), editing=True)
        except ConfigError as err:
            return RedirectResponse(with_msg("/", str(err)), status_code=303)

    @app.post("/instances/save")
    @threaded
    def save_instance(request: Request):
        raw = request.state.form
        f = {k: raw.get(k, "") for k in ("id", "name", "description", "kind", "image", "command", "env", "ports",
                                          "volumes", "gpu_mode", "min_gpus", "max_gpus", "priority", "open_port",
                                          "open_path", "shm_size", "user_spec", "memory_limit", "editing",
                                          "drain_check", "drain_timeout", "drain_on_timeout", "drain_port",
                                          "drain_path", "drain_key", "app", "requires", "vram_gb")}
        f["gpu_list"] = [int(x) for x in raw.getlist("gpu_list") if str(x).isdigit()]
        f["autostart"] = raw.get("autostart") == "on"
        f["cpu_fallback"] = raw.get("cpu_fallback") == "on"
        f["auto_update"] = raw.get("auto_update") == "on"
        editing = f["editing"] == "1"
        try:
            inst = Instance(
                id=f["id"].strip(), name=f["name"].strip() or f["id"], description=f["description"].strip(),
                kind=f["kind"] or "container", image=f["image"], command=parse_command(f["command"]),
                env=parse_env(f["env"]), ports=parse_ports(f["ports"]), volumes=parse_volumes(f["volumes"]),
                gpu=GpuPolicy(mode=f["gpu_mode"] or "none", gpus=f["gpu_list"],
                              min_gpus=int(f["min_gpus"] or 1), max_gpus=int(f["max_gpus"] or 0),
                              priority=int(f["priority"] or 50),
                              vram_gb=float(f["vram_gb"]) if str(f["vram_gb"]).strip() and f["gpu_mode"] == "elastic" else 0,
                              cpu_fallback=f["cpu_fallback"] and (f["gpu_mode"] == "elastic")),
                autostart=f["autostart"], open_port=int(f["open_port"]) if str(f["open_port"]).strip() else None,
                open_path=f["open_path"] or "/", shm_size=f["shm_size"] or None, user=f["user_spec"] or None,
                memory_limit=f["memory_limit"] or None,
                drain=DrainPolicy(check=f["drain_check"] or "auto",
                                  timeout=int(f["drain_timeout"]) if str(f["drain_timeout"]).strip() else 600,
                                  on_timeout=f["drain_on_timeout"] or "stop",
                                  port=int(f["drain_port"]) if str(f["drain_port"]).strip() else None,
                                  path=f["drain_path"], key=f["drain_key"]),
                app=str(f["app"]).strip(), requires=re.split(r"[\s,]+", str(f["requires"]).strip()),
                auto_update=f["auto_update"])
            msg = svc.save_instance(inst, editing)
        except (Problem, ConfigError, ValueError) as err:
            resp = editor(request, f, editing, error=err.msg if isinstance(err, Problem) else message(err))
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/", msg)})

    @app.post("/instances/{iid}/delete")
    def delete_instance(iid: str):
        try:
            msg = svc.delete_instance(iid)
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/", msg)})

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
                "models_error": err, "files": files, "file_notes": notes, "missing": svc.missing_files(iid),
                "failures": svc.start_failures(iid),
                "loading": sorted(app.state.loading.get(iid, set())),
                "resolved": llamaswap.resolved_aliases(cat, preview_gpus), "preview_gpus": preview_gpus,
                "preview": llamaswap.render(cat, preview_gpus, threads,
                                            engine.placement(inst, cat, st.gpus if running else
                                                             [g.index for g in engine.snapshot().gpus])),
                "placement": svc.placement(iid, cat), "tuning": svc.tuning(iid),
                "pending": bool(running and st.config_hash != engine.config_hash(inst)),
                "suggestions": svc.suggestions(iid), "download_dir": svc.models_dir(inst)}

    @app.get("/instances/{iid}/models", response_class=HTMLResponse)
    def models_page(request: Request, iid: str):
        try:
            return page(request, "models.html", **models_ctx(iid))
        except Problem as err:
            return RedirectResponse(with_msg("/", err.msg), status_code=303)

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
                return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})
            if op == "download":
                msg = svc.download_suggestion(iid, key)
            elif op == "cancel":
                msg = svc.cancel_download(iid, key)
            elif op == "discard":
                msg = svc.discard_download(iid, key)
            elif op == "forget":
                msg = svc.forget_link(iid, key)
            else:
                return PlainTextResponse("unknown operation", status_code=400)
        except Problem as err:
            return problem(err)
        return toast(suggestions_part(request, iid), msg)

    @app.post("/instances/{iid}/models/from-link")
    @threaded
    def model_from_link(request: Request, iid: str):
        form = request.state.form
        try:
            msg = svc.add_from_link(iid, str(form.get("url", "")), str(form.get("name", "")),
                                    str(form.get("mmproj_url", "")))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

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
    @threaded
    def save_model(request: Request, iid: str):
        raw = request.state.form
        inst = store.instance(iid)
        original = str(raw.get("original_id", ""))
        f = {k: str(raw.get(k, "")) for k in ("id", "name", "description", "file", "ctx_size", "parallel",
                                               "sampling", "extra_args", "kv_cache", "mmproj", "aliases_text",
                                               "role", "thinking", "mtp", "mtp_draft", "split_mode", "tensor_split",
                                               "ttl", "gpus")}
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
                                 gpus=int(f["gpus"] or 0),
                                 ttl=int(f["ttl"] or 0), keep_loaded=f["keep_loaded"], preload=f["preload"])
            msg = svc.save_model(iid, model, original)
        except (Problem, ValueError, ConfigError) as err:
            files, notes = llamaswap.local_model_files(inst)
            resp = page(request, "model_form.html", inst=inst, f=f, editing=bool(original), files=files,
                        file_notes=notes, error=err.msg if isinstance(err, Problem) else message(err), original=original)
            resp.headers["HX-Retarget"] = "#editor"
            resp.headers["HX-Reselect"] = "#editor"
            return resp
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

    @app.post("/instances/{iid}/models/{mid}/delete")
    def delete_model(iid: str, mid: str):
        try:
            msg = svc.delete_model(iid, mid)
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

    @app.post("/instances/{iid}/models/{mid}/{op}")
    def model_op(iid: str, mid: str, op: str):
        if op not in ("load", "unload"):
            return PlainTextResponse("unknown operation", status_code=400)
        try:
            if op == "unload":
                return toast(Response(status_code=204, headers={"HX-Trigger-After-Settle": "refresh"}),
                             svc.unload_model(iid, mid))
            # back to the page, which follows the load until it is done or has failed
            return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models",
                                                                              svc.load_model(iid, mid))})
        except Problem as err:
            return problem(err)

    @app.post("/instances/{iid}/tuning")
    @threaded
    def tuning_decide(request: Request, iid: str):
        raw = request.state.form
        try:
            msg = svc.tuning_decide(iid, str(raw.get("key", "")), str(raw.get("action", "")))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

    @app.post("/instances/{iid}/catalog/settings")
    @threaded
    def catalog_settings(request: Request, iid: str):
        raw = request.state.form
        try:
            unset = svc.server(iid)[1].fit_gpus is None    # left unticked: still offered as a suggestion
            msg = svc.server_settings(iid, threads=int(raw.get("threads") or 0),
                                      llama_server=str(raw.get("llama_server", "")),
                                      health_timeout=int(raw.get("health_timeout") or 900),
                                      fit_gpus=True if raw.get("fit_gpus") == "on" else None if unset else False,
                                      auto_tune=raw.get("auto_tune") == "on",
                                      new_key=raw.get("new_key") == "on")
        except ValueError:
            return toast(Response(status_code=204), "Threads and the start-up wait are whole numbers.", "error")
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

    @app.post("/instances/{iid}/catalog/auto")
    @threaded
    def catalog_auto(request: Request, iid: str):
        raw = request.state.form
        try:
            autos = [AutoAlias(alias=str(alias).strip(), default_model=str(default), rules=parse_auto_rules(rules))
                     for alias, default, rules in zip(raw.getlist("alias"), raw.getlist("default_model"),
                                                      raw.getlist("rules")) if str(alias).strip()]
            msg = svc.set_auto_aliases(iid, autos)
        except ValueError as err:
            return toast(Response(status_code=204), message(err), "error")
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg(f"/instances/{iid}/models", msg)})

    # ------------------------------------------------------------------ updates (done by update.sh on the host)
    def update_page(request: Request, confirm: bool = False) -> Response:
        u = updates.status(store.root)
        return page(request, "partials/update.html", u=u, confirm=confirm,
                    log_tail=updates.live_log(store.root) if u.busy else "")

    @app.get("/ui/update", response_class=HTMLResponse)
    def update_dialog(request: Request, confirm: str = ""):
        return update_page(request, confirm == "1")

    @app.get("/ui/credits", response_class=HTMLResponse)
    def credits_dialog(request: Request):
        apps = [t for t in tpl.TEMPLATES if t.license]
        return page(request, "partials/credits.html", credits=credits, apps=apps, models=model_library.SUGGESTIONS)

    @app.get("/ui/app-updates", response_class=HTMLResponse)
    def app_updates_dialog(request: Request):
        return page(request, "partials/app_updates.html", updates=svc.app_updates(), names=svc.names())

    @app.post("/ui/app-updates/apply-all")
    def app_updates_apply_all(request: Request):
        try:
            msg = svc.apply_all_app_updates()
        except Problem as err:
            return toast(app_updates_dialog(request), err.msg, err.kind)
        resp = app_updates_dialog(request)
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": msg, "kind": "info"}})
        return resp

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

    @app.post("/ui/update/branch")
    @threaded
    def update_branch(request: Request):
        raw = request.state.form
        try:
            msg = svc.switch_branch(str(raw.get("branch", "")))
        except Problem as err:
            return problem(err)
        resp = update_page(request)          # the dialog, now showing the progress
        resp.headers["HX-Trigger"] = json.dumps({"refresh": True, "toast": {"msg": msg, "kind": "info"}})
        return resp

    @app.get("/ui/update/log", response_class=HTMLResponse)
    def update_log(request: Request):
        u = updates.status(store.root)
        return page(request, "partials/logs.html", title="Update log", refresh="/ui/update/log" if u.busy else "",
                    text=updates.log_text(store.root) or "Nothing yet: the updater writes here once it starts.")

    @app.post("/ui/update/credentials")
    @threaded
    def update_credentials(request: Request):
        raw = request.state.form
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

    # ------------------------------------------------------------------ power (restart GPU Shepherd, restart / shut down the server)
    boot_id = secrets.token_hex(4)          # changes when this process restarts: the waiting dialog reloads the page then

    def power_page(request: Request, confirm: str = "", done: str = "", message: str = "") -> Response:
        return page(request, "partials/power.html", p=svc.power_status(), confirm=confirm, done=done, message=message,
                    boot=boot_id)

    @app.get("/ui/power", response_class=HTMLResponse)
    def power_dialog(request: Request, confirm: str = ""):
        return power_page(request, confirm if confirm in POWER_ACTIONS else "")

    @app.get("/ui/power/back")
    def power_back(boot: str = ""):
        # polled by the waiting dialog: once this is a different process, reload the page
        return Response(status_code=204, headers={"HX-Refresh": "true"} if boot and boot != boot_id else {})

    @app.post("/ui/power/{action}")
    def power_request(request: Request, action: str):
        if action not in POWER_ACTIONS:
            return PlainTextResponse("unknown action", status_code=400)
        try:
            msg = svc.power(action)
        except Problem as err:
            return problem(err)
        return toast(power_page(request, done=action, message=msg), msg)

    # ------------------------------------------------------------------ tensor unlock (gpu-unlock.sh on the host)
    def unlock_card_page(request: Request) -> Response:
        return page(request, "partials/unlock_card.html", x=svc.unlock_status(), unlock_url=gpu_unlock.URL)

    @app.get("/ui/unlock/card", response_class=HTMLResponse)
    def unlock_card(request: Request):
        return unlock_card_page(request)

    @app.get("/ui/unlock/log", response_class=HTMLResponse)
    def unlock_log(request: Request):
        x = svc.unlock_status()
        return page(request, "partials/logs.html", title="Tensor unlock log", refresh="/ui/unlock/log" if x.busy else "",
                    text=gpu_unlock.log_text(store.root) or "Nothing yet: the server writes here once a change starts.")

    @app.post("/ui/unlock/{action}")
    @threaded
    def unlock_request(request: Request, action: str):
        if action not in gpu_unlock.ACTIONS:
            return PlainTextResponse("unknown action", status_code=400)
        raw = request.state.form
        try:
            msg = svc.request_unlock(action, [str(c) for c in raw.getlist("cards")])
        except Problem as err:
            return problem(err)
        return toast(unlock_card_page(request), msg)

    # ------------------------------------------------------------------ settings
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return page(request, "settings.html", s=store.settings(), docker=backend.ping(), data_dir=str(store.root),
                    host_data_dir=str(engine.host_data_dir), initial=auth.initial_password_file(store).exists(),
                    error="", u=updates.status(store.root), api_token=auth.api_token(store),
                    image_checked=svc.image_checked(), x=svc.unlock_status(), unlock_url=gpu_unlock.URL)

    @app.post("/settings/password")
    @threaded
    def change_password(request: Request):
        raw = request.state.form
        try:
            msg = svc.change_password(*(str(raw.get(k, "")) for k in ("current", "new", "again")))
        except Problem as err:
            return problem(err)
        # the change ends every login (auth.session_stamp); this browser stays logged in
        request.session["stamp"] = auth.session_stamp(store.settings())
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/settings", msg)})

    @app.post("/settings/general")
    @threaded
    def general_settings(request: Request):
        raw = request.state.form
        try:
            msg = svc.set_containers(str(raw.get("bind_address", "")), str(raw.get("network", "")))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/settings", msg)})

    @app.post("/settings/api-token")
    def api_token_new():
        auth.new_api_token(store)
        log.info("API token replaced from the dashboard")
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/settings", "New API token made. The old one no longer works.")})

    # ------------------------------------------------------------------ knowledge (documents and skills)
    @app.get("/knowledge", response_class=HTMLResponse)
    def knowledge_page(request: Request, show: str = ""):
        b = svc.knowledge()
        if show in ("ready", "failed", "scanned", "skipped", "duplicate"):
            b["shown"] = [x for x in b["documents"] if x["status"] == show]
        else:
            show, b["shown"] = "", b["documents"]
        return page(request, "knowledge.html", b=b, show=show)

    @app.get("/books")
    def books_moved():                                  # the page's old name
        return RedirectResponse("/knowledge", status_code=301)

    @app.get("/knowledge/task", response_class=HTMLResponse)
    def knowledge_task(request: Request, was: int = 0):
        t = svc.knowledge_task.as_dict() if svc.knowledge_task else None
        if was and not (t and t["running"]):              # it just finished: show the page with the result
            return Response(status_code=204, headers={"HX-Refresh": "true"})
        return templates.TemplateResponse(request, "partials/knowledge_task.html", {"t": t})

    def knowledge_done(action):
        try:
            msg = action()
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/knowledge", msg)})

    @app.post("/knowledge/folder")
    @threaded
    def knowledge_folder(request: Request):
        raw = request.state.form
        return knowledge_done(lambda: svc.set_documents_folder(str(raw.get("path", ""))))

    @app.post("/knowledge/upload")
    @threaded
    def knowledge_upload(request: Request):
        raw = request.state.form
        files = [(f.filename, f.file) for f in raw.getlist("file") if getattr(f, "filename", "")]
        return knowledge_done(lambda: svc.add_documents(files))

    @app.get("/knowledge/files")
    def knowledge_files():
        return svc.document_files()

    @app.put("/knowledge/files")
    async def knowledge_put_file(request: Request, path: str = "", offset: int | None = None, total: int | None = None):
        """One document or zip of them as the raw body, or a piece of one (the Knowledge page's upload
        sends them one by one, in pieces read into memory)."""
        size = request.headers.get("content-length")
        try:
            r = await svc.receive_document(path, request.stream(), int(size) if size and size.isdigit() else None,
                                           offset, total)
        except Problem as err:
            return JSONResponse({"error": err.msg} | err.data, status_code=err.status)
        return r

    @app.post("/knowledge/scan")
    def knowledge_scan():
        return knowledge_done(svc.scan_documents)

    @app.post("/knowledge/stop")
    def knowledge_stop():
        return knowledge_done(svc.stop_knowledge)

    @app.post("/knowledge/send")
    @threaded
    def knowledge_send(request: Request):
        raw = request.state.form
        emb = {"server": True, "keep": False}.get(str(raw.get("embeddings", "")))
        return knowledge_done(lambda: svc.send_documents(str(raw.get("instance", "")), str(raw.get("name", "")),
                                                 bool(raw.get("split")), emb))

    @app.post("/knowledge/ocr")
    def knowledge_ocr():
        return knowledge_done(svc.ocr_documents)

    @app.post("/knowledge/delete-all")
    @threaded
    def knowledge_delete_all(request: Request):
        raw = request.state.form
        return knowledge_done(lambda: svc.delete_documents(str(raw.get("status", ""))))

    @app.post("/knowledge/{key}/delete")
    def knowledge_delete(key: str):
        return knowledge_done(lambda: svc.delete_document(key))

    @app.post("/knowledge/clear")
    @threaded
    def knowledge_clear(request: Request):
        raw = request.state.form
        return knowledge_done(lambda: svc.clear_knowledge(tuple(raw.getlist("what"))))

    @app.post("/knowledge/{key}/skill")
    @threaded
    def knowledge_skill(request: Request, key: str):
        raw = request.state.form
        return knowledge_done(lambda: svc.make_skill(key, str(raw.get("server", ""))))

    def skill_dialog(request: Request, slug: str, **ctx):
        try:
            text = ctx.pop("text", None) or svc.skill_text(slug)
        except Problem as err:
            return problem(err)
        return templates.TemplateResponse(request, "partials/skill_edit.html", {"slug": slug, "text": text} | ctx)

    @app.get("/knowledge/skills/{slug}", response_class=HTMLResponse)
    def knowledge_skill_show(request: Request, slug: str):
        return skill_dialog(request, slug)

    @app.post("/knowledge/skills/{slug}", response_class=HTMLResponse)
    @threaded
    def knowledge_skill_save(request: Request, slug: str):
        text = str(request.state.form.get("text", ""))
        try:
            return skill_dialog(request, slug, saved=svc.save_skill(slug, text))
        except Problem as err:
            return skill_dialog(request, slug, text=text, error=err.msg)

    @app.post("/knowledge/skills/{slug}/install")
    @threaded
    def knowledge_skill_install(request: Request, slug: str):
        raw = request.state.form
        return knowledge_done(lambda: svc.install_skill(slug, str(raw.get("instance", ""))))

    @app.post("/knowledge/skills/{slug}/delete")
    def knowledge_skill_delete(slug: str):
        return knowledge_done(lambda: svc.delete_skill(slug))

    # ------------------------------------------------------------------ storage
    @app.get("/storage", response_class=HTMLResponse)
    def storage_page(request: Request):
        return page(request, "storage.html", st=svc.storage(), drives=drives_info())

    def storage_done(action):
        """Run one clean-up, then show the Storage page again with what happened."""
        try:
            msg = action()
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/storage", msg)})

    @app.post("/storage/images/remove-unused")
    def storage_remove_images():
        return storage_done(svc.remove_unused_images)

    @app.post("/storage/orphans/remove")
    def storage_remove_orphans():
        return storage_done(svc.remove_orphans)

    @app.post("/storage/files/trash")
    @threaded
    def storage_trash_files(request: Request):
        raw = request.state.form
        return storage_done(lambda: svc.trash_files([str(p) for p in raw.getlist("path")]))

    @app.post("/storage/leftover/trash")
    @threaded
    def storage_trash_leftover(request: Request):
        raw = request.state.form
        return storage_done(lambda: svc.trash_leftover(str(raw.get("name", ""))))

    @app.post("/storage/trash/restore")
    @threaded
    def storage_restore(request: Request):
        raw = request.state.form
        return storage_done(lambda: svc.restore_trash(str(raw.get("id", ""))))

    @app.post("/storage/trash/purge")
    @threaded
    def storage_purge(request: Request):
        raw = request.state.form        # no id = empty the whole trash
        return storage_done(lambda: svc.purge_trash(str(raw.get("id", ""))))

    def move_form(raw) -> tuple[str, str]:
        """(source, dest); with home=1 the dest is where the folder was before it was moved."""
        source = str(raw.get("source", ""))
        return source, svc.home_of(source) if raw.get("home") else str(raw.get("dest", ""))

    def picker_context(source: str) -> dict:
        m = svc._movable(source)
        return {"source": m.path, "label": m.label, "name": Path(m.path).name}

    def picker_roots(found: list[drives.Drive]) -> list[dict]:
        """Where the tree starts: each mounted drive (not the system's), GPU Shepherd's own folder
        (on the drive it was installed on), then the whole machine."""
        roots = [{"path": d.mountpoint, "title": d.name, "meta": f"{d.fstype} · {gb(d.size)}"}
                 for d in found if d.state == "mounted" and not d.system and d.mountpoint.startswith("/")]
        home = str(svc.engine.host_data_dir)
        return roots + [{"path": home, "title": "GPU Shepherd's folder", "meta": "the install drive"},
                        {"path": "/", "title": "This machine", "meta": "all folders"}]

    @app.get("/storage/pick", response_class=HTMLResponse)
    def storage_pick(request: Request, source: str = "", path: str = ""):
        """The "where to?" dialog: drives, and a folder tree."""
        try:
            ctx = picker_context(source)
            info = drives_info()
        except Problem as err:
            return problem(err)
        resp = page(request, "partials/folder_picker.html", **ctx, **info, picking=True, roots=picker_roots(info["found"]),
                    start=drives.clean_path(path) if path else "")
        resp.headers["HX-Retarget"] = "#modal-body"
        resp.headers["HX-Reswap"] = "innerHTML"
        resp.headers["HX-Trigger-After-Swap"] = "open-modal"
        return resp

    def drives_info() -> dict:
        d = svc.drives()
        return {"found": [drives.Drive(**x) for x in d["drives"]], "drives_error": d["error"], "auto_mount": d["auto_mount"],
                "wsl": d.get("wsl", False)}

    @app.get("/storage/drives", response_class=HTMLResponse)
    def storage_drives(request: Request, source: str = "", picking: str = ""):
        return page(request, "partials/drives.html", **drives_info(), source=source, picking=bool(picking))

    @app.post("/storage/drives/mount", response_class=HTMLResponse)
    @threaded
    def storage_drive_mount(request: Request):
        raw = request.state.form
        source, picking = str(raw.get("source", "")), bool(raw.get("picking"))
        kind, msg = "info", ""
        try:
            msg = svc.mount_drive(str(raw.get("device", "")))
        except Problem as err:
            msg, kind = err.msg, err.kind
        resp = page(request, "partials/drives.html", **drives_info(), source=source, picking=picking)
        return toast(resp, msg, kind)

    @app.post("/storage/drives/auto")
    @threaded
    def storage_drives_auto(request: Request):
        raw = request.state.form
        return storage_done(lambda: svc.set_auto_mount(raw.get("on") == "1"))

    @app.get("/storage/tree", response_class=HTMLResponse)
    def storage_tree(request: Request, path: str = "/", source: str = "", root: str = ""):
        """The folders inside one folder, for the tree (or, with root, the tree starting at it)."""
        try:
            li = svc.browse(path)
        except Problem as err:
            return problem(err)
        return page(request, "partials/folder_tree.html", li=li, source=source, root=bool(root),
                    title=next((d.name for d in svc.scan_drives(False) if d.mountpoint == li.path), ""), meta="")

    @app.get("/storage/pick/info", response_class=HTMLResponse)
    def storage_pick_info(request: Request, source: str = "", pick: str = "", name: str = ""):
        """What picking this folder would mean: the new place, room on the disk, or why not."""
        if not pick:
            return HTMLResponse('<p class="muted small">Pick a folder in the list.</p>')
        try:
            info = svc.pick_place(source, pick, name)
        except Problem as err:
            info = {"error": err.msg}
        return page(request, "partials/pick_info.html", p=info)

    @app.post("/storage/move/check", response_class=HTMLResponse)
    @threaded
    def storage_move_check(request: Request):
        """The checked move, to confirm in a dialog."""
        try:
            info = svc.check_move(*move_form(request.state.form))
        except Problem as err:
            return problem(err)
        resp = page(request, "partials/move_confirm.html", m=info)
        resp.headers["HX-Retarget"] = "#modal-body"
        resp.headers["HX-Reswap"] = "innerHTML"
        resp.headers["HX-Trigger-After-Swap"] = "open-modal"
        return resp

    @app.post("/storage/move")
    @threaded
    def storage_move(request: Request):
        try:
            msg, _job = svc.move_folder(*move_form(request.state.form))
        except Problem as err:
            return problem(err)
        return Response(status_code=204, headers={"HX-Redirect": with_msg("/", msg)})

    @app.post("/storage/moves/{op}")
    @threaded
    def storage_move_op(request: Request, op: str):
        mid = str(request.state.form.get("id", ""))
        if op == "back":
            return storage_done(lambda: svc.move_back(mid))
        if op == "delete":
            return storage_done(lambda: svc.delete_leftover(mid))
        return PlainTextResponse("unknown operation", status_code=404)

    @app.post("/storage/trash/days")
    @threaded
    def storage_trash_days(request: Request):
        raw = request.state.form
        try:
            days = int(str(raw.get("days", "")).strip())
        except ValueError:
            return problem(Problem("Days must be a whole number (0 keeps things until you empty the trash)."))
        return storage_done(lambda: svc.set_trash_days(days))

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
    if autostart is not False:            # tests turn the background work off
        svc.start_measuring()
        svc.watch_drives()
        svc.watch_images()

    return app
