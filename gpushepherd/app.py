"""Web UI (FastAPI + Jinja2 + htmx). Server-rendered pages; htmx swaps in fragments."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import __version__, auth
from .backend import Backend, DockerBackend, FakeBackend
from .config import (AutoAlias, AutoRule, Catalog, CatalogModel, ConfigError, GpuPolicy, Instance, Store,
                     format_command, format_env, format_ports, format_volumes, friendly_errors, parse_command,
                     parse_env, parse_ports, parse_volumes)
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


def create_app(data_dir: str | None = None, backend: Backend | None = None, demo: bool | None = None,
               autostart: bool | None = None) -> FastAPI:
    demo = demo if demo is not None else os.environ.get("SHEPHERD_DEMO") == "1"
    store = Store(data_dir or os.environ.get("SHEPHERD_DATA_DIR", "/opt/gpu-shepherd"))
    if backend is None:
        backend = FakeBackend(slow=0.5) if demo else DockerBackend()
    engine = Engine(store, backend)
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
    templates.env.globals.update(version=__version__, gpu_text=gpu_text, demo=demo)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ------------------------------------------------------------------ security
    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if not path.startswith(PUBLIC):
                if not request.session.get("user"):
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

    def dashboard_ctx() -> dict:
        snap = engine.snapshot()
        cards = []
        order = sorted(snap.instances.values(), key=lambda i: (-i.gpu.priority if i.gpu.mode != "none" else 1, i.name.lower()))
        for inst in order:
            st = snap.states.get(inst.id)
            card = {"inst": inst, "state": st, "running": bool(st and st.running),
                    "status": (st.status if st else "not created"),
                    "gpus": snap.allocations.get(inst.id, []),
                    "pending": bool(st and st.running and st.config_hash and st.config_hash != engine.config_hash(inst)),
                    "loaded": [], "models_error": "",
                    "loading": sorted(app.state.loading.get(inst.id, set()))}
            if inst.kind == "llama-swap" and card["running"]:
                try:
                    client = llama_client(inst.id)
                    known = {m.id for m in store.catalog(inst.id).models}
                    card["loaded"] = sorted(client.running(known))
                except Exception:  # noqa: BLE001
                    card["models_error"] = "model server not answering yet"
            cards.append(card)
        gpus = [{"gpu": g, "users": snap.gpu_users(g.index)} for g in snap.gpus]
        return {"cards": cards, "gpus": gpus, "gpu_error": snap.gpu_error, "problems": snap.problems,
                "job": jobs.current, "last_job": (jobs.history[0] if jobs.history else None)}

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
            jobs.submit(f"{verb} {names.get(iid, iid)}", lambda logf: engine.execute(p, logf))
        except Busy as err:
            return toast(Response(status_code=204), str(err), "warn")
        if back:
            return Response(status_code=204, headers={"HX-Redirect": f"{back}?msg={verb} {names.get(iid, iid)}…"})
        resp = page(request, "partials/dashboard.html", **dashboard_ctx())
        resp.headers["HX-Trigger"] = json.dumps({"toast": {"msg": f"{verb} {names.get(iid, iid)}…", "kind": "info"},
                                                 "close-modal": True})
        return resp

    # ------------------------------------------------------------------ login
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return page(request, "login.html", error="", first_run=auth.initial_password_file(store).exists())

    @app.post("/login")
    async def login(request: Request):
        form = await request.form()
        s = store.settings()
        user, password = str(form.get("user", "")), str(form.get("password", ""))
        if secrets.compare_digest(user, s.admin_user) and auth.verify_password(password, s.password_hash):
            request.session.clear()
            request.session["user"] = user
            return RedirectResponse("/", status_code=303)
        return page(request, "login.html", status=401, error="Wrong user name or password.",
                    first_run=auth.initial_password_file(store).exists())

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return Response(status_code=204, headers={"HX-Redirect": "/login"})

    # ------------------------------------------------------------------ dashboard
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard.html", **dashboard_ctx(), docker=backend.ping())

    @app.get("/ui/dashboard", response_class=HTMLResponse)
    def dashboard_partial(request: Request):
        return page(request, "partials/dashboard.html", **dashboard_ctx())

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

    # ------------------------------------------------------------------ instance editor
    def blank_instance(kind: str) -> dict:
        if kind == "llama-swap":
            return {"kind": "llama-swap", "image": "ghcr.io/mostlygeek/llama-swap:cuda", "ports": "8080:8080",
                    "gpu_mode": "elastic", "min_gpus": 1, "max_gpus": 0, "priority": 50, "open_path": "/ui",
                    "volumes": "/srv/models:/models:ro", "name": "LLM server", "id": "llm"}
        return {"kind": "container", "gpu_mode": "none", "min_gpus": 1, "max_gpus": 0, "priority": 50, "open_path": "/"}

    def form_from_instance(inst: Instance) -> dict:
        return {"id": inst.id, "name": inst.name, "description": inst.description, "kind": inst.kind,
                "image": inst.image, "command": format_command(inst.command), "env": format_env(inst.env),
                "ports": format_ports(inst.ports), "volumes": format_volumes(inst.volumes),
                "gpu_mode": inst.gpu.mode, "gpu_list": inst.gpu.gpus, "min_gpus": inst.gpu.min_gpus,
                "max_gpus": inst.gpu.max_gpus, "priority": inst.gpu.priority, "autostart": inst.autostart,
                "open_port": inst.open_port or "", "open_path": inst.open_path, "shm_size": inst.shm_size or "",
                "user_spec": inst.user or "", "memory_limit": inst.memory_limit or ""}

    def editor(request: Request, form: dict, editing: bool, error: str = "", status: int = 200):
        gpus, _ = engine.snapshot().gpus, None
        gpu_count = max(len(gpus), 1)
        return page(request, "instance_form.html", status=status, f=form, editing=editing, error=error,
                    gpu_indices=list(range(max(gpu_count, max([0] + list(form.get("gpu_list") or [])) + 1))))

    @app.get("/instances/new", response_class=HTMLResponse)
    def new_instance(request: Request, kind: str = "container"):
        return editor(request, blank_instance(kind), editing=False)

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
                                          "open_path", "shm_size", "user_spec", "memory_limit", "editing")}
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
                memory_limit=f["memory_limit"] or None)
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
        return page(request, "model_form.html", inst=inst, f=f, editing=bool(m), files=files, file_notes=notes,
                    error="")

    @app.post("/instances/{iid}/models/save")
    async def save_model(request: Request, iid: str):
        raw = await request.form()
        inst, cat = store.instance(iid), store.catalog(iid)
        original = str(raw.get("original_id", ""))
        f = {k: str(raw.get(k, "")) for k in ("id", "name", "description", "file", "ctx_size", "parallel",
                                               "sampling", "extra_args", "kv_cache", "mmproj", "aliases_text")}
        try:
            model = CatalogModel(id=f["id"].strip(), name=f["name"].strip(), description=f["description"].strip(),
                                 file=f["file"], ctx_size=int(f["ctx_size"] or 32768),
                                 parallel=int(f["parallel"] or 1), sampling=f["sampling"],
                                 extra_args=f["extra_args"], kv_cache=f["kv_cache"] or "f16",
                                 mmproj=f["mmproj"].strip(),
                                 aliases=[a.strip() for a in f["aliases_text"].split(",") if a.strip()])
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

    # ------------------------------------------------------------------ settings
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return page(request, "settings.html", s=store.settings(), docker=backend.ping(), data_dir=str(store.root),
                    host_data_dir=str(engine.host_data_dir), initial=auth.initial_password_file(store).exists(),
                    error="")

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
        return JSONResponse({
            "version": __version__,
            "gpus": [g.__dict__ | {"users": snap.gpu_users(g.index)} for g in snap.gpus],
            "instances": {i: {"name": inst.name, "running": bool(snap.states.get(i) and snap.states[i].running),
                              "gpus": snap.allocations.get(i, []), "policy": inst.gpu.model_dump()}
                          for i, inst in snap.instances.items()},
            "job": jobs.current.title if jobs.current else None,
        })

    # ------------------------------------------------------------------ autostart
    if autostart if autostart is not None else os.environ.get("SHEPHERD_AUTOSTART", "1") == "1":
        try:
            p = engine.autostart_plan()
            if not p.error and not p.is_noop:
                jobs.submit("Autostart", lambda logf: engine.execute(p, logf))
        except Exception as err:  # noqa: BLE001
            log.warning("autostart skipped: %s", err)

    return app
