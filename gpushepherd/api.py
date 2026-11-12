"""The JSON API under /api/v1: everything the dashboard can do, for scripts and the `gpu-shepherd`
command (cli.py). It calls the same service.Shepherd methods as the dashboard's routes.

Authentication is the API token (auth.api_token, the `api-token` file in the data folder) sent as
`Authorization: Bearer <token>`; a logged-in browser may also GET. app.py's Guard checks it before
any route here runs. Errors are {"error": "..."} with a 4xx status. Reference: docs/api.md.
"""
from __future__ import annotations

import functools
import inspect
import json
import time
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__, auth, changelog, host as host_mod, instance_templates as tpl, updates
from .config import AutoAlias, CatalogModel, ConfigError, Instance
from .connectors import llamaswap
from .jobs import Job
from .service import Problem, Shepherd, message


def router(svc: Shepherd, build: dict) -> APIRouter:
    r = APIRouter()
    store, engine, jobs = svc.store, svc.engine, svc.jobs

    def fail(err: Exception) -> JSONResponse:
        if isinstance(err, Problem):
            return JSONResponse({"error": err.msg} | err.data, status_code=err.status)
        return JSONResponse({"error": message(err)}, status_code=400)

    def body(request: Request, kind=dict):
        """The request's JSON body (read by `handle` before the route runs)."""
        try:
            data = json.loads(request.state.raw)
        except ValueError as err:
            raise Problem("send a JSON body") from err
        if not isinstance(data, kind):
            raise Problem(f"the JSON body must be {'an object' if kind is dict else 'a list'}")
        return data

    def _piece(offset, total) -> tuple[int | None, int | None]:
        if offset in (None, ""):
            return None, None
        if not (str(offset).isdigit() and str(total or "").isdigit()):
            raise Problem("offset and total are byte counts")
        return int(offset), int(total)

    def handle(fn):
        """Run a route, answering a Problem or a validation error as {"error": ...}. Routes are plain
        functions run in a worker thread, with the body read first (request.state.raw), so one that
        waits for Docker, a model server or a job never holds up the dashboard. The two that take a
        file as it arrives are `async` and read the request themselves."""
        streams = inspect.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def wrapper(request: Request, **kw):
            try:
                if streams:
                    return await fn(request, **kw)
                request.state.raw = await request.body()
                return await run_in_threadpool(fn, request, **kw)
            except (Problem, ConfigError, ValueError) as err:
                return fail(err)
        return wrapper

    def job_json(job: Job, lines: bool = False) -> dict:
        out = {"id": job.id, "title": job.title, "status": job.status, "error": job.error,
               "started": job.started, "finished": job.finished, "seconds": job.seconds,
               "waiting_for": dict(job.waiting)}
        if lines:
            out["log"] = job.lines
        return out

    def instance_json(inst: Instance, snap=None, notes: dict | None = None) -> dict:
        snap = snap or engine.snapshot()
        st = snap.states.get(inst.id)
        running = bool(st and st.running)
        if notes is None:
            try:
                notes = engine.gpu_notes(snap) if inst.gpu.mode != "none" else {}
            except Exception:  # noqa: BLE001 - only a hint
                notes = {}
        return inst.model_dump(mode="json") | {"state": {
            "running": running, "status": st.status if st else "not created",
            "status_text": st.status_text if st else "", "gpus": snap.allocations.get(inst.id, []),
            "container": engine.container_name(inst.id),
            "restart_needed": bool(running and st.config_hash and st.config_hash != engine.config_hash(inst)),
            "gpu_note": notes.get(inst.id, "")}}

    # ------------------------------------------------------------------ state
    def state():
        snap = engine.snapshot()
        machine = host_mod.query()
        job = jobs.current
        return {
            "version": __version__,
            "commit": build.get("commit", ""),
            "update_available": updates.status(store.root).behind,
            "gpus": [g.__dict__ | {"users": snap.gpu_users(g.index)} for g in snap.gpus],
            "gpu_error": snap.gpu_error,
            "instances": {i: {"name": inst.name, "running": bool(snap.states.get(i) and snap.states[i].running),
                              "gpus": snap.allocations.get(i, []), "policy": inst.gpu.model_dump()}
                          for i, inst in snap.instances.items()},
            "problems": snap.problems,
            "host": {"cpu_pct": machine.cpu_pct, "cpu_threads": machine.cpu_threads, "load": machine.load,
                     "mem_total": machine.mem_total, "mem_used": machine.mem_used,
                     "mem_available": machine.mem_available},
            "job": job.title if job else None,
            "waiting_for": dict(job.waiting) if job else {},
        }

    r.get("/api/state")(state)           # the original read-only endpoint, kept for existing scripts
    r.get("/api/v1/state")(state)

    # ------------------------------------------------------------------ instances
    @r.get("/api/v1/instances")
    def list_instances():
        snap = engine.snapshot()
        try:
            notes = engine.gpu_notes(snap)
        except Exception:  # noqa: BLE001 - only a hint
            notes = {}
        return [instance_json(i, snap, notes) for i in sorted(snap.instances.values(), key=lambda i: i.id)]

    @r.get("/api/v1/instances/{iid}")
    @handle
    def get_instance(request: Request, iid: str):
        return instance_json(svc.instance(iid))

    @r.put("/api/v1/instances/{iid}")
    @handle
    def put_instance(request: Request, iid: str):
        data = body(request)
        data.pop("state", None)                       # as GET returns it
        data.setdefault("id", iid)
        if data["id"] != iid:
            raise Problem("the id in the body differs from the one in the address")
        data.setdefault("name", iid)
        editing = store.instance_path(iid).exists()
        msg = svc.save_instance(Instance.model_validate(data), editing)
        return JSONResponse({"message": msg, "instance": instance_json(svc.instance(iid))},
                            status_code=200 if editing else 201)

    @r.delete("/api/v1/instances/{iid}")
    @handle
    def delete_instance(request: Request, iid: str):
        return {"message": svc.delete_instance(iid)}

    @r.get("/api/v1/instances/{iid}/logins")
    @handle
    def instance_logins(request: Request, iid: str):
        return svc.logins(iid)

    @r.post("/api/v1/instances/{iid}/logins")
    @handle
    def reset_login(request: Request, iid: str):
        data = body(request) if request.state.raw else {}
        msg, password = svc.reset_login(iid, str(data.get("user") or ""), str(data.get("password") or ""))
        return {"message": msg, "password": password}

    @r.post("/api/v1/instances/{iid}/tuning")
    @handle
    def tuning_decide(request: Request, iid: str):
        data = body(request)
        return {"message": svc.tuning_decide(iid, str(data.get("key", "")), str(data.get("action", "")))}

    @r.post("/api/v1/instances/{iid}/{action}")
    @handle
    def instance_action(request: Request, iid: str, action: str):
        data = body(request) if request.state.raw else {}
        c = svc.change(action, iid, confirm=bool(data.get("confirm")))
        if c.noop:
            return {"message": c.text, "noop": True}
        if c.needs_confirm:
            why = "This also changes other instances." if c.plan.side_effects else "Memory may not be enough."
            return JSONResponse({"error": f"{why} Send {{\"confirm\": true}} to go ahead.", "needs_confirm": True,
                                 "plan": c.lines, "warnings": c.warnings}, status_code=409)
        return JSONResponse({"message": c.text, "plan": c.lines, "warnings": c.warnings, "job": job_json(c.job)},
                            status_code=202)

    @r.get("/api/v1/instances/{iid}/logs")
    @handle
    def instance_logs(request: Request, iid: str):
        return {"logs": svc.logs(iid)}

    # ------------------------------------------------------------------ jobs
    @r.get("/api/v1/jobs")
    def list_jobs():
        return [job_json(j) for j in jobs.recent()]

    @r.get("/api/v1/jobs/{jid}")
    @handle
    def get_job(request: Request, jid: int):
        return job_json(svc.job(jid), lines=True)

    @r.post("/api/v1/jobs/{jid}/{op}")
    @handle
    def control_job(request: Request, jid: int, op: str):
        return {"message": svc.control_job(jid, op)}

    # ------------------------------------------------------------------ App Center
    def app_json(t: tpl.Template, installed: list[Instance]) -> dict:
        return {"key": t.key, "title": t.title, "group": t.group, "blurb": t.blurb, "points": list(t.points),
                "kind": t.kind, "image": t.image, "gpu": t.gpu, "needs": t.needs, "docs": t.docs,
                "needs_llm_server": bool(t.connects), "installed": [i.id for i in installed],
                "also_installs": [tpl.get(c).title for c in t.companions], "after_install": t.after_install,
                "experimental": t.experimental, "license": t.license}

    @r.get("/api/v1/apps")
    def list_apps():
        installed = tpl.installed(store.instances()[0])
        return [app_json(t, installed[t.key]) for _, group in tpl.groups() for t in group]

    @r.get("/api/v1/apps/{key}")
    @handle
    def get_app(request: Request, key: str):
        t, answers, m = svc.app_defaults(key)
        gpus = engine.snapshot().gpus
        out = app_json(t, tpl.installed(m.instances)[t.key])
        out["answers"] = {k: answers.get(k) for k in tpl.answer_keys(t)}
        out["choices"] = {
            "gpu": tpl.gpu_choices(gpus, t),
            "offload": tpl.OFFLOAD_CHOICES if tpl.slots(t) else {},
            "server": ({tpl.NEW_SERVER: "Install a Llama Swap server with it"} | {k: v["name"] for k, v in m.servers.items()})
            if t.connects else {},
            "comfyui": {i.id: i.name for i in m.instances.values() if i.kind == "container" and i.gpu.mode != "none"}
            if t.connects == "openwebui" else {},
            "speech": tpl.speech_servers(m) if t.connects == "openwebui" else {},
            "model_slots": {f"{s.key}_file": s.label for s in tpl.slots(t)},
        }
        return out

    @r.post("/api/v1/apps/{key}/instance")
    @handle
    def app_instance(request: Request, key: str):
        data = body(request) if request.state.raw else {}
        return svc.app_instance(key, data).model_dump(mode="json")

    @r.post("/api/v1/apps/{key}/install")
    @handle
    def install_app(request: Request, key: str):
        done = svc.install(key, body(request) if request.state.raw else {}, fill_defaults=True)
        also = "".join(f" and {x.name}" for x in [done.bundle, *done.companions] if x)
        after = f" {tpl.get(key).after_install}" if tpl.get(key).after_install else ""
        fetching = f" Downloading {', '.join(done.downloads)}." if done.downloads else ""
        return JSONResponse({"message": f"Installed {done.inst.name}{also}.{fetching}{after}",
                             "instance": done.inst.model_dump(mode="json"),
                             "models": [x.id for x in done.models], "downloads": done.downloads,
                             "llm_server": done.bundle.id if done.bundle else None,
                             "companions": [x.id for x in done.companions]}, status_code=201)

    # ------------------------------------------------------------------ model servers
    @r.get("/api/v1/instances/{iid}/models")
    @handle
    def get_models(request: Request, iid: str):
        inst, cat = svc.server(iid)
        loaded, err = svc.loaded(iid, cat)
        files, notes = llamaswap.local_model_files(inst)
        return {"models": [m.model_dump(mode="json") for m in cat.models],
                "auto_aliases": [a.model_dump(mode="json") for a in cat.auto_aliases],
                "settings": {"threads": cat.threads, "llama_server": cat.llama_server,
                             "health_timeout": cat.health_timeout, "fit_gpus": cat.fit_gpus,
                             "auto_tune": cat.auto_tune,
                             "api_key": cat.api_key},
                "placement": svc.placement(iid, cat),
                "loaded": loaded, "loading": sorted(svc.loading.get(iid, set())), "error": err,
                "files": files, "file_notes": notes, "missing": svc.missing_files(iid),
                "start_failures": svc.start_failures(iid)}

    @r.put("/api/v1/instances/{iid}/models/{mid}")
    @handle
    def put_model(request: Request, iid: str, mid: str):
        data = body(request)
        data.setdefault("id", mid)
        _, cat = svc.server(iid)
        exists = any(m.id == mid for m in cat.models)
        msg = svc.save_model(iid, CatalogModel.model_validate(data), original=mid if exists else "")
        return JSONResponse({"message": msg}, status_code=200 if exists else 201)

    @r.delete("/api/v1/instances/{iid}/models/{mid}")
    @handle
    def delete_model(request: Request, iid: str, mid: str, files: bool = False):
        return {"message": svc.delete_model(iid, mid, files)}

    @r.post("/api/v1/instances/{iid}/models/{mid}/{op}")
    @handle
    def model_op(request: Request, iid: str, mid: str, op: str):
        if op == "load":
            return JSONResponse({"message": svc.load_model(iid, mid)}, status_code=202)
        if op == "unload":
            return {"message": svc.unload_model(iid, mid)}
        raise Problem("unknown operation", status=404)

    @r.get("/api/v1/instances/{iid}/suggestions")
    @handle
    def get_suggestions(request: Request, iid: str):
        return svc.suggestions(iid)

    @r.post("/api/v1/instances/{iid}/suggestions/{key}/{op}")
    @handle
    def suggestion_op(request: Request, iid: str, key: str, op: str):
        if op == "add":
            return JSONResponse({"message": svc.add_suggestion(iid, key)}, status_code=201)
        if op == "download":
            return JSONResponse({"message": svc.download_suggestion(iid, key)}, status_code=202)
        if op == "cancel":
            return {"message": svc.cancel_download(iid, key)}
        if op == "discard":
            return {"message": svc.discard_download(iid, key)}
        if op == "forget":
            return {"message": svc.forget_link(iid, key)}
        if op == "delete":
            return {"message": svc.delete_download(iid, key)}
        raise Problem("unknown operation", status=404)

    @r.post("/api/v1/instances/{iid}/models/from-link")
    @handle
    def model_from_link(request: Request, iid: str):
        data = body(request, object)
        if not isinstance(data, dict) or not data.get("url"):
            raise Problem('send {"url": "https://.../model.gguf"} (and optionally "name", "mmproj_url")')
        msg = svc.add_from_link(iid, str(data["url"]), str(data.get("name", "")), str(data.get("mmproj_url", "")))
        return JSONResponse({"message": msg}, status_code=201)

    @r.get("/api/v1/app-updates")
    @handle
    def get_app_updates(request: Request):
        return [{"instance": u.iid, "key": u.key, "title": u.title, "changes": u.lines, "afterwards": u.manual}
                for u in svc.app_updates()]

    @r.post("/api/v1/app-updates/apply-all")
    @handle
    def apply_all_app_updates(request: Request):
        return {"message": svc.apply_all_app_updates()}

    @r.post("/api/v1/app-updates/{iid}/{key}/{op}")
    @handle
    def app_update_op(request: Request, iid: str, key: str, op: str):
        if op == "apply":
            return {"message": svc.apply_app_update(iid, key)}
        if op == "skip":
            return {"message": svc.skip_app_update(iid, key)}
        raise Problem("unknown operation", status=404)

    @r.get("/api/v1/image-updates")
    @handle
    def get_image_updates(request: Request):
        instances, names = store.instances()[0], svc.names()
        return {"checked": svc.image_checked() or None, "every_hours": store.settings().image_check_hours,
                "instances": [{"instance": iid, "name": names[iid], "image": instances[iid].image,
                               "state": st.state, "auto_update": instances[iid].auto_update, "error": st.error,
                               "checked": st.checked or None, "updated": st.updated or None}
                              for iid, st in sorted(svc.image_status().items())]}

    @r.post("/api/v1/image-updates/check")
    @handle
    def check_image_updates(request: Request):
        return {"message": svc.check_images()}

    @r.post("/api/v1/image-updates/update-all")
    @handle
    def update_images(request: Request):
        """Update several apps in one change: {"instances": [...]}, or every app with a new
        version when there is no body."""
        data = body(request) if request.state.raw.strip() else {}
        ids = data.get("instances")
        if ids is not None and not (isinstance(ids, list) and ids and all(isinstance(i, str) for i in ids)):
            raise Problem('send {"instances": ["webui", ...]} or no body for every app')
        msg, job = svc.update_images(ids)
        return JSONResponse({"message": msg, "job": job_json(job)}, status_code=202)

    @r.post("/api/v1/image-updates/{iid}/update")
    @handle
    def update_image(request: Request, iid: str):
        msg, job = svc.update_image(iid)
        return JSONResponse({"message": msg, "job": job_json(job)}, status_code=202)

    @r.put("/api/v1/instances/{iid}/auto-update")
    @handle
    def auto_update(request: Request, iid: str):
        data = body(request)
        if not isinstance(data.get("on"), bool):
            raise Problem('send {"on": true} or {"on": false}')
        return {"message": svc.set_auto_update(iid, data["on"])}

    @r.patch("/api/v1/instances/{iid}/server-settings")
    @handle
    def server_settings(request: Request, iid: str):
        data = body(request)
        unknown = set(data) - {"threads", "llama_server", "health_timeout", "fit_gpus", "auto_tune", "new_api_key"}
        if unknown:
            raise Problem(f"unknown setting(s): {', '.join(sorted(unknown))}")
        return {"message": svc.server_settings(iid, threads=data.get("threads"), llama_server=data.get("llama_server"),
                                               health_timeout=data.get("health_timeout"),
                                               fit_gpus=data.get("fit_gpus"), auto_tune=data.get("auto_tune"),
                                               new_key=bool(data.get("new_api_key")))}

    @r.get("/api/v1/instances/{iid}/tuning")
    @handle
    def get_tuning(request: Request, iid: str):
        return svc.tuning(iid)

    @r.put("/api/v1/instances/{iid}/auto-aliases")
    @handle
    def put_auto_aliases(request: Request, iid: str):
        autos = [AutoAlias.model_validate(a) for a in body(request, list)]
        return {"message": svc.set_auto_aliases(iid, autos)}

    # ------------------------------------------------------------------ settings
    def settings_json() -> dict:
        s = store.settings()
        return {"admin_user": s.admin_user, "bind_address": s.bind_address, "network": s.network,
                "container_prefix": s.container_prefix, "data_dir": str(store.root),
                "host_data_dir": str(engine.host_data_dir), "trash_days": s.trash_days,
                "image_check_hours": s.image_check_hours, "hf_suggestions": s.hf_suggestions}

    @r.get("/api/v1/settings")
    def get_settings():
        return settings_json()

    @r.patch("/api/v1/settings")
    @handle
    def patch_settings(request: Request):
        data = body(request)
        unknown = set(data) - {"bind_address", "network", "trash_days", "image_check_hours", "hf_suggestions"}
        if unknown:
            raise Problem(f"unknown or read-only setting(s): {', '.join(sorted(unknown))}")
        msgs = []
        if "trash_days" in data:
            days = str(data["trash_days"])
            if not days.isdigit():
                raise Problem("trash_days must be a whole number of days (0 = keep until emptied)")
            msgs.append(svc.set_trash_days(int(days)))
        if "image_check_hours" in data:
            hours = str(data["image_check_hours"])
            if not hours.isdigit():
                raise Problem("image_check_hours must be a whole number of hours (0 = only when asked)")
            msgs.append(svc.set_image_check_hours(int(hours)))
        if "hf_suggestions" in data:
            if not isinstance(data["hf_suggestions"], bool):
                raise Problem("hf_suggestions must be true or false")
            msgs.append(svc.set_hf_suggestions(data["hf_suggestions"]))
        if {"bind_address", "network"} & set(data):
            msgs.append(svc.set_containers(data.get("bind_address"), data.get("network")))
        return {"message": " ".join(msgs), "settings": settings_json()}

    # ------------------------------------------------------------------ storage and trash
    @r.get("/api/v1/storage")
    @handle
    def get_storage(request: Request):
        return svc.storage()

    @r.get("/api/v1/storage/drives")
    @handle
    def get_drives(request: Request):
        """The machine's drives (a new one is mounted first if Settings say so)."""
        return svc.drives()

    @r.post("/api/v1/storage/drives/mount")
    @handle
    def mount_drive(request: Request):
        return {"message": svc.mount_drive(str(body(request).get("device", "")))}

    @r.put("/api/v1/storage/drives/auto")
    @handle
    def auto_mount(request: Request):
        return {"message": svc.set_auto_mount(bool(body(request).get("enabled")))}

    @r.get("/api/v1/storage/browse")
    @handle
    def browse_folders(request: Request, path: str = "/"):
        """The folders inside `path` on this machine, and the room on its disk."""
        li = svc.browse(path)
        if li.error:
            raise Problem(li.error, status=404)
        return asdict(li)

    @r.post("/api/v1/storage/move")
    @handle
    def storage_move(request: Request):
        """Move a folder the Storage report lists under "movable" to `dest`, or with "home": true
        back to where it was before it was moved (its "home"). Without confirm it only checks, and
        answers what the move would do."""
        data = body(request)
        source, dest = str(data.get("source", "")), str(data.get("dest", ""))
        if data.get("home"):
            dest = svc.home_of(source)
        if not data.get("confirm"):
            info = svc.check_move(source, dest)
            return JSONResponse({"error": 'Send {"confirm": true} to move it.', "needs_confirm": True, **info},
                                status_code=409)
        msg, job = svc.move_folder(source, dest)
        return JSONResponse({"message": msg, "job": job_json(job)}, status_code=202)

    @r.get("/api/v1/moves")
    @handle
    def get_moves(request: Request):
        return svc.move_history()

    @r.post("/api/v1/moves/{mid}/back")
    @handle
    def move_back(request: Request, mid: str):
        return {"message": svc.move_back(mid)}

    @r.delete("/api/v1/moves/{mid}/leftover")
    @handle
    def delete_move_leftover(request: Request, mid: str):
        return {"message": svc.delete_leftover(mid)}

    @r.post("/api/v1/storage/{what}")
    @handle
    def storage_cleanup(request: Request, what: str):
        if what == "images":
            return {"message": svc.remove_unused_images()}
        if what == "orphans":
            return {"message": svc.remove_orphans()}
        if what == "files":
            data = body(request)
            paths = data.get("paths")
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise Problem('send {"paths": [...]} with paths the storage report lists')
            return {"message": svc.trash_files(paths)}
        if what == "leftovers":
            data = body(request)
            return {"message": svc.trash_leftover(str(data.get("name", "")))}
        raise Problem("unknown clean-up (images, orphans, files or leftovers)", status=404)

    @r.get("/api/v1/trash")
    @handle
    def get_trash(request: Request):
        return svc.storage()["trash"]

    @r.post("/api/v1/trash/{eid}/restore")
    @handle
    def restore_trash(request: Request, eid: str):
        return {"message": svc.restore_trash(eid)}

    @r.delete("/api/v1/trash/{eid}")
    @handle
    def purge_trash_entry(request: Request, eid: str):
        return {"message": svc.purge_trash(eid)}

    @r.delete("/api/v1/trash")
    @handle
    def empty_trash(request: Request):
        return {"message": svc.purge_trash()}

    @r.post("/api/v1/settings/password")
    @handle
    def change_password(request: Request):
        data = body(request)
        return {"message": svc.change_password(str(data.get("current", "")), str(data.get("new", "")))}

    @r.post("/api/v1/token")
    def new_token():
        return {"message": "New API token made. The old one no longer works.", "token": auth.new_api_token(store)}

    # ------------------------------------------------------------------ updates
    @r.get("/api/v1/updates")
    def get_updates():
        u = updates.status(store.root)
        return {"running_version": __version__, "commit": u.running.get("commit", ""),
                "available": u.available, "behind": u.behind, "new_version": u.new_version,
                "checked_at": u.checked_at, "state": u.state, "step": u.step, "message": u.message,
                "busy": u.busy, "pending": u.pending, "progress": [{"step": s, "state": m} for s, m in u.progress],
                "commits": u.commits, "local_changes": u.local_changes,
                "branch": u.branch, "branches": u.branches,
                "dashboard_updates": u.dashboard_updates, "credentials_user": u.credentials, "command": u.command}

    # ------------------------------------------------------------------ power (the top menu's Power dialog)
    @r.get("/api/v1/power")
    def get_power():
        return svc.power_status()

    @r.post("/api/v1/power/{action}")
    @handle
    def power(request: Request, action: str):
        """restart (GPU Shepherd only), reboot or shutdown (the server)."""
        return JSONResponse({"message": svc.power(action)}, status_code=202)

    @r.post("/api/v1/updates/credentials")
    @handle
    def update_credentials(request: Request):
        data = body(request)
        return JSONResponse({"message": svc.update_credentials(str(data.get("user", "")), str(data.get("token", "")),
                                                               clear=bool(data.get("clear")))}, status_code=202)

    @r.post("/api/v1/updates/branch")
    @handle
    def update_branch(request: Request):
        """Follow another branch of the repository: {"branch": "main"}. Rebuilds GPU Shepherd from it."""
        return JSONResponse({"message": svc.switch_branch(str(body(request).get("branch", "")))},
                            status_code=202)

    @r.post("/api/v1/updates/{action}")
    @handle
    def update_request(request: Request, action: str):
        return JSONResponse({"message": svc.request_update(action)}, status_code=202)

    @r.get("/api/v1/updates/log")
    def update_log():
        return {"log": updates.log_text(store.root), "busy": updates.status(store.root).busy, "time": time.time()}

    @r.get("/api/v1/gpu-unlock")
    def get_gpu_unlock():
        """Experimental tensor unlock (gpu-unlock on the host): the cards, now and after the next reboot."""
        x = svc.unlock_status()
        return {"installed": x.installed, "installed_version": x.installed_version, "version": x.version,
                "state": x.state, "note": x.note, "run": x.run, "message": x.message, "busy": x.busy,
                "service": x.service, "blocker": x.blocker, "reboot_needed": x.reboot_needed,
                "checked_at": x.checked_at, "command": x.command,
                "cards": [{"pci": c.pci, "name": c.name, "family": c.family, "tier": c.tier, "supported": c.supported,
                           "unlocked": c.unlocked, "next_boot": c.configured, "note": c.note} for c in x.cards]}

    @r.post("/api/v1/gpu-unlock/{action}")
    @handle
    def gpu_unlock_request(request: Request, action: str):
        """check, install, uninstall, or set {"cards": ["0000:01:00.0"]} ([] unlocks none). Applies at the next reboot."""
        data = body(request) if action == "set" else {}
        cards = data.get("cards") or []
        if not isinstance(cards, list):
            return JSONResponse({"error": "cards must be a list of PCI addresses"}, status_code=400)
        return JSONResponse({"message": svc.request_unlock(action, [str(c) for c in cards])}, status_code=202)

    @r.get("/api/v1/changelog")
    def get_changelog():
        ours = changelog.read(changelog.find())
        latest = changelog.read(updates.folder(store.root) / "changelog.md")
        shown = latest if latest and (not ours or changelog.key(latest[0].version) >= changelog.key(ours[0].version)) else ours
        return [{"version": x.version, "date": x.date, "items": x.items,
                 "running": x.version == __version__,
                 "new": changelog.key(x.version) > changelog.key(__version__)} for x in shown]

    # ------------------------------------------------------------------ knowledge (documents) and skills
    # (/api/v1/books/... is the same, for older clients)
    def embeddings_choice(v) -> bool | None:
        if v in (None, "", "ask"):
            return None
        if v in ("server", True):
            return True
        if v in ("keep", False):
            return False
        raise Problem("embeddings: server (use the model server's embeddings model), keep, or leave it out")

    @r.get("/api/v1/knowledge")
    @handle
    def get_knowledge(request: Request):
        """The documents folder, its documents, the work running now, the apps documents can go to,
        and where it is all kept."""
        return svc.knowledge()

    @r.put("/api/v1/knowledge/folder")
    @handle
    def knowledge_folder(request: Request):
        return {"message": svc.set_documents_folder(str(body(request).get("path", "")))}

    @r.post("/api/v1/knowledge/upload")
    @handle
    async def knowledge_upload(request: Request):
        """Add documents: a multipart form with one or more `file` fields. A scan starts afterwards."""
        form = await request.form()
        files = [(f.filename, f.file) for f in form.getlist("file") if hasattr(f, "filename")]
        return JSONResponse({"message": await run_in_threadpool(svc.add_documents, files)}, status_code=202)

    @r.get("/api/v1/knowledge/files")
    @handle
    def knowledge_files(request: Request):
        """The document files already in the documents folder: {"folder", "files": {path: size}}."""
        return svc.document_files()

    @r.put("/api/v1/knowledge/files")
    @handle
    async def knowledge_put_file(request: Request):
        """Add one document, or a zip of them (unpacked into a folder named after it), sent as the raw body:
        PUT /api/v1/knowledge/files?path=Folder/Book.epub. Folders in the path are kept. Doesn't scan:
        POST /api/v1/knowledge/scan when all are in. Returns {"added": [paths], "left_out": n}.
        A big file can go in pieces: &offset=BYTE&total=SIZE on each (see Service.receive_document)."""
        size, q = request.headers.get("content-length"), request.query_params
        return await svc.receive_document(q.get("path", ""), request.stream(), int(size) if size and size.isdigit() else None,
                                          *_piece(q.get("offset"), q.get("total")))

    @r.post("/api/v1/knowledge/scan")
    @handle
    def knowledge_scan(request: Request):
        return JSONResponse({"message": svc.scan_documents()}, status_code=202)

    @r.post("/api/v1/knowledge/stop")
    @handle
    def knowledge_stop(request: Request):
        return {"message": svc.stop_knowledge()}

    @r.post("/api/v1/knowledge/send")
    @handle
    def knowledge_send(request: Request):
        """Put the ready documents into an Open WebUI knowledge base: {"instance", "name", "split",
        "embeddings": "server" | "keep", "sync": true | false}. sync true keeps it in sync from now on:
        every scan sends new documents and takes out the ones gone from the documents folder."""
        d = body(request)
        sync = d.get("sync")
        return JSONResponse({"message": svc.send_documents(str(d.get("instance", "")), str(d.get("name", "") or "Documents"),
                                                        bool(d.get("split")), embeddings_choice(d.get("embeddings")),
                                                        None if sync is None else bool(sync))},
                            status_code=202)

    @r.delete("/api/v1/knowledge/sync")
    @handle
    def knowledge_sync_stop(request: Request):
        """Stop keeping an Open WebUI knowledge base in sync (what is in it stays)."""
        return {"message": svc.stop_sync()}

    @r.get("/api/v1/knowledge/openwebui/{iid}")
    @handle
    def knowledge_openwebui(request: Request, iid: str):
        """What is in an Open WebUI's knowledge: {"bases": [{"id", "name", "files", "ours", "sent", "missing",
        "sync", "path"}], "sync", "stale"}. "path" is the knowledge base's page in Open WebUI."""
        return svc.openwebui_knowledge(iid)

    @r.post("/api/v1/knowledge/{key}/skill")
    @handle
    def knowledge_skill(request: Request, key: str):
        """Draft an agent skill from a document with a model server's chat model: {"server"} (optional)."""
        d = body(request) if request.state.raw else {}
        return JSONResponse({"message": svc.make_skill(key, str(d.get("server", "")))}, status_code=202)

    @r.post("/api/v1/knowledge/ocr")
    @handle
    def knowledge_ocr(request: Request):
        """Read the scanned PDFs (status "scanned") with OCR, in the background."""
        return JSONResponse({"message": svc.ocr_documents()}, status_code=202)

    @r.post("/api/v1/knowledge/delete")
    @handle
    def knowledge_delete_all(request: Request):
        """Delete every document with a status, in the background: {"status": "duplicate" | "failed"}."""
        return JSONResponse({"message": svc.delete_documents(str(body(request).get("status", "")))},
                            status_code=202)

    @r.delete("/api/v1/knowledge/{key}")
    @handle
    def knowledge_delete(request: Request, key: str):
        """Delete one document: its files, its text, and the copies sent to a running Open WebUI."""
        return {"message": svc.delete_document(key)}

    @r.post("/api/v1/knowledge/clear")
    @handle
    def knowledge_clear(request: Request):
        """Start over: {"what": ["documents", "converted", "skills", "apps"]} (all of them if left out).
        See Service.clear_knowledge for what each one removes."""
        d = body(request) if request.state.raw else {}
        what = d.get("what") or list(svc.CLEAR)
        if isinstance(what, str):
            what = [w.strip() for w in what.split(",")]
        return JSONResponse({"message": svc.clear_knowledge(tuple(what))}, status_code=202)

    @r.get("/api/v1/skills/{slug}")
    @handle
    def get_skill(request: Request, slug: str):
        return {"slug": slug, "text": svc.skill_text(slug)}

    @r.put("/api/v1/skills/{slug}")
    @handle
    def put_skill(request: Request, slug: str):
        return {"message": svc.save_skill(slug, str(body(request).get("text", "")))}

    @r.post("/api/v1/skills/{slug}/install")
    @handle
    def install_skill(request: Request, slug: str):
        return {"message": svc.install_skill(slug, str(body(request).get("instance", "")))}

    @r.delete("/api/v1/skills/{slug}")
    @handle
    def delete_skill(request: Request, slug: str):
        return {"message": svc.delete_skill(slug)}

    for route in list(r.routes):                     # the old names still work
        if getattr(route, "path", "").startswith("/api/v1/knowledge"):
            r.add_api_route("/api/v1/books" + route.path[len("/api/v1/knowledge"):], route.endpoint,
                            methods=list(route.methods), include_in_schema=False)
    return r
