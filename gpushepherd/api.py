"""The JSON API under /api/v1: everything the dashboard can do, for scripts and the `gpu-shepherd`
command (cli.py). It calls the same service.Shepherd methods as the dashboard's routes.

Authentication is the API token (auth.api_token, the `api-token` file in the data folder) sent as
`Authorization: Bearer <token>`; a logged-in browser may also GET. app.py's Guard checks it before
any route here runs. Errors are {"error": "..."} with a 4xx status. Reference: docs/api.md.
"""
from __future__ import annotations

import functools
import time
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

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
            return JSONResponse({"error": err.msg}, status_code=err.status)
        return JSONResponse({"error": message(err)}, status_code=400)

    async def body(request: Request, kind=dict):
        try:
            data = await request.json()
        except ValueError as err:
            raise Problem("send a JSON body") from err
        if not isinstance(data, kind):
            raise Problem(f"the JSON body must be {'an object' if kind is dict else 'a list'}")
        return data

    def handle(fn):
        """Run a route, answering a Problem or a validation error as {"error": ...}."""
        @functools.wraps(fn)
        async def wrapper(request: Request, **kw):
            try:
                result = fn(request, **kw)
                if hasattr(result, "__await__"):
                    result = await result
                return result
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

    def instance_json(inst: Instance, snap=None) -> dict:
        snap = snap or engine.snapshot()
        st = snap.states.get(inst.id)
        running = bool(st and st.running)
        return inst.model_dump(mode="json") | {"state": {
            "running": running, "status": st.status if st else "not created",
            "status_text": st.status_text if st else "", "gpus": snap.allocations.get(inst.id, []),
            "container": engine.container_name(inst.id),
            "restart_needed": bool(running and st.config_hash and st.config_hash != engine.config_hash(inst))}}

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
        return [instance_json(i, snap) for i in sorted(snap.instances.values(), key=lambda i: i.id)]

    @r.get("/api/v1/instances/{iid}")
    @handle
    def get_instance(request: Request, iid: str):
        return instance_json(svc.instance(iid))

    @r.put("/api/v1/instances/{iid}")
    @handle
    async def put_instance(request: Request, iid: str):
        data = await body(request)
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
    async def reset_login(request: Request, iid: str):
        data = await body(request) if await request.body() else {}
        msg, password = svc.reset_login(iid, str(data.get("user") or ""), str(data.get("password") or ""))
        return {"message": msg, "password": password}

    @r.post("/api/v1/instances/{iid}/tuning")
    @handle
    async def tuning_decide(request: Request, iid: str):
        data = await body(request)
        return {"message": svc.tuning_decide(iid, str(data.get("key", "")), str(data.get("action", "")))}

    @r.post("/api/v1/instances/{iid}/{action}")
    @handle
    async def instance_action(request: Request, iid: str, action: str):
        data = await body(request) if await request.body() else {}
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

    # ------------------------------------------------------------------ App Store
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
            "gpu": tpl.gpu_choices(gpus),
            "offload": tpl.OFFLOAD_CHOICES if tpl.slots(t) else {},
            "server": ({tpl.NEW_SERVER: "Install an LLM server with it"} | {k: v["name"] for k, v in m.servers.items()})
            if t.connects else {},
            "comfyui": {i.id: i.name for i in m.instances.values() if i.kind == "container" and i.gpu.mode != "none"}
            if t.connects == "openwebui" else {},
            "speech": tpl.speech_servers(m) if t.connects == "openwebui" else {},
            "model_slots": {f"{s.key}_file": s.label for s in tpl.slots(t)},
        }
        return out

    @r.post("/api/v1/apps/{key}/instance")
    @handle
    async def app_instance(request: Request, key: str):
        data = await body(request) if await request.body() else {}
        return svc.app_instance(key, data).model_dump(mode="json")

    @r.post("/api/v1/apps/{key}/install")
    @handle
    async def install_app(request: Request, key: str):
        done = svc.install(key, await body(request) if await request.body() else {}, fill_defaults=True)
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
    async def put_model(request: Request, iid: str, mid: str):
        data = await body(request)
        data.setdefault("id", mid)
        _, cat = svc.server(iid)
        exists = any(m.id == mid for m in cat.models)
        msg = svc.save_model(iid, CatalogModel.model_validate(data), original=mid if exists else "")
        return JSONResponse({"message": msg}, status_code=200 if exists else 201)

    @r.delete("/api/v1/instances/{iid}/models/{mid}")
    @handle
    def delete_model(request: Request, iid: str, mid: str):
        return {"message": svc.delete_model(iid, mid)}

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
        raise Problem("unknown operation", status=404)

    @r.post("/api/v1/instances/{iid}/models/from-link")
    @handle
    async def model_from_link(request: Request, iid: str):
        body = await request.json()
        if not isinstance(body, dict) or not body.get("url"):
            raise Problem('send {"url": "https://.../model.gguf"} (and optionally "name", "mmproj_url")')
        msg = svc.add_from_link(iid, str(body["url"]), str(body.get("name", "")), str(body.get("mmproj_url", "")))
        return JSONResponse({"message": msg}, status_code=201)

    @r.get("/api/v1/app-updates")
    @handle
    def get_app_updates(request: Request):
        return [{"instance": u.iid, "key": u.key, "title": u.title, "changes": u.lines, "afterwards": u.manual}
                for u in svc.app_updates()]

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

    @r.post("/api/v1/image-updates/{iid}/update")
    @handle
    def update_image(request: Request, iid: str):
        msg, job = svc.update_image(iid)
        return JSONResponse({"message": msg, "job": job_json(job)}, status_code=202)

    @r.put("/api/v1/instances/{iid}/auto-update")
    @handle
    async def auto_update(request: Request, iid: str):
        data = await body(request)
        if not isinstance(data.get("on"), bool):
            raise Problem('send {"on": true} or {"on": false}')
        return {"message": svc.set_auto_update(iid, data["on"])}

    @r.patch("/api/v1/instances/{iid}/server-settings")
    @handle
    async def server_settings(request: Request, iid: str):
        data = await body(request)
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
    async def put_auto_aliases(request: Request, iid: str):
        autos = [AutoAlias.model_validate(a) for a in await body(request, list)]
        return {"message": svc.set_auto_aliases(iid, autos)}

    # ------------------------------------------------------------------ settings
    def settings_json() -> dict:
        s = store.settings()
        return {"admin_user": s.admin_user, "bind_address": s.bind_address, "network": s.network,
                "container_prefix": s.container_prefix, "data_dir": str(store.root),
                "host_data_dir": str(engine.host_data_dir), "trash_days": s.trash_days,
                "image_check_hours": s.image_check_hours}

    @r.get("/api/v1/settings")
    def get_settings():
        return settings_json()

    @r.patch("/api/v1/settings")
    @handle
    async def patch_settings(request: Request):
        data = await body(request)
        unknown = set(data) - {"bind_address", "network", "trash_days", "image_check_hours"}
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
    async def mount_drive(request: Request):
        return {"message": svc.mount_drive(str((await body(request)).get("device", "")))}

    @r.put("/api/v1/storage/drives/auto")
    @handle
    async def auto_mount(request: Request):
        return {"message": svc.set_auto_mount(bool((await body(request)).get("enabled")))}

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
    async def storage_move(request: Request):
        """Move a folder the Storage report lists under "movable" to `dest`. Without confirm it
        only checks, and answers what the move would do."""
        data = await body(request)
        source, dest = str(data.get("source", "")), str(data.get("dest", ""))
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
    async def storage_cleanup(request: Request, what: str):
        if what == "images":
            return {"message": svc.remove_unused_images()}
        if what == "orphans":
            return {"message": svc.remove_orphans()}
        if what == "files":
            data = await body(request)
            paths = data.get("paths")
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise Problem('send {"paths": [...]} with paths the storage report lists')
            return {"message": svc.trash_files(paths)}
        if what == "leftovers":
            data = await body(request)
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
    async def change_password(request: Request):
        data = await body(request)
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
                "dashboard_updates": u.dashboard_updates, "credentials_user": u.credentials, "command": u.command}

    @r.post("/api/v1/updates/credentials")
    @handle
    async def update_credentials(request: Request):
        data = await body(request)
        return JSONResponse({"message": svc.update_credentials(str(data.get("user", "")), str(data.get("token", "")),
                                                               clear=bool(data.get("clear")))}, status_code=202)

    @r.post("/api/v1/updates/{action}")
    @handle
    def update_request(request: Request, action: str):
        return JSONResponse({"message": svc.request_update(action)}, status_code=202)

    @r.get("/api/v1/updates/log")
    def update_log():
        return {"log": updates.log_text(store.root), "busy": updates.status(store.root).busy, "time": time.time()}

    @r.get("/api/v1/changelog")
    def get_changelog():
        ours = changelog.read(changelog.find())
        latest = changelog.read(updates.folder(store.root) / "changelog.md")
        shown = latest if latest and (not ours or changelog.key(latest[0].version) >= changelog.key(ours[0].version)) else ours
        return [{"version": x.version, "date": x.date, "items": x.items,
                 "running": x.version == __version__,
                 "new": changelog.key(x.version) > changelog.key(__version__)} for x in shown]

    return r
