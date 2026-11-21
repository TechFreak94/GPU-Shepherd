"""An MCP server at /api/v1/mcp, so an AI agent (Claude Code, OpenCode, Cline, Cursor...) can look
after this GPU Shepherd: check GPUs and apps, start and stop them, install apps, add models.

It speaks MCP's Streamable HTTP transport, answering each JSON-RPC POST with plain JSON (no
sessions, no event stream). Each tool is a fixed call to the JSON API (api.py), made in-process
with the caller's own `Authorization` header, so a tool does exactly what the endpoint does and
app.py's Guard checks the API token as for any other API request.

The tools are a stable contract: their names and inputs only ever gain optional fields, and
TOOLS_VERSION goes up when tools are added. Some things are left out on purpose and stay in the
dashboard: passwords and the API token, reboot and shutdown, remote access, tensor unlock,
deleting instances, files or trash, and moving storage.

The Claude Code plugin that uses it is in agent-plugin/ (see its README).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import __version__

TOOLS_VERSION = 2
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")   # newest first
API = "/api/v1"
LOG_LINES = 200

INSTRUCTIONS = """\
GPU Shepherd shares one machine's GPUs between self-hosted AI apps (instances), each in its own container.
Starting an instance can stop or move others to free GPUs. Rules:
- Look before you change: get_status or list_instances first.
- start_instance, stop_instance and restart_instance answer with a plan when they would also stop or move
  other instances, or memory may not be enough. Show the user that plan and only send confirm=true after
  they agree. Never confirm on your own.
- Ask the user before any change (start, stop, restart, install, model load or unload, updates).
- GPU Shepherd makes one change at a time; a change runs as a job. Follow it with get_job.
- Passwords, the API token, reboot/shutdown, deleting and storage moves are not available here:
  send the user to the dashboard for those."""


def seg(value: Any) -> str:
    """One path segment, so an ID can't reach another endpoint."""
    text = str(value)
    if not text or text in (".", ".."):
        raise ValueError("an ID can't be empty")
    return quote(text, safe="")


def obj(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


ID = {"type": "string", "description": "Instance ID, as list_instances shows it (e.g. 'llm', 'comfyui')."}
CONFIRM = {"type": "boolean", "default": False,
           "description": "Go ahead with a plan that also stops or moves other instances. Only after the user agreed."}
MODEL = {"type": "string", "description": "Model ID on that model server, as list_models shows it."}


@dataclass
class Tool:
    name: str
    title: str
    description: str
    schema: dict
    call: Callable[[dict], tuple]          # args -> (method, path, json body or None)
    read_only: bool = False
    destructive: bool = False
    shape: Callable[[Any, dict], Any] | None = None    # trims an answer for the agent
    annotations: dict = field(init=False)

    def __post_init__(self):
        self.annotations = {"title": self.title, "readOnlyHint": self.read_only,
                            "destructiveHint": self.destructive, "idempotentHint": self.read_only,
                            "openWorldHint": False}

    def listing(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description,
                "inputSchema": self.schema, "annotations": self.annotations}


def tail_logs(answer: Any, args: dict) -> Any:
    if isinstance(answer, dict) and isinstance(answer.get("logs"), str):
        n = max(1, min(int(args.get("lines") or LOG_LINES), 5000))
        lines = answer["logs"].splitlines()
        answer = {"logs": "\n".join(lines[-n:]), "lines_shown": min(n, len(lines)), "lines_total": len(lines)}
    return answer


def get(path: str) -> Callable[[dict], tuple]:
    return lambda a: ("GET", path.format(**{k: seg(v) for k, v in a.items() if isinstance(v, str)}), None)


def action(verb: str) -> Callable[[dict], tuple]:
    return lambda a: ("POST", f"{API}/instances/{seg(a['id'])}/{verb}", {"confirm": bool(a.get("confirm"))})


TOOLS: list[Tool] = [
    # ------------------------------------------------------------------ looking
    Tool("get_status", "GPUs and instances at a glance",
         "The GPUs (memory in MiB, use, which instance holds each), host CPU and RAM, each instance's running "
         "state and GPUs, and the change running now (if any).",
         obj({}), get(f"{API}/state"), read_only=True),
    Tool("list_instances", "List instances",
         "Every instance (app) with its settings and state: running, status, GPUs, container, restart_needed.",
         obj({}), get(f"{API}/instances"), read_only=True),
    Tool("get_instance", "Show one instance", "One instance's settings and state.",
         obj({"id": ID}, ("id",)), get(f"{API}/instances/{{id}}"), read_only=True),
    Tool("get_logs", "Read an instance's log", "The last lines of an instance's container log.",
         obj({"id": ID, "lines": {"type": "integer", "minimum": 1, "maximum": 5000, "default": LOG_LINES,
                                  "description": "How many of the last lines to show."}}, ("id",)),
         get(f"{API}/instances/{{id}}/logs"), read_only=True, shape=tail_logs),
    Tool("get_priority", "Show the GPU priority order",
         "The apps that use GPUs, first = keeps its GPUs when there are not enough. Each has rank, running state, "
         "GPUs and the 0-100 priority number behind the order.",
         obj({}), get(f"{API}/priority"), read_only=True),
    Tool("set_priority_order", "Set the GPU priority order",
         "Put apps in priority order, first = keeps its GPUs first. Apps left out go after those named. "
         "Nothing restarts. Ask the user first.",
         obj({"order": {"type": "array", "items": ID, "description": "Instance IDs, most important first."}},
             ("order",)),
         lambda a: ("PUT", f"{API}/priority", {"order": list(a["order"])})),
    Tool("list_jobs", "List recent changes",
         "Recent changes (jobs), newest first: id, title, status (running, done, failed, cancelled), error, "
         "waiting_for (running work a stop waits for).",
         obj({}), get(f"{API}/jobs"), read_only=True),
    Tool("get_job", "Follow a change", "One job with its log lines. Use it to follow a start, stop or update.",
         obj({"job_id": {"type": "integer"}}, ("job_id",)),
         lambda a: ("GET", f"{API}/jobs/{int(a['job_id'])}", None), read_only=True),
    Tool("list_apps", "List App Center apps",
         "The apps that can be installed: key, title, what each needs, which instances are already installed.",
         obj({}), get(f"{API}/apps"), read_only=True),
    Tool("get_app", "Show an app's install options",
         "One app, the answers its install form starts with and the choices for each (gpu, server, model files...).",
         obj({"key": {"type": "string", "description": "App key from list_apps."}}, ("key",)),
         get(f"{API}/apps/{{key}}"), read_only=True),
    Tool("list_models", "List a model server's models",
         "For a model server (kind llama-swap): its models, auto tags, which are loaded, and the .gguf files it sees.",
         obj({"id": ID}, ("id",)), get(f"{API}/instances/{{id}}/models"), read_only=True),
    Tool("list_suggested_models", "List suggested models",
         "Suggested models for a model server, best fit for this machine first, with their download state.",
         obj({"id": ID}, ("id",)), get(f"{API}/instances/{{id}}/suggestions"), read_only=True),
    Tool("get_storage", "Show disk space",
         "Free space per disk, folder sizes, app images, files worth a look, and the trash.",
         obj({}), get(f"{API}/storage"), read_only=True),
    Tool("list_app_updates", "List app setting updates",
         "Settings an app's template gained after it was installed, waiting to be applied.",
         obj({}), get(f"{API}/app-updates"), read_only=True),
    Tool("list_image_updates", "List app image updates",
         "Newer releases of the apps' Docker images, per instance.",
         obj({}), get(f"{API}/image-updates"), read_only=True),
    Tool("get_dashboard_update", "Show GPU Shepherd's own update",
         "The running GPU Shepherd version and whether a newer one is available.",
         obj({}), get(f"{API}/updates"), read_only=True),

    # ------------------------------------------------------------------ changing
    Tool("start_instance", "Start an instance",
         "Start an instance. If it would also stop or move other instances (or memory may not be enough) "
         "nothing happens and the answer has needs_confirm and the plan: show it to the user, and call again "
         "with confirm=true only if they agree. Otherwise it answers with a job to follow.",
         obj({"id": ID, "confirm": CONFIRM}, ("id",)), action("start")),
    Tool("stop_instance", "Stop an instance",
         "Stop an instance, waiting for the work it is doing to finish. Same confirm rules as start_instance.",
         obj({"id": ID, "confirm": CONFIRM}, ("id",)), action("stop"), destructive=True),
    Tool("restart_instance", "Restart an instance",
         "Restart an instance (also applies changed settings). Same confirm rules as start_instance.",
         obj({"id": ID, "confirm": CONFIRM}, ("id",)), action("restart"), destructive=True),
    Tool("cancel_job", "Call off a change", "Call off the running change before anything is stopped.",
         obj({"job_id": {"type": "integer"}}, ("job_id",)),
         lambda a: ("POST", f"{API}/jobs/{int(a['job_id'])}/cancel", None)),
    Tool("stop_now", "Stop without waiting",
         "While a change waits for running work (a reply being written, a picture being made): stop without "
         "waiting longer. That work is lost.",
         obj({"job_id": {"type": "integer"}}, ("job_id",)),
         lambda a: ("POST", f"{API}/jobs/{int(a['job_id'])}/skip", None), destructive=True),
    Tool("install_app", "Install an app",
         "Install an App Center app as a new instance. answers holds only what differs from get_app's answers, "
         "e.g. {\"gpu\": \"pin:1\", \"port\": 8189} or {\"server\": \"new\"}. It is not started.",
         obj({"key": {"type": "string", "description": "App key from list_apps."},
              "answers": {"type": "object", "description": "Install answers that differ from the defaults."}},
             ("key",)),
         lambda a: ("POST", f"{API}/apps/{seg(a['key'])}/install", a.get("answers") or {})),
    Tool("add_suggested_model", "Add a suggested model",
         "Add a suggested model to a model server with its makers' settings, downloading it in the background.",
         obj({"id": ID, "key": {"type": "string", "description": "Suggestion key from list_suggested_models."}},
             ("id", "key")),
         lambda a: ("POST", f"{API}/instances/{seg(a['id'])}/suggestions/{seg(a['key'])}/add", None)),
    Tool("add_model_from_link", "Add a model by its link",
         "Add a model by the link of its .gguf file (e.g. a Hugging Face file or model page); it downloads "
         "in the background.",
         obj({"id": ID, "url": {"type": "string"}, "name": {"type": "string"},
              "mmproj_url": {"type": "string", "description": "Optional vision projector .gguf link."}},
             ("id", "url")),
         lambda a: ("POST", f"{API}/instances/{seg(a['id'])}/models/from-link",
                    {k: a[k] for k in ("url", "name", "mmproj_url") if a.get(k)})),
    Tool("load_model", "Load a model", "Load a model on a running model server now, in the background.",
         obj({"id": ID, "model": MODEL}, ("id", "model")),
         lambda a: ("POST", f"{API}/instances/{seg(a['id'])}/models/{seg(a['model'])}/load", None)),
    Tool("unload_model", "Unload a model",
         "Unload a model from a model server ('_all' unloads every model). Refused while it is answering.",
         obj({"id": ID, "model": MODEL}, ("id", "model")),
         lambda a: ("POST", f"{API}/instances/{seg(a['id'])}/models/{seg(a['model'])}/unload", None),
         destructive=True),
    Tool("apply_app_update", "Apply an app setting update", "Apply one update from list_app_updates.",
         obj({"id": ID, "key": {"type": "string", "description": "Update key from list_app_updates."}},
             ("id", "key")),
         lambda a: ("POST", f"{API}/app-updates/{seg(a['id'])}/{seg(a['key'])}/apply", None)),
    Tool("check_image_updates", "Check for app image updates", "Look in the registries for new app releases now.",
         obj({}), lambda a: ("POST", f"{API}/image-updates/check", None)),
    Tool("update_app_image", "Update an app's image",
         "Download an app's new image and restart it, waiting for running work. Answers with a job.",
         obj({"id": ID}, ("id",)),
         lambda a: ("POST", f"{API}/image-updates/{seg(a['id'])}/update", None), destructive=True),
    Tool("check_dashboard_update", "Check for a GPU Shepherd update", "Ask the host to check for a new version.",
         obj({}), lambda a: ("POST", f"{API}/updates/check", None)),
    Tool("update_dashboard", "Update GPU Shepherd",
         "Update GPU Shepherd itself. The dashboard restarts; instances keep running.",
         obj({}), lambda a: ("POST", f"{API}/updates/update", None)),
]
BY_NAME = {t.name: t for t in TOOLS}


def rpc_error(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def text_result(data: Any, error: bool = False) -> dict:
    text = data if isinstance(data, str) else json.dumps(data, indent=1, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": error}


async def call_tool(request: Request, name: str, args: dict) -> dict:
    tool = BY_NAME.get(name)
    if tool is None:
        raise LookupError(f"no tool '{name}'")
    if not isinstance(args, dict):
        return text_result("arguments must be an object", True)
    missing = [k for k in tool.schema["required"] if args.get(k) in (None, "")]
    if missing:
        return text_result(f"missing: {', '.join(missing)}", True)
    try:
        method, path, body = tool.call(args)
    except (ValueError, TypeError, KeyError) as err:
        return text_result(f"invalid arguments: {err}", True)
    headers = {"Authorization": request.headers.get("Authorization", "")}
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gpu-shepherd", timeout=600) as client:
        resp = await client.request(method, path, headers=headers, json=body)
    try:
        answer = resp.json()
    except ValueError:
        answer = resp.text
    if resp.status_code == 409 and isinstance(answer, dict) and answer.get("needs_confirm"):
        # not a failure: the plan for the user to agree to
        answer = {"needs_confirm": True, "plan": answer.get("plan", []), "warnings": answer.get("warnings", []),
                  "message": "Nothing changed yet. Show the user this plan; if they agree, call again with confirm=true."}
        return text_result(answer)
    if tool.shape and resp.status_code < 400:
        answer = tool.shape(answer, args)
    if resp.status_code >= 400:
        return text_result({"status": resp.status_code} | (answer if isinstance(answer, dict) else {"error": answer}),
                           True)
    return text_result(answer)


async def handle(request: Request, msg: Any) -> dict | None:
    """One JSON-RPC message; None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return rpc_error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    rid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
    if "id" not in msg:
        return None                     # notifications/initialized, cancelled...: nothing to answer
    if method == "initialize":
        asked = params.get("protocolVersion") if isinstance(params, dict) else None
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "gpu-shepherd", "title": "GPU Shepherd", "version": __version__},
            "instructions": INSTRUCTIONS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": [t.listing() for t in TOOLS],
                                                         "_meta": {"toolsVersion": TOOLS_VERSION}}}
    if method == "tools/call":
        if not isinstance(params, dict):
            return rpc_error(rid, -32602, "params must be an object")
        try:
            result = await call_tool(request, str(params.get("name", "")), params.get("arguments") or {})
        except LookupError as err:
            return rpc_error(rid, -32602, str(err))
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return rpc_error(rid, -32601, f"method not found: {method}")


def router() -> APIRouter:
    r = APIRouter()

    @r.post(f"{API}/mcp")
    async def mcp(request: Request):
        try:
            msg = json.loads(await request.body())
        except ValueError:
            return JSONResponse(rpc_error(None, -32700, "parse error"), status_code=400)
        if isinstance(msg, list):       # a batch (protocol 2025-03-26)
            answers = [a for a in [await handle(request, m) for m in msg] if a is not None]
            return JSONResponse(answers) if answers else Response(status_code=202)
        answer = await handle(request, msg)
        return JSONResponse(answer) if answer is not None else Response(status_code=202)

    @r.get(f"{API}/mcp")
    def mcp_stream():
        # no server-to-client event stream: every answer comes back on its POST
        return Response(status_code=405, headers={"Allow": "POST"})

    @r.delete(f"{API}/mcp")
    def mcp_end():
        return Response(status_code=405, headers={"Allow": "POST"})

    return r
