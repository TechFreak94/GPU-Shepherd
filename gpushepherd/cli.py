"""The `gpu-shepherd` commands that manage a running GPU Shepherd: a small client of the JSON API
(api.py), so a command does exactly what the same button on the dashboard does.

It finds the app at http://127.0.0.1:<port> and reads the API token from the data folder, which is
what `docker exec gpu-shepherd gpu-shepherd ...` sees. From elsewhere, pass --url and --token (or
SHEPHERD_URL and SHEPHERD_TOKEN). Reference: docs/cli.md.
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

from .config import CatalogModel, ConfigError, parse_command, parse_env, parse_ports, parse_volumes

# instance fields set as text, as the dashboard's form takes them (lines separated by ";" here)
TEXT_FIELDS = {"ports": parse_ports, "volumes": parse_volumes, "env": parse_env, "command": parse_command}
NEW_MODEL = CatalogModel(id="new", file="/models/new.gguf").model_dump(mode="json") | {"id": "", "file": ""}


class Failed(Exception):
    """Stop with this message and exit status 1."""


# --------------------------------------------------------------------------- talking to the app

class Client:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def call(self, method: str, path: str, body=None, ok=(200, 201, 202)) -> dict | list:
        try:
            r = httpx.request(method, self.url + path, json=body, timeout=60,
                              headers={"Authorization": f"Bearer {self.token}"})
        except httpx.HTTPError as err:
            raise Failed(f"GPU Shepherd isn't answering at {self.url} ({err}). Is it running? "
                         "Pass --url if it listens somewhere else.") from err
        try:
            data = r.json()
        except ValueError:
            data = {"error": r.text.strip() or f"HTTP {r.status_code}"}
        if r.status_code == 401:
            raise Failed("The API token was refused. Run this on the server as root (it reads the token from the "
                         "data folder), or pass the token from Settings with --token.")
        if r.status_code not in ok:
            raise Failed(data.get("error", f"HTTP {r.status_code}") if isinstance(data, dict) else str(data))
        return data

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None, ok=(200, 201, 202)):
        return self.call("POST", path, body, ok)


def connect(args) -> Client:
    url = args.url or os.environ.get("SHEPHERD_URL") or f"http://127.0.0.1:{args.port}"
    token = args.token or os.environ.get("SHEPHERD_TOKEN", "")
    if not token:
        path = Path(args.data_dir) / "api-token"
        try:
            token = path.read_text(encoding="utf-8").strip()
        except PermissionError as err:
            raise Failed(f"Can't read {path}: run with sudo, or pass --token.") from err
        except OSError as err:
            raise Failed(f"No API token in {path}. Start GPU Shepherd once (it makes one), or pass --data-dir "
                         "or --token.") from err
    return Client(url, token)


# --------------------------------------------------------------------------- output helpers

def out(args, data, text=None) -> None:
    """Print data as JSON with --json, otherwise as text (a string, or a function that prints)."""
    if args.json:
        print(json.dumps(data, indent=2))
    elif callable(text):
        text(data)
    elif text is not None:
        print(text)
    elif isinstance(data, dict) and "message" in data:
        print(data["message"])
    else:
        print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip())


def table(rows: list[list], head: list[str]) -> None:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(x) for x in col) for col in zip(head, *rows)] if rows else [len(h) for h in head]
    for r in [head] + rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def gb(n) -> str:
    return f"{(n or 0) / 2**30:.1f} GB"


def gpus_text(gpus: list[int]) -> str:
    return ",".join(map(str, gpus)) if gpus else "-"


def ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def pairs(items: list[str]) -> dict[str, str]:
    """KEY=VALUE arguments."""
    found = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise Failed(f"'{item}': write it as KEY=VALUE")
        found[key.strip()] = value
    return found


def typed(value: str, like):
    """A KEY=VALUE value converted to the type of the value it replaces."""
    if isinstance(like, bool):
        v = value.strip().lower()
        if v not in ("true", "false", "yes", "no", "on", "off", "1", "0"):
            raise Failed(f"'{value}': use true or false")
        return v in ("true", "yes", "on", "1")
    if isinstance(like, int):
        try:
            return int(value)
        except ValueError as err:
            raise Failed(f"'{value}' is not a whole number") from err
    if isinstance(like, list):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return [int(x) if x.isdigit() else x for x in items]
    if like is None:
        return value.strip() or None
    return value


def apply_pairs(obj: dict, changes: dict[str, str]) -> dict:
    """Set dotted keys (gpu.mode=elastic) in a copy of obj, typed like what is there."""
    obj = json.loads(json.dumps(obj))
    for key, value in changes.items():
        if key in TEXT_FIELDS:
            try:
                parsed = TEXT_FIELDS[key](value if key == "command" else value.replace(";", "\n"))
            except ConfigError as err:
                raise Failed(str(err)) from err
            obj[key] = [x.model_dump() for x in parsed] if key in ("ports", "volumes") else parsed
            continue
        *path, last = key.split(".")
        node = obj
        for part in path:
            if not isinstance(node.get(part), dict):
                raise Failed(f"unknown setting '{key}'")
            node = node[part]
        if last not in node:
            raise Failed(f"unknown setting '{key}'")
        node[last] = typed(value, node[last])
    return obj


def follow(c: Client, job: dict) -> int:
    """Print a job's log as it runs. Returns the exit status."""
    seen = 0
    try:
        while True:
            j = c.get(f"/api/v1/jobs/{job['id']}")
            for line in j["log"][seen:]:
                print(line, flush=True)
            seen = len(j["log"])
            if j["status"] != "running":
                return 0 if j["status"] == "done" else 1
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStill running in GPU Shepherd. Follow it with `gpu-shepherd jobs show {job['id']}`; "
              f"`gpu-shepherd jobs skip {job['id']}` stops waiting for running work, "
              f"`gpu-shepherd jobs cancel {job['id']}` calls it off.")
        return 130


# --------------------------------------------------------------------------- commands

def cmd_status(args, c: Client) -> int:
    s = c.get("/api/v1/state")

    def show(s):
        print(f"GPU Shepherd {s['version']}" + (f"  (update available: {s['update_available']} new)"
                                                if s.get("update_available") else ""))
        h = s["host"]
        print(f"CPU {h['cpu_pct']}% of {h['cpu_threads']} threads · RAM {gb(h['mem_used'])} of {gb(h['mem_total'])}")
        if s.get("gpu_error"):
            print(f"GPUs: {s['gpu_error']}")
        for g in s["gpus"]:
            print(f"GPU {g['index']}  {g['name']}  {g['mem_used'] / 1024:.1f}/{g['mem_total'] / 1024:.1f} GB  "
                  f"{g['util']:.0f}%  {', '.join(g['users']) or 'free'}")
        print()
        table([[i, v["name"], "running" if v["running"] else "stopped", gpus_text(v["gpus"]), v["policy"]["mode"]]
               for i, v in s["instances"].items()], ["ID", "NAME", "STATE", "GPUS", "GPU MODE"])
        if s.get("job"):
            print(f"\nNow: {s['job']}" + (f" (waiting for {', '.join(s['waiting_for'])})" if s["waiting_for"] else ""))
        for p in s.get("problems", []):
            print(f"Problem: {p}")
    out(args, s, show)
    return 0


def cmd_list(args, c: Client) -> int:
    items = c.get("/api/v1/instances")
    out(args, items, lambda items: table(
        [[i["id"], i["name"], i["app"] or i["kind"], i["state"]["status"] + (" (restart to apply)"
          if i["state"]["restart_needed"] else ""), gpus_text(i["state"]["gpus"]),
          ",".join(str(p["host"]) for p in i["ports"]) or "-"] for i in items],
        ["ID", "NAME", "APP", "STATE", "GPUS", "PORTS"]))
    return 0


def cmd_show(args, c: Client) -> int:
    inst = c.get(f"/api/v1/instances/{args.id}")
    if not args.json:
        inst.pop("state", None)           # so the output can be edited and fed to `apply`
    out(args, inst)
    return 0


def cmd_change(args, c: Client) -> int:
    path = f"/api/v1/instances/{args.id}/{args.command}"
    r = c.post(path, {}, ok=(200, 202, 409))
    if r.get("needs_confirm"):
        print("This also changes other instances:")
        for line in r["plan"]:
            print(f"  {line}")
        if not (args.yes or ask("Go ahead?")):
            print("Nothing changed. Add -y to go ahead without asking.")
            return 1
        r = c.post(path, {"confirm": True})
    elif "error" in r:
        raise Failed(r["error"])
    if args.json:
        out(args, r)
        return 0
    print(r["message"])
    if r.get("noop") or args.no_wait:
        return 0
    return follow(c, r["job"])


def cmd_logs(args, c: Client) -> int:
    r = c.get(f"/api/v1/instances/{args.id}/logs")
    out(args, r, r["logs"])
    return 0


def read_file(name: str) -> dict:
    try:
        text = sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as err:
        raise Failed(f"can't read {name}: {err}") from err
    if not isinstance(data, dict):
        raise Failed(f"{name}: expected the settings of one instance (as `gpu-shepherd show` prints them)")
    return data


def cmd_apply(args, c: Client) -> int:
    data = read_file(args.file)
    iid = args.id or data.get("id")
    if not iid:
        raise Failed("the file has no id: add one, or pass --id")
    out(args, c.call("PUT", f"/api/v1/instances/{iid}", data | {"id": iid}))
    return 0


def cmd_set(args, c: Client) -> int:
    inst = c.get(f"/api/v1/instances/{args.id}")
    inst.pop("state", None)
    out(args, c.call("PUT", f"/api/v1/instances/{args.id}", apply_pairs(inst, pairs(args.changes))))
    return 0


def cmd_delete(args, c: Client) -> int:
    if not (args.yes or ask(f"Delete {args.id}? It is stopped and moved to the trash with its app data folder.")):
        print("Nothing deleted. Add -y to delete without asking.")
        return 1
    out(args, c.call("DELETE", f"/api/v1/instances/{args.id}"))
    return 0


def cmd_apps(args, c: Client) -> int:
    action = args.action or "list"
    if action == "list":
        apps = c.get("/api/v1/apps")
        out(args, apps, lambda apps: table(
            [[a["key"], a["title"], a["group"], ", ".join(a["installed"]) or "-"] for a in apps],
            ["APP", "TITLE", "GROUP", "INSTALLED AS"]))
        return 0
    if not args.key:
        raise Failed(f"which app? e.g. gpu-shepherd apps {action} comfyui")
    if action == "show":
        a = c.get(f"/api/v1/apps/{args.key}")

        def show(a):
            print(f"{a['title']}: {a['blurb']}")
            for p in a["points"]:
                print(f"  - {p}")
            if a["needs"]:
                print(f"Needs: {a['needs']}")
            if a.get("also_installs"):
                print(f"Also installs: {', '.join(a['also_installs'])} (starts and stops with it)")
            if a.get("after_install"):
                print(f"Afterwards: {a['after_install']}")
            print("\nInstall answers (change any with KEY=VALUE):")
            for k, v in a["answers"].items():
                print(f"  {k}={'' if v is None else v}")
            for name, choices in a["choices"].items():
                if choices:
                    print(f"\n{name}:")
                    for k, v in choices.items():
                        print(f"  {k:<14} {v}")
        out(args, a, show)
        return 0
    answers = pairs(args.answers)
    if action == "install":
        out(args, c.post(f"/api/v1/apps/{args.key}/install", answers))
        return 0
    if action == "manual":
        out(args, c.post(f"/api/v1/apps/{args.key}/instance", answers))
        return 0
    raise Failed(f"unknown apps command '{action}'")


def cmd_app_updates(args, c: Client) -> int:
    if args.action in (None, "list"):
        def show(rows):
            for u in rows:
                print(f"{u['instance']} {u['key']}: {u['title']}")
                for line in u["changes"]:
                    print(f"  - {line}")
                for n in u["afterwards"]:
                    print(f"  Afterwards: {n}")
            if not rows:
                print("Every installed app is up to date.")
        out(args, c.get("/api/v1/app-updates"), show)
        return 0
    if args.action == "apply" and args.all:
        done: set[tuple[str, str]] = set()
        while todo := [u for u in c.get("/api/v1/app-updates") if (u["instance"], u["key"]) not in done]:
            for u in todo:               # an embeddings model on the server brings the chat apps' updates next
                done.add((u["instance"], u["key"]))
                out(args, c.post(f"/api/v1/app-updates/{u['instance']}/{u['key']}/apply"))
        return 0
    if not (args.instance and args.key):
        raise Failed(f"which update? e.g. gpu-shepherd app-updates {args.action} webui env-1a2b3c4d")
    out(args, c.post(f"/api/v1/app-updates/{args.instance}/{args.key}/{args.action}"))
    return 0


def model_table(m: dict) -> None:
    loaded = m["loaded"]
    table([[x["id"], x["file"], x["ctx_size"], ", ".join(x["aliases"]) or "-",
            loaded.get(x["id"], "loading" if x["id"] in m["loading"] else "-")] for x in m["models"]],
          ["MODEL", "FILE", "CONTEXT", "ALIASES", "LOADED"])
    for a in m["auto_aliases"]:
        rules = ", ".join(f"{r['min_gpus']}+ GPUs: {r['model']}" for r in a["rules"])
        print(f"Auto name {a['alias']}: {a['default_model']}" + (f"; {rules}" if rules else ""))
    if m["error"]:
        print(m["error"])


def cmd_models(args, c: Client) -> int:
    base = f"/api/v1/instances/{args.server}"
    a = args.action
    if a == "list":
        out(args, c.get(f"{base}/models"), model_table)
    elif a == "files":
        m = c.get(f"{base}/models")
        out(args, m["files"], lambda files: (table([[f["container_path"], f"{f['size_gb']} GB"] for f in files],
                                                   ["FILE", "SIZE"]), [print(n) for n in m["file_notes"]]))
    elif a == "suggested":
        out(args, c.get(f"{base}/suggestions"), lambda rows: table(
            [[x["key"], x["name"], x["fit"], x["status"], "yes" if x["in_catalog"] else "-"] for x in rows],
            ["KEY", "MODEL", "ON THIS MACHINE", "DOWNLOAD", "ADDED"]))
    elif a in ("add-suggested", "download", "stop-download", "discard-download"):
        op = {"add-suggested": "add", "download": "download", "stop-download": "cancel",
              "discard-download": "discard"}[a]
        out(args, c.post(f"{base}/suggestions/{args.key}/{op}"))
    elif a == "show":
        m = next((x for x in c.get(f"{base}/models")["models"] if x["id"] == args.model), None)
        if not m:
            raise Failed(f"no model '{args.model}'")
        out(args, m)
    elif a == "set":
        current = c.get(f"{base}/models")["models"]
        m = next((x for x in current if x["id"] == args.model), None) or NEW_MODEL | {"id": args.model}
        changes = pairs(args.changes)
        offload = {r["min_gpus"]: r for r in m["offload"]}
        for key in [k for k in changes if k.startswith("offload.")]:
            _, n, *field = key.split(".")
            if not n.isdigit() or field not in ([], ["gpu_layers"], ["cpu_moe"]):
                raise Failed(f"'{key}': use offload.N.gpu_layers, offload.N.cpu_moe, or offload.N= to remove it")
            value = changes.pop(key)
            if not field:
                if value.strip():
                    raise Failed(f"'{key}': only offload.{n}= (empty) can be set without a field")
                offload.pop(int(n), None)
                continue
            rule = offload.setdefault(int(n), {"min_gpus": int(n), "gpu_layers": None, "cpu_moe": 0})
            rule[field[0]] = (int(value) if value.strip() else (None if field[0] == "gpu_layers" else 0))
        m = apply_pairs(m, changes) | {"offload": sorted(offload.values(), key=lambda r: r["min_gpus"])}
        out(args, c.call("PUT", f"{base}/models/{args.model}", m))
    elif a == "delete":
        out(args, c.call("DELETE", f"{base}/models/{args.model}"))
    elif a == "load":
        out(args, c.post(f"{base}/models/{args.model}/load"))
    elif a == "unload":
        out(args, c.post(f"{base}/models/{args.model or '_all'}/unload"))
    elif a == "settings":
        changes = pairs(args.changes)
        if not changes and not args.new_api_key:
            s = c.get(f"{base}/models")["settings"]
            out(args, s)
            return 0
        body = {k: (int(v) if k in ("threads", "health_timeout") and v.strip().isdigit() else v)
                for k, v in changes.items()} | ({"new_api_key": True} if args.new_api_key else {})
        out(args, c.call("PATCH", f"{base}/server-settings", body))
    elif a in ("auto-set", "auto-delete"):
        autos = c.get(f"{base}/models")["auto_aliases"]
        autos = [x for x in autos if x["alias"] != args.alias]
        if a == "auto-set":
            if not args.default:
                raise Failed("give the model it uses by default, e.g. auto-set llm coder small-model 2=big-model")
            rules = []
            for rule in args.rules:
                n, sep, model = rule.partition("=")
                if not sep or not n.rstrip("+").isdigit():
                    raise Failed(f"rule '{rule}': write it as 2=model-id (2 or more GPUs use model-id)")
                rules.append({"min_gpus": int(n.rstrip("+")), "model": model})
            autos.append({"alias": args.alias, "default_model": args.default, "rules": rules})
        out(args, c.call("PUT", f"{base}/auto-aliases", autos))
    return 0


def cmd_jobs(args, c: Client) -> int:
    a = args.action or "list"
    if a == "list":
        jobs = c.get("/api/v1/jobs")
        out(args, jobs, lambda jobs: table(
            [[j["id"], j["status"], f"{j['seconds']}s", j["title"], j["error"]] for j in jobs],
            ["JOB", "STATUS", "TIME", "WHAT", "ERROR"]))
        return 0
    if not args.job:
        raise Failed(f"which job? e.g. gpu-shepherd jobs {a} 3")
    if a == "show":
        if args.json:
            out(args, c.get(f"/api/v1/jobs/{args.job}"))
            return 0
        return follow(c, {"id": args.job})
    out(args, c.post(f"/api/v1/jobs/{args.job}/{a}"))
    return 0


def cmd_settings(args, c: Client) -> int:
    if args.action in (None, "show"):
        out(args, c.get("/api/v1/settings"))
    else:
        out(args, c.call("PATCH", "/api/v1/settings", pairs(args.changes)))
    return 0


def cmd_storage(args, c: Client) -> int:
    a = args.action or "show"
    if a == "show":
        def text(s):
            table([[d["path"], gb(d["free"]), gb(d["total"]), "LOW" if d["low"] else ""] for d in s["disks"]],
                  ["DISK", "FREE", "SIZE", ""])
            print()
            table([[f["label"], f["path"], "not visible" if f["size"] is None else gb(f["size"])] for f in s["folders"]],
                  ["FOLDER", "PATH", "SIZE"])
            rows = [[i["image"], gb(i["size"]), ", ".join(i["used_by"]) or "unused"] for i in s["images"]]
            if rows:
                print()
                table(rows, ["IMAGE", "SIZE", "USED BY"])
            rows = [[f["why"], gb(f["size"]), f["path"]] for f in s["files"]]
            rows += [["leftover", gb(f["size"]), f["path"]] for f in s["leftover_folders"]]
            rows += [["container", "-", n] for n in s["orphans"]]
            if rows:
                print("\nWorth a look (unused = in no model list, partial = unfinished download):")
                table(rows, ["WHAT", "SIZE", "PATH"])
        out(args, c.get("/api/v1/storage"), text)
    elif a in ("remove-images", "remove-orphans"):
        out(args, c.post(f"/api/v1/storage/{a.removeprefix('remove-')}"))
    elif a == "trash-files":
        if not args.items:
            raise Failed("which files? e.g. gpu-shepherd storage trash-files /srv/models/old.gguf")
        out(args, c.call("POST", "/api/v1/storage/files", {"paths": args.items}))
    elif a == "trash-folder":
        if len(args.items) != 1:
            raise Failed("which app data folder? e.g. gpu-shepherd storage trash-folder old-app")
        out(args, c.call("POST", "/api/v1/storage/leftovers", {"name": args.items[0]}))
    return 0


def cmd_trash(args, c: Client) -> int:
    a = args.action or "list"
    if a == "list":
        out(args, c.get("/api/v1/trash"), lambda rows: table(
            [[e["id"], e["title"], gb(e["size"]), f"{e['age_days']}d"] for e in rows], ["ID", "WHAT", "SIZE", "AGE"]))
        return 0
    if a == "empty":
        if not (args.yes or ask("Delete everything in the trash for good?")):
            print("Nothing deleted. Add -y to empty it without asking.")
            return 1
        out(args, c.call("DELETE", "/api/v1/trash"))
        return 0
    if not args.id:
        raise Failed(f"which one? e.g. gpu-shepherd trash {a} ID (see gpu-shepherd trash)")
    if a == "restore":
        out(args, c.post(f"/api/v1/trash/{args.id}/restore"))
    else:
        out(args, c.call("DELETE", f"/api/v1/trash/{args.id}"))
    return 0


def cmd_password(args, c: Client) -> int:
    current = getpass.getpass("Current password: ")
    new = getpass.getpass("New password (10+ characters): ")
    if getpass.getpass("New password again: ") != new:
        raise Failed("The new passwords do not match.")
    out(args, c.post("/api/v1/settings/password", {"current": current, "new": new}))
    return 0


def cmd_update(args, c: Client) -> int:
    a = args.action or "status"
    if a == "status":
        u = c.get("/api/v1/updates")

        def show(u):
            print(f"Running {u['running_version']}" + (f" ({u['commit'][:7]})" if u["commit"] else ""))
            if not u["checked_at"]:
                print("Not checked yet." + ("" if u["dashboard_updates"] else f" Run: {u['command']} --check"))
            elif u["available"]:
                print(f"Update available: {u['new_version'] or str(u['behind']) + ' new commit(s)'}")
                for x in u["commits"]:
                    print(f"  - {x.get('subject', '')}")
            else:
                print("Up to date.")
            if u["busy"]:
                print(f"Now: {u['state'] or u['pending']} {u['step']}".rstrip())
            if u["message"]:
                print(u["message"])
            if not u["dashboard_updates"]:
                print(f"Updating from the dashboard or this command is off; on the host run: {u['command']}")
        out(args, u, show)
    elif a in ("check", "install"):
        if a == "install" and not (args.yes or ask("Update GPU Shepherd now? It restarts; instances keep running.")):
            print("Nothing changed. Add -y to update without asking.")
            return 1
        out(args, c.post(f"/api/v1/updates/{'update' if a == 'install' else a}"))
    elif a == "log":
        r = c.get("/api/v1/updates/log")
        out(args, r, r["log"] or "Nothing yet: the updater writes here once it starts.")
    elif a == "set-token":
        if not args.user:
            raise Failed("give your GitHub user name: gpu-shepherd update set-token <user>")
        token = sys.stdin.readline().strip() if not sys.stdin.isatty() else getpass.getpass("GitHub access token: ")
        out(args, c.post("/api/v1/updates/credentials", {"user": args.user, "token": token}))
    elif a == "clear-token":
        out(args, c.post("/api/v1/updates/credentials", {"clear": True}))
    return 0


def cmd_changelog(args, c: Client) -> int:
    releases = c.get("/api/v1/changelog")

    def show(releases):
        for r in releases[:args.limit]:
            mark = " (running)" if r["running"] else " (new)" if r["new"] else ""
            print(f"{r['version']}{mark}  {r['date']}")
            for item in r["items"]:
                print(f"  - {item}")
    out(args, releases, show)
    return 0


def cmd_token(args, c: Client | None) -> int:
    from . import auth
    from .config import Store
    store = Store(args.data_dir)
    token = auth.new_api_token(store) if args.new else auth.api_token(store)
    print(token)
    if args.new:
        print("The old token no longer works.", file=sys.stderr)
    return 0


COMMANDS = {"status": cmd_status, "list": cmd_list, "show": cmd_show, "start": cmd_change, "stop": cmd_change,
            "restart": cmd_change, "logs": cmd_logs, "apply": cmd_apply, "set": cmd_set, "delete": cmd_delete,
            "apps": cmd_apps, "app-updates": cmd_app_updates, "models": cmd_models, "jobs": cmd_jobs,
            "settings": cmd_settings, "storage": cmd_storage, "trash": cmd_trash,
            "password": cmd_password, "update": cmd_update, "changelog": cmd_changelog, "token": cmd_token}


def add_commands(sub) -> None:
    """The management commands, as argparse subcommands of `gpu-shepherd`."""
    sub.add_parser("status", help="GPUs, host, instances and the change that is running")
    sub.add_parser("list", help="instances with their state, GPUs and ports")
    p = sub.add_parser("show", help="an instance's settings as YAML (edit them and `apply`)")
    p.add_argument("id")
    for name, what in (("start", "start an instance"), ("stop", "stop an instance (waits for running work)"),
                       ("restart", "restart an instance (applies saved changes)")):
        p = sub.add_parser(name, help=what)
        p.add_argument("id")
        p.add_argument("-y", "--yes", action="store_true", help="go ahead when it also stops or moves other instances")
        p.add_argument("--no-wait", action="store_true", help="don't follow the change's log")
    p = sub.add_parser("logs", help="an instance's container log")
    p.add_argument("id")
    p = sub.add_parser("apply", help="create or replace an instance from a YAML or JSON file (- = stdin)")
    p.add_argument("file")
    p.add_argument("--id", help="the instance ID, if the file has none")
    p = sub.add_parser("set", help="change an instance's settings: KEY=VALUE, e.g. gpu.mode=elastic autostart=true")
    p.add_argument("id")
    p.add_argument("changes", nargs="+", metavar="KEY=VALUE")
    p = sub.add_parser("delete", help="delete an instance (it goes to the trash with its app data folder)")
    p.add_argument("id")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask")

    p = sub.add_parser("apps", help="the App Store: list, show KEY, install KEY [KEY=VALUE...], manual KEY")
    p.add_argument("action", nargs="?", choices=("list", "show", "install", "manual"))
    p.add_argument("key", nargs="?")
    p.add_argument("answers", nargs="*", metavar="KEY=VALUE")

    p = sub.add_parser("app-updates", help="settings added to apps since they were installed: list, "
                                           "apply INSTANCE KEY (or --all), skip INSTANCE KEY")
    p.add_argument("action", nargs="?", choices=("list", "apply", "skip"))
    p.add_argument("instance", nargs="?")
    p.add_argument("key", nargs="?")
    p.add_argument("--all", action="store_true", help="apply every update")

    p = sub.add_parser("models", help="a model server's models, auto names and settings")
    msub = p.add_subparsers(dest="action", required=True, metavar="ACTION")
    for name, what in (("list", "models, aliases, auto names and what is loaded"),
                       ("files", "the .gguf files the server can see")):
        msub.add_parser(name, help=what).add_argument("server")
    for name, what in (("show", "one model's settings"), ("delete", "remove a model"),
                       ("load", "load a model now")):
        q = msub.add_parser(name, help=what)
        q.add_argument("server")
        q.add_argument("model")
    msub.add_parser("suggested", help="suggested models: how each fits this machine, downloaded or not") \
        .add_argument("server")
    for name, what in (("add-suggested", "add a suggested model with its settings, downloading it if needed"),
                       ("download", "download (or resume) a suggested model's files"),
                       ("stop-download", "stop a download; starting it again resumes"),
                       ("discard-download", "stop a download and delete its partial files")):
        q = msub.add_parser(name, help=what)
        q.add_argument("server")
        q.add_argument("key", help="its key, from `models suggested`")
    q = msub.add_parser("unload", help="unload a model, or every model")
    q.add_argument("server")
    q.add_argument("model", nargs="?")
    q = msub.add_parser("set", help="add or change a model: KEY=VALUE, e.g. file=/models/x.gguf ctx_size=32768")
    q.add_argument("server")
    q.add_argument("model")
    q.add_argument("changes", nargs="*", metavar="KEY=VALUE")
    q = msub.add_parser("settings", help="show or change threads, llama_server, health_timeout")
    q.add_argument("server")
    q.add_argument("changes", nargs="*", metavar="KEY=VALUE")
    q.add_argument("--new-api-key", action="store_true", help="give the server a new API key")
    q = msub.add_parser("auto-set", help="add or replace an auto name: ALIAS DEFAULT [N=MODEL...]")
    q.add_argument("server")
    q.add_argument("alias")
    q.add_argument("default", nargs="?")
    q.add_argument("rules", nargs="*", metavar="N=MODEL")
    q = msub.add_parser("auto-delete", help="remove an auto name")
    q.add_argument("server")
    q.add_argument("alias")

    p = sub.add_parser("jobs", help="recent changes: list, show N, skip N (stop waiting), cancel N")
    p.add_argument("action", nargs="?", choices=("list", "show", "skip", "cancel"))
    p.add_argument("job", nargs="?", type=int)
    p = sub.add_parser("settings", help="show, or set bind_address=... network=... trash_days=...")
    p.add_argument("action", nargs="?", choices=("show", "set"))
    p.add_argument("changes", nargs="*", metavar="KEY=VALUE")
    p = sub.add_parser("storage", help="disk space: show, remove-images, remove-orphans, trash-files PATH..., "
                                       "trash-folder NAME")
    p.add_argument("action", nargs="?", choices=("show", "remove-images", "remove-orphans", "trash-files",
                                                 "trash-folder"))
    p.add_argument("items", nargs="*")
    p = sub.add_parser("trash", help="deleted instances and files: list, restore ID, delete ID, empty")
    p.add_argument("action", nargs="?", choices=("list", "restore", "delete", "empty"))
    p.add_argument("id", nargs="?")
    p.add_argument("-y", "--yes", action="store_true", help="empty without asking")
    sub.add_parser("password", help="change the admin password (asks for it)")
    p = sub.add_parser("update", help="status, check, install, log, set-token USER, clear-token")
    p.add_argument("action", nargs="?", choices=("status", "check", "install", "log", "set-token", "clear-token"))
    p.add_argument("user", nargs="?")
    p.add_argument("-y", "--yes", action="store_true", help="install without asking")
    p = sub.add_parser("changelog", help="what changed in each version")
    p.add_argument("--limit", type=int, default=5, help="how many versions (default 5)")
    p = sub.add_parser("token", help="print the API token (on the server; reads the data folder)")
    p.add_argument("--new", action="store_true", help="make a new one; the old one stops working")


def run(args) -> int:
    try:
        if args.command == "settings" and args.action == "set" and not args.changes:
            raise Failed("what to change? e.g. gpu-shepherd settings set bind_address=127.0.0.1")
        return COMMANDS[args.command](args, None if args.command == "token" else connect(args))
    except Failed as err:
        print(f"gpu-shepherd: {err}", file=sys.stderr)
        return 1
