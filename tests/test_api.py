"""The JSON API (/api/v1) and the `gpu-shepherd` commands that use it."""
import shutil
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, cli
from gpushepherd.__main__ import main
from gpushepherd.app import DemoLlamaClient, create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store
from gpushepherd.service import Problem

EXAMPLE = Path(__file__).parent / "fixtures" / "stack"
HX = {"HX-Request": "true"}


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    backend = FakeBackend()
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=False)
    DemoLlamaClient.loaded.clear()
    token = auth.api_token(Store(tmp_path))
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})
    return app, client, backend, tmp_path


def wait(app):
    for _ in range(500):
        if not app.state.jobs.busy:
            return
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_token_required(stack):
    app, client, _, data = stack
    anon = TestClient(app)
    assert anon.get("/api/v1/state").status_code == 401
    assert anon.get("/api/v1/state", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/v1/state").json()["gpus"][0]["name"] == "Test GPU"
    # a logged-in browser may read, but not change anything (no cross-site forms)
    anon.post("/login", data={"user": "admin", "password": "correct-horse"})
    assert anon.get("/api/state").status_code == 200
    assert anon.post("/api/v1/instances/llm/start").status_code == 401
    # a new token from Settings replaces the old one at once
    assert anon.post("/settings/api-token", headers=HX).status_code == 204
    assert client.get("/api/v1/state").status_code == 401
    assert (data / "api-token").stat().st_mode & 0o777 == 0o600


def test_start_asks_before_moving_others(stack):
    app, client, backend, _ = stack
    r = client.post("/api/v1/instances/llm/start")
    assert r.status_code == 202 and r.json()["job"]["title"] == "Starting LLM server"
    wait(app)
    assert client.post("/api/v1/instances/llm/start").json()["noop"] is True
    r = client.post("/api/v1/instances/comfyui/start")
    assert r.status_code == 409 and r.json()["needs_confirm"] and any("Restart LLM server" in x for x in r.json()["plan"])
    r = client.post("/api/v1/instances/comfyui/start", json={"confirm": True})
    assert r.status_code == 202
    wait(app)
    job = client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "done" and any("Starting ComfyUI" in x for x in job["log"])
    inst = client.get("/api/v1/instances/comfyui").json()
    assert inst["state"]["running"] and inst["state"]["gpus"] == [0]
    assert client.post("/api/v1/instances/nope/start").status_code == 404
    assert client.post("/api/v1/instances/llm/explode").status_code == 400


def test_instances_put_and_delete(stack):
    app, client, backend, data = stack
    r = client.put("/api/v1/instances/whisper", json={"name": "Whisper", "image": "ghcr.io/x/whisper:1",
                                                       "ports": [{"host": 9000, "container": 9000}]})
    assert r.status_code == 201 and r.json()["instance"]["state"]["status"] == "not created"
    inst = client.get("/api/v1/instances/whisper").json()
    inst["gpu"] = {"mode": "fixed", "gpus": [1]}
    assert client.put("/api/v1/instances/whisper", json=inst).status_code == 200
    assert yaml.safe_load((data / "instances" / "whisper.yaml").read_text())["gpu"]["gpus"] == [1]
    r = client.put("/api/v1/instances/whisper", json=inst | {"ports": [{"host": 8080, "container": 1}]})
    assert r.status_code == 400 and "already used by 'llm'" in r.json()["error"]
    r = client.put("/api/v1/instances/whisper", json=inst | {"gpu": {"mode": "fixed"}})
    assert r.status_code == 400 and "needs at least one GPU" in r.json()["error"]
    assert client.delete("/api/v1/instances/whisper").json()["message"].startswith("Deleted Whisper.")
    assert client.get("/api/v1/instances/whisper").status_code == 404


def test_app_store_install(stack):
    _, client, _, data = stack
    apps = {a["key"]: a for a in client.get("/api/v1/apps").json()}
    assert apps["comfyui"]["installed"] == ["comfyui"] and apps["hermes"]["needs_llm_server"]
    shown = client.get("/api/v1/apps/comfyui").json()
    assert shown["answers"]["id"] == "comfyui-2" and "hass_url" not in shown["answers"]
    assert "pin:1" in shown["choices"]["gpu"]
    manual = client.post("/api/v1/apps/comfyui/instance", json={"gpu": "pin:1"}).json()
    assert manual["gpu"]["mode"] == "fixed" and not (data / "instances" / "comfyui-2.yaml").exists()
    r = client.post("/api/v1/apps/comfyui/install", json={"id": "comfy2", "port": "8199", "gpu": "pin:1"})
    assert r.status_code == 201, r.text
    saved = yaml.safe_load((data / "instances" / "comfy2.yaml").read_text())
    assert saved["gpu"]["gpus"] == [1] and saved["app"] == "comfyui"
    r = client.post("/api/v1/apps/comfyui/install", json={"id": "comfy2"})
    assert r.status_code == 400 and "already exists" in r.json()["error"]
    assert client.post("/api/v1/apps/nope/install", json={}).status_code == 404
    lc = client.get("/api/v1/apps/librechat").json()
    assert lc["also_installs"] == ["LibreChat database"] and "first account" in lc["after_install"]
    assert not lc["experimental"] and client.get("/api/v1/apps/langflow").json()["experimental"]
    assert lc["license"] == "MIT"
    r = client.post("/api/v1/apps/librechat/install", json={"server": "llm"})
    assert r.status_code == 201 and r.json()["companions"] == ["librechat-mongodb"], r.text
    assert "Installed LibreChat and LibreChat database." in r.json()["message"]
    assert "librechat-mongodb" not in {a["key"] for a in client.get("/api/v1/apps").json()}


def test_models_and_server_settings(stack):
    _, client, _, data = stack
    base = "/api/v1/instances/llm"
    before = client.get(f"{base}/models").json()
    first = before["models"][0]["id"]
    r = client.put(f"{base}/models/tiny", json={"file": "/models/tiny.gguf", "ctx_size": 4096})
    assert r.status_code == 201
    assert client.put(f"{base}/auto-aliases", json=before["auto_aliases"] + [
        {"alias": "quick", "default_model": "tiny", "rules": [{"min_gpus": 2, "model": first}]}]).status_code == 200
    # renaming follows into the auto names
    assert client.put(f"{base}/models/tiny", json={"id": "tiny2", "file": "/models/tiny.gguf"}).status_code == 200
    autos = {a["alias"]: a for a in client.get(f"{base}/models").json()["auto_aliases"]}
    assert autos["quick"]["default_model"] == "tiny2"
    r = client.delete(f"{base}/models/tiny2")
    assert r.status_code == 409 and "quick" in r.json()["error"]
    r = client.put(f"{base}/auto-aliases", json=[{"alias": "x", "default_model": "missing"}])
    assert r.status_code == 400 and "unknown model" in r.json()["error"]
    key = before["settings"]["api_key"]
    assert client.patch(f"{base}/server-settings", json={"threads": 6, "new_api_key": True}).status_code == 200
    settings = client.get(f"{base}/models").json()["settings"]
    assert settings["threads"] == 6 and settings["api_key"] != key
    assert client.patch(f"{base}/server-settings", json={"bogus": 1}).status_code == 400
    assert client.get("/api/v1/instances/comfyui/models").status_code == 400   # not a model server


def test_models_load_and_unload(stack):
    app, client, _, _ = stack
    client.post("/api/v1/instances/llm/start")
    wait(app)
    mid = client.get("/api/v1/instances/llm/models").json()["models"][0]["id"]
    assert client.post(f"/api/v1/instances/llm/models/{mid}/load").status_code == 202
    for _ in range(100):
        if client.get("/api/v1/instances/llm/models").json()["loaded"]:
            break
        time.sleep(0.01)
    assert mid in client.get("/api/v1/instances/llm/models").json()["loaded"]
    assert client.post("/api/v1/instances/llm/models/_all/unload").json()["message"] == "Model unloaded."
    assert client.post("/api/v1/instances/llm/models/nope/load").status_code == 404


def test_models_with_missing_files_are_flagged(stack, tmp_path_factory):
    app, client, _, data = stack
    models = tmp_path_factory.mktemp("models")
    path = data / "instances" / "llm.yaml"
    path.write_text(path.read_text().replace("host: /srv/models", f"host: {models}"))
    (models / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    missing = client.get("/api/v1/instances/llm/models").json()["missing"]
    assert missing == {"qwen3-coder-next": ["/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"]}
    client.post("/api/v1/instances/llm/start")
    wait(app)
    r = client.post("/api/v1/instances/llm/models/qwen3-coder-next/load")
    assert r.status_code == 409 and "Qwen3-Coder-Next-UD-Q4_K_XL.gguf" in r.json()["error"]
    browser = TestClient(app)
    browser.post("/login", data={"user": "admin", "password": "correct-horse"})
    assert "file missing" in browser.get("/instances/llm/models").text
    # a folder GPU Shepherd can't see says nothing either way
    path.write_text(path.read_text().replace(f"host: {models}", "host: /nonexistent/models"))
    assert client.get("/api/v1/instances/llm/models").json()["missing"] == {}


def test_settings_password_and_updates(stack):
    _, client, _, data = stack
    assert client.patch("/api/v1/settings", json={"bind_address": "127.0.0.1"}).json()["settings"]["bind_address"] == "127.0.0.1"
    assert client.patch("/api/v1/settings", json={"bind_address": "nope"}).status_code == 400
    assert client.patch("/api/v1/settings", json={"admin_user": "x"}).status_code == 400
    assert client.post("/api/v1/settings/password", json={"current": "bad", "new": "x" * 12}).status_code == 403
    assert client.post("/api/v1/settings/password", json={"current": "correct-horse", "new": "brand-new-pass"}).status_code == 200
    assert auth.verify_password("brand-new-pass", Store(data).settings().password_hash)
    u = client.get("/api/v1/updates").json()
    assert u["running_version"] and u["dashboard_updates"] is False
    r = client.post("/api/v1/updates/check")
    assert r.status_code == 409 and "update.sh" in r.json()["error"]
    assert isinstance(client.get("/api/v1/changelog").json(), list)


# --------------------------------------------------------------------------- the command

@pytest.fixture
def run(stack, monkeypatch, capsys):
    """Run `gpu-shepherd ...` against the test app; returns (exit status, output)."""
    app, client, _, data = stack

    def request(method, url, json=None, timeout=None, headers=None):
        return client.request(method, url.removeprefix("http://127.0.0.1:8090"), json=json)
    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr(cli, "ask", lambda q: False)

    def go(*argv):
        monkeypatch.setattr("sys.argv", ["gpu-shepherd", "--data-dir", str(data), *argv])
        try:
            main()
            code = 0
        except SystemExit as err:
            code = err.code
        wait(app)
        res = capsys.readouterr()
        return code, res.out + res.err
    return go


def test_cli_status_start_and_set(run, stack):
    _, _, backend, data = stack
    code, text = run("status")
    assert code == 0 and "GPU 0  Test GPU" in text and "llm" in text
    assert run("start", "llm")[0] == 0
    code, text = run("start", "comfyui")
    assert code == 1 and "Restart LLM server" in text and "Nothing changed" in text
    code, text = run("start", "comfyui", "-y")
    assert code == 0 and "Done." in text and backend.containers["shepherd-comfyui"].gpus == [0]
    code, text = run("set", "comfyui", "gpu.priority=70", "autostart=false", "env=A=1;B=two", "ports=8188:8188;9999:9")
    assert code == 0 and "Restart it to apply" in text
    saved = yaml.safe_load((data / "instances" / "comfyui.yaml").read_text())
    assert saved["gpu"]["priority"] == 70 and saved["autostart"] is False and saved["env"] == {"A": "1", "B": "two"}
    assert [p["host"] for p in saved["ports"]] == [8188, 9999]
    assert run("set", "comfyui", "gpu.nope=1")[1].strip() == "gpu-shepherd: unknown setting 'gpu.nope'"


def test_cli_show_apply_delete(run, stack, tmp_path):
    _, _, _, data = stack
    code, text = run("show", "webui")
    assert code == 0 and "state" not in yaml.safe_load(text)
    f = tmp_path / "copy.yaml"
    f.write_text(text.replace("id: webui", "id: webui2").replace("host: 3000", "host: 3001"))
    code, text = run("apply", str(f))
    assert code == 0 and (data / "instances" / "webui2.yaml").exists()
    assert run("delete", "webui2")[0] == 1                 # asks first
    assert run("delete", "webui2", "-y")[0] == 0 and not (data / "instances" / "webui2.yaml").exists()


def test_cli_apps_models_settings(run, stack):
    _, _, _, data = stack
    code, text = run("apps", "install", "comfyui", "id=comfy3", "port=8190")
    assert code == 0 and "Installed ComfyUI" in text and (data / "instances" / "comfy3.yaml").exists()
    code, text = run("models", "set", "llm", "tiny", "file=/models/tiny.gguf", "aliases=small,mini",
                     "offload.2.cpu_moe=999")
    assert code == 0, text
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    tiny = next(m for m in cat["models"] if m["id"] == "tiny")
    assert tiny["aliases"] == ["small", "mini"] and tiny["offload"] == [{"min_gpus": 2, "gpu_layers": None, "cpu_moe": 999}]
    assert run("models", "auto-set", "llm", "quick", "tiny", "2=tiny")[0] == 0
    code, text = run("models", "list", "llm")
    assert "Auto name quick: tiny; 2+ GPUs: tiny" in text
    assert run("models", "auto-delete", "llm", "quick")[0] == 0
    assert run("models", "delete", "llm", "tiny")[0] == 0
    assert run("models", "settings", "llm", "threads=4")[0] == 0
    assert yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())["threads"] == 4
    assert run("settings", "set", "network=lab-net")[0] == 0 and Store(data).settings().network == "lab-net"
    code, text = run("--json", "jobs")
    assert code == 0 and text.startswith("[")


def test_cli_token_and_errors(run, stack, monkeypatch):
    _, _, _, data = stack
    code, text = run("token")
    assert code == 0 and text.strip() == (data / "api-token").read_text().strip()
    code, text = run("token", "--new")
    assert (data / "api-token").read_text().strip() in text
    monkeypatch.setenv("SHEPHERD_TOKEN", "wrong")
    code, text = run("status")
    assert code == 1 and "token was refused" in text


def test_suggestions_api_and_cli(run, stack):
    app, client, backend, data = stack
    rows = client.get("/api/v1/instances/llm/suggestions").json()
    assert rows[0]["fits"] and {"key", "fit", "status", "files", "in_catalog"} <= set(rows[0])
    assert all(f["url"].startswith("https://huggingface.co/") for r in rows for f in r["files"])
    r = client.post("/api/v1/instances/llm/suggestions/gemma-4-e4b/add")
    assert r.status_code == 201 and "Added gemma-4-e4b" in r.json()["message"]
    assert client.post("/api/v1/instances/llm/suggestions/gemma-4-e4b/add").status_code == 409
    assert client.post("/api/v1/instances/llm/suggestions/gemma-4-e4b/nope").status_code == 404
    code, text = run("models", "suggested", "llm")
    assert code == 0 and "qwen3.5-9b" in text and "ON THIS MACHINE" in text
    code, text = run("models", "download", "llm", "qwen3-embedding-0.6b")
    assert code == 0 and "Downloading Qwen3 Embedding" in text
    assert run("models", "stop-download", "llm", "qwen3-embedding-0.6b")[0] == 0
    code, text = run("models", "add-suggested", "llm", "qwen3-embedding-0.6b")
    assert code == 0 and "Added qwen3-embedding-0.6b" in text
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert {m["id"] for m in cat["models"]} >= {"gemma-4-e4b", "qwen3-embedding-0.6b"}


def test_installing_with_a_suggested_model_downloads_it(stack, tmp_path_factory):
    app, client, backend, data = stack
    models = tmp_path_factory.mktemp("models")
    r = client.post("/api/v1/apps/llm-server/install", json={
        "id": "llm2", "port": 8091, "models_dir": str(models),
        "chat_file": "/models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf"})
    assert r.status_code == 201 and r.json()["downloads"] == ["Qwen3.5 9B"]
    assert "Downloading Qwen3.5 9B" in r.json()["message"]
    for _ in range(100):
        if "shepherd-download-qwen3.5-9b" in backend.helper_runs:
            break
        time.sleep(0.01)
    assert "shepherd-download-qwen3.5-9b" in backend.helper_runs


def test_delete_moves_an_app_and_its_database_to_the_trash(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    appdata = data / "appdata" / "chat"
    vol = lambda sub, inside: {"host": str(appdata / sub), "container": inside}    # noqa: E731
    assert client.put("/api/v1/instances/chat-db", json={"name": "Chat DB", "image": "mongo:7", "app": "mongodb",
                                                          "volumes": [vol("db", "/data/db")]}).status_code == 201
    assert client.put("/api/v1/instances/chat", json={"name": "Chat", "image": "x/chat:1", "requires": ["chat-db"],
                                                       "volumes": [vol("data", "/data")]}).status_code == 201
    (appdata / "db").mkdir(parents=True, exist_ok=True)
    (appdata / "db" / "store.bin").write_bytes(b"x" * 1000)
    (data / "generated" / "chat").mkdir(parents=True)
    r = client.delete("/api/v1/instances/chat-db")
    assert r.status_code == 409 and "Chat needs Chat DB" in r.json()["error"]
    client.post("/api/v1/instances/chat/start")
    wait(app)
    assert backend.running["shepherd-chat"] and backend.running["shepherd-chat-db"]
    msg = client.delete("/api/v1/instances/chat").json()["message"]
    assert "with Chat DB" in msg and "stops once its running work finishes" in msg
    wait(app)
    assert "shepherd-chat" not in backend.containers and "shepherd-chat-db" not in backend.containers
    assert "stop shepherd-chat" in backend.calls                                     # stopped the normal way first
    assert not (data / "instances" / "chat.yaml").exists() and not (data / "instances" / "chat-db.yaml").exists()
    assert not appdata.exists() and not (data / "generated" / "chat").exists()
    [entry] = svc.trash_entries()
    assert entry.instances == ["chat", "chat-db"] and entry.title == "Chat, Chat DB" and entry.size >= 1000
    # put back, then delete again and empty the trash
    assert "Put Chat, Chat DB back" in svc.restore_trash(entry.id)
    assert (appdata / "db" / "store.bin").exists() and client.get("/api/v1/instances/chat").status_code == 200
    assert svc.trash_entries() == []
    assert client.delete("/api/v1/instances/chat").json()["message"].startswith("Deleted Chat (with Chat DB).")
    entry = svc.trash_entries()[0]
    client.put("/api/v1/instances/chat", json={"name": "New chat", "image": "x/chat:2"})
    with pytest.raises(Problem, match="something else is there now"):
        svc.restore_trash(entry.id)
    assert "freeing" in svc.purge_trash(entry.id) and svc.trash_entries() == []
    with pytest.raises(Problem):
        svc.purge_trash(entry.id)


def test_old_trash_is_emptied_by_itself(stack):
    app, client, _, data = stack
    svc = app.state.svc
    client.put("/api/v1/instances/whisper", json={"name": "Whisper", "image": "x/w:1"})
    client.delete("/api/v1/instances/whisper")
    [entry] = svc.trash_entries()
    manifest = data / "trash" / entry.id / "manifest.yaml"
    raw = yaml.safe_load(manifest.read_text())
    manifest.write_text(yaml.safe_dump(raw | {"deleted": raw["deleted"] - 31 * 86400}))
    assert "keeps things" in svc.set_trash_days(0) and len(svc.trash_entries()) == 1
    svc.set_trash_days(30)
    assert svc.trash_entries() == []
    with pytest.raises(Problem):
        svc.set_trash_days(-1)


def test_app_logins(stack):
    app, client, backend, data = stack
    base = "/api/v1/instances"
    # the LLM server: any user name, its API key as the password (llama-swap's browser login)
    got = client.get(f"{base}/llm/logins").json()
    key = client.get(f"{base}/llm/models").json()["settings"]["api_key"]
    assert got["accounts"][0]["password"] == key and got["accounts"][0]["user"] == "(any user name)" and not got["can_reset"]
    assert "New API key" in client.post(f"{base}/llm/logins", json={"password": "long-enough"}).json()["error"]

    # Hermes Agent: one login in its environment, shown; a new one restarts it
    r = client.post("/api/v1/apps/hermes/install", json={"id": "agent", "server": "llm", "password": "first-pass"})
    assert r.status_code == 201, r.text
    got = client.get(f"{base}/agent/logins").json()
    assert got["accounts"] == [{"user": "admin", "role": "dashboard", "password": "first-pass"}] and got["can_reset"]
    assert client.post(f"{base}/agent/logins", json={"password": "short"}).status_code == 400
    r = client.post(f"{base}/agent/logins", json={"user": "me", "password": "second-pass"}).json()
    assert r["password"] == "second-pass" and "next start" in r["message"]
    env = yaml.safe_load((data / "instances" / "agent.yaml").read_text())["env"]
    assert env["HERMES_DASHBOARD_BASIC_AUTH_USERNAME"] == "me" and env["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"] == "second-pass"
    client.post(f"{base}/agent/start")
    wait(app)
    assert "restarting" in client.post(f"{base}/agent/logins", json={"user": "me"}).json()["message"]
    wait(app)

    # Open WebUI: accounts in its database, read and changed by a script in its running container
    assert client.post("/api/v1/apps/open-webui/install", json={"id": "webui2", "server": "llm"}).status_code == 201
    got = client.get(f"{base}/webui2/logins").json()
    assert "Start" in got["status"] and not got["can_reset"] and not backend.execs
    assert client.post(f"{base}/webui2/logins", json={"user": "a@b.c"}).status_code == 409
    client.post(f"{base}/webui2/start")
    wait(app)
    backend.exec_reply = lambda name, cmd, env: (0, 'note\n[{"user": "a@b.c", "role": "admin"}]')
    assert client.get(f"{base}/webui2/logins").json()["accounts"] == [{"user": "a@b.c", "role": "admin", "password": ""}]
    backend.exec_reply = lambda name, cmd, env: (0, '{"updated": 1}' if env.get("GS_USER") == "a@b.c" else '{"updated": 0}')
    r = client.post(f"{base}/webui2/logins", json={"user": "a@b.c"}).json()
    name, cmd, env, _ = backend.execs[-1]
    assert name.endswith("webui2") and env == {"GS_USER": "a@b.c", "GS_PASSWORD": r["password"]}
    assert r["password"] not in " ".join(cmd) and len(r["password"]) >= 12
    assert client.post(f"{base}/webui2/logins", json={"user": "x@y.z"}).status_code == 404
    backend.exec_reply = lambda name, cmd, env: (1, "Traceback\nsqlite3.OperationalError: no such table: auth")
    assert "no such table" in client.get(f"{base}/webui2/logins").json()["error"]

    # AnythingLLM without multi-user mode: its one password lives in server.env
    assert client.post("/api/v1/apps/anythingllm/install", json={"id": "allm", "server": "llm"}).status_code == 201
    got = client.get(f"{base}/allm/logins").json()
    assert got["accounts"] == [] and "no password" in got["status"] and got["can_reset"]
    r = client.post(f"{base}/allm/logins", json={"password": "doc-pass-123"}).json()
    vols = yaml.safe_load((data / "instances" / "allm.yaml").read_text())["volumes"]
    env_path = Path(next(v["host"] for v in vols if v["container"] == "/app/server/.env"))
    env_file = env_path.read_text()
    assert "AUTH_TOKEN='doc-pass-123'" in env_file and "JWT_SECRET='" in env_file and "LLM_PROVIDER" in env_file
    assert client.get(f"{base}/allm/logins").json()["accounts"][0]["password"] == "doc-pass-123"


def test_cli_logins(run, stack):
    _, client, _, _ = stack
    client.post("/api/v1/apps/hermes/install", json={"id": "agent", "server": "llm", "password": "first-pass"})
    code, text = run("logins", "agent")
    assert code == 0 and "admin (dashboard)  password: first-pass" in text and "environment" in text
    code, text = run("logins", "agent", "--reset")
    assert code == 0 and "Password: " in text and "first-pass" not in text
    code, text = run("logins", "llm")
    assert code == 0 and "(any user name) (API key)  password: sk-" in text
    assert run("logins", "llm", "--reset")[0] == 1
    assert run("logins", "nope")[0] == 1
