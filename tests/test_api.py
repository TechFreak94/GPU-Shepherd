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
    assert client.delete("/api/v1/instances/whisper").json()["message"] == "Deleted Whisper."
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
