"""End-to-end UI flows against the in-memory backend and fake GPUs."""
import re
import shutil
import time
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml
from fastapi.testclient import TestClient

import gpushepherd
import gpushepherd.gpu as gpu_mod
from gpushepherd import instance_templates as tpl
from gpushepherd.app import DemoLlamaClient, create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

EXAMPLE = Path(__file__).parent / "fixtures" / "stack"   # ComfyUI on GPU 0, elastic LLM, Hermes, Open WebUI
HX = {"HX-Request": "true"}


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    backend = FakeBackend()
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=False)
    client = TestClient(app)
    DemoLlamaClient.loaded.clear()
    return app, client, backend, tmp_path


def login(client):
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303


def wait(app):
    for _ in range(500):
        if not app.state.jobs.busy:
            return
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def gpus_of(backend, iid):
    return backend.containers[f"shepherd-{iid}"].gpus


def act(client, app, iid, action, confirm=False):
    r = client.post(f"/ui/instances/{iid}/{action}", headers=HX, data={"confirm": "1"} if confirm else {})
    wait(app)
    return r


def test_login_required(stack):
    _, client, _, _ = stack
    assert client.get("/", follow_redirects=False).status_code == 303
    r = client.get("/ui/dashboard", headers=HX)
    assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    assert client.post("/login", data={"user": "admin", "password": "nope"}).status_code == 401
    assert client.get("/healthz").json()["ok"]


def test_state_changes_need_htmx_header(stack):
    _, client, _, _ = stack
    login(client)
    assert client.post("/ui/instances/llm/start").status_code == 403


def test_gpu_handoff_flow(stack):
    app, client, backend, data = stack
    login(client)
    page = client.get("/")
    assert page.status_code == 200 and "ComfyUI" in page.text and "LLM server" in page.text

    # LLM alone: both GPUs, coder -> Qwen3 Coder Next
    act(client, app, "llm", "start")
    assert gpus_of(backend, "llm") == [0, 1]
    cfg = yaml.safe_load((data / "generated" / "llm" / "llama-swap.yaml").read_text())
    assert cfg["models"]["qwen3.6-35b-a3b"]["aliases"] == ["qwen", "chat"]
    assert cfg["models"]["qwen3-coder-next"]["aliases"] == ["qwen-coder", "coder"]
    assert "--n-cpu-moe 38" in cfg["models"]["qwen3-coder-next"]["cmd"]
    spec = backend.containers["shepherd-llm"]
    assert spec.env["LLAMA_SWAP_API_KEY"].startswith("sk-")
    assert (str(data / "generated" / "llm" / "llama-swap.yaml"), "/app/config.yaml", True) in spec.volumes

    # starting ComfyUI moves the LLM: needs confirmation first
    r = act(client, app, "comfyui", "start")
    assert r.headers.get("HX-Retarget") == "#modal-body"
    assert "Restart LLM server to move it from GPUs 0,1 to GPU 1" in r.text
    assert "shepherd-comfyui" not in backend.containers
    act(client, app, "comfyui", "start", confirm=True)
    assert gpus_of(backend, "comfyui") == [0] and gpus_of(backend, "llm") == [1]
    cfg = yaml.safe_load((data / "generated" / "llm" / "llama-swap.yaml").read_text())
    assert cfg["models"]["qwen3.6-35b-a3b"]["aliases"] == ["qwen", "chat", "coder"]
    assert "--n-cpu-moe 30" in cfg["models"]["qwen3.6-35b-a3b"]["cmd"]      # one GPU: more of it in RAM

    # CPU-only Hermes starts without bothering anyone
    r = act(client, app, "hermes", "start")
    assert "HX-Retarget" not in r.headers and gpus_of(backend, "hermes") == []

    # stopping ComfyUI gives the LLM both GPUs back
    act(client, app, "comfyui", "stop", confirm=True)
    assert backend.running["shepherd-comfyui"] is False and gpus_of(backend, "llm") == [0, 1]

    state = client.get("/api/state").json()
    assert state["instances"]["llm"]["gpus"] == [0, 1] and state["instances"]["comfyui"]["running"] is False
    dash = client.get("/ui/dashboard", headers=HX).text
    assert dash.count('<a href="#inst-llm" title="Go to LLM server">LLM server</a><span class="on">assigned</span>') == 2  # both GPU cards


def test_create_edit_and_validation(stack):
    app, client, backend, data = stack
    login(client)
    form = {"editing": "0", "kind": "container", "id": "whisper", "name": "Whisper", "image": "ghcr.io/x/whisper:1",
            "ports": "9000:9000", "volumes": "/srv/whisper:/data", "env": "MODEL=large", "command": "serve --fast",
            "gpu_mode": "fixed", "gpu_list": "1", "priority": "60", "min_gpus": "1", "max_gpus": "0", "open_path": "/"}
    r = client.post("/instances/save", headers=HX, data=form)
    assert r.status_code == 204 and unquote(r.headers["HX-Redirect"]).startswith("/?msg=Saved Whisper")
    saved = yaml.safe_load((data / "instances" / "whisper.yaml").read_text())
    assert saved["gpu"] == {"mode": "fixed", "gpus": [1], "min_gpus": 1, "max_gpus": 0, "priority": 60}
    assert saved["command"] == ["serve", "--fast"]

    bad = dict(form, id="other", ports="3000:8080")         # webui already uses 3000
    r = client.post("/instances/save", headers=HX, data=bad)
    assert r.status_code == 200 and "already used by" in r.text and r.headers["HX-Retarget"] == "#editor"
    bad = dict(form, id="other2", ports="9001:1", volumes="relative:/x")
    assert "absolute path" in client.post("/instances/save", headers=HX, data=bad).text

    # editing a running instance flags "restart to apply"
    act(client, app, "hermes", "start")
    form_h = {"editing": "1", "kind": "container", "id": "hermes", "name": "Hermes", "image": "nousresearch/hermes-agent:v2",
              "gpu_mode": "none", "priority": "50", "min_gpus": "1", "max_gpus": "0", "command": "gateway run"}
    assert client.post("/instances/save", headers=HX, data=form_h).status_code == 204
    assert "Saved changes not applied" in client.get("/ui/dashboard", headers=HX).text
    act(client, app, "hermes", "restart")
    assert "Saved changes not applied" not in client.get("/ui/dashboard", headers=HX).text

    r = client.post("/instances/whisper/delete", headers=HX)
    assert r.status_code == 204 and not (data / "instances" / "whisper.yaml").exists()


def test_model_offload_settings(stack):
    app, client, backend, data = stack
    login(client)
    form = client.get("/instances/llm/models/edit?model=qwen3.6-35b-a3b").text
    assert form.count('name="offload_gpus"') == 2                     # one row per GPU on this machine
    assert 'name="offload_cpu_moe" min="0" max="999" value="30"' in form
    base = {"original_id": "qwen3.6-35b-a3b", "id": "qwen3.6-35b-a3b", "file": "/models/q.gguf",
            "ctx_size": "65536", "parallel": "1", "kv_cache": "f16"}
    r = client.post("/instances/llm/models/save", headers=HX, data={
        **base, "offload_gpus": ["1", "2"], "offload_gpu_layers": ["", "99"], "offload_cpu_moe": ["24", ""]})
    assert r.status_code == 204
    model = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())["models"][0]
    assert model["offload"] == [{"min_gpus": 1, "gpu_layers": None, "cpu_moe": 24},
                                {"min_gpus": 2, "gpu_layers": 99, "cpu_moe": 0}]
    assert "--n-gpu-layers 99" in client.get("/instances/llm/models").text   # config preview, 2 GPUs

    bad = client.post("/instances/llm/models/save", headers=HX, data={
        **base, "offload_gpus": ["1", "2"], "offload_gpu_layers": ["", ""], "offload_cpu_moe": ["lots", ""]})
    assert "error-box" in bad.text and "use whole numbers" in bad.text
    assert 'value="lots"' in bad.text                                  # the form comes back as typed


def test_model_catalogue_editing(stack):
    app, client, backend, data = stack
    login(client)
    assert "Qwen3.6 35B-A3B" in client.get("/instances/llm/models").text
    r = client.post("/instances/llm/models/save", headers=HX, data={
        "original_id": "", "id": "gemma4-26b", "name": "Gemma 4 26B", "file": "/models/gemma.gguf",
        "ctx_size": "65536", "parallel": "1", "kv_cache": "q8_0", "sampling": "--temp 1.0", "extra_args": "",
        "mmproj": "", "aliases_text": "gemma", "description": ""})
    assert r.status_code == 204
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert [m["id"] for m in cat["models"]] == ["qwen3.6-35b-a3b", "qwen3-coder-next", "gemma4-26b"]

    bad = client.post("/instances/llm/models/save", headers=HX, data={"original_id": "", "id": "x",
                                                                     "file": "gemma.gguf", "ctx_size": "100"})
    assert "error-box" in bad.text

    # auto names: "auto" -> gemma with 3+ GPUs
    r = client.post("/instances/llm/catalog/auto", headers=HX, data={
        "alias": "auto", "default_model": "qwen3.6-35b-a3b", "rules": "2=qwen3-coder-next, 3=gemma4-26b"})
    assert r.status_code == 204
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert cat["auto_aliases"][0]["rules"][1] == {"min_gpus": 3, "model": "gemma4-26b"}
    r = client.post("/instances/llm/catalog/auto", headers=HX, data={"alias": "auto", "default_model": "qwen3.6-35b-a3b",
                                                                    "rules": "two gpus please"})
    assert "rule 'two gpus please'" in r.headers["HX-Trigger"]

    # a model used by an auto name cannot be removed
    r = client.post("/instances/llm/models/gemma4-26b/delete", headers=HX)
    assert "Cannot delete" in r.headers["HX-Trigger"]

    # renaming follows into auto names
    r = client.post("/instances/llm/models/save", headers=HX, data={
        "original_id": "gemma4-26b", "id": "gemma4", "name": "Gemma 4 26B", "file": "/models/gemma.gguf",
        "ctx_size": "65536", "parallel": "1", "kv_cache": "q8_0"})
    assert r.status_code == 204
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert cat["auto_aliases"][0]["rules"][1]["model"] == "gemma4"

    # load / unload while running
    act(client, app, "llm", "start")
    client.post("/instances/llm/models/qwen3-coder-next/load", headers=HX)
    for _ in range(100):
        if "qwen3-coder-next" in DemoLlamaClient.loaded:
            break
        time.sleep(0.01)
    assert "<strong>qwen3-coder-next</strong> loaded" in client.get("/ui/dashboard", headers=HX).text
    client.post("/instances/llm/models/qwen3-coder-next/unload", headers=HX)
    assert not DemoLlamaClient.loaded


def test_dashboard_details_and_machine_panels(stack):
    app, client, backend, _ = stack
    login(client)
    act(client, app, "hermes", "start")
    dash = client.get("/ui/dashboard", headers=HX).text
    assert "<strong>CPU</strong>" in dash and "<strong>Memory (RAM)</strong>" in dash
    assert '<ul class="usage" aria-label="Memory use by instance">' in dash and 'href="#inst-hermes"' in dash
    card = dash[dash.index('id="inst-hermes"'):]
    card = card[:card.index("</article>")]
    assert "nousresearch/hermes-agent:stable" in card and "shepherd-hermes" in card
    assert 'href="http://testserver:9119/"' in card and '>testserver:9119</a>' in card and "limit 4g" in card and "Up " in card
    assert "HERMES_UID" in card and "change-me-please" not in card     # setting names, never their values
    assert "share of the whole machine" in card and "Advanced details" in card
    assert all(f'id="tip-hermes-{k}"' in card for k in ("gpu", "use", "addr", "state", "image", "folders", "env"))
    # the long explanations live in tooltips that hover, focus and tap can all reach
    assert 'aria-describedby="tip-hermes-about"' in card and 'role="tooltip" id="tip-hermes-about"' in card
    assert card.index("Home lab") > card.index('id="tip-hermes-about"')     # the description is in the tooltip
    stopped = dash[dash.index('id="inst-webui"'):]
    assert "Never started" in stopped[:stopped.index("</article>")]
    state = client.get("/api/state").json()
    assert state["host"]["mem_total"] > 0


def test_password_change(stack):
    app, client, _, data = stack
    login(client)
    r = client.post("/settings/password", headers=HX, data={"current": "wrong", "new": "x" * 12, "again": "x" * 12})
    assert "wrong" in r.headers["HX-Trigger"]
    r = client.post("/settings/password", headers=HX, data={"current": "correct-horse", "new": "brand-new-pass",
                                                             "again": "brand-new-pass"})
    assert unquote(r.headers["HX-Redirect"]).startswith("/settings")
    client.post("/logout", headers=HX)
    r = client.post("/login", data={"user": "admin", "password": "brand-new-pass"}, follow_redirects=False)
    assert r.status_code == 303


def test_autostart_brings_up_marked_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    backend = FakeBackend()
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=True)
    wait(app)
    assert gpus_of(backend, "comfyui") == [0] and gpus_of(backend, "llm") == [1]
    assert "shepherd-hermes" in backend.containers and "shepherd-webui" not in backend.containers


class FlakyBackend(FakeBackend):
    """Fails to start one container, records the order of operations."""
    def __init__(self, fail_name):
        super().__init__()
        self.fail_name = fail_name

    def create_and_start(self, spec):
        if spec.name == self.fail_name:
            self.calls.append(f"fail {spec.name}")
            raise RuntimeError("image is broken")
        super().create_and_start(spec)


def test_images_pulled_first_and_failures_do_not_strand_others(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    backend = FlakyBackend("shepherd-comfyui")
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=False)
    client = TestClient(app)
    login(client)
    act(client, app, "llm", "start")
    backend.calls.clear()
    act(client, app, "comfyui", "start", confirm=True)
    calls = backend.calls
    # every image is checked before the LLM is stopped
    first_stop = next(n for n, c in enumerate(calls) if c.startswith("stop"))
    assert all(n < first_stop for n, c in enumerate(calls) if c.startswith("image"))
    # ComfyUI failed, but the LLM was still restarted (on GPU 1) instead of being left stopped
    assert "fail shepherd-comfyui" in calls and gpus_of(backend, "llm") == [1]
    assert backend.running["shepherd-llm"] is True
    job = app.state.jobs.history[0]
    assert job.status == "failed" and "ComfyUI: image is broken" in job.error


def test_generated_password_logs_in(tmp_path, monkeypatch):
    """No SHEPHERD_ADMIN_PASSWORD: the password in INITIAL_PASSWORD.txt is the one that works."""
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.delenv("SHEPHERD_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    client = TestClient(create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False))
    text = (tmp_path / "INITIAL_PASSWORD.txt").read_text()
    password = text.split("password: ", 1)[1].splitlines()[0]
    assert not set(password) & set("0O1lI-_")
    r = client.post("/login", data={"user": "admin", "password": f" {password}\n"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/", follow_redirects=False).status_code == 200


def test_reset_password(stack, monkeypatch):
    app, client, _, data = stack
    from gpushepherd import auth
    user, password = auth.reset_admin(app.state.store)
    assert password in (data / "INITIAL_PASSWORD.txt").read_text()
    assert client.post("/login", data={"user": "admin", "password": "correct-horse"}).status_code == 401
    r = client.post("/login", data={"user": user, "password": password}, follow_redirects=False)
    assert r.status_code == 303


def test_login_without_cookie_explains(stack):
    app, client, _, data = stack
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.headers["location"] == "/?login=1"
    client.cookies.clear()                       # a browser that drops the session cookie
    r = client.get("/?login=1", follow_redirects=False)
    assert r.headers["location"] == "/login?nocookie=1"
    assert "did not keep the login cookie" in client.get("/login?nocookie=1").text


def test_guided_setup(stack, tmp_path_factory):
    app, client, backend, data = stack
    login(client)
    page = client.get("/instances/new").text            # the old address opens the App Store
    for key in ("llm-server", "comfyui", "open-webui", "hermes"):
        assert f'href="/apps/{key}"' in page and f'href="/apps/{key}?mode=manual"' in page
    assert 'href="/instances/new/custom"' in page and "llm-coder" not in page and "llm-chat" not in page
    # old "Add model server" and template links still work; the chat and coding servers are one app now
    assert 'name="kind" value="llama-swap"' in client.get("/instances/new?kind=llama-swap").text
    assert "Install LLM server" in client.get("/instances/new/llm-coder").text
    r = client.get("/apps/llm-chat?mode=manual", follow_redirects=False)
    assert r.headers["location"] == "/apps/llm-server?mode=manual"

    models = tmp_path_factory.mktemp("models")
    with open(models / "Big-MoE-Q4.gguf", "wb") as fh:
        fh.truncate(10 * 1_073_741_824)                   # sparse: 10 GB on two 8 GB GPUs
    (models / "Qwen3-8B-Q4.gguf").write_bytes(b"gguf")
    form = client.get("/apps/llm-server").text
    assert 'value="llm-2"' in form and 'value="8081"' in form and 'value="/srv/models"' in form
    assert 'name="chat_file"' in form and 'name="coder_file"' in form and 'name="coder_ctx" min="512" value="65536"' in form

    answers = {"template": "llm-server", "name": "Coder", "id": "coder", "port": "8081", "gpu": "all",
               "models_dir": str(models), "coder_file": "/models/Big-MoE-Q4.gguf", "coder_ctx": "65536",
               "chat_file": "/models/Qwen3-8B-Q4.gguf", "chat_ctx": "32768", "offload": "auto", "action": "create"}
    r = client.post("/instances/guided", headers=HX, data=answers)
    assert r.status_code == 204 and unquote(r.headers["HX-Redirect"]).startswith("/instances/coder/models")
    assert "with qwen3-8b-q4, big-moe-q4" in unquote(r.headers["HX-Redirect"])
    inst = yaml.safe_load((data / "instances" / "coder.yaml").read_text())
    assert inst["kind"] == "llama-swap" and inst["gpu"]["mode"] == "elastic"
    assert inst["volumes"] == [{"host": str(models), "container": "/models", "read_only": True}]
    cat = yaml.safe_load((data / "catalogs" / "coder.yaml").read_text())
    assert [(x["id"], x["aliases"]) for x in cat["models"]] == [("qwen3-8b-q4", ["chat"]), ("big-moe-q4", ["coder"])]
    assert [(o["min_gpus"], o["cpu_moe"]) for o in cat["models"][1]["offload"]] == [(1, 999), (2, 0)]

    # the same ID again is refused, and the form keeps the answers
    r = client.post("/instances/guided", headers=HX, data=answers)
    assert "already exists" in r.text and r.headers["HX-Retarget"] == "#editor"

    # Open WebUI wires itself to a model server, with its API key
    r = client.post("/instances/guided", headers=HX, data={
        "template": "open-webui", "name": "Chat 2", "id": "chat2", "port": "3001", "gpu": "none",
        "appdata_dir": str(data / "appdata" / "chat2"), "server": "coder", "comfyui": "comfyui", "action": "create"})
    assert r.status_code == 204
    env = yaml.safe_load((data / "instances" / "chat2.yaml").read_text())["env"]
    assert env["OPENAI_API_BASE_URLS"] == "http://coder:8080/v1" and env["OPENAI_API_KEYS"] == cat["api_key"]
    assert (data / "appdata" / "chat2" / "data").is_dir()

    # "Customize every field" opens the full form filled in, without saving
    r = client.post("/instances/guided", headers=HX, data={
        "template": "comfyui", "name": "ComfyUI 2", "id": "comfyui-2", "port": "8189", "gpu": "pin:1",
        "models_dir": "/srv/models", "appdata_dir": "/opt/gpu-shepherd/appdata/comfyui-2", "action": "advanced"})
    assert r.status_code == 200 and 'action="/instances/save"' not in r.text
    assert 'hx-post="/instances/save"' in r.text and "8189:8188" in r.text and "yanwk/comfyui-boot" in r.text
    assert not (data / "instances" / "comfyui-2.yaml").exists()


def test_app_store(stack):
    app, client, backend, data = stack
    login(client)
    page = client.get("/apps").text
    assert "App Store" in page and 'data-filter="chat-and-agents"' in page
    # the example stack already has Open WebUI, ComfyUI, Hermes and a chat model server
    for key in ("open-webui", "comfyui", "hermes", "llm-server"):
        card = page.split(f'id="app-{key}"')[1].split("</article>")[0]
        assert "1 installed" in card and "Install another" in card, key

    # automatic and manual are two ways into the same app
    auto = client.get("/apps/hermes").text
    assert 'href="/apps/hermes?mode=manual"' in auto and 'aria-current="page"' in auto
    assert 'name="password"' in auto and 'name="hass_token"' in auto
    manual = client.get("/apps/hermes?mode=manual").text
    assert 'hx-post="/instances/save"' in manual and "Install Hermes Agent" in manual
    assert "nousresearch/hermes-agent:latest" in manual and "hermes-2" in manual and "gateway run" in manual
    assert "OPENAI_BASE_URL=http://llm:8080/v1" in manual and 'name="app" value="hermes"' in manual
    assert client.get("/apps/nope", follow_redirects=False).status_code == 303

    # installing by hand keeps which app it is
    form = {"editing": "0", "kind": "container", "app": "comfyui", "id": "comfy-b", "name": "ComfyUI B",
            "image": "yanwk/comfyui-boot:cu130-slim-v2", "ports": "8190:8188", "gpu_mode": "none",
            "volumes": f"{data}/appdata/comfy-b/output:/root/ComfyUI/output"}
    assert client.post("/instances/save", headers=HX, data=form).status_code == 204
    assert yaml.safe_load((data / "instances" / "comfy-b.yaml").read_text())["app"] == "comfyui"
    assert (data / "appdata" / "comfy-b" / "output").is_dir()
    assert "2 installed" in client.get("/apps").text.split('id="app-comfyui"')[1].split("</article>")[0]


def test_app_needs_an_llm_server_first(stack, tmp_path_factory):
    """No LLM server yet: apps that need one say so and offer to install it first, then come back."""
    app, client, backend, data = stack
    login(client)
    assert "Add model server" not in client.get("/").text
    assert "needs an LLM server" not in client.get("/apps/open-webui").text       # the example has one
    for f in (data / "instances").glob("*.yaml"):
        f.unlink()
    assert "Add model server" not in client.get("/").text
    store_page = client.get("/apps").text
    for key in ("open-webui", "hermes"):
        assert "none is installed yet" in store_page.split(f'id="app-{key}"')[1].split("</article>")[0]
    assert "none is installed yet" not in store_page.split('id="app-comfyui"')[1].split("</article>")[0]
    for url in ("/apps/hermes", "/apps/hermes?mode=manual"):
        page = client.get(url).text
        assert "Hermes Agent needs an LLM server" in page and 'href="/apps/llm-server?then=hermes"' in page
    assert "needs an LLM server" not in client.get("/apps/comfyui").text

    # the LLM server installed "first" goes on to Hermes, which then picks it
    page = client.get("/apps/llm-server?then=hermes").text
    assert 'name="then" value="hermes"' in page and "Next: Hermes Agent" in page and "and continue" in page
    assert 'name="then"' not in client.get("/apps/llm-server?then=comfyui").text   # only apps that need a server
    models = tmp_path_factory.mktemp("models")
    r = client.post("/instances/guided", headers=HX, data={
        "template": "llm-server", "then": "hermes", "name": "LLM server", "id": "llm", "port": "8080", "gpu": "all",
        "models_dir": str(models), "offload": "auto", "action": "create"})
    assert r.status_code == 204 and unquote(r.headers["HX-Redirect"]).startswith("/apps/hermes?msg=Installed LLM server")
    page = client.get(r.headers["HX-Redirect"]).text
    assert "needs an LLM server" not in page and '<option value="llm" selected>' in page
    assert "Now set up Hermes Agent" in page


def test_app_store_installs_a_model_server_too(stack, tmp_path_factory):
    """Hermes on a machine without a model server: the server comes with it, and Hermes is wired to it."""
    app, client, backend, data = stack
    login(client)
    for f in (data / "instances").glob("*.yaml"):
        f.unlink()
    models = tmp_path_factory.mktemp("models")
    (models / "Qwen3-8B-Q4.gguf").write_bytes(b"gguf")
    page = client.get("/apps/hermes").text
    assert 'value="new" selected' in page and 'data-server="new"' in page

    answers = {"template": "hermes", "name": "Hermes", "id": "hermes", "port": "9119", "gpu": "none",
               "appdata_dir": str(data / "appdata" / "hermes"), "server": "new", "models_dir": str(models),
               "chat_file": "/models/Qwen3-8B-Q4.gguf", "login_user": "admin", "password": "short",
               "hass_url": "http://ha.lan:8123", "hass_token": "tok", "action": "create"}
    r = client.post("/instances/guided", headers=HX, data=answers)
    assert "at least 8 characters" in r.text and not (data / "instances" / "llm.yaml").exists()
    assert "A new LLM server" in r.text           # no half-made server in the list

    r = client.post("/instances/guided", headers=HX, data=answers | {"password": "long-enough-pw"})
    assert r.status_code == 204, re.findall(r"error-box.*?</div>", r.text)
    assert "Installed Hermes and LLM server" in unquote(r.headers["HX-Redirect"])
    srv = yaml.safe_load((data / "instances" / "llm.yaml").read_text())
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert srv["kind"] == "llama-swap" and srv["app"] == "llm-server" and srv["ports"][0]["host"] == 8080
    assert cat["models"][0]["file"] == "/models/Qwen3-8B-Q4.gguf" and cat["models"][0]["aliases"] == ["chat"]
    hermes = yaml.safe_load((data / "instances" / "hermes.yaml").read_text())
    assert hermes["command"] == ["gateway", "run"] and hermes["app"] == "hermes"
    env = hermes["env"]
    assert env["OPENAI_BASE_URL"] == "http://llm:8080/v1" and env["OPENAI_API_KEY"] == cat["api_key"]
    assert env["HERMES_MODEL"] == "chat" and env["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"] == "long-enough-pw"
    assert env["HASS_URL"] == "http://ha.lan:8123" and env["HASS_TOKEN"] == "tok"
    config = (data / "appdata" / "hermes" / "data" / "config.yaml")
    assert yaml.safe_load(config.read_text())["model"] == {
        "provider": "custom", "base_url": "http://llm:8080/v1", "api_key": cat["api_key"], "default": "chat"}
    assert config.stat().st_mode & 0o777 == 0o600


def test_swap_waits_for_running_work(stack):
    """Starting ComfyUI moves the LLM; while the LLM is answering, nothing stops until it is done."""
    from gpushepherd import drain
    app, client, backend, _ = stack
    login(client)
    act(client, app, "llm", "start")
    engine = app.state.engine
    engine.drain_interval = 0.02
    busy = {"llm": 1}
    engine.activity = lambda inst: drain.Activity(busy.get(inst.id, 0), f"{busy.get(inst.id, 0)} request (coder)")

    # Cancel change: everything stays as it was
    client.post("/ui/instances/comfyui/start", headers=HX, data={"confirm": "1"})
    job = app.state.jobs.current
    for _ in range(200):
        if job.waiting:
            break
        time.sleep(0.01)
    dash = client.get("/ui/dashboard", headers=HX).text
    assert "waiting for running work to finish" in dash and "1 request (coder)" in dash
    assert "Finishing: 1 request (coder)" in dash and f"/ui/jobs/{job.id}/skip" in dash
    assert client.get("/api/state").json()["waiting_for"]["llm"]["detail"] == "1 request (coder)"
    assert client.post(f"/ui/jobs/{job.id}/cancel", headers=HX).status_code == 204
    wait(app)
    assert job.status == "cancelled" and gpus_of(backend, "llm") == [0, 1]
    assert "shepherd-comfyui" not in backend.containers
    assert "was called off" in client.get("/ui/dashboard", headers=HX).text

    # the work finishes by itself: then the swap goes ahead
    client.post("/ui/instances/comfyui/start", headers=HX, data={"confirm": "1"})
    time.sleep(0.1)
    assert backend.running["shepherd-llm"] is True and app.state.jobs.busy
    busy["llm"] = 0
    wait(app)
    assert gpus_of(backend, "comfyui") == [0] and gpus_of(backend, "llm") == [1]
    assert any("LLM server finished its work" in ln for ln in app.state.jobs.history[0].lines)

    # Stop now: stops without waiting any longer
    busy["llm"] = 2
    client.post("/ui/instances/llm/stop", headers=HX)
    job = app.state.jobs.current
    time.sleep(0.1)
    client.post(f"/ui/jobs/{job.id}/skip", headers=HX)
    wait(app)
    assert job.status == "done" and backend.running["shepherd-llm"] is False


def test_unload_refuses_while_answering(stack):
    from gpushepherd import drain
    app, client, _, _ = stack
    login(client)
    act(client, app, "llm", "start")
    app.state.engine.activity = lambda inst: drain.Activity(1, "1 request (coder)")
    r = client.post("/instances/llm/models/_all/unload", headers=HX)
    assert "Still answering 1 request (coder)" in r.headers["HX-Trigger"]
    app.state.engine.activity = lambda inst: drain.Activity(0)
    assert "Model unloaded." in client.post("/instances/llm/models/_all/unload", headers=HX).headers["HX-Trigger"]


def test_drain_settings_in_the_editor(stack):
    app, client, _, data = stack
    login(client)
    form = client.get("/instances/comfyui/edit").text
    assert 'name="drain_check"' in form and 'value="600"' in form
    inst = app.state.store.instance("comfyui")
    fields = {"editing": "1", "id": "comfyui", "name": inst.name, "kind": "container", "image": inst.image,
              "ports": "8188:8188", "gpu_mode": "fixed", "gpu_list": "0", "priority": "80",
              "drain_check": "http", "drain_timeout": "1800", "drain_on_timeout": "cancel",
              "drain_path": "/queue", "drain_key": "queue_running"}
    r = client.post("/instances/save", headers=HX, data=fields)
    assert r.status_code == 204, r.text
    saved = yaml.safe_load((data / "instances" / "comfyui.yaml").read_text())
    assert saved["drain"] == {"check": "http", "port": None, "path": "/queue", "key": "queue_running",
                              "timeout": 1800, "on_timeout": "cancel"}
    r = client.post("/instances/save", headers=HX, data={**fields, "drain_path": ""})
    assert "needs a path" in r.text


def test_theme_cookie(stack):
    _, client, _, _ = stack
    page = client.get("/login")
    assert "data-theme=" not in page.text.split(">", 2)[1] and "data-theme-toggle" in page.text
    login(client)
    client.cookies.set("theme", "dark")
    page = client.get("/settings")
    assert '<html lang="en" data-theme="dark">' in page.text
    assert 'data-theme-set="dark" aria-pressed="true"' in page.text
    client.cookies.set("theme", "neon\" onload=\"x")
    assert '<html lang="en">' in client.get("/").text


def test_dashboard_kind_tags(stack):
    """Every card says what kind of instance it is: the model server and each UI."""
    app, client, backend, data = stack
    login(client)
    dash = client.get("/ui/dashboard", headers=HX).text
    card = lambda iid: dash.split(f'id="inst-{iid}"')[1].split("</h2>")[0]
    assert '<span class="tag kind server">Model Server</span>' in card("llm")
    assert '<span class="tag kind ui">Chat UI</span>' in card("webui")
    assert '<span class="tag kind server">Image Server</span>' in card("comfyui")
    assert '<span class="tag kind ui">Agent UI</span>' in card("hermes")
    assert "model server</span>" not in dash


def test_older_chat_ui_name_shows_as_open_webui(stack, monkeypatch):
    """Open WebUI used to be installed as "Chat UI", the same as its tag: its card says Open WebUI,
    and the better name alone does not ask for a restart."""
    app, client, backend, data = stack
    login(client)
    assert "name: Chat UI" in (data / "instances" / "webui.yaml").read_text()
    with monkeypatch.context() as old_version:         # started before names were fixed up
        old_version.setattr(tpl, "app_name", lambda inst: inst.name)
        act(client, app, "webui", "start")
    dash = client.get("/ui/dashboard", headers=HX).text
    card = dash.split('id="inst-webui"')[1].split("</article>")[0]
    assert '<h2>Open WebUI<span class="tag kind ui">Chat UI</span>' in card
    assert "not applied" not in card
    assert client.get("/instances/webui/edit").text.count('value="Open WebUI"') == 1


def test_librechat_installs_its_database_and_starts_with_it(stack):
    app, client, backend, data = stack
    login(client)
    page = client.get("/apps").text
    assert 'id="app-librechat"' in page and 'id="app-anythingllm"' in page
    assert "librechat-mongodb" not in page                     # companions are not on the store
    assert page.count(">Experimental</span>") == 1 and "Experimental." in client.get("/apps/langflow").text
    assert "License: MIT" in client.get("/apps/langflow").text and "License: Apache-2.0" in page
    assert ">Credits</button>" in client.get("/ui/update").text
    cr = client.get("/ui/credits").text
    assert "htmx" in cr and "Langflow" in cr and "SSPL" in cr and "unsloth/Qwen3.5-9B-GGUF" in cr
    form = client.get("/apps/librechat").text
    assert "LibreChat database" in form and "mongo:8.0" in form and "the first account becomes the admin" in form
    r = client.post("/instances/guided", headers=HX, data={
        "template": "librechat", "name": "LibreChat", "id": "librechat", "port": "3080", "gpu": "none",
        "appdata_dir": str(data / "appdata" / "librechat"), "server": "llm", "action": "create"})
    assert r.status_code == 204 and "Installed LibreChat and LibreChat database" in unquote(r.headers["HX-Redirect"])
    inst = yaml.safe_load((data / "instances" / "librechat.yaml").read_text())
    db = yaml.safe_load((data / "instances" / "librechat-mongodb.yaml").read_text())
    assert inst["requires"] == ["librechat-mongodb"] and db["app"] == "librechat-mongodb" and db["ports"] == []
    yml = (data / "appdata" / "librechat" / "librechat.yaml")
    assert yml.is_file() and "${LLM_API_KEY}" in yml.read_text() and (data / "appdata" / "librechat" / "uploads").is_dir()
    dash = client.get("/").text
    assert ">Database</span>" in dash.split('id="inst-librechat-mongodb"')[1]

    act(client, app, "librechat", "start")
    assert {"shepherd-librechat", "shepherd-librechat-mongodb"} <= set(backend.containers)
    act(client, app, "librechat", "stop")
    assert not backend.running["shepherd-librechat"] and not backend.running["shepherd-librechat-mongodb"]
    assert backend.calls.index("stop shepherd-librechat") < backend.calls.index("stop shepherd-librechat-mongodb")

    # by hand: the full form keeps "Starts with", and saving makes the missing database too
    manual = client.get("/apps/librechat?mode=manual").text
    assert 'name="requires" value="librechat-2-mongodb"' in manual and "/app/librechat.yaml" in manual
    vols = f"{data}/appdata/lc3/librechat.yaml:/app/librechat.yaml\n{data}/appdata/lc3/uploads:/app/uploads"
    r = client.post("/instances/save", headers=HX, data={
        "editing": "0", "kind": "container", "app": "librechat", "id": "lc3", "name": "LC3",
        "image": "librechat/librechat:latest", "ports": "3090:3080", "gpu_mode": "none", "volumes": vols,
        "requires": "lc3-mongodb"})
    assert r.status_code == 204
    assert yaml.safe_load((data / "instances" / "lc3-mongodb.yaml").read_text())["volumes"][0]["host"] == \
        f"{data}/appdata/lc3/mongodb"
    assert (data / "appdata" / "lc3" / "librechat.yaml").is_file()      # a file, not a folder Docker would make


def test_anythingllm_settings_file(stack):
    app, client, backend, data = stack
    login(client)
    r = client.post("/instances/guided", headers=HX, data={
        "template": "anythingllm", "name": "AnythingLLM", "id": "anythingllm", "port": "3001", "gpu": "none",
        "appdata_dir": str(data / "appdata" / "anythingllm"), "server": "llm", "action": "create"})
    assert r.status_code == 204 and "make a workspace" in unquote(r.headers["HX-Redirect"])
    env = (data / "appdata" / "anythingllm" / "server.env").read_text()
    assert "GENERIC_OPEN_AI_BASE_PATH='http://llm:8080/v1'" in env and "GENERIC_OPEN_AI_MODEL_PREF='chat'" in env
    assert (data / "appdata" / "anythingllm" / "storage").is_dir()
    act(client, app, "anythingllm", "start")
    assert "shepherd-anythingllm" in backend.containers


def test_suggested_models_page(stack, tmp_path_factory):
    app, client, backend, data = stack
    login(client)
    models = tmp_path_factory.mktemp("models")
    inst = yaml.safe_load((data / "instances" / "llm.yaml").read_text())
    inst["volumes"][0]["host"] = str(models)
    (data / "instances" / "llm.yaml").write_text(yaml.safe_dump(inst))
    page = client.get("/instances/llm/models").text
    assert "Suggested models" in page and "Qwen3.5 9B" in page and "Fits on one GPU" in page
    assert f"Downloads go to <span class=\"mono\">{models}</span>" in page
    assert 'hx-post="/instances/llm/suggestions/qwen3.5-9b/add"' in page and "and downloads it (6.9 GB)" in page

    backend.helper_seconds = 60                   # a download that keeps running
    r = client.post("/instances/llm/suggestions/qwen3.5-9b/add", headers=HX)
    assert r.status_code == 204 and "Added qwen3.5-9b; downloading it (6.9 GB)" in unquote(r.headers["HX-Redirect"])
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    added = next(m for m in cat["models"] if m["id"] == "qwen3.5-9b")
    assert added["file"] == "/models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf" and added["thinking"] == "off"
    for _ in range(100):
        if "shepherd-download-qwen3.5-9b" in backend.helper_runs:
            break
        time.sleep(0.01)
    part = client.get("/instances/llm/suggestions", headers=HX).text
    assert 'hx-trigger="every 3s"' in part and "Downloading" in part and "/suggestions/qwen3.5-9b/cancel" in part
    assert "already on LLM server" in client.post("/instances/llm/suggestions/qwen3.5-9b/add", headers=HX).headers["HX-Trigger"]
    r = client.post("/instances/llm/suggestions/qwen3.5-9b/cancel", headers=HX)
    assert "Stopped downloading" in r.headers["HX-Trigger"] and "/suggestions/qwen3.5-9b/download" in r.text
    assert "no suggested model" in client.post("/instances/llm/suggestions/nope/download", headers=HX).headers["HX-Trigger"]

    # an embeddings model gets the "embed" name and stays loaded beside the others
    client.post("/instances/llm/suggestions/qwen3-embedding-0.6b/add", headers=HX)
    preview = client.get("/instances/llm/models").text
    assert "kept-loaded" in preview and "--embeddings" in preview
    cat = yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())
    assert next(m for m in cat["models"] if m["role"] == "embed")["aliases"] == ["embed"]


def test_model_form_new_settings(stack):
    app, client, backend, data = stack
    login(client)
    form = client.get("/instances/llm/models/edit").text
    for field in ('name="thinking"', 'name="mtp"', 'name="mtp_draft"', 'name="role"', 'name="split_mode"',
                  'name="tensor_split"', 'name="ttl"', 'name="keep_loaded"', 'name="preload"', 'value="bf16"'):
        assert field in form
    r = client.post("/instances/llm/models/save", headers=HX, data={
        "original_id": "", "id": "fast", "file": "/models/fast.gguf", "ctx_size": "32768", "parallel": "2",
        "kv_cache": "q5_1", "thinking": "off", "mtp": "2", "mtp_draft": "/models/mtp-fast.gguf", "role": "chat",
        "split_mode": "row", "tensor_split": "8,12", "ttl": "900", "preload": "on"})
    assert r.status_code == 204
    m = next(x for x in yaml.safe_load((data / "catalogs" / "llm.yaml").read_text())["models"] if x["id"] == "fast")
    assert (m["kv_cache"], m["thinking"], m["mtp"], m["split_mode"], m["tensor_split"], m["ttl"], m["preload"],
            m["keep_loaded"]) == ("q5_1", "off", 2, "row", "8,12", 900, True, False)
    edit = client.get("/instances/llm/models/edit?model=fast").text
    assert 'name="mtp" min="0" max="6" value="2"' in edit and 'value="8,12"' in edit
    bad = client.post("/instances/llm/models/save", headers=HX, data={
        "original_id": "", "id": "x", "file": "/models/x.gguf", "ctx_size": "32768", "tensor_split": "half"})
    assert "error-box" in bad.text and "share per GPU" in bad.text


def test_install_form_offers_suggested_models(stack):
    app, client, backend, data = stack
    login(client)
    form = client.get("/apps/llm-server").text
    assert 'name="embed_file"' in form and "Embeddings model" in form
    assert 'data-pick-file="/models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf" data-pick-target="chat_file"' in form
    assert 'data-pick-file="/models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf" data-pick-target="embed_file"' in form
    webui = client.get("/apps/open-webui").text            # its bundled server: chat models only
    assert "/models/qwen3.5-9b/" in webui and "qwen3-embedding" not in webui and "Qwen3 Coder Next" not in webui


def test_app_updates_notice_and_dialog(stack):
    _, client, _, data = stack
    login(client)
    assert "1 app update:" in client.get("/ui/dashboard").text       # the LLM server's embeddings model
    dlg = client.get("/ui/app-updates").text
    assert "Add an embeddings model" in dlg and "/ui/app-updates/llm/embed-model/apply" in dlg
    r = client.post("/ui/app-updates/llm/embed-model/apply", headers=HX)
    assert "Added qwen3-embedding-0.6b" in r.headers["HX-Trigger"]
    assert "documents" in r.text and "webui" in r.text               # now Open WebUI can use it
    key = r.text.split("/ui/app-updates/webui/")[1].split("/")[0]
    r = client.post(f"/ui/app-updates/webui/{key}/apply", headers=HX)
    assert "Admin Panel" in r.headers["HX-Trigger"] and "up to date" in r.text
    env = yaml.safe_load((data / "instances" / "webui.yaml").read_text())["env"]
    assert env["RAG_EMBEDDING_MODEL"] == "embed" and env["DEFAULT_MODELS"] == "chat"
    assert "app update" not in client.get("/ui/dashboard").text


def test_unreadable_settings_still_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    store = Store(tmp_path)
    create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    store.save_settings(store.settings())                     # settings.yaml.prev now has the login
    store.settings_path.write_text("admin_user: [oops\n")
    client = TestClient(create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False))
    login(client)
    assert "settings.yaml.broken" in client.get("/").text


def test_a_bad_model_list_does_not_break_the_dashboard(stack):
    app, client, backend, data = stack
    login(client)
    (data / "catalogs" / "llm.yaml").write_text("models: [oops\n")
    r = client.get("/")
    assert r.status_code == 200 and "catalogs/llm.yaml: not valid YAML" in r.text


def test_docker_going_away_shows_a_notice(stack):
    app, client, backend, data = stack
    login(client)

    def gone():
        raise RuntimeError("connection refused")
    backend.states = gone
    r = client.get("/")
    assert r.status_code == 200 and "Docker is not reachable" in r.text
    assert client.get("/healthz?full=1").status_code == 200


def test_full_health_check_fails_when_the_dashboard_does(stack, monkeypatch):
    app, client, backend, data = stack
    assert client.get("/healthz?full=1").json()["ok"]

    def broken():
        raise RuntimeError("bug")
    monkeypatch.setattr("gpushepherd.app.host_mod.query", broken)
    assert client.get("/healthz?full=1").status_code == 500
    assert client.get("/healthz").status_code == 200


def test_storage_page_and_trash(stack):
    app, client, _, data = stack
    login(client)
    r = client.get("/storage")
    assert r.status_code == 200 and "Storage" in r.text and "The trash is empty" in r.text
    assert client.post("/instances/hermes/delete", headers=HX).status_code == 204
    wait(app)
    r = client.get("/storage")
    assert r.status_code == 200 and "Hermes" in r.text and "Restore" in r.text and "Empty trash" in r.text
    assert 'href="/storage"' in client.get("/settings").text


def test_storage_cleanup_routes(stack):
    _, client, _, _ = stack
    login(client)
    r = client.post("/storage/images/remove-unused", headers=HX)
    assert r.status_code == 204 and unquote(r.headers["HX-Redirect"]).startswith("/storage?msg=")
    r = client.post("/storage/trash/days", headers=HX, data={"days": "7"})
    assert unquote(r.headers["HX-Redirect"]).startswith("/storage?msg=")
    assert 'value="7"' in client.get("/storage").text
    r = client.post("/storage/trash/days", headers=HX, data={"days": "x"})
    assert r.status_code == 204 and "toast" in r.headers["HX-Trigger"]


def test_confirm_modal_shows_memory_warnings(stack, monkeypatch):
    app, client, _, _ = stack
    login(client)
    monkeypatch.setattr(app.state.svc, "memory_warnings", lambda *a, **k: ["test warning"])
    r = client.post("/ui/instances/llm/start", headers=HX)
    wait(app)
    assert "test warning" in r.text and "Continue" in r.text


def test_logins_dialog(stack):
    app, client, backend, tmp = stack
    login(client)
    hermes = tmp / "instances" / "hermes.yaml"
    hermes.write_text(hermes.read_text() + "app: hermes\n")
    dash = client.get("/", headers=HX).text
    assert "/ui/instances/hermes/logins" in dash and "/ui/instances/llm/logins" in dash
    llm = client.get("/ui/instances/llm/logins", headers=HX).text
    assert "(any user name)" in llm and "sk-" in llm and "Set new password" not in llm and "New API key" in llm
    dlg = client.get("/ui/instances/hermes/logins", headers=HX).text
    assert "Logins: Hermes" in dlg and "<strong>admin</strong>" in dlg
    r = client.post("/ui/instances/hermes/logins", data={"user": "admin", "password": "brand-new-pass-1"}, headers=HX)
    assert "Saved" in r.text and "brand-new-pass-1" in r.text


def test_restart_from_the_models_page_goes_back_there(stack):
    app, client, _, _ = stack
    login(client)
    act(client, app, "llm", "start", confirm=True)
    r = client.post("/ui/instances/llm/restart", headers=HX,
                    data={"confirm": "1", "back": "/instances/llm/models"})
    wait(app)
    # the message has "…" in it, which a header can only carry encoded
    assert r.status_code == 204 and unquote(r.headers["HX-Redirect"]).startswith(
        "/instances/llm/models?msg=Restarting LLM server")
