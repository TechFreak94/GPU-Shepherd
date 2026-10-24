"""End-to-end UI flows against the in-memory backend and fake GPUs."""
import shutil
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import gpushepherd
import gpushepherd.gpu as gpu_mod
from gpushepherd.app import DemoLlamaClient, create_app
from gpushepherd.backend import FakeBackend

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
    assert r.status_code == 204 and r.headers["HX-Redirect"].startswith("/?msg=Saved Whisper")
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
    assert 'href="http://testserver:9119/"' in card and '>testserver:9119</a>' in card and "· limit 4g" in card and "Up " in card
    assert "HERMES_UID" in card and "change-me-please" not in card     # setting names, never their values
    assert "% of the machine" in card
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
    assert r.headers["HX-Redirect"].startswith("/settings")
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
    page = client.get("/instances/new").text
    for key in ("llm-chat", "llm-coder", "comfyui", "open-webui", "custom"):
        assert f'href="/instances/new/{key}' in page
    # old "Add model server" links still reach the full form
    assert 'name="kind" value="llama-swap"' in client.get("/instances/new?kind=llama-swap").text

    models = tmp_path_factory.mktemp("models")
    with open(models / "Big-MoE-Q4.gguf", "wb") as fh:
        fh.truncate(10 * 1_073_741_824)                   # sparse: 10 GB on two 8 GB GPUs
    form = client.get("/instances/new/llm-coder").text
    assert 'value="coder"' in form and 'value="8081"' in form and 'value="/srv/models"' in form

    answers = {"template": "llm-coder", "name": "Coder", "id": "coder", "port": "8081", "gpu": "all",
               "models_dir": str(models), "file": "/models/Big-MoE-Q4.gguf", "ctx_size": "65536",
               "offload": "auto", "action": "create"}
    r = client.post("/instances/guided", headers=HX, data=answers)
    assert r.status_code == 204 and r.headers["HX-Redirect"].startswith("/instances/coder/models")
    inst = yaml.safe_load((data / "instances" / "coder.yaml").read_text())
    assert inst["kind"] == "llama-swap" and inst["gpu"]["mode"] == "elastic"
    assert inst["volumes"] == [{"host": str(models), "container": "/models", "read_only": True}]
    cat = yaml.safe_load((data / "catalogs" / "coder.yaml").read_text())
    assert cat["models"][0]["id"] == "big-moe-q4" and cat["models"][0]["aliases"] == ["coder"]
    assert [(o["min_gpus"], o["cpu_moe"]) for o in cat["models"][0]["offload"]] == [(1, 999), (2, 0)]

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
