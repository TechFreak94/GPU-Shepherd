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

PRESETS = Path(gpushepherd.__file__).parent / "presets"
EXAMPLE = PRESETS / "zimage-2x8gb"
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
    assert dash.count('<span class="small on">LLM server</span>') == 2     # both GPU cards show the LLM


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
    assert [m["id"] for m in cat["models"]] == ["qwen3.6-35b-a3b", "qwen3-coder-next", "muse-glimmer", "gemma4-26b"]

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
    assert "Loaded: <strong>qwen3-coder-next</strong>" in client.get("/ui/dashboard", headers=HX).text
    client.post("/instances/llm/models/qwen3-coder-next/unload", headers=HX)
    assert not DemoLlamaClient.loaded


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
