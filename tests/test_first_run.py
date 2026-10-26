"""The published image's first start: creating a preset and downloading its models (first_run.py)."""
import time


from gpushepherd import first_run, model_library
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store
from gpushepherd.downloads import Downloads, script


def test_creates_the_starter_preset_once(tmp_path):
    store = Store(tmp_path)
    made = first_run.create_preset(store, "starter", str(tmp_path / "models"))
    assert made and made.name == "Starter"
    instances, problems = store.instances()
    assert not problems and set(instances) == {"llm", "comfyui", "webui"}
    assert not any(i.autostart for i in instances.values())
    text = "".join(f.read_text() for f in (tmp_path / "instances").glob("*.yaml"))
    assert "/srv/models" not in text and "REPLACE_WITH_LLM_API_KEY" not in text
    assert f"{tmp_path}/models/comfyui" in text and (tmp_path / "appdata" / "open-webui").is_dir()
    key = store.catalog("llm").api_key
    assert key.startswith("sk-") and "sk-demo" not in key and key in text
    assert (tmp_path / "PRESET_NOTES.txt").is_file()
    # never again, even after every instance is deleted
    for f in (tmp_path / "instances").glob("*.yaml"):
        f.unlink()
    assert first_run.create_preset(store, "starter") is None
    assert not any((tmp_path / "instances").glob("*.yaml"))


def test_skips_none_unknown_and_existing_setups(tmp_path):
    assert first_run.create_preset(Store(tmp_path / "a"), "") is None
    assert not (tmp_path / "a" / first_run.MARKER).exists()          # unset: install.sh's job
    assert first_run.create_preset(Store(tmp_path / "b"), "none") is None
    assert first_run.create_preset(Store(tmp_path / "c"), "no-such-preset") is None
    store = Store(tmp_path / "d")
    (tmp_path / "d" / "instances" / "mine.yaml").write_text("name: Mine\nimage: x\n")
    assert first_run.create_preset(store, "starter") is None
    assert [f.name for f in (tmp_path / "d" / "instances").glob("*.yaml")] == ["mine.yaml"]


def test_downloads_use_suggested_models_where_they_can(tmp_path):
    items = first_run.downloads(first_run.PRESETS / "starter")
    keys = [x.key for x in items]
    assert "qwen3.5-9b" in keys and "qwen3-embedding-0.6b" in keys
    bundle = next(x for x in items if isinstance(x, first_run.Bundle))
    assert [bundle.path(f) for f in bundle.files] == ["comfyui/checkpoints/sd_xl_base_1.0.safetensors"]
    assert "mkdir -p /models/comfyui/checkpoints" in script(bundle)


def test_download_starts_helpers(tmp_path):
    store = Store(tmp_path)
    made = first_run.create_preset(store, "starter")
    dl = Downloads(FakeBackend())
    assert first_run.download(made, dl, want=False) == []
    started = first_run.download(made, dl, want=True)
    assert len(started) == 3
    s = next(s for s in model_library.SUGGESTIONS if s.key == "qwen3.5-9b")
    deadline = time.time() + 5
    while dl.status(s, made.models_dir).state == "starting" and time.time() < deadline:
        time.sleep(0.05)
    assert dl.status(s, made.models_dir).state in ("downloading", "ready")


def test_app_creates_the_preset_on_first_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_PRESET", "starter")
    monkeypatch.setenv("SHEPHERD_DOWNLOAD_MODELS", "0")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    create_app(data_dir=str(tmp_path), backend=FakeBackend(), autostart=False)
    assert (tmp_path / "instances" / "llm.yaml").is_file() and (tmp_path / "models").is_dir()
