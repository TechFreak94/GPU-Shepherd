"""WSL2: GPUs named by UUID, no automatic mounting, copies to Windows drives."""
import yaml

from gpushepherd import gpu as gpu_mod, host, moves
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Catalog, CatalogModel, GpuPolicy, Instance, Store
from gpushepherd.connectors.llamaswap import render
from gpushepherd.engine import Engine


def test_detects_wsl_from_the_kernel_name(tmp_path, monkeypatch):
    monkeypatch.delenv("SHEPHERD_WSL", raising=False)
    (tmp_path / "version").write_text("Linux version 5.15.153.1-microsoft-standard-WSL2 (root@host)\n")
    monkeypatch.setattr(host, "PROC", str(tmp_path))
    monkeypatch.setattr(host, "_wsl", None)
    assert host.is_wsl()
    (tmp_path / "version").write_text("Linux version 6.8.0-49-generic (buildd@lcy02)\n")
    monkeypatch.setattr(host, "_wsl", None)
    assert not host.is_wsl()
    monkeypatch.setenv("SHEPHERD_WSL", "1")
    assert host.is_wsl()


def test_render_names_a_models_gpus_by_uuid():
    c = Catalog(instance_id="llm", models=[CatalogModel(id="s", file="/m/s.gguf")])
    cfg = yaml.safe_load(render(c, 2, 8, {"s": [1]}, ["GPU-a", "GPU-b"]))["models"]
    assert cfg["s"]["env"] == ["CUDA_VISIBLE_DEVICES=GPU-b"]


def _engine(tmp_path, monkeypatch, wsl):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_WSL", "1" if wsl else "0")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    store = Store(str(tmp_path / "data"))
    inst = Instance(id="app", name="App", image="img", ports=[{"host": 8188, "container": 8188}],
                    gpu=GpuPolicy(mode="elastic"))
    store.save_instance(inst)
    return Engine(store, FakeBackend()), inst


def test_wsl_containers_are_told_their_gpus(tmp_path, monkeypatch):
    engine, inst = _engine(tmp_path, monkeypatch, wsl=True)
    assert engine.build_spec(inst, [1]).env["CUDA_VISIBLE_DEVICES"] == "GPU-fake-1"


def test_elsewhere_docker_picks_the_gpus(tmp_path, monkeypatch):
    engine, inst = _engine(tmp_path, monkeypatch, wsl=False)
    assert "CUDA_VISIBLE_DEVICES" not in engine.build_spec(inst, [1]).env


def test_windows_drives(monkeypatch):
    monkeypatch.setenv("SHEPHERD_WSL", "1")
    assert moves.windows_drive("/mnt/c/models") and moves.windows_drive("/mnt/d")
    assert not moves.windows_drive("/mnt/data/models")
    monkeypatch.setenv("SHEPHERD_WSL", "0")
    assert not moves.windows_drive("/mnt/c/models")
    assert moves.Check(windows=True).warnings
