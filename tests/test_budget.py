"""Memory checks before a start (budget.py) and the warnings they add to a plan."""
import time

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, budget, host
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Catalog, CatalogModel, Instance, OffloadRule, Store, VolumeMap

GB = 10**9


def wait(app):
    for _ in range(500):
        if not app.state.jobs.busy:
            return
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def server(tmp_path, *models):
    inst = Instance(id="llm", name="LLM", image="x", kind="llama-swap",
                    volumes=[VolumeMap(host=str(tmp_path), container="/models")])
    return inst, Catalog(instance_id="llm", models=list(models))


def test_server_ram_counts_what_the_gpus_cannot_hold(tmp_path):
    sparse(tmp_path / "big.gguf", 30 * GB)
    sparse(tmp_path / "embed.gguf", 1 * GB)
    big = CatalogModel(id="big", file="/models/big.gguf")
    embed = CatalogModel(id="embed", file="/models/embed.gguf", keep_loaded=True, role="embed")
    inst, cat = server(tmp_path, big, embed)
    assert budget.server_ram(inst, cat, 1, 24 * GB) == pytest.approx(31 * GB - 0.9 * 24 * GB + GB)
    assert budget.server_ram(inst, cat, 2, 48 * GB) == GB                      # fits: just the server itself
    big.offload = [OffloadRule(min_gpus=1, cpu_moe=30)]                        # experts kept in RAM on purpose
    assert budget.server_ram(inst, cat, 2, 48 * GB) == 15 * GB + GB
    assert budget.server_ram(inst, cat, 0, 0) == 31 * GB + GB                  # no GPU: all of it
    assert budget.server_ram(*server(tmp_path / "none", big), 1, 24 * GB) == 0  # files not visible: unknown


def test_too_big_needs_an_offload_rule(tmp_path):
    sparse(tmp_path / "big.gguf", 30 * GB)
    big = CatalogModel(id="big", file="/models/big.gguf")
    inst, cat = server(tmp_path, big)
    [msg] = budget.too_big(inst, cat, 1, 24 * GB)
    assert "big (30.0 GB) is bigger than the GPU memory LLM gets with 1 GPU (24.0 GB)" in msg
    assert budget.too_big(inst, cat, 2, 48 * GB) == []
    big.offload = [OffloadRule(min_gpus=1, gpu_layers=20)]
    assert budget.too_big(inst, cat, 1, 24 * GB) == []


def test_ram_needed_prefers_what_it_used_before():
    app = Instance(id="comfy", name="ComfyUI", image="x", memory_limit="8g")
    assert budget.ram_needed(app, None, 1, 0, peak=0) == 8 * 2**30
    assert budget.ram_needed(app, None, 1, 0, peak=12 * GB) == 12 * GB
    assert budget.parse_size("512m") == 512 * 2**20 and budget.parse_size("lots") == 0


def test_peaks_are_kept_and_only_written_when_they_grow(tmp_path):
    p = budget.Peaks(tmp_path)
    p.record({"a": 100})
    p.record({"a": 103})                       # under 5% more: not written
    assert budget.Peaks(tmp_path).get() == {"a": 100}
    p.record({"a": 200, "b": 5})
    assert budget.Peaks(tmp_path).get() == {"a": 200, "b": 5}
    p.forget("a")
    assert p.get() == {"b": 5}


def test_start_warns_and_asks_when_ram_runs_short(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x24576:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    models = tmp_path / "models"
    sparse(models / "big.gguf", 40 * GB)
    data = tmp_path / "data"
    data.mkdir()
    app = create_app(data_dir=str(data), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(data))}"})
    client.put("/api/v1/instances/llm", json={"name": "LLM", "image": "x", "kind": "llama-swap",
                                               "gpu": {"mode": "elastic", "min_gpus": 1},
                                               "volumes": [{"host": str(models), "container": "/models"}]})
    client.put("/api/v1/instances/llm/models/big", json={"id": "big", "file": "/models/big.gguf",
                                                          "offload": [{"min_gpus": 1, "gpu_layers": 30}]})
    monkeypatch.setattr(host, "mem_available", lambda: (32 * GB, 10 * GB))
    r = client.post("/api/v1/instances/llm/start", json={})
    assert r.status_code == 409 and "Memory may not be enough" in r.json()["error"]
    assert "LLM may need about 21.0 GB of system RAM, and about 10.0 GB is free" in r.json()["warnings"][0]
    r = client.post("/api/v1/instances/llm/start", json={"confirm": True})
    assert r.status_code == 202
    wait(app)
    monkeypatch.setattr(host, "mem_available", lambda: (128 * GB, 100 * GB))
    assert client.post("/api/v1/instances/llm/stop", json={"confirm": True}).status_code == 202
    wait(app)
    assert client.post("/api/v1/instances/llm/start", json={}).status_code == 202   # plenty: no question
