"""Suggested adjustments: measuring models, working out suggestions, and keeping the decisions."""
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, tuning
from gpushepherd.app import DemoLlamaClient, create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Catalog, CatalogModel, GpuPolicy, Instance, OffloadRule, Store

from test_placement import write_gguf

GiB = 2**30
TWO = [8 * GiB, 8 * GiB]
FOUR = [8 * GiB] * 4


def cat(*models, **kw):
    return Catalog(instance_id="llm", models=list(models), **({"fit_gpus": True} | kw))


def m(mid, **kw):
    return CatalogModel(id=mid, file=f"/models/{mid}.gguf", **kw)


def test_unchosen_fit_setting_is_offered_with_what_it_changes():
    c = cat(m("small"), fit_gpus=None)
    s = tuning.suggest(c, {}, {"small": 4 * GiB}, FOUR)
    assert [x.key for x in s] == ["llm::fit_gpus"]
    assert "small on one GPU" in s[0].reason and "take 1 of the 4 GPUs" in s[0].reason
    assert tuning.suggest(cat(m("small")), {}, {"small": 4 * GiB}, FOUR) == []


def test_suggestions_from_measurements():
    big = m("big")                                            # estimate: two GPUs
    needs = {"big": 9 * GiB}
    # it really needs less: one GPU
    s = tuning.suggest(cat(big), {"big": {2: {"vram": int(5.5 * GiB)}}}, needs, FOUR)
    assert [(x.model, x.value, x.old) for x in s] == [("big", 1, 0)] and "fits on one GPU" in s[0].reason
    # nearly full: one more GPU
    s = tuning.suggest(cat(big), {"big": {2: {"vram": int(13.5 * GiB)}}}, needs, FOUR)
    assert s[0].value == 3 and "run it out of memory" in s[0].reason
    # faster on another count, once there are enough answers
    fast = {"big": {2: {"tps": 20.0, "n": 5}, 3: {"tps": 30.0, "n": 5}}}
    assert tuning.suggest(cat(big), fast, needs, FOUR)[0].value == 3
    few = {"big": {2: {"tps": 20.0, "n": 5}, 3: {"tps": 30.0, "n": 1}}}
    assert tuning.suggest(cat(big), few, needs, FOUR) == []
    # nothing for models on the CPU
    cpu = m("embed", offload=[OffloadRule(min_gpus=1, gpu_layers=0)])
    assert tuning.suggest(cat(cpu), {"embed": {4: {"vram": 1}}}, {}, FOUR) == []


def test_metrics_speeds_only_counts_new_answers():
    items = [{"model": "a", "tokens_per_second": 40.0, "output_tokens": 200, "timestamp": "2026-10-29T10:00:00Z"},
             {"model": "a", "output_tokens": 100, "duration_ms": 2000, "timestamp": "2026-10-29T10:01:00Z"},
             {"model": "a", "tokens_per_second": 99.0, "output_tokens": 3, "timestamp": "2026-10-29T10:02:00Z"}]
    speeds, newest = tuning.metrics_speeds(items, -1)
    assert speeds == [("a", 40.0), ("a", 50.0)]
    assert tuning.metrics_speeds(items, newest)[0] == []


def test_measurements_keep_peaks_and_recent_average(tmp_path):
    ms = tuning.Measurements(tmp_path)
    ms.record("llm", "a", 1, vram=5 * GiB)
    ms.record("llm", "a", 1, vram=4 * GiB, tps=30)
    ms.record("llm", "a", 1, tps=40)
    again = tuning.Measurements(tmp_path).get("llm")
    assert again == {"a": {1: {"vram": 5 * GiB, "tps": 35.0, "n": 2}}}


def test_apply_dismiss_undo_and_auto_tune(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "4x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    models = tmp_path / "models"
    models.mkdir()
    meta = {"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096}
    write_gguf(models / "small.gguf", 4.0, **meta)
    backend = FakeBackend()
    app = create_app(data_dir=str(tmp_path / "data"), backend=backend, demo=True, autostart=False)
    svc = app.state.svc
    inst = Instance(id="llm", name="LLM", image="img", kind="llama-swap", ports=[{"host": 8080, "container": 8080}],
                    volumes=[{"host": str(models), "container": "/models"}], gpu=GpuPolicy(mode="elastic"))
    svc.store.save_instance(inst)
    svc.store.save_catalog(Catalog(instance_id="llm", llama_server="/app/llama-server",
                                   models=[CatalogModel(id="small", file="/models/small.gguf", ctx_size=8192)]))
    [s] = svc.tuning("llm")
    assert s["key"] == "llm::fit_gpus" and s["status"] == "new"
    assert svc.engine.plan("start", "llm").allocations == {"llm": [0, 1, 2, 3]}      # not chosen: as before
    assert "Dismissed" in svc.tuning_decide("llm", s["key"], "dismiss")
    assert svc.tuning("llm")[0]["status"] == "dismissed" and svc.store.catalog("llm").fit_gpus is None
    svc.tuning_decide("llm", s["key"], "apply")
    assert svc.store.catalog("llm").fit_gpus is True and svc.tuning("llm")[0]["status"] == "applied"
    assert svc.engine.plan("start", "llm").allocations == {"llm": [0]}
    svc.tuning_decide("llm", s["key"], "undo")
    assert svc.store.catalog("llm").fit_gpus is None and svc.tuning("llm")[0]["status"] == "dismissed"
    svc.tuning_decide("llm", s["key"], "apply")

    # running: the model used far less than estimated, measured from the GPUs it had to itself
    svc.engine.execute(svc.engine.plan("start", "llm"), lambda _m: None)
    cat = svc.store.catalog("llm")
    cat.models[0].gpus = 2                                     # set by hand to two GPUs
    svc.store.save_catalog(cat)
    svc.engine.execute(svc.engine.plan("restart", "llm"), lambda _m: None)
    DemoLlamaClient.loaded.clear()
    DemoLlamaClient.loaded["small"] = "ready"
    monkeypatch.setattr(gpu_mod, "query", lambda max_age=2.0: (
        [gpu_mod.Gpu(i, "Test GPU", 2500, 8192, 0) for i in range(4)], ""))
    svc.measure()
    assert svc.measurements.get("llm")["small"][2]["vram"] == 2 * 2500 * 2**20
    new = [x for x in svc.tuning("llm") if x["status"] == "new"]
    assert [(x["model"], x["value"], x["old"]) for x in new] == [("small", 1, 2)]
    cat = svc.store.catalog("llm")
    cat.auto_tune = True
    svc.store.save_catalog(cat)
    svc.measure()                                              # applies it by itself, and keeps it listed
    assert svc.store.catalog("llm").models[0].gpus == 1
    applied = [x for x in svc.tuning("llm") if x["model"] == "small"]
    assert applied[0]["status"] == "applied"
    svc.tuning_decide("llm", applied[0]["key"], "undo")
    assert svc.store.catalog("llm").models[0].gpus == 2
    DemoLlamaClient.loaded.clear()


def test_api_and_new_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    c = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(tmp_path))}"})
    r = c.post("/api/v1/apps/llm-server/install", json={"id": "llm"})
    assert r.status_code == 201, r.text
    assert app.state.svc.store.catalog("llm").fit_gpus is True          # new servers start with it on
    assert c.get("/api/v1/instances/llm/tuning").json() == []
    r = c.patch("/api/v1/instances/llm/server-settings", json={"auto_tune": True})
    assert r.status_code == 200 and c.get("/api/v1/instances/llm/models").json()["settings"]["auto_tune"] is True
    r = c.post("/api/v1/instances/llm/tuning", json={"key": "nope", "action": "apply"})
    assert r.status_code == 404
