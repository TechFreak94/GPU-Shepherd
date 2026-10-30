"""Small models on few GPUs: GGUF headers, placement, the planner's limit and llama-swap's config."""
import struct

import yaml

from gpushepherd import budget, gguf, placement
from gpushepherd.backend import FakeBackend
from gpushepherd.config import AutoAlias, AutoRule, Catalog, CatalogModel, GpuPolicy, Instance, OffloadRule, Store
from gpushepherd.connectors.llamaswap import render
from gpushepherd.engine import Engine
from gpushepherd.scheduler import plan

GiB = 2**30


def write_gguf(path, size_gb=0.0, **meta):
    """A GGUF header with these keys (ints, strings, int lists), a big skipped array, then padding."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    items = [("tokenizer.ggml.tokens", ["a", "bb"] * 1000), ("tokenizer.ggml.scores", [0.5] * 5000), *meta.items()]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(items))
    for key, v in items:
        out += s(key)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + s(v)
        elif isinstance(v, list) and v and isinstance(v[0], str):
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x) for x in v)
        elif isinstance(v, list) and v and isinstance(v[0], float):
            out += struct.pack("<IIQ", 9, 6, len(v)) + struct.pack(f"<{len(v)}f", *v)
        elif isinstance(v, list):
            out += struct.pack("<IIQ", 9, 4, len(v)) + struct.pack(f"<{len(v)}I", *v)
        else:
            out += struct.pack("<II", 4, v)
    with open(path, "wb") as fh:
        fh.write(out)
        if size_gb:
            fh.truncate(int(size_gb * 1e9))            # sparse: no real disk space
    return path




def test_reads_kv_shape(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", **{"general.architecture": "llama", "llama.block_count": 32,
                                           "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
                                           "llama.embedding_length": 4096})
    sh = gguf.shape(str(p))
    assert sh == gguf.Shape(kv_layers=32, kv_width=8 * 256)
    assert gguf.kv_bytes(str(p), 32768) == 32768 * 32 * 8 * 256 * 2          # 4 GiB at f16
    assert gguf.kv_bytes(str(p), 32768, "q4_0") < gguf.kv_bytes(str(p), 32768) / 3
    hybrid = write_gguf(tmp_path / "h.gguf", **{"general.architecture": "qwen35", "qwen35.block_count": 40,
                                                "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 256,
                                                "qwen35.attention.value_length": 256,
                                                "qwen35.full_attention_interval": 4})
    assert gguf.shape(str(hybrid)).kv_layers == 10
    per_layer = write_gguf(tmp_path / "p.gguf", **{"general.architecture": "x", "x.block_count": 4,
                                                   "x.attention.head_count_kv": [8, 0, 8, 0],
                                                   "x.attention.key_length": 128, "x.attention.value_length": 128})
    assert gguf.shape(str(per_layer)) == gguf.Shape(kv_layers=2, kv_width=8 * 256)
    (tmp_path / "bad.gguf").write_bytes(b"not a model")
    assert gguf.shape(str(tmp_path / "bad.gguf")) is None
    assert gguf.kv_bytes(str(tmp_path / "bad.gguf"), 1000) == 1000 * gguf.FALLBACK_PER_TOKEN


def cat(*models, **kw):
    return Catalog(instance_id="llm", models=list(models), **({"fit_gpus": True} | kw))


def m(mid, **kw):
    return CatalogModel(id=mid, file=f"/models/{mid}.gguf", **kw)


EIGHT_GB = [8 * GiB] * 4


def test_fewest_gpus_that_hold_it():
    assert placement.fits_on(5 * GiB, EIGHT_GB) == 1
    assert placement.fits_on(7 * GiB, EIGHT_GB) == 2          # the buffers on each GPU count too
    assert placement.fits_on(40 * GiB, EIGHT_GB) == 0
    small, mid, huge = m("small"), m("mid"), m("huge")
    needs = {"small": 5 * GiB, "mid": 16 * GiB, "huge": 50 * GiB}
    assert placement.place(cat(small, mid, huge), needs, EIGHT_GB) == {"small": [0], "mid": [0, 1, 2]}
    # unknown size, a GPU / RAM split for this count, or the setting off: every GPU, as before
    split = m("split", offload=[OffloadRule(min_gpus=4, cpu_moe=10)])
    assert placement.place(cat(m("x"), split), {"split": 5 * GiB}, EIGHT_GB) == {}
    assert placement.place(cat(small, fit_gpus=False), needs, EIGHT_GB) == {}
    assert placement.place(cat(m("small", gpus=2)), needs, EIGHT_GB) == {"small": [0, 1]}


def test_kept_loaded_models_get_their_own_gpus():
    c = cat(m("chat"), m("embed", keep_loaded=True, role="embed"), m("rerank", keep_loaded=True, role="rerank"))
    needs = {"chat": 5 * GiB, "embed": 1 * GiB, "rerank": 1 * GiB}
    assert placement.place(c, needs, EIGHT_GB) == {"chat": [0], "embed": [3], "rerank": [2]}
    assert placement.place(c, needs, EIGHT_GB[:2]) == {"chat": [0], "embed": [1], "rerank": [0]}
    cpu = cat(m("chat"), m("embed", keep_loaded=True, offload=[OffloadRule(min_gpus=1, gpu_layers=0)]))
    assert placement.place(cpu, needs, EIGHT_GB) == {"chat": [0]}
    assert placement.wants(cpu, needs, EIGHT_GB) == 1


def test_how_many_gpus_a_server_can_use():
    needs = {"small": 5 * GiB, "mid": 16 * GiB, "huge": 50 * GiB, "embed": 1 * GiB}
    assert placement.wants(cat(m("small")), needs, EIGHT_GB) == 1
    assert placement.wants(cat(m("small"), m("mid")), needs, EIGHT_GB) == 3
    assert placement.wants(cat(m("small"), m("embed", keep_loaded=True)), needs, EIGHT_GB) == 2
    assert placement.wants(cat(m("small"), m("huge")), needs, EIGHT_GB) is None     # uses them all
    assert placement.wants(cat(m("small"), m("unknown")), needs, EIGHT_GB) is None
    assert placement.wants(cat(m("small"), fit_gpus=False), needs, EIGHT_GB) is None
    auto = [AutoAlias(alias="chat", default_model="small", rules=[AutoRule(min_gpus=3, model="small")])]
    assert placement.wants(cat(m("small"), auto_aliases=auto), needs, EIGHT_GB) == 3


def test_planner_leaves_gpus_a_server_cannot_use():
    s = {"llm": Instance(id="llm", name="LLM", image="i", gpu=GpuPolicy(mode="elastic", min_gpus=1)),
         "big": Instance(id="big", name="Big", image="i", gpu=GpuPolicy(mode="elastic", min_gpus=1, priority=20))}
    p = plan([0, 1, 2, 3], s, {}, "rebalance", extra_wanted={"llm", "big"}, usable={"llm": 1})
    assert p.allocations == {"llm": [0], "big": [1, 2, 3]}
    s["llm"].gpu.min_gpus = 2                                           # never below its minimum
    assert plan([0, 1, 2, 3], s, {}, "start", "llm", usable={"llm": 1}).allocations == {"llm": [0, 1]}


def test_render_runs_a_model_on_its_gpus():
    c = cat(m("small", tensor_split="1,2,3,4"), m("big"))
    cfg = yaml.safe_load(render(c, 4, 8, {"small": [2, 3]}))["models"]
    assert cfg["small"]["env"] == ["CUDA_VISIBLE_DEVICES=2,3"]
    assert "--tensor-split 3,4" in cfg["small"]["cmd"]
    assert "env" not in cfg["big"] and "--tensor-split" not in cfg["big"]["cmd"]
    one = yaml.safe_load(render(cat(m("s", split_mode="row")), 4, 8, {"s": [0]}))["models"]["s"]
    assert "--split-mode" not in one["cmd"]                               # one GPU: nothing to split


def test_engine_places_real_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "4x8192:Test GPU")
    import gpushepherd.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    models = tmp_path / "models"
    models.mkdir()
    meta = {"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096}
    write_gguf(models / "small.gguf", 4.0, **meta)                        # + 1 GiB of context: one GPU
    write_gguf(models / "mid.gguf", 12.0, **meta)                         # two GPUs
    store = Store(str(tmp_path / "data"))
    inst = Instance(id="llm", name="LLM", image="img", kind="llama-swap", ports=[{"host": 8080, "container": 8080}],
                    volumes=[{"host": str(models), "container": "/models"}], gpu=GpuPolicy(mode="elastic"))
    store.save_instance(inst)
    store.save_catalog(cat(m("small", ctx_size=8192), llama_server="/app/llama-server"))
    engine = Engine(store, FakeBackend())
    assert budget.model_memory(inst, m("small", ctx_size=8192)) == int(4e9) + 8192 * 32 * 8 * 256 * 2
    p = engine.plan("start", "llm")
    assert p.allocations == {"llm": [0]}                                  # GPUs 1-3 stay free
    store.save_catalog(cat(m("small", ctx_size=8192), m("mid", ctx_size=8192), llama_server="/app/llama-server"))
    assert engine.plan("start", "llm").allocations == {"llm": [0, 1]}
    engine.execute(engine.plan("start", "llm"), lambda _m: None)
    cfg = yaml.safe_load((store.generated_dir("llm") / "llama-swap.yaml").read_text())["models"]
    assert cfg["small"]["env"] == ["CUDA_VISIBLE_DEVICES=0"] and "env" not in cfg["mid"]
