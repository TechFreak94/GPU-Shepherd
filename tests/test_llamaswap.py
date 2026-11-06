import pytest
import yaml
from pydantic import ValidationError

from gpushepherd.config import AutoAlias, AutoRule, Catalog, CatalogModel, OffloadRule
from gpushepherd.connectors.llamaswap import render, resolved_aliases, start_failures


def catalog():
    return Catalog(
        instance_id="llm", api_key="sk-test", llama_server="/app/llama-server",
        models=[
            CatalogModel(id="qwen", file="/models/q.gguf", ctx_size=131072, parallel=2,
                         sampling="--temp 0.6 --top-k 20", aliases=["q"]),
            CatalogModel(id="muse", file="/models/my model.gguf", kv_cache="q8_0", mmproj="/models/mmproj.gguf"),
        ],
        auto_aliases=[AutoAlias(alias="auto", default_model="qwen", rules=[AutoRule(min_gpus=2, model="muse")])],
    )


def test_render_is_valid_llama_swap_config():
    cfg = yaml.safe_load(render(catalog(), gpu_count=1, threads=8))
    assert cfg["apiKeys"] == ["${env.LLAMA_SWAP_API_KEY}"]
    q = cfg["models"]["qwen"]["cmd"]
    for part in ("/app/llama-server", "--port ${PORT}", "--jinja", "--threads 8", "--model /models/q.gguf",
                 "--ctx-size 131072", "--parallel 2", "--temp 0.6 --top-k 20"):
        assert part in q
    m = cfg["models"]["muse"]["cmd"]
    assert "--model '/models/my model.gguf'" in m          # quoted path
    assert "--cache-type-k q8_0 --cache-type-v q8_0" in m
    assert "--mmproj /models/mmproj.gguf --no-mmproj-offload" in m
    assert "sk-test" not in render(catalog(), 1, 8)          # the key is passed as env, never written


def test_auto_alias_follows_gpu_count():
    one = yaml.safe_load(render(catalog(), 1, 8))["models"]
    two = yaml.safe_load(render(catalog(), 2, 8))["models"]
    assert one["qwen"]["aliases"] == ["q", "auto"] and "aliases" not in one["muse"]
    assert two["qwen"]["aliases"] == ["q"] and two["muse"]["aliases"] == ["auto"]
    assert resolved_aliases(catalog(), 4) == {"auto": "muse"}


def test_offload_follows_gpu_count():
    cat = catalog()
    cat.models[0].offload = [OffloadRule(min_gpus=1, cpu_moe=30), OffloadRule(min_gpus=2, cpu_moe=14),
                             OffloadRule(min_gpus=4, gpu_layers=99)]
    cmd = {n: yaml.safe_load(render(cat, n, 8))["models"]["qwen"]["cmd"] for n in (1, 2, 3, 4)}
    assert "--n-cpu-moe 30" in cmd[1] and "--n-gpu-layers" not in cmd[1]
    assert "--n-cpu-moe 14" in cmd[2] and "--n-cpu-moe 14" in cmd[3]      # a rule covers larger counts
    assert "--n-gpu-layers 99" in cmd[4] and "--n-cpu-moe" not in cmd[4]
    # before the extra flags, so a hand-written flag there still wins
    assert cmd[1].index("--n-cpu-moe") < cmd[1].index("--temp")
    muse = yaml.safe_load(render(cat, 1, 8))["models"]["muse"]["cmd"]
    assert "--n-cpu-moe" not in muse and "--n-gpu-layers" not in muse    # no rules: llama.cpp defaults


def test_offload_rules_are_sorted_and_unique():
    m = CatalogModel(id="x", file="/m/x.gguf", offload=[{"min_gpus": 2, "cpu_moe": 5}, {"min_gpus": 1, "gpu_layers": 0}])
    assert [r.min_gpus for r in m.offload] == [1, 2]
    assert m.offload_for(1).gpu_layers == 0 and m.offload_for(3).cpu_moe == 5
    with pytest.raises(ValidationError):
        CatalogModel(id="x", file="/m/x.gguf", offload=[{"min_gpus": 1}, {"min_gpus": 1, "cpu_moe": 2}])


LOG = """\
[INFO] <coder> Health check passed on http://localhost:5801/health
0.12.345.678 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1520.00 MiB on device 1: cudaMalloc failed: out of memory
0.12.345.700 E alloc_tensor_range: failed to allocate CUDA1 buffer of size 1593835520
[WARN] group: starting coder failed: upstream command exited prematurely
0.01.000.000 E gguf_init_from_file: failed to open GGUF file '/models/gone.gguf' (No such file or directory)
[WARN] group: starting gone failed: upstream command exited prematurely
[WARN] group: starting quiet failed: health check timed out
[WARN] group: starting stranger failed: upstream command exited prematurely
[INFO] <quiet> Health check passed on http://localhost:5802/health
"""


def test_start_failures_name_the_reason_and_clear_once_a_model_starts():
    failed = start_failures(LOG, {"coder", "gone", "quiet"})
    assert set(failed) == {"coder", "gone"}                  # quiet started later; stranger isn't in the catalogue
    assert failed["coder"].startswith("Out of GPU memory: it needed 1.6 GB more on one GPU.")
    assert "No such file or directory" in failed["gone"]
    assert start_failures(LOG + "[INFO] <coder> Health check passed\n", {"coder"}) == {}
    assert start_failures("[WARN] group: starting x failed: health check timed out\n", {"x"}) == \
        {"x": "health check timed out"}


def test_tensor_split_on_many_gpus():
    cat = catalog()
    cat.models[0].tensor_split = "1,1,2,2"
    cmd = {n: yaml.safe_load(render(cat, n, 8))["models"]["qwen"]["cmd"] for n in (1, 3, 4, 6)}
    assert "--tensor-split" not in cmd[1]
    assert "--tensor-split 1,1,2" in cmd[3] and "--tensor-split 1,1,2,2" in cmd[4]
    assert "--tensor-split" not in cmd[6]           # fewer shares than GPUs: llama.cpp splits by memory
    assert "for 6 GPU(s)" in render(cat, 6, 8)


def test_no_gpu_keeps_every_layer_in_ram():
    cat = catalog()
    cat.models[0].offload = [OffloadRule(min_gpus=1, gpu_layers=20, cpu_moe=10)]
    cfg = yaml.safe_load(render(cat, 0, 8))["models"]
    assert "--n-gpu-layers 0" in cfg["qwen"]["cmd"] and "--n-cpu-moe" not in cfg["qwen"]["cmd"]
    assert "--n-gpu-layers 0" in cfg["muse"]["cmd"]
    assert "--n-gpu-layers 0" not in yaml.safe_load(render(cat, 1, 8))["models"]["muse"]["cmd"]
