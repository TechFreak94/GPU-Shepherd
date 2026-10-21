import yaml

from gpushepherd.config import AutoAlias, AutoRule, Catalog, CatalogModel
from gpushepherd.connectors.llamaswap import render, resolved_aliases


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
