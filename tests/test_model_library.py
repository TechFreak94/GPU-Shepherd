"""Suggested models, their downloads, and the model settings they fill in."""
import time
from dataclasses import replace

import pytest
import yaml
from pydantic import ValidationError

from gpushepherd import downloads, instance_templates as tpl, model_library as ml
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Catalog, CatalogModel, Instance, PortMap
from gpushepherd.connectors.llamaswap import render


def test_suggestions_are_generic_and_complete():
    keys = [s.key for s in ml.SUGGESTIONS]
    assert len(keys) == len(set(keys))
    for s in ml.SUGGESTIONS:
        assert s.uses and set(s.uses) <= {"chat", "coder", "embed"}
        assert all(f.url.startswith("https://huggingface.co/") and f.url.endswith("/" + f.name) for f in s.files)
        assert s.model.name.endswith(".gguf") and s.page.startswith("https://huggingface.co/")
        assert not s.moe or s.layers                # MoE models say how many expert layers can go to RAM
        m = ml.catalog_model(s, [8.0, 8.0])
        assert m.id == s.key and m.file == f"/models/{s.key}/{s.model.name}"
        assert ml.for_file(m.file) is s
        assert (m.mmproj != "") == s.vision and (m.mtp_draft != "") == (s.draft is not None)
        assert "personal" not in s.key


def test_fit_for_machines():
    small = ml.get("qwen3.5-9b")
    assert ml.fit(small, [8.0], 16).text == "Fits on one GPU"
    dense = ml.get("qwen3.8-27b")
    assert ml.fit(dense, [8.0, 8.0], 64).text == "Fits on 2 GPUs"
    assert not ml.fit(dense, [8.0], 64).ok                         # dense models don't spill into RAM
    moe = ml.get("qwen3-coder-30b-a3b")
    one = ml.fit(moe, [8.0], 64)
    assert one.ok and one.gpus == 1 and 10 < one.ram_gb < 16 and one.text.startswith("One GPU + about")
    assert "(one GPU: about" in ml.fit(moe, [8.0, 8.0], 64).text
    big = ml.get("qwen3-coder-next")
    assert ml.fit(big, [8.0, 8.0], 64).ok                          # the 64 GB target
    assert not ml.fit(big, [8.0, 8.0], 32).ok and "Too big" in ml.fit(big, [8.0, 8.0], 32).text
    assert ml.fit(ml.get("qwen3-embedding-0.6b"), [8.0], 16).text.startswith("Runs on the CPU")
    assert ml.fit(small, [], 16).ok                                # no GPU info: say what it needs
    order = [s.key for s, _ in ml.for_slot("coder", [8.0], 64)]
    assert order.index("qwen3.5-9b") < order.index("qwen3-coder-30b-a3b") < order.index("qwen3.8-27b")


def test_fit_and_offload_on_many_gpus():
    four = [8.0] * 4
    assert ml.fit(ml.get("qwen3.5-9b"), four, 64).text == "Fits on one GPU"
    assert ml.fit(ml.get("qwen3-coder-30b-a3b"), four, 64).text == "Fits on 3 GPUs"
    big = ml.fit(ml.get("qwen3-coder-next"), four, 64)
    assert big.ok and big.gpus == 4 and big.text.startswith("4 GPUs + about")
    assert ml.fit(ml.get("qwen3-coder-next"), [24.0] * 8, 64).text.startswith("Fits on")
    # a dense model with layers in RAM on one GPU: a rule for every count up to the machine's
    rules = ml.offload(ml.get("muse-glimmer-30b"), four)
    assert [r.min_gpus for r in rules] == [1, 2, 3, 4] and rules[0].gpu_layers
    assert all(r.gpu_layers is None for r in rules[1:])


def test_offload_rules_only_where_llama_cpp_does_not_fit_it_itself():
    # MoE: llama.cpp's own fit keeps the experts that don't fit in RAM and balances the GPUs;
    # --n-cpu-moe would turn that off and overfill the last GPU
    moe = ml.get("qwen3-coder-30b-a3b")
    assert ml.offload(moe, [8.0, 8.0]) == [] and ml.offload(moe, [8.0]) == []
    m = ml.catalog_model(moe, [8.0, 8.0])
    assert "--n-cpu-moe" not in render(Catalog(instance_id="x", models=[m]), 2, 8)
    assert ml.offload(ml.get("qwen3.8-27b"), [8.0, 8.0]) == []     # dense: nothing to move
    muse = ml.get("muse-glimmer-30b")                             # dense: one 8 GB GPU is too small
    assert ml.fit(muse, [8.0, 8.0], 64).text == "Fits on 2 GPUs"
    assert not ml.fit(muse, [8.0], 64).ok                          # most of it in RAM: too slow
    assert [(r.min_gpus, r.gpu_layers) for r in ml.offload(muse, [8.0, 8.0])] == [(1, 26), (2, None)]
    assert ml.offload(muse, [24.0]) == []
    just_too_big = replace(muse, model=replace(muse.model, gb=15.88))  # a few layers go to RAM
    assert ml.fit(just_too_big, [8.0, 8.0], 64).text.startswith("2 GPUs with 43 of its 52 layers")
    assert [(r.min_gpus, r.gpu_layers) for r in ml.offload(just_too_big, [8.0, 8.0])] == [(1, 21), (2, 43)]
    assert [(r.min_gpus, r.gpu_layers) for r in ml.offload(ml.get("qwen3-embedding-0.6b"), [8.0])] == [(1, 0)]


def test_render_new_settings():
    m = ml.catalog_model(ml.get("gemma-4-e4b"), [8.0])
    embed = ml.catalog_model(ml.get("qwen3-embedding-0.6b"), [8.0], aliases=["embed"])
    big = CatalogModel(id="big", file="/models/big.gguf", thinking="on", split_mode="row", tensor_split="8,12",
                       ttl=600, kv_cache="bf16")
    cat = Catalog(instance_id="llm", models=[m.model_copy(update={"preload": True}), embed, big])
    one = yaml.safe_load(render(cat, 1, 8))
    cmd = one["models"]["gemma-4-e4b"]["cmd"]
    assert "--chat-template-kwargs '{\"enable_thinking\": false}'" in cmd
    assert "--model-draft /models/gemma-4-e4b/mtp-gemma-4-E4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 2" in cmd
    assert "--mmproj /models/gemma-4-e4b/mmproj-F16.gguf" in cmd and "--temp 1.0 --top-p 0.95 --top-k 64" in cmd
    e = one["models"]["qwen3-embedding-0.6b"]["cmd"]
    assert "--embeddings" in e and "--n-gpu-layers 0" in e and "--pooling last" in e and "enable_thinking" not in e
    b = one["models"]["big"]
    assert "--split-mode" not in b["cmd"] and "--tensor-split" not in b["cmd"]   # only with 2+ GPUs
    assert "--cache-type-k bf16" in b["cmd"] and b["ttl"] == 600 and "enable_thinking\": true" in b["cmd"]
    assert one["groups"]["kept-loaded"] == {"swap": False, "exclusive": False, "persistent": True,
                                            "members": ["qwen3-embedding-0.6b"]}
    assert one["groups"]["models"]["members"] == ["gemma-4-e4b", "big"]
    assert one["hooks"] == {"on_startup": {"preload": ["gemma-4-e4b", "qwen3-embedding-0.6b"]}}
    two = yaml.safe_load(render(cat, 2, 8))["models"]["big"]["cmd"]
    assert "--split-mode row --tensor-split 8,12" in two
    plain = yaml.safe_load(render(Catalog(instance_id="llm", models=[CatalogModel(id="a", file="/models/a.gguf")]), 1, 8))
    assert "groups" not in plain and "hooks" not in plain and "ttl" not in plain["models"]["a"]


def test_new_settings_are_checked():
    with pytest.raises(ValidationError, match="share per GPU"):
        CatalogModel(id="a", file="/models/a.gguf", tensor_split="8;12")
    with pytest.raises(ValidationError, match="inside the container"):
        CatalogModel(id="a", file="/models/a.gguf", mtp_draft="mtp.gguf")
    with pytest.raises(ValidationError):
        CatalogModel(id="a", file="/models/a.gguf", mtp=9)
    assert CatalogModel(id="a", file="/models/a.gguf", tensor_split=" 8, 12 ").tensor_split == "8,12"
    old = CatalogModel.model_validate({"id": "a", "file": "/models/a.gguf"})     # older catalogues still load
    assert old.role == "chat" and old.thinking == "auto" and not old.keep_loaded and old.mtp == 0


def test_download_script_resumes_and_quotes():
    s = ml.get("qwen3.5-9b")
    sh = downloads.script(s)
    assert "mkdir -p /models/qwen3.5-9b" in sh
    assert "curl -fsSL --retry 5 --retry-delay 10 -C - -o /models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf.part" in sh
    assert "mv /models/qwen3.5-9b/mmproj-F16.gguf.part /models/qwen3.5-9b/mmproj-F16.gguf" in sh
    assert sh.count("if [ ! -s") == 2                               # files already there are skipped


def wait_for(fn, state):
    for _ in range(200):
        if fn().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"never reached {state}: {fn()}")


def test_downloads_run_in_a_helper(tmp_path):
    backend = FakeBackend()
    backend.helper_seconds = 0.2
    d = downloads.Downloads(backend)
    s = ml.get("qwen3-embedding-0.6b")
    assert d.status(s, str(tmp_path)).state == "missing"
    (tmp_path / s.key).mkdir()
    (tmp_path / s.key / (s.model.name + ".part")).write_bytes(b"x" * 320_000_000)
    assert "Downloading Qwen3 Embedding 0.6B (0.6 GB)" in d.start(s, str(tmp_path))
    wait_for(lambda: d.status(s, str(tmp_path)), "downloading")
    st = d.status(s, str(tmp_path))
    assert st.pct == 50 and "0.3 of 0.6 GB (50%)" in st.text
    kind, script, _ = backend.helper_runs["shepherd-download-qwen3-embedding-0.6b"]
    assert kind == "download" and "Qwen3-Embedding-0.6B-Q8_0.gguf" in script
    assert "already downloading" in d.start(s, str(tmp_path))
    wait_for(lambda: d.status(s, str(tmp_path)), "ready")           # the helper exited 0
    assert "Stopped" in d.cancel(s) and "shepherd-download-qwen3-embedding-0.6b" not in backend.helper_runs
    (tmp_path / s.key / s.model.name).write_bytes(b"gguf")
    assert d.status(s, str(tmp_path)).state == "ready"             # the file is there
    backend.helper_exit, backend.helper_seconds = 6, 0
    (tmp_path / s.key / s.model.name).unlink()
    d.start(s, str(tmp_path))
    wait_for(lambda: d.status(s, str(tmp_path)), "failed")
    with pytest.raises(ValueError):
        d.start(s, "relative/path")


def test_finished_downloads_are_tidied_and_partials_can_be_discarded(tmp_path, monkeypatch):
    backend = FakeBackend()
    d = downloads.Downloads(backend)
    s = ml.get("qwen3-embedding-0.6b")
    d.start(s, str(tmp_path))
    wait_for(lambda: d.status(s, str(tmp_path)), "ready")
    name = d.name(s)
    assert name in backend.helper_runs
    (tmp_path / s.key).mkdir()
    (tmp_path / s.key / s.model.name).write_bytes(b"gguf")
    assert d.status(s, str(tmp_path)).state == "ready" and name not in backend.helper_runs   # helper removed
    # a stopped download: its .part file can be thrown away
    (tmp_path / s.key / s.model.name).unlink()
    part = tmp_path / s.key / (s.model.name + ".part")
    with open(part, "wb") as f:
        f.truncate(50_000_000)                                     # sparse: 50 MB fetched so far
    st = d.status(s, str(tmp_path))
    assert st.state == "missing" and st.partial
    assert "Threw away" in d.discard(s, str(tmp_path)) and not part.exists()
    assert not d.status(s, str(tmp_path)).partial
    # a folder GPU Shepherd can't see is cleaned up by a helper container
    d.discard(s, "/nonexistent/models")
    script, volumes = backend.tasks[-1]
    assert script.startswith("rm -f /models/qwen3-embedding-0.6b/") and volumes == [("/nonexistent/models", "/models", False)]


def test_downloads_need_room_on_the_disk(tmp_path, monkeypatch):
    s = ml.get("qwen3-embedding-0.6b")
    usage = type("U", (), {"free": int(1e9)})
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda _p: usage)
    with pytest.raises(ValueError, match="Not enough disk space"):
        downloads.Downloads(FakeBackend()).start(s, str(tmp_path))
    usage.free = int(10e9)
    assert downloads.room_short(s, str(tmp_path)) == ""
    assert downloads.room_short(s, "/nonexistent") == ""        # can't tell: let it try


def test_install_form_uses_a_suggestion_and_the_embed_slot():
    t = tpl.get("llm-server")
    m = tpl.Machine(gpu_mem_gb=[8.0, 8.0], instances={}, servers={}, model_sizes={}, ram_gb=64)
    assert [s.key for s in tpl.slots(t)] == ["chat", "coder", "embed"]
    f = tpl.defaults(t, m, "/srv/models", "/a") | {
        "chat_file": "/models/qwen3.8-27b/Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "coder_file": "/models/qwen3.8-27b/Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "embed_file": "/models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf"}
    _, models = tpl.build(t, f, m)
    chat, embed = models
    assert chat.id == "qwen3.8-27b" and chat.aliases == ["chat", "coder"] and chat.ctx_size == 65536
    assert chat.sampling == ml.QWEN_THINKING and chat.mmproj.endswith("mmproj-F16.gguf")
    assert embed.role == "embed" and embed.keep_loaded and embed.aliases == ["embed"]
    assert embed.preload and not chat.preload       # ready for the chat apps' documents at start
    # any other file in the embed slot is still set up as an embeddings model on the CPU
    other = tpl.build(t, f | {"embed_file": "my-embedder.gguf"}, m)[1][1]
    assert other.role == "embed" and other.keep_loaded and other.preload and other.offload[0].gpu_layers == 0
    with pytest.raises(Exception, match="can't also chat"):
        tpl.build(t, f | {"embed_file": f["chat_file"]}, m)


def test_chat_apps_use_the_servers_embeddings_model():
    llm = Instance(id="llm", name="LLM", image="x", kind="llama-swap", ports=[PortMap(host=8090, container=8080)])
    info = {"api_key": "sk-abc", "name": "LLM", "default_model": "chat", "ctx": 32768, "embed": "embed"}
    m = tpl.Machine(gpu_mem_gb=[8.0], instances={"llm": llm}, servers={"llm": info}, model_sizes={})
    t = tpl.get("open-webui")
    env = tpl.build(t, tpl.defaults(t, m, "/m", "/a"), m)[0].env
    assert env["RAG_EMBEDDING_ENGINE"] == "openai" and env["RAG_EMBEDDING_MODEL"] == "embed"
    assert env["RAG_OPENAI_API_BASE_URL"] == "http://llm:8080/v1" and env["RAG_OPENAI_API_KEY"] == "sk-abc"
    t = tpl.get("anythingllm")
    seed = tpl.seed_files(t, tpl.defaults(t, m, "/m", "/a"), m)["server.env"]
    assert "EMBEDDING_ENGINE='generic-openai'" in seed and "EMBEDDING_MODEL_PREF='embed'" in seed
    m.servers["llm"] = info | {"embed": ""}
    assert "RAG_EMBEDDING_ENGINE" not in tpl.build(tpl.get("open-webui"), tpl.defaults(tpl.get("open-webui"), m, "/m", "/a"), m)[0].env
    assert "EMBEDDING_ENGINE='native'" in tpl.seed_files(t, tpl.defaults(t, m, "/m", "/a"), m)["server.env"]


def test_suggested_embeddings_model_loads_at_start_with_the_embeddings_flag():
    s = ml.get("qwen3-embedding-0.6b")
    m = ml.catalog_model(s, [8.0], aliases=["embed"])
    assert m.preload and m.keep_loaded
    cfg = yaml.safe_load(render(Catalog(instance_id="llm", models=[m]), 1, 4))
    assert "--embeddings" in cfg["models"][m.id]["cmd"]
    assert cfg["hooks"]["on_startup"]["preload"] == [m.id]
