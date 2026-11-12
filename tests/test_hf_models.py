"""Suggested models from Hugging Face: what is asked, which file is picked, and what is offered."""
import json

from gpushepherd import hf_models as hf, model_library as ml

GB = 10**9


def _row(repo, params, arch="llama", ctx=131072, pipeline="text-generation", trending=10.0):
    return {"id": repo, "pipeline_tag": pipeline, "downloads": 1234, "likes": 5, "trendingScore": trending,
            "tags": ["gguf", "license:apache-2.0"],
            "gguf": {"total": params, "architecture": arch, "context_length": ctx, "chat_template": "..."}}


def _f(path, gb):
    return {"type": "file", "path": path, "size": 100, "lfs": {"size": int(gb * GB)}}


TREES = {
    "unsloth/Small-9B-GGUF": [_f("Small-9B-UD-Q4_K_XL.gguf", 6), _f("Small-9B-Q8_0.gguf", 9.5),
                              _f("mmproj-F32.gguf", 1.8), _f("mmproj-F16.gguf", 0.9),
                              {"type": "file", "path": "README.md", "size": 10}],
    "unsloth/Big-Coder-80B-A3B-GGUF": [_f("UD-Q4_K_XL/Big-Coder-80B-A3B-UD-Q4_K_XL-00001-of-00002.gguf", 30),
                                       _f("UD-Q4_K_XL/Big-Coder-80B-A3B-UD-Q4_K_XL-00002-of-00002.gguf", 16),
                                       _f("Big-Coder-80B-A3B-UD-Q2_K_XL.gguf", 28)],
    "unsloth/Dense-70B-GGUF": [_f("Dense-70B-Q4_K_M.gguf", 42), _f("Dense-70B-UD-Q2_K_XL.gguf", 26)],
    "unsloth/Tiny-1B-GGUF": [_f("Tiny-1B-Q4_K_M.gguf", 0.8), _f("Tiny-1B-Q8_0.gguf", 1.3)],
    "bartowski/Org_Small-9B-GGUF": [_f("Org_Small-9B-Q4_K_M.gguf", 5.6)],
}
LISTS = {
    "unsloth": [_row("unsloth/Small-9B-GGUF", 9 * GB, "qwen35", pipeline="image-text-to-text", trending=50),
                _row("unsloth/Big-Coder-80B-A3B-GGUF", 80 * GB, "qwen3next", trending=40),
                _row("unsloth/Dense-70B-GGUF", 70 * GB, trending=30),
                _row("unsloth/Tiny-1B-GGUF", 1 * GB, "gemma3", trending=20),
                _row("unsloth/Some-Embedding-GGUF", 1 * GB, pipeline="feature-extraction"),
                _row("unsloth/Thing-Abliterated-GGUF", 8 * GB),
                _row("unsloth/FLUX.2-dev-GGUF", 12 * GB, pipeline="text-to-image")],
    "bartowski": [_row("bartowski/Org_Small-9B-GGUF", 9 * GB, trending=60)],
}


def fake_get(url, params=None):
    if url.endswith("/tree/main"):
        return TREES[url.split("/api/models/", 1)[1].rsplit("/tree/", 1)[0]]
    author = dict(params)["author"]
    if author == "ggml-org":
        raise OSError("offline")
    return LISTS.get(author, [])


def cands():
    return hf.fetch(fake_get)


def test_fetch_keeps_chat_models_one_per_base_model():
    got = cands()
    assert [c["repo"] for c in got] == ["unsloth/Small-9B-GGUF", "unsloth/Big-Coder-80B-A3B-GGUF",
                                         "unsloth/Dense-70B-GGUF", "unsloth/Tiny-1B-GGUF"]
    small = got[0]
    assert small["license"] == "apache-2.0" and small["vision"] and small["ctx"] == 131072
    assert {f["path"] for f in small["files"]} == {"Small-9B-UD-Q4_K_XL.gguf", "Small-9B-Q8_0.gguf",
                                                     "mmproj-F32.gguf", "mmproj-F16.gguf"}
    assert hf.base_name("bartowski/Qwen_Qwen3.8-27B-GGUF") == hf.base_name("unsloth/Qwen3.8-27B-GGUF") == "qwen3.8-27b"


def test_quant_names():
    assert hf.quant("Small-9B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert hf.quant("Org_Small-9B-Q4_K_M.gguf") == "Q4_K_M"
    assert hf.quant("UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00002.gguf") == "UD-Q4_K_XL"
    assert hf.quant("gpt-oss-120b-MXFP4_MOE.gguf") == "MXFP4_MOE"
    assert hf.quant("model.gguf") == ""


def test_picks_fit_the_machine():
    c = cands()
    by = {s.key: s for s in hf.picks(c, {}, [8.0, 8.0], 64, known=())}
    small = by["hf-small-9b"]
    assert small.model.name == "Small-9B-UD-Q4_K_XL.gguf" and small.mmproj.name == "mmproj-F16.gguf"
    assert small.model.url == "https://huggingface.co/unsloth/Small-9B-GGUF/resolve/main/Small-9B-UD-Q4_K_XL.gguf"
    assert "Trending on Hugging Face" in small.blurb and small.license == "apache-2.0"
    big = by["hf-big-coder-80b-a3b"]                      # MoE: experts in RAM, split file kept whole
    assert big.moe and "coder" in big.uses and big.model.name.endswith("-00001-of-00002.gguf") and len(big.shards) == 1
    assert ml.fit(big, [8.0, 8.0], 64).ok
    assert "hf-dense-70b" not in by                      # dense and too big for 16 GB of GPUs: not offered
    assert by["hf-tiny-1b"].model.name == "Tiny-1B-Q8_0.gguf"     # small: 8-bit
    # with less RAM the MoE model's 4-bit file no longer fits, its 2-bit one does
    lean = {s.key: s for s in hf.picks(c, {}, [8.0, 8.0], 40, known=())}
    assert lean["hf-big-coder-80b-a3b"].model.name == "Big-Coder-80B-A3B-UD-Q2_K_XL.gguf"
    # big GPUs: the dense 70B fits at 4 bits
    roomy = {s.key: s for s in hf.picks(c, {}, [48.0, 48.0], 128, known=())}
    assert roomy["hf-dense-70b"].model.name == "Dense-70B-Q4_K_M.gguf"


def test_picks_skip_built_in_models_and_keep_added_ones():
    c = cands()
    known = (ml.Suggestion(key="small", name="Small", blurb="", uses=("chat",), page="https://huggingface.co/unsloth/Small-9B-GGUF",
                           model=ml.File("x.gguf", "https://huggingface.co/unsloth/Small-9B-GGUF/resolve/main/x.gguf", 6)),)
    keys = [s.key for s in hf.picks(c, {}, [8.0, 8.0], 64, known=known)]
    assert "hf-small-9b" not in keys
    kept = {"hf-dense-70b": {"cand": c[2], "quant": "Q4_K_M"}}            # added earlier on a bigger machine
    keys = [s.key for s in hf.picks([], kept, [8.0], 16, known=())]
    assert keys == ["hf-dense-70b"]
    assert len(hf.picks(c, {}, [80.0] * 4, 512, known=(), shown=2)) == 2


def test_library_caches_and_survives_offline(tmp_path):
    calls = []

    def fetcher():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("offline")
        return cands()
    lib = hf.Library(tmp_path, fetcher)
    assert lib.due() and lib.state() == ([], {})
    assert lib.refresh() == 4 and not lib.due()
    assert lib.refresh() == 0 and len(lib.state()[0]) == 4      # offline: the last list stays
    s = hf.picks(lib.state()[0], {}, [8.0, 8.0], 64, known=())[0]
    lib.keep(s)
    again = hf.Library(tmp_path)
    models, kept = again.state()
    assert len(models) == 4 and kept[s.key]["quant"] == hf.quant(s.model.name)
    assert json.loads((tmp_path / "hf-models.json").read_text())["checked"] > 0


def test_models_page_install_and_setting(tmp_path, tmp_path_factory, monkeypatch):
    import shutil
    import time
    from pathlib import Path
    from fastapi.testclient import TestClient
    from gpushepherd import auth, gpu as gpu_mod, host
    from gpushepherd.app import create_app
    from gpushepherd.backend import FakeBackend
    from gpushepherd.config import Store
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    monkeypatch.setattr(host, "ram_gb", lambda: 64.0)
    shutil.copytree(Path(__file__).parent / "fixtures" / "stack", tmp_path, dirs_exist_ok=True)
    backend = FakeBackend()
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=False)
    svc = app.state.svc
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(tmp_path))}"})
    assert not any(r["hf"] for r in client.get("/api/v1/instances/llm/suggestions").json())   # not asked yet
    svc.hf = hf.Library(tmp_path, cands)
    svc.hf.refresh()
    rows = {r["key"]: r for r in client.get("/api/v1/instances/llm/suggestions").json()}
    assert rows["hf-big-coder-80b-a3b"]["hf"] and rows["hf-big-coder-80b-a3b"]["fits"]
    assert "hf-dense-70b" not in rows and not rows["qwen3.5-9b"]["hf"]
    # installing a model server with one picks up its settings and downloads it
    models = tmp_path_factory.mktemp("models")
    r = client.post("/api/v1/apps/llm-server/install", json={
        "id": "llm2", "port": 8091, "models_dir": str(models),
        "chat_file": "/models/hf-small-9b/Small-9B-UD-Q4_K_XL.gguf"})
    assert r.status_code == 201 and r.json()["downloads"] == ["Small-9B"]
    for _ in range(100):
        if "shepherd-download-hf-small-9b" in backend.helper_runs:
            break
        time.sleep(0.01)
    assert "shepherd-download-hf-small-9b" in backend.helper_runs
    assert "hf-small-9b" in svc.hf.state()[1]                   # kept: stays listed after it stops trending
    # the setting turns the trending list off; kept models stay
    r = client.patch("/api/v1/settings", json={"hf_suggestions": False})
    assert r.status_code == 200 and r.json()["settings"]["hf_suggestions"] is False
    keys = {r["key"] for r in client.get("/api/v1/instances/llm/suggestions").json()}
    assert "hf-small-9b" in keys and "hf-big-coder-80b-a3b" not in keys
    assert client.patch("/api/v1/settings", json={"hf_suggestions": "yes"}).status_code == 400
