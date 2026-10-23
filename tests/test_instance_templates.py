"""Guided setup: the built-in templates and what they build."""
import pytest

from gpushepherd import instance_templates as tpl
from gpushepherd.config import ConfigError, Instance, PortMap


def machine(**kw):
    base = dict(gpu_mem_gb=[8.0, 8.0], instances={}, servers={}, model_sizes={})
    return tpl.Machine(**(base | kw))


def test_templates_are_generic_and_valid():
    keys = [t.key for t in tpl.TEMPLATES]
    assert len(keys) == len(set(keys))
    m = machine()
    for t in tpl.TEMPLATES:
        f = tpl.defaults(t, m, "/srv/models", "/opt/gpu-shepherd/appdata")
        inst, _ = tpl.build(t, f, m)
        assert inst.id == t.default_id and inst.ports[0].host == t.host_port
        assert all(v.host.startswith("/") for v in inst.volumes)


def test_defaults_avoid_taken_ids_and_ports():
    llm = Instance(id="llm", name="LLM", image="x", ports=[PortMap(host=8080, container=8080)])
    f = tpl.defaults(tpl.get("llm-chat"), machine(instances={"llm": llm}), "/m", "/a")
    assert f["id"] == "llm-2" and f["port"] == 8081 and f["appdata_dir"] == "/a/llm-2"


def test_gpu_choices():
    t, m = tpl.get("comfyui"), machine()
    f = tpl.defaults(t, m, "/m", "/a")
    assert tpl.build(t, f, m)[0].gpu.model_dump(include={"mode", "min_gpus", "max_gpus"}) == \
        {"mode": "elastic", "min_gpus": 1, "max_gpus": 1}
    assert tpl.build(t, f | {"gpu": "all"}, m)[0].gpu.max_gpus == 0
    pinned = tpl.build(t, f | {"gpu": "pin:1"}, m)[0].gpu
    assert pinned.mode == "fixed" and pinned.gpus == [1]
    assert tpl.build(t, f | {"gpu": "none"}, m)[0].gpu.mode == "none"


def test_comfyui_folders():
    t, m = tpl.get("comfyui"), machine()
    inst, model = tpl.build(t, tpl.defaults(t, m, "/srv/models/", "/data/appdata"), m)
    vols = {v.container: v.host for v in inst.volumes}
    assert vols["/root/ComfyUI/models"] == "/srv/models/comfyui"
    assert vols["/root/ComfyUI/output"] == "/data/appdata/comfyui/output"
    assert model is None and inst.gpu.priority == 60


def test_open_webui_connects_to_server_and_comfyui():
    llm = Instance(id="llm", name="LLM", image="x", kind="llama-swap", ports=[PortMap(host=8090, container=8080)])
    comfy = Instance(id="comfy", name="C", image="yanwk/comfyui-boot", ports=[PortMap(host=8188, container=8188)])
    m = machine(instances={"llm": llm, "comfy": comfy},
                servers={"llm": {"api_key": "sk-abc", "name": "LLM", "default_model": "chat"}})
    t = tpl.get("open-webui")
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["server"] == "llm" and f["comfyui"] == "comfy"
    env = tpl.build(t, f, m)[0].env
    assert env["OPENAI_API_BASE_URLS"] == "http://llm:8080/v1" and env["OPENAI_API_KEYS"] == "sk-abc"
    assert env["DEFAULT_MODELS"] == "chat" and env["COMFYUI_BASE_URL"] == "http://comfy:8188"
    bare = tpl.build(t, f | {"server": "", "comfyui": ""}, m)[0].env
    assert "OPENAI_API_KEYS" not in bare and "COMFYUI_BASE_URL" not in bare
    with pytest.raises(ConfigError):
        tpl.build(t, f | {"server": "nope"}, m)


def test_first_model():
    t = tpl.get("llm-coder")
    m = machine(model_sizes={"/models/big/Coder-Q4.gguf": 10.0})
    f = tpl.defaults(t, m, "/m", "/a") | {"file": "/models/big/Coder-Q4.gguf"}
    _, model = tpl.build(t, f, m)
    assert model.id == "coder-q4" and model.aliases == ["coder"] and model.ctx_size == 65536
    # 10 GB needs more than one 8 GB GPU but fits on two
    assert [(r.min_gpus, r.cpu_moe) for r in model.offload] == [(1, 999), (2, 0)]
    assert tpl.build(t, f | {"offload": "gpu"}, m)[1].offload == []
    assert [(r.min_gpus, r.cpu_moe) for r in tpl.build(t, f | {"offload": "moe"}, m)[1].offload] == [(1, 999)]
    assert tpl.build(t, f | {"file": ""}, m)[1] is None
    assert tpl.build(t, f | {"file": "x-00001-of-00003.gguf"}, m)[1].file == "/models/x-00001-of-00003.gguf"


def test_auto_offload():
    assert tpl.auto_offload(4.0, [8.0, 8.0]) == []                       # fits one GPU
    assert [r.cpu_moe for r in tpl.auto_offload(40.0, [8.0, 8.0])] == [999]
    assert tpl.auto_offload(None, [8.0]) == [] and tpl.auto_offload(5.0, []) == []
    assert tpl.model_id_from_file("/m/Qwen3-80B-Q4_K_M-00001-of-00003.gguf") == "qwen3-80b-q4_k_m"


def test_bad_answers():
    t, m = tpl.get("llm-chat"), machine()
    f = tpl.defaults(t, m, "/m", "/a")
    for bad, msg in (({"id": "Bad Id"}, "ID"), ({"port": "x"}, "Port"), ({"models_dir": "models"}, "Models folder"),
                     ({"file": "/models/a.gguf", "ctx_size": "lots"}, "Context")):
        with pytest.raises(ConfigError, match=msg):
            tpl.build(t, f | bad, m)
