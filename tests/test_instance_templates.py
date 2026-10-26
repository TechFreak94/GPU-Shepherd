"""The App Store's apps: the built-in templates and what they build."""
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
        if t.hidden:                       # companions are built by their app (test_librechat)
            assert t not in [x for _, g in tpl.groups() for x in g]
            continue
        f = tpl.defaults(t, m, "/srv/models", "/opt/gpu-shepherd/appdata")
        assert f["server"] == (tpl.NEW_SERVER if t.connects else "")   # no server yet: offer to install one
        inst, _ = tpl.build(t, f | {"server": ""}, m)
        assert inst.app == t.key and tpl.matches(t, inst)
        assert inst.id == t.default_id and inst.ports[0].host == t.host_port
        assert all(v.host.startswith("/") for v in inst.volumes)


def test_defaults_avoid_taken_ids_and_ports():
    llm = Instance(id="llm", name="LLM", image="x", ports=[PortMap(host=8080, container=8080)])
    f = tpl.defaults(tpl.get("llm-server"), machine(instances={"llm": llm}), "/m", "/a")
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
    inst, models = tpl.build(t, tpl.defaults(t, m, "/srv/models/", "/data/appdata"), m)
    vols = {v.container: v.host for v in inst.volumes}
    assert vols["/root/ComfyUI/models"] == "/srv/models/comfyui"
    assert vols["/root/ComfyUI/output"] == "/data/appdata/comfyui/output"
    assert models == [] and inst.gpu.priority == 60


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


def test_models_picked_while_installing():
    t = tpl.get("llm-server")
    m = machine(model_sizes={"/models/big/Coder-Q4.gguf": 10.0})
    f = tpl.defaults(t, m, "/m", "/a") | {"coder_file": "/models/big/Coder-Q4.gguf"}
    assert f["chat_ctx"] == 32768 and f["coder_ctx"] == 65536
    [model] = tpl.build(t, f, m)[1]
    assert model.id == "coder-q4" and model.aliases == ["coder"] and model.ctx_size == 65536
    assert "--temp 0.3" in model.sampling
    # 10 GB needs more than one 8 GB GPU but fits on two
    assert [(r.min_gpus, r.cpu_moe) for r in model.offload] == [(1, 999), (2, 0)]
    assert tpl.build(t, f | {"offload": "gpu"}, m)[1][0].offload == []
    assert [(r.min_gpus, r.cpu_moe) for r in tpl.build(t, f | {"offload": "moe"}, m)[1][0].offload] == [(1, 999)]
    assert tpl.build(t, f | {"coder_file": ""}, m)[1] == []
    assert tpl.build(t, f | {"coder_file": "x-00001-of-00003.gguf"}, m)[1][0].file == "/models/x-00001-of-00003.gguf"

    # a chat model and a coding model
    both = tpl.build(t, f | {"chat_file": "Qwen3-8B.gguf", "chat_ctx": "16384"}, m)[1]
    assert [(x.id, x.aliases, x.ctx_size) for x in both] == [("qwen3-8b", ["chat"], 16384), ("coder-q4", ["coder"], 65536)]
    # one file for both: one model that answers to both names, with the longer context
    [one] = tpl.build(t, f | {"chat_file": "/models/big/Coder-Q4.gguf"}, m)[1]
    assert one.aliases == ["chat", "coder"] and one.ctx_size == 65536
    # two files with the same name in different folders get different ids
    two = tpl.build(t, f | {"chat_file": "/models/a/M.gguf", "coder_file": "/models/b/M.gguf"}, m)[1]
    assert [x.id for x in two] == ["m", "m-2"]


def test_renamed_apps():
    """The chat and coding model servers are one LLM server now; old keys and instances still find it."""
    assert tpl.get("llm-chat") is tpl.get("llm-coder") is tpl.get("llm-server")
    assert [x.key for x in tpl.TEMPLATES if x.kind == "llama-swap"] == ["llm-server"]


def test_auto_offload():
    assert tpl.auto_offload(4.0, [8.0, 8.0]) == []                       # fits one GPU
    assert [r.cpu_moe for r in tpl.auto_offload(40.0, [8.0, 8.0])] == [999]
    assert tpl.auto_offload(None, [8.0]) == [] and tpl.auto_offload(5.0, []) == []
    assert tpl.model_id_from_file("/m/Qwen3-80B-Q4_K_M-00001-of-00003.gguf") == "qwen3-80b-q4_k_m"


def test_bad_answers():
    t, m = tpl.get("llm-server"), machine()
    f = tpl.defaults(t, m, "/m", "/a")
    for bad, msg in (({"id": "Bad Id"}, "ID"), ({"port": "x"}, "Port"), ({"models_dir": "models"}, "Models folder"),
                     ({"coder_file": "/models/a.gguf", "coder_ctx": "lots"}, "Coding model: enter the context")):
        with pytest.raises(ConfigError, match=msg):
            tpl.build(t, f | bad, m)


def test_hermes():
    llm = Instance(id="llm", name="LLM", image="x", kind="llama-swap", ports=[PortMap(host=8090, container=8080)])
    m = machine(instances={"llm": llm}, servers={"llm": {"api_key": "sk-abc", "name": "LLM", "default_model": "coder"}})
    t = tpl.get("hermes")
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["server"] == "llm" and f["login_user"] == "admin" and len(f["password"]) >= 12
    inst, _ = tpl.build(t, f, m)
    assert inst.command == ["gateway", "run"] and inst.ports[0].container == 9119 and inst.gpu.mode == "none"
    assert {v.container: v.host for v in inst.volumes} == {"/opt/data": "/a/hermes/data"}
    assert inst.env["OPENAI_BASE_URL"] == "http://llm:8080/v1" and inst.env["HERMES_MODEL"] == "coder"
    assert inst.env["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"] == f["password"] and "HASS_TOKEN" not in inst.env
    seed = tpl.seed_files(t, f, m)["data/config.yaml"]
    assert "provider: custom" in seed and "base_url: http://llm:8080/v1" in seed and "default: coder" in seed
    assert tpl.seed_files(t, f | {"server": ""}, m) == {} and tpl.seed_files(tpl.get("open-webui"), f, m) == {}
    for bad, msg in (({"password": "short"}, "8 characters"), ({"hass_url": "ha.lan"}, "Home Assistant address"),
                     ({"hass_url": "http://ha.lan:8123"}, "token")):
        with pytest.raises(ConfigError, match=msg):
            tpl.build(t, f | bad, m)
    assert tpl.build(t, f | {"hass_token": "t"}, m)[0].env["HASS_URL"] == "http://homeassistant.local:8123"


def test_bundled_server():
    webui = Instance(id="webui", name="Chat UI", image="ghcr.io/open-webui/open-webui:main",
                     ports=[PortMap(host=8080, container=8080)])
    m = machine(instances={"webui": webui})
    t = tpl.get("open-webui")
    f = tpl.defaults(t, m, "/srv/models", "/a") | {"id": "llm", "port": "3000", "chat_file": "Qwen3-8B.gguf"}
    assert tpl.bundled_server(f | {"server": ""}, m, "/a") is None
    srv, cat = tpl.bundled_server(f, m, "/a")
    assert srv.id == "llm-2" and srv.ports[0].host == 8081 and srv.kind == "llama-swap" and srv.app == "llm-server"
    assert cat.instance_id == "llm-2" and cat.models[0].file == "/models/Qwen3-8B.gguf"
    env = tpl.build(t, f | {"server": srv.id}, m)[0].env
    assert env["OPENAI_API_BASE_URLS"] == "http://llm-2:8080/v1" and env["OPENAI_API_KEYS"] == cat.api_key
    assert env["DEFAULT_MODELS"] == "chat"
    with pytest.raises(ConfigError, match="not been set up"):
        tpl.build(t, f, m)


def test_installed_counts_older_instances():
    old = {"a": Instance(id="a", name="A", image="yanwk/comfyui-boot:cu128"),
           "b": Instance(id="b", name="B", image="x", kind="llama-swap", description="my server"),
           "c": Instance(id="c", name="C", image="x", kind="llama-swap", app="llm-coder"),
           "d": Instance(id="d", name="D", image="x", app="hermes")}
    got = {k: [i.id for i in v] for k, v in tpl.installed(old).items()}
    assert got == {"llm-server": ["b", "c"], "comfyui": ["a"], "open-webui": [], "hermes": ["d"],
                   "librechat": [], "anythingllm": []}


def test_kind_tags():
    assert tpl.tag_for(Instance(id="a", name="A", image="x", kind="llama-swap")) == "Model Server"
    assert tpl.tag_for(Instance(id="b", name="B", image="ghcr.io/open-webui/open-webui:v0.6")) == "Chat UI"
    assert tpl.tag_for(Instance(id="c", name="C", image="x", app="hermes")) == "Agent UI"
    assert tpl.tag_for(Instance(id="d", name="D", image="nginx:latest")) == "App"
    assert tpl.tag_for(Instance(id="e", name="E", image="mongo:8.0", app="librechat-mongodb")) == "Database"
    assert all(t.tag for t in tpl.TEMPLATES)


def server_machine(**kw):
    llm = Instance(id="llm", name="LLM", image="x", kind="llama-swap", ports=[PortMap(host=8090, container=8080)])
    return machine(instances={"llm": llm} | kw,
                   servers={"llm": {"api_key": "sk-abc", "name": "LLM", "default_model": "chat", "ctx": 32768}})


def test_librechat():
    m, t = server_machine(), tpl.get("librechat")
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["server"] == "llm" and "appdata_dir" in tpl.answer_keys(t)
    inst, _ = tpl.build(t, f, m)
    assert inst.requires == ["librechat-mongodb"] and inst.ports[0].container == 3080 and inst.gpu.mode == "none"
    env = inst.env
    assert env["MONGO_URI"] == "mongodb://librechat-mongodb:27017/LibreChat"
    assert env["LLM_BASE_URL"] == "http://llm:8080/v1" and env["LLM_API_KEY"] == "sk-abc"
    assert len(env["CREDS_KEY"]) == 64 and len(env["CREDS_IV"]) == 32 and env["JWT_SECRET"] != env["JWT_REFRESH_SECRET"]
    assert env["SESSION_COOKIE_SECURE"] == "false" and env["ALLOW_REGISTRATION"] == "true"
    vols = {v.container: v.host for v in inst.volumes}
    assert vols["/app/librechat.yaml"] == "/a/librechat/librechat.yaml" and vols["/app/uploads"] == "/a/librechat/uploads"
    assert tpl.appdata_dir(t, inst) == "/a/librechat"
    (db,) = tpl.companions(t, inst)
    assert db.id == "librechat-mongodb" and db.app == "librechat-mongodb" and db.gpu.mode == "none" and db.ports == []
    assert db.image.startswith("mongo:") and {v.container: v.host for v in db.volumes} == {"/data/db": "/a/librechat/mongodb"}
    seed = tpl.seed_files(t, f, m)
    assert list(seed) == ["librechat.yaml"] and "${LLM_BASE_URL}" in seed["librechat.yaml"] and "sk-abc" not in seed["librechat.yaml"]
    # no model server: still a database and a settings file, just no endpoint address
    bare = tpl.build(t, f | {"server": ""}, m)[0]
    assert "LLM_BASE_URL" not in bare.env and bare.requires == ["librechat-mongodb"]
    taken = server_machine(**{"librechat-mongodb": Instance(id="librechat-mongodb", name="x", image="mongo")})
    with pytest.raises(ConfigError, match="already exists"):
        tpl.build(t, f, taken)
    with pytest.raises(ConfigError, match="too long"):
        tpl.build(t, f | {"id": "a" * 40}, m)


def test_anythingllm():
    m, t = server_machine(), tpl.get("anythingllm")
    f = tpl.defaults(t, m, "/m", "/a")
    inst, _ = tpl.build(t, f, m)
    assert inst.ports[0].container == 3001 and inst.requires == [] and inst.env["STORAGE_DIR"] == "/app/server/storage"
    assert "--no-sandbox" in inst.env["ANYTHINGLLM_CHROMIUM_ARGS"]
    assert "GENERIC_OPEN_AI_API_KEY" not in inst.env            # the connection is in its own settings file
    vols = {v.container: v.host for v in inst.volumes}
    assert vols == {"/app/server/storage": "/a/anythingllm/storage", "/app/server/.env": "/a/anythingllm/server.env"}
    env = tpl.seed_files(t, f, m)["server.env"]
    assert "LLM_PROVIDER='generic-openai'" in env and "GENERIC_OPEN_AI_BASE_PATH='http://llm:8080/v1'" in env
    assert "GENERIC_OPEN_AI_API_KEY='sk-abc'" in env and "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT='32768'" in env
    assert "LLM_PROVIDER" not in tpl.seed_files(t, f | {"server": ""}, m)["server.env"]
