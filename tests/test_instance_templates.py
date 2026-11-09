"""The App Center's apps: the built-in templates and what they build."""
import base64
import pytest

from gpushepherd import instance_templates as tpl
from gpushepherd.config import ConfigError, Instance, PortMap
from gpushepherd.gpu import Gpu


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
    capped = tpl.build(t, f | {"gpu": "up:3"}, m)[0].gpu
    assert (capped.mode, capped.min_gpus, capped.max_gpus) == ("elastic", 1, 3)
    with pytest.raises(ConfigError):
        tpl.build(t, f | {"gpu": "up:x"}, m)


def test_gpu_choices_grow_with_the_machine():
    def gpus(n):
        return [Gpu(i, "Test GPU", 0, 8192, 0) for i in range(n)]
    assert list(tpl.gpu_choices(gpus(1))) == ["one", "all", "none"]
    assert list(tpl.gpu_choices(gpus(2))) == ["one", "all", "pin:0", "pin:1", "none"]
    four = tpl.gpu_choices(gpus(4))
    assert list(four) == ["one", "up:2", "up:3", "all", "pin:0", "pin:1", "pin:2", "pin:3", "none"]
    assert four["up:3"].startswith("Up to 3 free GPUs") and "8 GB" in four["pin:3"]


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
    """The chat and coding model servers are one Llama Swap now; old keys and instances still find it."""
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
                   "librechat": [], "anythingllm": [], "langflow": [], "ollama": [], "invokeai": [],
                   "speaches": [], "vllm": [], "sillytavern": [], "open-notebook": [], "opencode": []}


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


def test_langflow():
    m, t = server_machine(), tpl.get("langflow")
    assert t.experimental and "OpenAI API Base" in t.after_install
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["login_user"] == "admin" and len(f["password"]) >= 8
    inst, _ = tpl.build(t, f, m)
    assert inst.ports[0].container == 7860 and inst.gpu.mode == "none" and not inst.requires
    env = inst.env
    assert env["LANGFLOW_SUPERUSER"] == "admin" and env["LANGFLOW_SUPERUSER_PASSWORD"] == f["password"]
    assert env["LANGFLOW_AUTO_LOGIN"] == "false" and env["LANGFLOW_DATABASE_URL"] == "sqlite:////app/data/langflow.db"
    assert env["OPENAI_API_KEY"] == "sk-abc" and env["LLM_BASE_URL"] == "http://llm:8080/v1"
    assert env["LANGFLOW_SSRF_ALLOWED_HOSTS"] == "llm" and env["LANGFLOW_PROVIDER_CREDENTIAL_ALLOWED_HOSTS"] == "http://llm"
    assert len(base64.urlsafe_b64decode(env["LANGFLOW_SECRET_KEY"])) == 32          # a Fernet key
    assert {v.container: v.host for v in inst.volumes} == {"/app/data": "/a/langflow/data"}
    assert tpl.seed_files(t, f, m) == {}
    assert tpl.connected_server(inst, m) == "llm"
    want = tpl.app_env(t, "llm", m)
    assert want["LANGFLOW_SSRF_ALLOWED_HOSTS"] == "llm" and "LANGFLOW_SECRET_KEY" not in want
    bare = tpl.build(t, f | {"server": ""}, m)[0].env
    assert "OPENAI_API_KEY" not in bare and "LANGFLOW_SSRF_ALLOWED_HOSTS" not in bare


def test_speaches_gives_open_webui_a_voice():
    m = server_machine()
    sp_t = tpl.get("speaches")
    sp, _ = tpl.build(sp_t, tpl.defaults(sp_t, m, "/m", "/a"), m)
    assert sp.gpu.max_gpus == 1 and len(sp.env["API_KEY"]) >= 24 and tpl.STT_MODEL in sp.env["PRELOAD_MODELS"]
    assert tpl.build(sp_t, tpl.defaults(sp_t, m, "/m", "/a"), m)[0].env["API_KEY"] != sp.env["API_KEY"]
    m.instances["speaches"] = sp
    t = tpl.get("open-webui")
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["speech"] == "speaches" and "speech" in tpl.answer_keys(t)
    env = tpl.build(t, f, m)[0].env
    assert env["AUDIO_STT_OPENAI_API_BASE_URL"] == env["AUDIO_TTS_OPENAI_API_BASE_URL"] == "http://speaches:8000/v1"
    assert env["AUDIO_TTS_OPENAI_API_KEY"] == sp.env["API_KEY"] and env["AUDIO_TTS_VOICE"] == tpl.TTS_VOICE
    assert "AUDIO_STT_ENGINE" not in tpl.build(t, f | {"speech": ""}, m)[0].env
    with pytest.raises(ConfigError, match="Speaches"):
        tpl.build(t, f | {"speech": "llm"}, m)


def test_ollama_and_invokeai():
    m = machine()
    ol = tpl.build(tpl.get("ollama"), tpl.defaults(tpl.get("ollama"), m, "/m", "/a"), m)[0]
    assert ol.kind == "container" and ol.gpu.max_gpus == 0 and ol.ports[0].container == 11434
    assert ol.gpu.cpu_fallback and ol.env["OLLAMA_CONTEXT_LENGTH"] == "32768"
    assert {v.container: v.host for v in ol.volumes} == {"/root/.ollama/models": "/m/ollama"}
    inv = tpl.build(tpl.get("invokeai"), tpl.defaults(tpl.get("invokeai"), m, "/m", "/a"), m)[0]
    assert inv.gpu.max_gpus == 1 and inv.gpu.priority == 60
    assert {v.container: v.host for v in inv.volumes} == {"/invokeai": "/a/invokeai/invokeai",
                                                         "/invokeai/models": "/m/invokeai"}
    assert tpl.tag_for(ol) == "Model Server" and tpl.tag_for(inv) == "Image UI"


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


def test_app_name_replaces_a_name_that_only_repeats_the_tag():
    webui = lambda iid, name: Instance(id=iid, name=name, image="ghcr.io/open-webui/open-webui:main")
    assert tpl.app_name(webui("webui", "Chat UI")) == "Open WebUI"
    assert tpl.app_name(webui("webui-2", "Chat UI 2")) == "Open WebUI 2"
    assert tpl.app_name(webui("family", "Family chat")) == "Family chat"     # the user's own name
    assert tpl.app_name(webui("webui", "Chat UI 2")) == "Chat UI 2"
    assert tpl.app_name(Instance(id="x", name="App", image="nginx")) == "App"
    assert tpl.app_name(Instance(id="h", name="Agent UI", image="x", app="hermes")) == "Hermes Agent"
    assert all(t.default_name != t.tag for t in tpl.TEMPLATES)


def test_sillytavern():
    import json
    m, t = server_machine(), tpl.get("sillytavern")
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["login_user"] == "admin" and len(f["password"]) >= 8
    inst, _ = tpl.build(t, f, m)
    env = inst.env
    assert env["SILLYTAVERN_BASICAUTHMODE"] == "true" and env["SILLYTAVERN_WHITELISTMODE"] == "false"
    assert env["SILLYTAVERN_BASICAUTHUSER_USERNAME"] == "admin" and env["SILLYTAVERN_BASICAUTHUSER_PASSWORD"] == f["password"]
    assert env["LLM_BASE_URL"] == "http://llm:8080/v1" and tpl.connected_server(inst, m) == "llm"
    assert {v.container: v.host for v in inst.volumes}["/home/node/app/data"] == "/a/sillytavern/data"
    seed = tpl.seed_files(t, f, m)
    settings = json.loads(seed["data/default-user/settings.json"])
    assert settings["oai_settings"] == {"chat_completion_source": "custom", "custom_url": "http://llm:8080/v1",
                                        "custom_model": "chat"} and settings["power_user"]["auto_connect"]
    assert json.loads(seed["data/default-user/secrets.json"]) == {"api_key_custom": "sk-abc"}
    assert tpl.seed_files(t, f | {"server": ""}, m) == {}          # no server: SillyTavern's own defaults
    assert tpl.app_env(t, "llm", m)["LLM_API_KEY"] == "sk-abc"


def test_opencode():
    import json
    m, t = server_machine(), tpl.get("opencode")
    m.servers["llm"]["models"] = {"chat": 32768, "coder": 65536}
    f = tpl.defaults(t, m, "/m", "/a")
    assert f["login_user"] == "admin" and [s.key for s in tpl.slots(t)] == ["coder"]
    inst, _ = tpl.build(t, f, m)
    env = inst.env
    assert inst.command == ["web", "--hostname", "0.0.0.0", "--port", "4096"] and inst.gpu.mode == "none"
    assert env["OPENCODE_SERVER_USERNAME"] == "admin" and env["OPENCODE_SERVER_PASSWORD"] == f["password"]
    assert env["LLM_BASE_URL"] == "http://llm:8080/v1" and env["LLM_API_KEY"] == "sk-abc"
    assert tpl.connected_server(inst, m) == "llm" and tpl.app_env(t, "llm", m)["LLM_API_KEY"] == "sk-abc"
    assert {v.container: v.host for v in inst.volumes}["/workspace"] == "/a/opencode/workspace"
    config = json.loads(tpl.seed_files(t, f, m)["config/opencode.json"])
    local = config["provider"]["local"]
    assert local["options"] == {"baseURL": "{env:LLM_BASE_URL}", "apiKey": "{env:LLM_API_KEY}"}
    assert local["npm"] == "@ai-sdk/openai-compatible" and set(local["models"]) == {"chat", "coder"}
    assert local["models"]["coder"]["limit"] == {"context": 65536, "output": 16384}
    assert config["model"] == config["small_model"] == "local/coder" and config["autoupdate"] is False
    del m.servers["llm"]["models"]                    # a server without a coding model: its chat model
    assert json.loads(tpl.seed_files(t, f, m)["config/opencode.json"])["model"] == "local/chat"
    assert tpl.seed_files(t, f | {"server": ""}, m) == {}


def test_opencode_bundled_server_gets_the_coding_model():
    m, t = machine(), tpl.get("opencode")
    f = tpl.defaults(t, m, "/srv/models", "/a") | {"coder_file": "Qwen3-Coder.gguf"}
    srv, cat = tpl.bundled_server(f, m, "/a", t.bundled_slot)
    assert cat.models[0].aliases == ["coder"] and cat.models[0].ctx_size == 65536
    assert m.servers[srv.id]["default_model"] == "coder"
    assert tpl.build(t, f | {"server": srv.id}, m)[0].env["LLM_API_KEY"] == cat.api_key


def test_open_notebook():
    m, t = server_machine(), tpl.get("open-notebook")
    f = tpl.defaults(t, m, "/m", "/a")
    assert "login_user" not in tpl.answer_keys(t) and "password" in tpl.answer_keys(t)
    inst, _ = tpl.build(t, f | {"login_user": ""}, m)
    env = inst.env
    assert inst.requires == ["notebook-surrealdb"] and env["SURREAL_URL"] == "ws://notebook-surrealdb:8000/rpc"
    assert env["OPEN_NOTEBOOK_PASSWORD"] == f["password"] and len(env["OPEN_NOTEBOOK_ENCRYPTION_KEY"]) >= 24
    assert env["OPENAI_COMPATIBLE_BASE_URL"] == "http://llm:8080/v1" and env["OPENAI_COMPATIBLE_API_KEY"] == "sk-abc"
    (db,) = tpl.companions(t, inst)
    assert db.id == "notebook-surrealdb" and db.ports == [] and db.user == "0:0"
    assert db.env == {"SURREAL_USER": "root", "SURREAL_PASS": env["SURREAL_PASS"]}     # the app's password
    assert {v.container: v.host for v in db.volumes} == {"/mydata": "/a/notebook/surrealdb"}
    with pytest.raises(ConfigError, match="password"):
        tpl.build(t, f | {"password": "short"}, m)


def test_vllm():
    def gpus(n):
        return [Gpu(i, "Test GPU", 0, 8192, 0) for i in range(n)]
    m, t = machine(), tpl.get("vllm")
    assert list(tpl.gpu_choices(gpus(1), t)) == ["one"]
    assert list(tpl.gpu_choices(gpus(4), t)) == ["one", "gpus:2", "gpus:4", "pin:0", "pin:1", "pin:2", "pin:3"]
    f = tpl.defaults(t, m, "/srv/models", "/a")
    assert {"hf_model", "hf_token", "max_len"} <= set(tpl.answer_keys(t))
    inst, models = tpl.build(t, f, m)
    assert not models and inst.kind == "container" and inst.shm_size == "8g" and len(inst.env["VLLM_API_KEY"]) >= 24
    assert inst.command[0] == tpl.VLLM_MODEL and inst.command[inst.command.index("--tensor-parallel-size") + 1] == "1"
    assert {v.container: v.host for v in inst.volumes} == {"/root/.cache/huggingface": "/srv/models/huggingface"}
    two, _ = tpl.build(t, f | {"gpu": "gpus:2", "hf_model": "org/Model-AWQ", "max_len": "4096", "hf_token": "hf_x"}, m)
    assert (two.gpu.mode, two.gpu.min_gpus, two.gpu.max_gpus) == ("elastic", 2, 2) and two.env["HF_TOKEN"] == "hf_x"
    assert two.command[two.command.index("--tensor-parallel-size") + 1] == "2" and "4096" in two.command
    for bad in ({"gpu": "all"}, {"gpu": "none"}, {"hf_model": "not a model"}, {"max_len": "lots"}):
        with pytest.raises(ConfigError):
            tpl.build(t, f | bad, m)
    # chat apps can use it as a model server, under the name chat
    info = tpl.vllm_server(two)
    assert info == {"api_key": two.env["VLLM_API_KEY"], "name": "vLLM", "default_model": "chat", "ctx": 4096, "embed": "",
                    "models": {"chat": 4096}}
    m2 = machine(instances={"vllm": two}, servers={"vllm": info})
    web = tpl.build(tpl.get("open-webui"), tpl.defaults(tpl.get("open-webui"), m2, "/m", "/a") | {"server": "vllm"}, m2)[0]
    assert web.env["OPENAI_API_BASE_URLS"] == "http://vllm:8000/v1" and web.env["DEFAULT_MODELS"] == "chat"
    assert tpl.vllm_server(Instance(id="x", name="x", image="x")) is None


def test_every_client_app_can_use_ollama():
    """Ollama is a model server like the others: every app that connects to one can pick it, through
    its OpenAI-compatible API under /v1, asking for the model chat."""
    m = machine()
    ollama, _ = tpl.build(tpl.get("ollama"), tpl.defaults(tpl.get("ollama"), m, "/srv/models", "/a"), m)
    info = tpl.ollama_server(ollama)
    assert info == {"api_key": tpl.OLLAMA_KEY, "name": "Ollama", "default_model": "chat", "ctx": 32768, "embed": "",
                    "models": {"chat": 32768}}
    assert tpl.ollama_server(Instance(id="x", name="x", image="x")) is None
    m = machine(instances={"ollama": ollama}, servers={"ollama": info})
    url = "http://ollama:11434/v1"
    clients = [t for t in tpl.TEMPLATES if t.connects]
    assert {t.key for t in clients} >= {"open-webui", "librechat", "anythingllm", "langflow", "sillytavern",
                                        "open-notebook", "hermes", "opencode"}
    for t in clients:
        assert "Llama Swap server" not in t.needs
        a = tpl.defaults(t, m, "/srv/models", "/a") | {"server": "ollama", "password": "long-enough-pw"}
        inst, _ = tpl.build(t, a, m)
        seeds = tpl.seed_files(t, a, m)
        assert any(url in v for v in [*inst.env.values(), *seeds.values()]), t.key
        assert tpl.connected_server(inst, m, seeds.get("server.env", "")) == "ollama", t.key
