"""App updates: instances installed before an app changed get what it sets up today."""
import yaml

from gpushepherd import app_updates as au, instance_templates as tpl
from gpushepherd.config import Catalog, CatalogModel, Instance, PortMap, VolumeMap

from test_api import run, stack, wait  # noqa: F401 - fixtures


def machine(embed="embed"):
    llm = Instance(id="llm", name="LLM", image="x", kind="llama-swap", ports=[PortMap(host=8090, container=8080)])
    info = {"api_key": "sk-abc", "name": "LLM", "default_model": "chat", "ctx": 32768, "embed": embed}
    return tpl.Machine(gpu_mem_gb=[8.0], instances={"llm": llm}, servers={"llm": info}, model_sizes={})


def test_open_webui_gets_the_embeddings_settings_it_lacks():
    m = machine()
    webui = Instance(id="webui", name="Open WebUI", image="ghcr.io/open-webui/open-webui:main", app="open-webui",
                     env={"OPENAI_API_BASE_URLS": "http://llm:8080/v1", "OPENAI_API_KEYS": "sk-abc",
                          "DEFAULT_MODELS": "my-model", "ENABLE_OLLAMA_API": "false"})
    [u] = au.for_instance(webui, m)
    assert u.kind == "env" and set(u.env) == {"RAG_EMBEDDING_ENGINE", "RAG_OPENAI_API_BASE_URL",
                                              "RAG_OPENAI_API_KEY", "RAG_EMBEDDING_MODEL"}
    assert "Admin Panel" in u.manual[0] and not any("sk-abc" in x for x in u.lines)   # keys aren't shown
    assert au.for_instance(webui.model_copy(update={"env": webui.env | u.env}), m) == []
    assert au.for_instance(webui, machine(embed="")) == []          # nothing to point at yet
    assert au.for_instance(webui.model_copy(update={"skipped_updates": [u.key]}), m) == []
    # a value the user changed is never touched, and an unconnected app gets no connection
    alone = Instance(id="w2", name="W", image="ghcr.io/open-webui/open-webui:main", env={"ENABLE_OLLAMA_API": "true"})
    assert au.for_instance(alone, m) == []


def test_anythingllm_settings_file_switches_from_the_built_in_embedder():
    m = machine()
    inst = Instance(id="allm", name="AnythingLLM", image="mintplexlabs/anythingllm:latest", app="anythingllm",
                    env=dict(tpl.get("anythingllm").env), volumes=[VolumeMap(host="/d/server.env", container="/app/server/.env")])
    old = tpl._anythingllm_env("http://llm:8080/v1", {"api_key": "sk-abc", "default_model": "chat", "ctx": 8192})
    [u] = au.for_instance(inst, m, settings_file=old)
    assert u.kind == "file" and "EMBEDDING_ENGINE='native'" not in u.file_text
    assert "EMBEDDING_ENGINE='generic-openai'" in u.file_text and "LLM_PROVIDER='generic-openai'" in u.file_text
    assert au.for_instance(inst, m, settings_file=u.file_text) == []
    picked = old.replace("'native'", "'ollama'")                    # chosen in its web page: left alone
    assert au.for_instance(inst, m, settings_file=picked) == []


def test_model_server_gets_an_embeddings_model_and_suggested_settings():
    m = machine()
    srv = m.instances["llm"]
    old = CatalogModel(id="coder", file="/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf", sampling="--temp 1.0")
    cat = Catalog(instance_id="llm", models=[old, CatalogModel(id="other", file="/models/mine.gguf")])
    ups = {u.kind: u for u in au.for_instance(srv, m, cat)}
    assert set(ups) <= {"embed-model", "model"} and "embed-model" in ups
    if "model" in ups:                                              # only fields still at their defaults
        assert ups["model"].model.sampling == "--temp 1.0" and ups["model"].model.id == "coder"
    cat.models.append(CatalogModel(id="e", file="/models/e.gguf", role="embed"))
    assert all(u.kind != "embed-model" for u in au.for_instance(srv, m, cat))


def test_apply_and_skip_from_the_api_and_cli(run, stack):  # noqa: F811
    app, client, _, data = stack
    rows = client.get("/api/v1/app-updates").json()
    assert [r["key"] for r in rows if r["instance"] == "llm"] == ["embed-model"]
    assert not any(r["instance"] == "webui" for r in rows)          # its server has no embeddings model yet
    r = client.post("/api/v1/app-updates/llm/embed-model/apply")
    assert r.status_code == 200 and "Added qwen3-embedding-0.6b" in r.json()["message"]
    web = [r for r in client.get("/api/v1/app-updates").json() if r["instance"] == "webui"]
    assert len(web) == 1 and "Admin Panel" in web[0]["afterwards"][0]
    code, text = run("app-updates", "skip", "webui", web[0]["key"])
    assert code == 0 and "Skipped" in text
    assert yaml.safe_load((data / "instances" / "webui.yaml").read_text())["skipped_updates"] == [web[0]["key"]]
    code, text = run("app-updates")
    assert code == 0 and "webui" not in text
    assert client.post("/api/v1/app-updates/webui/nope/apply").status_code == 404
    code, text = run("app-updates", "apply", "--all")
    assert code == 0 and client.get("/api/v1/app-updates").json() == []


def test_open_webui_book_settings_and_query_prefix():
    m = machine()
    m.servers["llm"]["embed_query_prefix"] = tpl.query_prefix("/models/Qwen3-Embedding-0.6B-Q8_0.gguf")
    assert m.servers["llm"]["embed_query_prefix"].startswith("Instruct: ") and tpl.query_prefix("nomic.gguf") == ""
    t = tpl.get("open-webui")
    env = tpl.build(t, tpl.defaults(t, m, "/m", "/a"), m)[0].env
    assert env["CHUNK_SIZE"] == "1500" and env["RAG_TOP_K"] == "8" and env["RAG_EMBEDDING_QUERY_PREFIX"].endswith("Query: ")
    # an existing install isn't offered them: Open WebUI only reads them at its first start
    webui = Instance(id="webui", name="Open WebUI", image="ghcr.io/open-webui/open-webui:main", app="open-webui",
                     env={k: v for k, v in env.items() if k not in tpl.OPENWEBUI_DOCS})
    assert au.for_instance(webui, m) == []


def test_apply_all_follows_on_to_the_updates_it_brings(stack):  # noqa: F811
    app, client, _, _ = stack
    from fastapi.testclient import TestClient
    b = TestClient(app)
    b.post("/login", data={"user": "admin", "password": "correct-horse"})
    r = b.post("/ui/app-updates/apply-all", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "Applied" in r.headers["HX-Trigger"]
    assert client.get("/api/v1/app-updates").json() == []         # the server's model, then the chat app's settings
    assert "up to date" in client.post("/api/v1/app-updates/apply-all").json()["message"]


def comfy_machine(tmp_path, checkpoint="sd_xl_base_1.0.safetensors"):
    m = machine()
    m.instances["comfyui"] = Instance(id="comfyui", name="ComfyUI", image="yanwk/comfyui-boot:cu130-slim-v2",
                                      ports=[PortMap(host=8188, container=8188)])
    m.checkpoints["comfyui"] = [checkpoint] if checkpoint else []
    return m


def test_open_webui_is_offered_pictures_with_comfyui(tmp_path):
    m = comfy_machine(tmp_path)
    base = {"OPENAI_API_BASE_URLS": "http://llm:8080/v1", "OPENAI_API_KEYS": "sk-abc", "DEFAULT_MODELS": "chat",
            "ENABLE_OLLAMA_API": "false", **{k: "x" for k in ("RAG_EMBEDDING_ENGINE", "RAG_OPENAI_API_BASE_URL",
                                                                "RAG_OPENAI_API_KEY", "RAG_EMBEDDING_MODEL")}}
    webui = Instance(id="webui", name="Open WebUI", image="ghcr.io/open-webui/open-webui:main", env=base)
    [u] = au.for_instance(webui, m)                       # not connected yet: offered
    assert u.kind == "images" and u.title.startswith("Make and edit pictures")
    assert u.env["COMFYUI_BASE_URL"] == "http://comfyui:8188" and u.env["IMAGE_EDIT_ENGINE"] == "comfyui"
    assert u.env["IMAGE_GENERATION_MODEL"] == "sd_xl_base_1.0.safetensors" and not u.manual
    # an install from before the workflows only has the address: finished, nothing it has is changed
    old = webui.model_copy(update={"env": base | {"ENABLE_IMAGE_GENERATION": "true", "IMAGE_GENERATION_ENGINE":
                                                  "comfyui", "COMFYUI_BASE_URL": "http://comfyui:8188",
                                                  "IMAGE_SIZE": "768x768"}})
    [u] = au.for_instance(old, m)
    assert u.title.startswith("Finish connecting") and "IMAGE_SIZE" not in u.env and "COMFYUI_WORKFLOW_NODES" in u.env
    assert au.for_instance(old.model_copy(update={"env": old.env | u.env}), m) == []
    # pictures turned off, or another image server: left alone
    assert au.for_instance(webui.model_copy(update={"env": base | {"ENABLE_IMAGE_GENERATION": "false"}}), m) == []
    assert au.for_instance(webui.model_copy(update={"env": base | {"COMFYUI_BASE_URL": "http://elsewhere:1"}}), m) == []
    # no checkpoint yet: everything but the model, and where to pick one
    [u] = au.for_instance(webui, comfy_machine(tmp_path, ""))
    assert "IMAGE_GENERATION_MODEL" not in u.env and "checkpoints/" in u.manual[0]


def test_applying_the_pictures_update_sets_a_running_open_webui(stack):  # noqa: F811
    app, client, backend, data = stack
    from gpushepherd.connectors import comfyui
    svc = app.state.svc
    sent = []

    class Client:
        def __init__(self, base, token):
            pass

        def use_images(self, values):
            sent.append(values)
    svc.openwebui_client = Client
    backend.exec_reply = lambda name, cmd, env: (0, '{"token": "tok"}')
    ckpt = data / "models" / "comfyui" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "sd_xl_base_1.0.safetensors").write_bytes(b"x")
    f = data / "instances" / "comfyui.yaml"
    f.write_text(f.read_text().replace("/srv/models", str(data / "models")))
    [u] = [u for u in svc.app_updates() if u.iid == "webui" and u.kind == "images"]
    assert set(u.env) == {"IMAGE_GENERATION_MODEL", "IMAGE_EDIT_MODEL"} and u.title.startswith("Finish")
    msg = svc.apply_app_update("webui", u.key)
    assert "Start Open WebUI" in msg and not sent                 # stopped: the variables do it at its start
    env = svc.instance("webui").env
    assert env["COMFYUI_WORKFLOW_NODES"].startswith("[") and env["ENABLE_IMAGE_EDIT"] == "true"
    assert client.post("/api/v1/instances/webui/start").status_code in (200, 202)
    wait(app)
    msg = svc._send_images(svc.instance("webui"))
    assert "makes and edits pictures" in msg, msg
    [values] = sent
    assert values["ENABLE_IMAGE_EDIT"] is True and values["IMAGE_STEPS"] == 25
    assert values["IMAGE_GENERATION_MODEL"] == "sd_xl_base_1.0.safetensors"
    assert values["COMFYUI_WORKFLOW_NODES"] == comfyui.GENERATE_NODES and values["IMAGE_SIZE"] == "1024x1024"


def test_own_comfyui_workflow_replaces_the_built_in_one(tmp_path):
    import json
    from gpushepherd.connectors import comfyui
    own = tmp_path / comfyui.CUSTOM_DIR
    own.mkdir()
    flow = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "my-model.safetensors"}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "Prompt"}},
            "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 9}}}
    nodes = [{"type": "prompt", "key": "text", "node_ids": ["5"]}, {"type": "model", "key": "unet_name", "node_ids": ["1"]},
             {"type": "width", "key": "width", "node_ids": ["7"]}, {"type": "height", "key": "height", "node_ids": ["7"]},
             {"type": "steps", "key": "steps", "node_ids": ["8"]}]
    (own / "generate.json").write_text(json.dumps(flow))
    (own / "generate-nodes.json").write_text(json.dumps(nodes))
    (own / "edit.json").write_text(json.dumps(flow))
    (own / "edit-nodes.json").write_text(json.dumps([{"type": "prompt", "key": "nope", "node_ids": ["5"]}]))
    found = comfyui.custom(tmp_path)
    assert set(found) == {"generate"}                     # the edit map names an input its workflow lacks
    m = comfy_machine(tmp_path, "")
    built_in = tpl.comfyui_env("comfyui", m)              # what an earlier app update set
    webui = Instance(id="webui", name="Open WebUI", image="ghcr.io/open-webui/open-webui:main",
                     env={"OPENAI_API_BASE_URLS": "http://llm:8080/v1", "OPENAI_API_KEYS": "sk-abc",
                          "DEFAULT_MODELS": "chat", "ENABLE_OLLAMA_API": "false",
                          **{k: "x" for k in ("RAG_EMBEDDING_ENGINE", "RAG_OPENAI_API_BASE_URL",
                                              "RAG_OPENAI_API_KEY", "RAG_EMBEDDING_MODEL")}, **built_in})
    assert au.for_instance(webui, m) == []
    m.comfy_workflows["comfyui"] = found
    [u] = au.for_instance(webui, m)
    assert u.title == "Use your own ComfyUI workflow for pictures" and "COMFYUI_BASE_URL" not in u.env
    assert json.loads(u.env["COMFYUI_WORKFLOW"]) == flow and u.env["IMAGE_GENERATION_MODEL"] == "my-model.safetensors"
    assert u.env["IMAGE_SIZE"] == "768x1344" and u.env["IMAGE_STEPS"] == "9" and u.env["ENABLE_IMAGE_EDIT"] == "false"
    assert comfyui.from_env(webui.env | u.env)["ENABLE_IMAGE_EDIT"] is False
    assert au.for_instance(webui.model_copy(update={"env": webui.env | u.env}), m) == []


def test_pictures_button_sends_the_settings_again(stack):  # noqa: F811
    app, client, backend, data = stack
    svc = app.state.svc
    sent = []

    class Client:
        def __init__(self, base, token):
            pass

        def use_images(self, values):
            sent.append(values)
    svc.openwebui_client = Client
    backend.exec_reply = lambda name, cmd, env: (0, '{"token": "tok"}')
    r = client.post("/api/v1/instances/webui/send-images")
    assert r.status_code == 409, r.text                                                # stopped
    assert client.post("/api/v1/instances/llm/send-images").status_code == 400         # not an Open WebUI
    assert client.post("/api/v1/instances/webui/start").status_code in (200, 202)
    wait(app)
    from fastapi.testclient import TestClient
    b = TestClient(app)
    b.post("/login", data={"user": "admin", "password": "correct-horse"})
    assert 'send-images"' in b.get("/ui/dashboard").text
    r = client.post("/api/v1/instances/webui/send-images")
    assert r.status_code == 200 and "picture settings" in r.json()["message"]
    assert sent[0]["COMFYUI_BASE_URL"] == "http://comfyui:8188" and sent[0]["ENABLE_IMAGE_GENERATION"] is True
