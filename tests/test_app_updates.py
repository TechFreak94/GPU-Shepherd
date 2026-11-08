"""App updates: instances installed before an app changed get what it sets up today."""
import yaml

from gpushepherd import app_updates as au, instance_templates as tpl
from gpushepherd.config import Catalog, CatalogModel, Instance, PortMap, VolumeMap

from test_api import run, stack  # noqa: F401 - fixtures


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
