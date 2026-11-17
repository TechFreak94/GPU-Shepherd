"""The MCP server at /api/v1/mcp (mcp.py) and the agent plugin that points at it (agent-plugin/)."""
import json
import re
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Match

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, mcp
from gpushepherd.app import DemoLlamaClient, create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

ROOT = Path(__file__).parent.parent
EXAMPLE = Path(__file__).parent / "fixtures" / "stack"
PLUGIN = ROOT / "agent-plugin"


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    DemoLlamaClient.loaded.clear()
    token = auth.api_token(Store(tmp_path))
    return app, TestClient(app, headers={"Authorization": f"Bearer {token}"})


def rpc(client, method, params=None, rid=1):
    r = client.post("/api/v1/mcp", json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
    assert r.status_code == 200, r.text
    return r.json()


def call(client, name, **args):
    res = rpc(client, "tools/call", {"name": name, "arguments": args})["result"]
    return json.loads(res["content"][0]["text"]) if res["content"][0]["text"][:1] in "[{" else res["content"][0]["text"], res["isError"]


def wait(app):
    for _ in range(500):
        if not app.state.jobs.busy:
            return
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_needs_the_api_token(stack):
    app, client = stack
    anon = TestClient(app)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert anon.post("/api/v1/mcp", json=body).status_code == 401
    assert anon.post("/api/v1/mcp", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    # a logged-in browser may read the API, but not use the MCP server (it is a POST)
    anon.post("/login", data={"user": "admin", "password": "correct-horse"})
    assert anon.post("/api/v1/mcp", json=body).status_code == 401
    assert client.post("/api/v1/mcp", json=body).status_code == 200


def test_handshake(stack):
    _, client = stack
    res = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "1"}})["result"]
    assert res["protocolVersion"] == "2025-06-18"
    assert res["serverInfo"]["name"] == "gpu-shepherd" and "tools" in res["capabilities"]
    assert "confirm=true" in res["instructions"]
    assert rpc(client, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == mcp.PROTOCOLS[0]
    # notifications get no answer
    r = client.post("/api/v1/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202 and not r.content
    assert rpc(client, "ping")["result"] == {}
    assert rpc(client, "nope")["error"]["code"] == -32601
    assert client.post("/api/v1/mcp", content=b"{nope").json()["error"]["code"] == -32700
    assert client.get("/api/v1/mcp").status_code == 405
    batch = client.post("/api/v1/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                             {"jsonrpc": "2.0", "method": "notifications/initialized"}]).json()
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def sample(tool):
    args = {}
    for k, prop in tool.schema["properties"].items():
        args[k] = {"integer": 1, "boolean": False, "object": {}}.get(prop.get("type"), "x")
    return args


def test_every_tool_reaches_a_real_endpoint(stack):
    """The contract: each tool is a fixed call to an endpoint that exists, and the risky things stay out."""
    app, client = stack
    listed = rpc(client, "tools/list")["result"]["tools"]
    assert [t["name"] for t in listed] == [t.name for t in mcp.TOOLS]
    assert len({t.name for t in mcp.TOOLS}) == len(mcp.TOOLS)
    for t in listed:
        assert t["inputSchema"]["type"] == "object" and t["description"]
        assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])
    for tool in mcp.TOOLS:
        method, path, _ = tool.call(sample(tool))
        scope = {"type": "http", "path": path, "method": method}
        assert any(r.matches(scope)[0] == Match.FULL for r in app.router.routes), (tool.name, method, path)
        assert tool.read_only == (method == "GET"), tool.name
        assert method != "DELETE", tool.name
        for kept_out in ("password", "token", "power", "tailscale", "gpu-unlock", "trash", "storage/", "moves",
                         "credentials", "branch", "settings", "knowledge"):
            assert kept_out not in path, (tool.name, path)


def test_ids_stay_in_their_place():
    with pytest.raises(ValueError):
        mcp.seg("..")
    assert mcp.seg("../settings/password") == "..%2Fsettings%2Fpassword"


def test_look_then_start_with_confirm(stack):
    app, client = stack
    status, err = call(client, "get_status")
    assert not err and status["gpus"][0]["name"] == "Test GPU" and "llm" in status["instances"]
    insts, _ = call(client, "list_instances")
    assert {i["id"] for i in insts} >= {"llm", "comfyui"}

    started, err = call(client, "start_instance", id="llm")
    assert not err and started["job"]["title"] == "Starting Llama Swap"
    wait(app)
    job, _ = call(client, "get_job", job_id=started["job"]["id"])
    assert job["status"] == "done" and job["log"]

    # it would restart Llama Swap: nothing happens until the user agrees
    plan, err = call(client, "start_instance", id="comfyui")
    assert not err and plan["needs_confirm"] and any("Restart Llama Swap" in x for x in plan["plan"])
    assert app.state.jobs.current is None
    done, err = call(client, "start_instance", id="comfyui", confirm=True)
    assert not err and "job" in done
    wait(app)


def test_errors_and_logs(stack):
    _, client = stack
    answer, err = call(client, "get_instance", id="nope")
    assert err and answer["status"] == 404
    answer, err = call(client, "get_instance")
    assert err and "missing: id" in answer
    assert "error" in rpc(client, "tools/call", {"name": "delete_everything", "arguments": {}})
    assert mcp.tail_logs({"logs": "a\nb\nc"}, {"lines": 2}) == {"logs": "b\nc", "lines_shown": 2, "lines_total": 3}


def test_plugin_points_at_the_server():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    server = servers["gpu-shepherd"]
    assert server["type"] == "http" and server["url"].endswith("/api/v1/mcp")
    for ref in ("url", "api_token"):
        assert f"${{user_config.{ref}}}" in json.dumps(server)
        assert ref in manifest["userConfig"]
    assert manifest["userConfig"]["api_token"]["sensitive"] is True
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    entry = next(p for p in market["plugins"] if p["name"] == manifest["name"])
    assert (ROOT / entry["source"]).resolve() == PLUGIN.resolve()
    skill = (PLUGIN / "skills" / "gpu-shepherd" / "SKILL.md").read_text()
    named = set(re.findall(r"`([a-z]+_[a-z_]+)`", skill))
    verbs = ("get_", "list_", "start_", "stop_", "restart_", "cancel_", "install_", "add_", "load_", "unload_",
             "apply_", "check_", "update_")
    assert {"get_status", "start_instance"} <= named
    assert {n for n in named if n.startswith(verbs)} <= {t.name for t in mcp.TOOLS}
