"""Fixes from the November 2026 review: a helper that a running app needs says why it won't stop, a
link to something that no longer exists gives a page (not a server error), and a container Docker
keeps restarting still counts as holding its GPUs."""
import test_app
from test_app import HX, login
from test_docker_backend import make_backend

from gpushepherd.backend import LABEL_GPUS, LABEL_INSTANCE

stack = test_app.stack          # the same app fixture


def test_stopping_a_helper_a_running_app_needs_says_why(stack):
    app, client, backend, data = stack
    token = {"Authorization": "Bearer " + (data / "api-token").read_text().strip()}
    for iid, extra in (("chat-db", {}), ("chat", {"requires": ["chat-db"]})):
        r = client.put(f"/api/v1/instances/{iid}", headers=token,
                       json={"name": iid.title(), "image": f"x/{iid}:1", "gpu": {"mode": "none"}} | extra)
        assert r.status_code in (200, 201), r.text
    assert client.post("/api/v1/instances/chat/start", headers=token, json={}).status_code == 202
    test_app.wait(app)
    states = backend.states()
    assert states["chat"].running and states["chat-db"].running
    r = client.post("/api/v1/instances/chat-db/stop", headers=token, json={})
    assert r.status_code == 409 and "needed by Chat" in r.json()["error"]
    assert backend.states()["chat-db"].running


def test_an_unknown_instance_is_not_a_server_error(stack):
    app, client, backend, data = stack
    login(client)
    r = client.get("/ui/instances/nope/logs")
    assert r.status_code == 404 and "nope" in r.text and "<html" in r.text.lower()
    r = client.post("/ui/instances/nope/stop", headers=HX, data={})
    assert r.status_code < 500


def test_a_restarting_or_paused_container_still_holds_its_gpus():
    b = make_backend()
    b.api.containers = lambda all, filters: [
        {"Names": [f"/shepherd-{s}"], "State": s, "Labels": {LABEL_INSTANCE: s, LABEL_GPUS: "0"}}
        for s in ("running", "restarting", "paused", "exited", "created")]
    states = b.states()
    assert [s for s in states if states[s].running] == ["running", "restarting", "paused"]
