"""Routes that wait for Docker run in worker threads, not on the event loop (where they would hold
up every other request), and a page asks Docker once for what it shows many times."""
import asyncio

import test_app
from test_app import HX, login

stack = test_app.stack          # the same app fixture


def on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_api_and_form_routes_stay_off_the_event_loop(stack):
    app, client, backend, data = stack
    seen = []
    states = backend.states

    def watched():
        seen.append(on_event_loop())
        return states()
    backend.states = watched
    token = {"Authorization": "Bearer " + (data / "api-token").read_text().strip()}
    assert client.post("/api/v1/instances/webui/start", headers=token, json={}).status_code == 202
    test_app.wait(app)
    assert client.get("/api/v1/instances/webui", headers=token).status_code == 200
    login(client)
    assert client.post("/ui/instances/webui/stop", headers=HX, data={}).status_code in (200, 204)
    test_app.wait(app)
    assert seen and not any(seen)


def test_the_suggestion_list_asks_docker_for_downloads_once(stack):
    app, client, backend, data = stack
    svc, calls = app.state.svc, []
    helpers = backend.helpers

    def counted(kind):
        calls.append(kind)
        return helpers(kind)
    backend.helpers = counted
    assert len(svc.suggestions("llm")) > 5
    assert len(calls) == 1


def test_a_model_still_downloading_is_not_called_missing(stack, tmp_path):
    app, client, backend, data = stack
    svc = app.state.svc
    inst = svc.store.instance("llm")
    folder = tmp_path / "models-here"
    folder.mkdir()
    inst.volumes = [v for v in inst.volumes if v.container != "/models"] + [
        type(inst.volumes[0])(host=str(folder), container="/models", read_only=True)]
    svc.store.save_instance(inst)
    backend.helper_seconds = 60
    key = "qwen3-embedding-0.6b"
    svc.add_suggestion("llm", key)                      # adds it to the model list and starts its download
    for _ in range(200):                                # the helper starts in the background
        if any(h.running for h in backend.helpers("download").values()):
            break
        test_app.time.sleep(0.01)
    assert key not in svc.missing_files("llm")
    svc.cancel_download("llm", key)
    assert key in svc.missing_files("llm")              # stopped: now it really is missing
