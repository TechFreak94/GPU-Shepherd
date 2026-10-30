"""New releases of the apps' images: checks, the Update button, automatic updates."""
import yaml
from fastapi.testclient import TestClient

from gpushepherd import image_updates as iu
from gpushepherd.config import Instance

from test_api import HX, run, stack, wait  # noqa: F401 - fixtures

WEBUI = "ghcr.io/open-webui/open-webui:main"
NEW = "sha256:" + "a" * 64


def browser(app) -> TestClient:
    b = TestClient(app)
    b.post("/login", data={"user": "admin", "password": "correct-horse"})
    return b


def test_status_compares_the_registry_with_this_machine():
    inst = Instance(id="w", name="W", image=WEBUI)
    rec = {"id": "sha256:old", "local": ["sha256:1"], "remote": "sha256:1", "at": 100}
    assert iu.status(inst, "sha256:old", True, rec).state == "current"
    assert iu.status(inst, "sha256:old", True, rec | {"remote": "sha256:2"}).available
    # downloaded, but the container still runs the old image
    assert iu.status(inst, "sha256:older", True, rec).state == "restart"
    assert iu.status(inst, "sha256:older", False, rec).state == "current"      # its next start uses it
    assert iu.status(inst, "", True, None).state == "unknown"
    s = iu.status(inst, "", True, rec | {"error": "denied"})
    assert s.state == "unknown" and s.error == "denied"
    assert iu.status(inst, "", True, rec, updated=1000, now=2000).updated == 1000
    assert iu.status(inst, "", True, rec, updated=1000, now=1000 + iu.RECENT + 1).updated == 0
    assert iu.digests(["a/b@sha256:1", "c@sha256:1", "bad"]) == ["sha256:1"]
    assert iu.ago(0, now=7200) == "2 hours ago" and iu.ago(10, now=20) == "just now"


def test_check_update_and_dashboard_notice(stack, run):  # noqa: F811
    app, client, backend, data = stack
    svc = app.state.svc
    client.post("/api/v1/instances/webui/start")
    wait(app)
    assert "newest" in client.post("/api/v1/image-updates/check").json()["message"]
    backend.remote[WEBUI] = NEW
    backend.registry_error["yanwk/comfyui-boot:cu130-slim-v2"] = "pull access denied"
    msg = client.post("/api/v1/image-updates/check").json()["message"]
    assert "Open WebUI" in msg and "1 image could not be checked" in msg
    rows = {r["instance"]: r for r in client.get("/api/v1/image-updates").json()["instances"]}
    assert rows["webui"]["state"] == "available" and rows["webui"]["auto_update"] is False
    assert rows["comfyui"]["error"] == "pull access denied"
    html = browser(app).get("/ui/dashboard", headers=HX).text
    assert "New version available" in html and "/ui/instances/webui/update-image" in html
    old_id = backend.states()["webui"].image_id

    r = client.post("/api/v1/image-updates/webui/update")
    assert r.status_code == 202
    wait(app)
    job = client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "done", job["log"]
    assert f"pull {WEBUI}" in backend.calls and any("Restarting" in x or "Starting" in x for x in job["log"])
    assert backend.states()["webui"].image_id != old_id
    st = svc.image_status()["webui"]
    assert st.state == "current" and st.updated
    assert "Updated just now" in browser(app).get("/ui/dashboard", headers=HX).text
    assert client.post("/api/v1/image-updates/webui/update").status_code == 409     # nothing to update

    code, text = run("app-versions")
    assert code == 0 and "webui" in text and "manual" in text


def test_stopped_app_only_downloads(stack):  # noqa: F811
    app, client, backend, _ = stack
    client.post("/api/v1/image-updates/check")
    backend.remote["nousresearch/hermes-agent:stable"] = NEW
    client.post("/api/v1/image-updates/check")
    r = client.post("/api/v1/image-updates/hermes/update")
    wait(app)
    job = client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "done" and any("next start" in x for x in job["log"])
    assert not any(c.startswith("create") for c in backend.calls)


def test_automatic_updates_follow_the_setting(stack, run):  # noqa: F811
    app, client, backend, data = stack
    svc = app.state.svc
    client.post("/api/v1/instances/webui/start")
    wait(app)
    backend.remote[WEBUI] = NEW
    svc.check_images()
    svc.auto_update()
    assert not app.state.jobs.busy and f"pull {WEBUI}" not in backend.calls      # manual: left to the user
    code, text = run("app-versions", "auto", "webui", "on")
    assert code == 0 and "by itself" in text
    assert yaml.safe_load((data / "instances" / "webui.yaml").read_text())["auto_update"] is True
    hash_before = backend.states()["webui"].config_hash
    assert app.state.engine.config_hash(svc.instance("webui")) == hash_before      # no restart for the setting
    svc.auto_update()
    wait(app)
    assert f"pull {WEBUI}" in backend.calls and svc.image_status()["webui"].state == "current"
    n = len(backend.calls)
    svc.auto_update()                     # nothing new: nothing happens
    assert not app.state.jobs.busy and len(backend.calls) == n
    # the dashboard toggle
    r = browser(app).post("/ui/instances/webui/auto-update", data={"on": "0"}, headers=HX)
    assert r.status_code == 200 and svc.instance("webui").auto_update is False


def test_check_interval_setting(stack):  # noqa: F811
    app, client, _, _ = stack
    r = client.patch("/api/v1/settings", json={"image_check_hours": 0})
    assert r.status_code == 200 and r.json()["settings"]["image_check_hours"] == 0
    assert client.patch("/api/v1/settings", json={"image_check_hours": "x"}).status_code == 400
    app.state.svc.auto_update()           # 0 = never checks by itself
    assert app.state.svc.image_checked() == 0
    assert "Check now" in browser(app).get("/settings").text
