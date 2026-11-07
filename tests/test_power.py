"""The top menu's Power dialog: restart GPU Shepherd, or restart / shut down the server."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, updates
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

ROOT = Path(__file__).resolve().parent.parent
HX = {"HX-Request": "true"}


def write_status(root: Path, **fields):
    (root / "update").mkdir(parents=True, exist_ok=True)
    base = {"checked_at": "2026-10-23T10:00:00Z", "state": "idle", "repo": "/home/u/gpu-shepherd",
            "branch": "main", "dashboard_updates": True}
    (root / "update" / "status.json").write_text(json.dumps(base | fields))


@pytest.fixture
def power(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.delenv("SHEPHERD_INSTALL", raising=False)
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    restarts = []
    app.state.svc.restarter = lambda: restarts.append(1)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(tmp_path))}"})
    return app, client, tmp_path, restarts


def test_restart_shepherd_needs_no_service(power):
    app, client, root, restarts = power
    r = client.get("/api/v1/power").json()
    assert not r["host"] and "--enable-dashboard-updates" in r["hint"]
    r = client.post("/api/v1/power/restart")
    assert r.status_code == 202 and "Restarting GPU Shepherd" in r.json()["message"] and restarts == [1]
    assert not (root / "update" / "request").exists()


def test_server_power_goes_through_the_update_service(power):
    app, client, root, restarts = power
    r = client.post("/api/v1/power/reboot")
    assert r.status_code == 409 and "update service" in r.json()["error"]
    write_status(root)
    assert client.get("/api/v1/power").json()["host"]
    assert client.post("/api/v1/power/reboot").status_code == 202
    assert (root / "update" / "request").read_text() == "reboot\n"
    r = client.post("/api/v1/power/shutdown")              # one request at a time
    assert r.status_code == 409 and "running" in r.json()["error"]
    (root / "update" / "request").unlink()
    assert client.post("/api/v1/power/shutdown").status_code == 202
    assert (root / "update" / "request").read_text() == "poweroff\n"
    assert client.post("/api/v1/power/nope").status_code == 400
    assert restarts == []


def test_published_image_has_no_server_power(power, monkeypatch):
    app, client, root, _ = power
    monkeypatch.setenv("SHEPHERD_INSTALL", "image")
    write_status(root)
    r = client.get("/api/v1/power").json()
    assert not r["host"] and "sudo reboot" in r["hint"]


def test_not_during_a_change(power):
    app, client, root, restarts = power
    write_status(root)
    app.state.jobs.current = object()
    try:
        r = client.post("/api/v1/power/restart")
        assert r.status_code == 409 and "current change" in r.json()["error"] and restarts == []
        assert client.post("/api/v1/power/reboot").status_code == 409
    finally:
        app.state.jobs.current = None
    assert not (root / "update" / "request").exists()


def test_power_request_values():
    with pytest.raises(ValueError):
        updates.request(Path("/nonexistent"), "halt")


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
@pytest.mark.parametrize("action", ["reboot", "poweroff"])
def test_update_sh_runs_systemctl(tmp_path, action):
    bin_dir, data, clone = tmp_path / "bin", tmp_path / "data", tmp_path / "clone"
    bin_dir.mkdir(); data.mkdir(); clone.mkdir()
    for tool in ("docker", "systemctl"):
        (bin_dir / tool).write_text(f'#!/bin/sh\necho "{tool} $*" >> {tmp_path}/calls\n')
        (bin_dir / tool).chmod(0o755)
    shutil.copy(ROOT / "update.sh", clone / "update.sh")
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
    (clone / "compose.yaml").write_text(f"    environment:\n      SHEPHERD_DATA_DIR: {data}\n")
    updates.request(data, action)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    r = subprocess.run([str(clone / "update.sh"), "--from-dashboard"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"systemctl --no-block {action}" in (tmp_path / "calls").read_text()
    assert not (data / "update" / "request").exists()
    assert "the server" in (data / "update" / "log.txt").read_text()


# ---------------------------------------------------------------------- web UI

@pytest.fixture
def ui(power):
    app, _, root, restarts = power
    client = TestClient(app)
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303
    return client, root, restarts


def test_power_button_in_the_header(ui):
    client, _, _ = ui
    assert 'hx-get="/ui/power"' in client.get("/").text
    assert 'hx-get="/ui/power"' not in TestClient(client.app).get("/login").text


def test_power_dialog_lists_three_options(ui):
    client, root, _ = ui
    t = client.get("/ui/power", headers=HX).text
    for label in ("Restart GPU Shepherd", "Restart the server", "Shut down the server"):
        assert label in t
    assert "your apps keep running" in t
    assert "--enable-dashboard-updates" in t
    assert t.count("disabled") == 2                          # both server options, not the dashboard restart
    write_status(root)
    t = client.get("/ui/power", headers=HX).text
    assert "disabled" not in t and "--enable-dashboard-updates" not in t


def test_power_confirm_step(ui):
    client, root, _ = ui
    write_status(root)
    t = client.get("/ui/power?confirm=shutdown", headers=HX).text
    assert "Shut down the server?" in t and 'hx-post="/ui/power/shutdown"' in t and "danger" in t
    assert 'hx-post="/ui/power/restart"' in client.get("/ui/power?confirm=restart", headers=HX).text
    assert "hx-post" not in client.get("/ui/power?confirm=bogus", headers=HX).text


def test_power_restart_from_the_dialog(ui):
    client, root, restarts = ui
    r = client.post("/ui/power/restart", headers=HX)
    assert r.status_code == 200 and restarts == [1]
    assert "Restarting GPU Shepherd" in r.text and "/ui/power/back" in r.text and "data-quiet-offline" in r.text
    assert "toast" in r.headers["HX-Trigger"]
    assert not (root / "update" / "request").exists()


def test_power_reboot_and_shutdown_from_the_dialog(ui):
    client, root, _ = ui
    r = client.post("/ui/power/reboot", headers=HX)          # no update service yet
    assert r.status_code == 204 and "update service" in r.headers["HX-Trigger"]
    write_status(root)
    r = client.post("/ui/power/reboot", headers=HX)
    assert r.status_code == 200 and (root / "update" / "request").read_text() == "reboot\n"
    assert "/ui/power/back" in r.text
    (root / "update" / "request").unlink()
    r = client.post("/ui/power/shutdown", headers=HX)
    assert r.status_code == 200 and "turn the server back on" in r.text and "/ui/power/back" not in r.text


def test_power_back_reloads_only_after_a_restart(ui):
    client, _, _ = ui
    boot = client.post("/ui/power/restart", headers=HX).text.split("boot=")[1].split('"')[0]
    assert "HX-Refresh" not in client.get(f"/ui/power/back?boot={boot}").headers
    assert client.get("/ui/power/back?boot=other").headers["HX-Refresh"] == "true"


def test_power_ui_errors_and_login(ui):
    client, _, _ = ui
    assert client.post("/ui/power/explode", headers=HX).status_code == 400
    anon = TestClient(client.app)
    r = anon.get("/ui/power", headers=HX, follow_redirects=False)
    assert r.status_code in (303, 401) and "Restart the server" not in r.text
