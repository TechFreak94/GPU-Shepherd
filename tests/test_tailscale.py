"""Settings > Remote access: Tailscale on the host, through update.sh's service."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, tailscale, updates
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

ROOT = Path(__file__).resolve().parent.parent

STATUS = {"BackendState": "Running", "TailscaleIPs": ["100.64.0.7", "fd7a:115c:a1e0::7"],
          "Self": {"DNSName": "box.example-tailnet.ts.net.", "HostName": "box"},
          "CurrentTailnet": {"Name": "example-tailnet.ts.net"}, "CertDomains": ["box.example-tailnet.ts.net"]}


def service_on(root: Path):
    (root / "update").mkdir(parents=True, exist_ok=True)
    (root / "update" / "status.json").write_text(json.dumps(
        {"checked_at": "2026-10-23T10:00:00Z", "state": "idle", "repo": "/home/u/gpu-shepherd", "dashboard_updates": True}))


def host_says(root: Path, status: dict | None = None, serve: dict | None = None, **state):
    d = root / "tailscale"
    d.mkdir(parents=True, exist_ok=True)
    base = {"installed": True, "wsl": False, "port": 8090, "state": "idle", "message": "",
            "at": updates.datetime.now(updates.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (d / "state.json").write_text(json.dumps(base | state))
    if status is not None:
        (d / "status.json").write_text(json.dumps(status))
    if serve is not None:
        (d / "serve.json").write_text(json.dumps(serve))


@pytest.fixture
def ts(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.delenv("SHEPHERD_INSTALL", raising=False)
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(tmp_path))}"})
    return client, tmp_path


def test_status_from_the_host_files(tmp_path):
    host_says(tmp_path, STATUS, {"Web": {"box.example-tailnet.ts.net:443": {}}}, port=8091)
    st = tailscale.status(tmp_path)
    assert st.connected and st.name == "box.example-tailnet.ts.net" and st.ip == "100.64.0.7" and st.https
    assert st.dashboard_urls == ["https://box.example-tailnet.ts.net", "http://box.example-tailnet.ts.net:8091",
                                 "http://100.64.0.7:8091"]
    # logging in: the link shows; a garbled status file is ignored
    host_says(tmp_path, {"BackendState": "NeedsLogin", "AuthURL": "https://login.tailscale.com/a/abc"}, {}, state="busy")
    st = tailscale.status(tmp_path)
    assert st.busy and st.needs_login and st.auth_url.endswith("/a/abc") and st.dashboard_urls == []
    (tmp_path / "tailscale" / "status.json").write_text("not json")
    assert tailscale.status(tmp_path).backend == ""


def test_needs_the_update_service(ts):
    client, root = ts
    r = client.post("/api/v1/tailscale/install")
    assert r.status_code == 409 and "--enable-dashboard-updates" in r.json()["error"]
    assert not (root / "update" / "request").exists()


def test_requests_go_through_the_update_service(ts):
    client, root = ts
    service_on(root)
    host_says(root, installed=False)
    assert client.post("/api/v1/tailscale/up").status_code == 409            # not installed yet
    r = client.post("/api/v1/tailscale/install")
    assert r.status_code == 202 and (root / "update" / "request").read_text() == "tailscale\ninstall\n"
    # the Updates card doesn't take it for an update, but nothing else can be asked meanwhile
    assert not updates.status(root).pending and not updates.status(root).busy
    assert client.get("/api/v1/tailscale").json()["busy"]
    assert client.post("/api/v1/tailscale/install").status_code == 409
    (root / "update" / "request").unlink()
    host_says(root, {**STATUS, "CertDomains": []})
    r = client.post("/api/v1/tailscale/https-on")
    assert r.status_code == 409 and "HTTPS certificates" in r.json()["error"]
    host_says(root, STATUS)
    assert client.post("/api/v1/tailscale/https-on").status_code == 202
    assert client.post("/api/v1/tailscale/nope").status_code == 400
    with pytest.raises(ValueError):
        tailscale.request(root, "rm -rf")


def test_wsl_and_published_image(ts, monkeypatch):
    client, root = ts
    service_on(root)
    host_says(root, installed=False, wsl=True)
    r = client.post("/api/v1/tailscale/install")
    assert r.status_code == 409 and "Windows" in r.json()["error"]
    monkeypatch.setenv("SHEPHERD_INSTALL", "image")
    r = client.post("/api/v1/tailscale/refresh")
    assert r.status_code == 409 and "tailscale up" in r.json()["error"]


def test_settings_card(ts):
    client, root = ts
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303
    html = client.get("/settings").text
    assert "Remote access" in html and "--enable-dashboard-updates" in html
    service_on(root)
    client.get("/settings")                    # no report yet: asks the host for one
    assert (root / "update" / "request").read_text() == "tailscale\nrefresh\n"
    (root / "update" / "request").unlink()
    host_says(root, STATUS)
    html = client.get("/ui/tailscale/card", headers={"HX-Request": "true"}).text
    assert "http://box.example-tailnet.ts.net:8090" in html and "Serve over HTTPS" in html and "Disconnect" in html
    assert not (root / "update" / "request").exists()                       # fresh report: no new request
    host_says(root, {"BackendState": "NeedsLogin", "AuthURL": "https://login.tailscale.com/a/xyz"}, state="busy")
    html = client.get("/ui/tailscale/card", headers={"HX-Request": "true"}).text
    assert "https://login.tailscale.com/a/xyz" in html and "Log in to Tailscale" in html
    r = client.post("/ui/tailscale/down", headers={"HX-Request": "true"})
    assert r.status_code == 204 and "already being changed" in r.headers["HX-Trigger"]


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_update_sh_runs_tailscale(tmp_path):
    bin_dir, data, clone = tmp_path / "bin", tmp_path / "data", tmp_path / "clone"
    bin_dir.mkdir(); data.mkdir(); clone.mkdir()
    status = json.dumps(STATUS)
    (bin_dir / "tailscale").write_text(
        f'#!/bin/sh\necho "tailscale $*" >> {tmp_path}/calls\n'
        f'case "$1 $2" in "status --json") echo \'{status}\' ;; "serve status") echo "{{}}" ;; esac\n')
    (bin_dir / "systemctl").write_text(f'#!/bin/sh\necho "systemctl $*" >> {tmp_path}/calls\n')
    (bin_dir / "docker").write_text("#!/bin/sh\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    shutil.copy(ROOT / "update.sh", clone / "update.sh")
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
    (clone / "compose.yaml").write_text(f'    environment:\n      SHEPHERD_DATA_DIR: {data}\n    ports:\n      - "8091:8090"\n')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    def run(action):
        tailscale.request(data, action)
        r = subprocess.run([str(clone / "update.sh"), "--from-dashboard"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (data / "update" / "request").exists()
        return tailscale.status(data)

    st = run("refresh")
    assert st.checked and st.installed and st.connected and st.port == 8091 and st.run == "idle"
    assert not (data / "update" / "log.txt").exists()          # the update log is left alone
    st = run("https-on")
    assert "tailscale serve --bg --https=443 http://127.0.0.1:8091" in (tmp_path / "calls").read_text()
    assert st.message == "The dashboard is at https://box.example-tailnet.ts.net"
    st = run("up")                                             # already running: nothing to log in to
    assert st.message == "Connected." and "tailscale login" not in (tmp_path / "calls").read_text()
    st = run("logout")
    assert "tailscale logout" in (tmp_path / "calls").read_text() and "left your tailnet" in st.message
    assert "Tailscale, from the dashboard: logout" in tailscale.log_text(data)
