"""Build info, the update status update.sh writes, and the dashboard's update routes."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import buildinfo, updates
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend

ROOT = Path(__file__).resolve().parent.parent
HX = {"HX-Request": "true"}
SHA_A, SHA_B = "a" * 40, "b" * 40


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def head(cwd) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


# ---------------------------------------------------------------------- buildinfo

@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_buildinfo_reads_loose_and_packed_refs(tmp_path):
    git("init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "f").write_text("1")
    git("add", "f", cwd=tmp_path)
    git("commit", "-qm", "one", cwd=tmp_path)
    assert buildinfo.read(tmp_path) == {"commit": head(tmp_path), "branch": "main"}
    git("pack-refs", "--all", cwd=tmp_path)
    assert not (tmp_path / ".git/refs/heads/main").exists()
    assert buildinfo.read(tmp_path)["commit"] == head(tmp_path)
    git("checkout", "-q", "--detach", cwd=tmp_path)
    assert buildinfo.read(tmp_path) == {"commit": head(tmp_path), "branch": ""}


def test_buildinfo_outside_a_clone(tmp_path):
    assert buildinfo.read(tmp_path) == {}


def test_buildinfo_main_writes_file(tmp_path, monkeypatch):
    out = tmp_path / "x" / "build.json"
    assert buildinfo.main([str(tmp_path), str(out)]) == 0
    data = json.loads(out.read_text())
    assert "built" in data and "commit" not in data
    monkeypatch.setattr(buildinfo, "BUILD_INFO", out)
    assert buildinfo.load() == data


# ---------------------------------------------------------------------- status file

def write_status(root: Path, **fields):
    (root / "update").mkdir(parents=True, exist_ok=True)
    base = {"checked_at": "2026-10-23T10:00:00Z", "state": "idle", "message": "", "repo": "/home/u/gpu-shepherd",
            "branch": "main", "upstream": "origin/main", "current": SHA_A, "latest": SHA_B, "previous": "",
            "behind": 0, "ahead": 0, "local_changes": [], "commits": [], "dashboard_updates": True}
    (root / "update" / "status.json").write_text(json.dumps(base | fields))


def test_status_without_any_check(tmp_path):
    st = updates.status(tmp_path)
    assert not st.checked and not st.available and not st.busy and not st.rebuild_needed


def test_status_update_available(tmp_path):
    write_status(tmp_path, behind=2, commits=[{"sha": "bbbbbbb", "date": "2026-10-23", "subject": "New thing"}])
    st = updates.status(tmp_path)
    assert st.available and st.behind == 2 and st.commits[0]["subject"] == "New thing"
    assert st.command == "sudo /home/u/gpu-shepherd/update.sh"


def test_rebuild_needed_when_image_is_older_than_clone(tmp_path, monkeypatch):
    write_status(tmp_path, current=SHA_B)
    monkeypatch.setattr(buildinfo, "load", lambda: {"commit": SHA_A})
    assert updates.status(tmp_path).rebuild_needed
    monkeypatch.setattr(buildinfo, "load", lambda: {"commit": SHA_B})
    assert not updates.status(tmp_path).rebuild_needed


def test_request_and_stale_request(tmp_path):
    updates.request(tmp_path, "check")
    st = updates.status(tmp_path)
    assert st.pending == "check" and st.busy
    st = updates.status(tmp_path, now=time.time() + updates.STALE_REQUEST + 5)
    assert st.pending_stale and not st.busy
    with pytest.raises(ValueError):
        updates.request(tmp_path, "rm -rf")


def test_ago():
    t = time.mktime(time.strptime("2026-10-23T10:00:00", "%Y-%m-%dT%H:%M:%S")) - time.timezone
    assert updates.ago("2026-10-23T10:00:00Z", now=t + 10) == "just now"
    assert updates.ago("2026-10-23T10:00:00Z", now=t + 3 * 3600) == "3 hours ago"
    assert updates.ago("2026-10-23T10:00:00Z", now=t + 86400) == "1 day ago"
    assert updates.ago("garbage") == ""


# ---------------------------------------------------------------------- web UI

@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    monkeypatch.setattr(buildinfo, "load", lambda: {"commit": SHA_A, "branch": "main", "built": "2026-10-22T00:00:00Z"})
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app)
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303
    return client, tmp_path


def test_dashboard_shows_update_banner(ui):
    client, root = ui
    assert "Update available" not in client.get("/ui/dashboard").text
    write_status(root, behind=3, commits=[{"sha": "bbbbbbb", "date": "2026-10-23", "subject": "Faster starts"}])
    html = client.get("/ui/dashboard").text
    assert "Update available" in html and "Faster starts" in html and "Update…" in html
    write_status(root, behind=3, dashboard_updates=False)
    html = client.get("/ui/dashboard").text
    assert "Update available" in html and "Update…" not in html


def test_update_dialog_and_settings(ui):
    client, root = ui
    html = client.get("/settings").text
    assert "hasn't checked for updates yet" in html and "aaaaaaa" in html
    write_status(root, behind=1, commits=[{"sha": "bbbbbbb", "date": "2026-10-23", "subject": "Fix & <polish>"}],
                 dashboard_updates=False)
    html = client.get("/ui/update").text
    assert "Fix &amp; &lt;polish&gt;" in html and "--enable-dashboard-updates" in html and "Update now" not in html
    write_status(root, behind=1)
    assert "Update now" in client.get("/ui/update").text
    assert "Check now" in client.get("/ui/update/card").text


def test_update_request_needs_the_service(ui):
    client, root = ui
    write_status(root, behind=1, dashboard_updates=False)
    r = client.post("/ui/update/update", headers=HX)
    assert r.status_code == 204 and "off" in r.headers["HX-Trigger"]
    assert not (root / "update" / "request").exists()


def test_update_request_written(ui):
    client, root = ui
    write_status(root, behind=1)
    assert client.post("/ui/update/nope", headers=HX).status_code == 400
    r = client.post("/ui/update/update", headers=HX)
    assert r.status_code == 204 and "restarts" in r.headers["HX-Trigger"]
    assert (root / "update" / "request").read_text() == "update\n"
    assert "Updating GPU Shepherd" in client.get("/ui/dashboard").text
    r = client.post("/ui/update/check", headers=HX)          # one request at a time
    assert "already running" in r.headers["HX-Trigger"]


def test_update_blocked_by_local_changes(ui):
    client, root = ui
    write_status(root, behind=1, local_changes=["README.md"])
    r = client.post("/ui/update/update", headers=HX)
    assert "--stash" in r.headers["HX-Trigger"] and not (root / "update" / "request").exists()


def test_update_log_and_failed_state(ui):
    client, root = ui
    assert "Nothing yet" in client.get("/ui/update/log").text
    write_status(root, state="failed", message="the rebuild failed")
    (root / "update" / "log.txt").write_text("step 1\nboom\n")
    assert "the rebuild failed" in client.get("/ui/dashboard").text
    assert "boom" in client.get("/ui/update/log").text


def test_healthz_and_assets_carry_the_commit(ui):
    client, _ = ui
    assert client.get("/healthz").json()["commit"] == SHA_A
    assert f"app.css?v={SHA_A[:12]}" in client.get("/settings").text


def test_update_routes_need_login(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    client = TestClient(create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False))
    assert client.post("/ui/update/update", headers=HX).status_code == 401
    assert not (tmp_path / "update" / "request").exists()


# ---------------------------------------------------------------------- update.sh

@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_update_sh_check_update_and_rollback(tmp_path):
    """update.sh against a scratch remote, with a stand-in docker and curl."""
    bin_dir, data = tmp_path / "bin", tmp_path / "data"
    bin_dir.mkdir(); data.mkdir()
    for tool in ("docker", "curl"):
        (bin_dir / tool).write_text(f'#!/bin/sh\necho "{tool} $*" >> {tmp_path}/calls\n')
        (bin_dir / tool).chmod(0o755)
    seed, remote, clone = tmp_path / "seed", tmp_path / "remote.git", tmp_path / "clone"
    seed.mkdir()
    shutil.copy(ROOT / "update.sh", seed / "update.sh")
    (seed / "compose.example.yaml").write_text("services: {}\n")
    (seed / ".gitignore").write_text("compose.yaml\n")
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-qm", "first", cwd=seed)
    git("clone", "-q", "--bare", str(seed), str(remote), cwd=tmp_path)
    git("clone", "-q", str(remote), str(clone), cwd=tmp_path)
    (clone / "compose.yaml").write_text(f"    environment:\n      SHEPHERD_DATA_DIR: {data}\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    def run(*args):
        return subprocess.run([str(clone / "update.sh"), *args], env=env, capture_output=True, text=True)

    def status():
        return json.loads((data / "update" / "status.json").read_text())

    first = head(clone)
    assert run("--check").returncode == 0 and status()["behind"] == 0
    (seed / "NEW").write_text("x")
    git("add", "NEW", cwd=seed)
    git("commit", "-qm", "Add a new thing", cwd=seed)
    git("push", "-q", str(remote), "main", cwd=seed)

    r = run("--check")
    assert r.returncode == 10, r.stderr
    assert status()["behind"] == 1 and status()["commits"][0]["subject"] == "Add a new thing"
    assert head(clone) == first                       # --check changes nothing

    (clone / "compose.example.yaml").write_text("services: {mine: {}}\n")   # a local edit blocks the update
    r = run("--yes")
    assert r.returncode == 1 and status()["local_changes"] == ["compose.example.yaml"]
    r = run("--yes", "--stash")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (clone / "NEW").exists() and "mine" in (clone / "compose.example.yaml").read_text()
    assert status()["state"] == "done" and status()["previous"] == first
    assert "docker compose up -d --build gpu-shepherd" in (tmp_path / "calls").read_text()

    git("checkout", "-q", "--", "compose.example.yaml", cwd=clone)
    r = run("--yes", "--rollback")
    assert r.returncode == 0, r.stdout + r.stderr
    assert head(clone) == first and not (clone / "NEW").exists()
    assert run("--check").returncode == 10
