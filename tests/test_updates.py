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
from gpushepherd import buildinfo, changelog, updates
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
    assert "Check for updates" in client.get("/ui/update/card").text


def test_published_image_points_to_docker_compose_pull(ui, monkeypatch):
    client, root = ui
    monkeypatch.setenv("SHEPHERD_INSTALL", "image")
    html = client.get("/settings").text
    assert "runs the published image" in html and "docker compose pull" in html
    assert "update.sh" not in html and "Private repository" not in html
    assert "Check for updates" not in client.get("/ui/update").text
    r = client.post("/ui/update/check", headers=HX)
    assert r.status_code == 204 and "docker compose pull" in r.headers["HX-Trigger"]


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
    # the dialog switches to the progress view, which keeps reloading itself
    assert r.status_code == 200 and "Updating GPU Shepherd…" in r.text and 'hx-trigger="every 2s"' in r.text
    assert "Waiting for the update service" in r.text and "data-quiet-offline" in r.text
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
    (seed / "gpushepherd").mkdir()
    (seed / "gpushepherd" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (seed / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0 - 2026-10-01\n\n- First\n")
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
    assert status()["version"] == status()["latest_version"] == "0.1.0"
    assert status()["state"] == "idle" and status()["step"] == "check"
    (seed / "NEW").write_text("x")
    (seed / "gpushepherd" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (seed / "CHANGELOG.md").write_text("# Changelog\n\n## 0.2.0 - 2026-10-02\n\n- New thing\n\n## 0.1.0 - 2026-10-01\n\n- First\n")
    git("add", "-A", cwd=seed)
    git("commit", "-qm", "Add a new thing", cwd=seed)
    git("push", "-q", str(remote), "main", cwd=seed)

    r = run("--check")
    assert r.returncode == 10, r.stderr
    assert status()["behind"] == 1 and status()["commits"][0]["subject"] == "Add a new thing"
    assert "Version 0.2.0 is available (you have 0.1.0)" in r.stdout
    assert status()["version"] == "0.1.0" and status()["latest_version"] == "0.2.0"
    assert "## 0.2.0" in (data / "update" / "changelog.md").read_text()   # what the dashboard shows as new
    assert head(clone) == first                       # --check changes nothing

    (clone / "compose.example.yaml").write_text("services: {mine: {}}\n")   # a local edit blocks the update
    r = run("--yes")
    assert r.returncode == 1 and status()["local_changes"] == ["compose.example.yaml"]
    r = run("--yes", "--stash")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (clone / "NEW").exists() and "mine" in (clone / "compose.example.yaml").read_text()
    assert status()["state"] == "done" and status()["previous"] == first
    assert status()["step"] == "start" and status()["version"] == "0.2.0" and "0.2.0 is running" in status()["message"]
    assert "docker compose up -d --build gpu-shepherd" in (tmp_path / "calls").read_text()

    git("checkout", "-q", "--", "compose.example.yaml", cwd=clone)
    r = run("--yes", "--rollback")
    assert r.returncode == 0, r.stdout + r.stderr
    assert head(clone) == first and not (clone / "NEW").exists()
    assert run("--check").returncode == 10


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_update_sh_recovers_from_a_broken_update(tmp_path):
    """The new version doesn't start: update.sh puts back the old code, image and settings."""
    bin_dir, data = tmp_path / "bin", tmp_path / "data"
    bin_dir.mkdir(); data.mkdir()
    (data / "settings.yaml").write_text("network: before\n")
    # docker: reports a running image; the new build "breaks" the settings and doesn't answer;
    # starting the saved image (--no-build) fixes it. curl fails while it is broken.
    (bin_dir / "docker").write_text(f"""#!/bin/sh
echo "docker $*" >> {tmp_path}/calls
case "$*" in
  *"{{{{.Image}}}}"*) echo sha256:old ;;
  *"{{{{.Config.Image}}}}"*) echo gpu-shepherd:local ;;
  "image inspect"*) exit 0 ;;
  *--no-build*) rm -f {tmp_path}/broken ;;
  *--build*) touch {tmp_path}/broken; echo "network: mangled" > {data}/settings.yaml ;;
esac
""")
    (bin_dir / "curl").write_text(f"#!/bin/sh\n[ ! -e {tmp_path}/broken ]\n")
    (bin_dir / "sleep").write_text("#!/bin/sh\n")          # don't wait 90 s for a web UI that never comes
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    seed, remote, clone = tmp_path / "seed", tmp_path / "remote.git", tmp_path / "clone"
    seed.mkdir()
    shutil.copy(ROOT / "update.sh", seed / "update.sh")
    (seed / ".gitignore").write_text("compose.yaml\n")
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-qm", "first", cwd=seed)
    git("clone", "-q", "--bare", str(seed), str(remote), cwd=tmp_path)
    git("clone", "-q", str(remote), str(clone), cwd=tmp_path)
    (clone / "compose.yaml").write_text(f"    environment:\n      SHEPHERD_DATA_DIR: {data}\n")
    (seed / "NEW").write_text("x")
    git("add", "NEW", cwd=seed)
    git("commit", "-qm", "Broken release", cwd=seed)
    git("push", "-q", str(remote), "main", cwd=seed)
    first = head(clone)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    r = subprocess.run([str(clone / "update.sh"), "--yes"], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "Recovered" in out
    assert head(clone) == first and not (clone / "NEW").exists()
    assert (data / "settings.yaml").read_text() == "network: before\n"
    assert (data / "settings.yaml.before-restore").read_text() == "network: mangled\n"
    calls = (tmp_path / "calls").read_text()
    assert "docker tag sha256:old gpu-shepherd:previous" in calls
    assert "docker tag gpu-shepherd:previous gpu-shepherd:local" in calls
    st = json.loads((data / "update" / "status.json").read_text())
    assert st["state"] == "failed" and "went back" in st["message"] and st["behind"] == 1
    assert list((data / "update" / "backups").glob("settings-*.tar.gz"))


# ---------------------------------------------------------------------- versions, progress, changelog

def test_changelog_parse_and_inline():
    text = """# Changelog

Intro with a [link](docs/x.md).

## 0.3.0 - 2026-10-24

- New **thing** with `code`
  continued here
- See [docs](https://example.com/a?b=1&c=2)

## [0.2.0]

* Older <script>
### Notes
- still 0.2.0

## Something else

- not a release
"""
    rs = changelog.parse(text)
    assert [(r.version, r.date) for r in rs] == [("0.3.0", "2026-10-24"), ("0.2.0", "")]
    assert rs[0].items[0] == "New **thing** with `code` continued here"
    assert rs[1].items == ["Older <script>", "still 0.2.0"]
    assert changelog.inline(rs[0].items[0]) == "New <strong>thing</strong> with <code>code</code> continued here"
    assert 'href="https://example.com/a?b=1&amp;c=2"' in changelog.inline(rs[0].items[1])
    assert changelog.inline("[x](javascript:alert(1)) <b>") == "[x](javascript:alert(1)) &lt;b&gt;"
    assert [r.version for r in changelog.newer(rs, "0.2.0")] == ["0.3.0"]
    assert changelog.key("0.10.0") > changelog.key("0.9.9") > changelog.key("junk")


def test_progress_of_an_update(tmp_path):
    write_status(tmp_path, state="updating", step="rebuild", message="Rebuilding GPU Shepherd",
                 checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    st = updates.status(tmp_path)
    assert st.busy and st.action == "update"
    assert [m for _, m in st.progress] == ["done", "done", "done", "active", "todo"]
    write_status(tmp_path, state="failed", step="start", checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    st = updates.status(tmp_path)
    assert not st.busy and [m for _, m in st.progress] == ["done", "done", "done", "done", "failed"]
    write_status(tmp_path, state="updating", step="recover", checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    assert updates.status(tmp_path).progress[-1] == (updates.RECOVER[1], "active")
    # an updater that went silent half an hour ago is not "busy" any more
    write_status(tmp_path, state="updating", step="download", checked_at="2026-10-23T10:00:00Z")
    st = updates.status(tmp_path)
    assert st.stuck and not st.busy


def test_progress_of_a_check(tmp_path):
    updates.request(tmp_path, "check")
    st = updates.status(tmp_path)
    assert st.action == "check" and st.progress == [("Check for updates", "todo")]
    (tmp_path / "update" / "request").unlink()
    write_status(tmp_path, state="checking", step="check", checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    st = updates.status(tmp_path)
    assert st.busy and st.action == "check" and st.progress == [("Check for updates", "active")]
    write_status(tmp_path, state="idle", step="check")
    assert updates.status(tmp_path).progress == [("Check for updates", "done")]


def test_new_version_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "__version__", "0.7.0")
    (tmp_path / "update").mkdir()
    (tmp_path / "update" / "changelog.md").write_text(
        "## 0.9.0 - 2026-10-25\n\n- Nine\n\n## 0.8.0 - 2026-10-24\n\n- Eight\n\n## 0.7.0 - 2026-10-23\n\n- Seven\n")
    write_status(tmp_path, behind=4, version="0.7.0", latest_version="0.9.0")
    st = updates.status(tmp_path)
    assert st.new_version == "0.9.0" and [r.version for r in st.releases] == ["0.9.0", "0.8.0"]
    write_status(tmp_path, behind=1, version="0.7.0", latest_version="0.7.0")   # merged, not released yet
    assert updates.status(tmp_path).new_version == ""


def test_version_in_header_and_update_dialog(ui, monkeypatch):
    client, root = ui
    monkeypatch.setattr(updates, "__version__", "0.7.0")
    from gpushepherd import __version__
    assert f"v{__version__}" in client.get("/settings").text
    (root / "update").mkdir(exist_ok=True)
    (root / "update" / "changelog.md").write_text("## 0.8.0 - 2026-10-24\n\n- Shows **versions**\n\n## 0.7.0\n\n- Seven\n")
    write_status(root, behind=2, version="0.7.0", latest_version="0.8.0")
    html = client.get("/ui/update").text
    assert "<strong>0.8.0</strong> is available" in html and "Update to 0.8.0" in html
    assert "Shows <strong>versions</strong>" in html
    assert "GPU Shepherd 0.8.0 is available" in client.get("/ui/dashboard").text


def test_changelog_dialog(ui, monkeypatch):
    client, root = ui
    from gpushepherd import __version__
    html = client.get("/ui/changelog").text
    assert "Changelog" in html and f"{__version__}" in html and "running" in html
    major = int(__version__.split(".")[0]) + 1
    (root / "update").mkdir(exist_ok=True)
    (root / "update" / "changelog.md").write_text(f"## {major}.0.0 - 2027-01-01\n\n- Big one\n\n## {__version__}\n\n- Current\n")
    html = client.get("/ui/changelog").text
    assert "Big one" in html and "1 newer version is available" in html and ">new<" in html


def test_update_progress_dialog_shows_steps_and_log(ui):
    client, root = ui
    write_status(root, state="updating", step="download", message="Downloading the new version",
                 checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (root / "update" / "log.txt").write_text("".join(f"line {i}\n" for i in range(40)))
    html = client.get("/ui/update").text
    assert 'class="active">Download the new version' in html and "Downloading the new version…" in html
    assert "line 39" in html and "line 10\n" not in html            # only the end of the log
    assert 'hx-trigger="every 2s"' in html
    write_status(root, state="done", step="start", message="GPU Shepherd 0.8.0 is running (abc1234, 2026-10-24).",
                 checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    html = client.get("/ui/update").text
    assert "Update finished." in html and "Reload" in html and "every 2s" not in html


# ---------------------------------------------------------------------- GitHub token for a private repository

TOKEN = "github_pat_" + "A1b2" * 10


def test_request_credentials_file(tmp_path):
    updates.request_credentials(tmp_path, "TechFan-1", TOKEN)
    f = tmp_path / "update" / "credentials"
    assert f.read_text() == f"TechFan-1\n{TOKEN}\n" and (f.stat().st_mode & 0o777) == 0o600
    assert (tmp_path / "update" / "request").read_text() == "credentials\n"
    assert updates.status(tmp_path).action == "check"           # the host checks right after saving it
    updates.request_credentials(tmp_path, clear=True)
    assert f.read_text() == "clear\n"
    for user, token in (("bad user", TOKEN), ("ok", "short"), ("ok", TOKEN + "\nx"), ("-x", TOKEN)):
        with pytest.raises(ValueError):
            updates.request_credentials(tmp_path, user, token)


def test_credentials_in_settings(ui):
    client, root = ui
    write_status(root, dashboard_updates=False)
    html = client.get("/settings").text
    assert "Private repository" in html and "--set-token" in html and 'name="token"' not in html
    r = client.post("/ui/update/credentials", data={"user": "zach", "token": TOKEN}, headers=HX)
    assert "--set-token" in r.headers["HX-Trigger"] and not (root / "update" / "credentials").exists()
    write_status(root)
    assert 'name="token"' in client.get("/settings").text
    r = client.post("/ui/update/credentials", data={"user": "zach", "token": "nope"}, headers=HX)
    assert "access token" in r.headers["HX-Trigger"] and not (root / "update" / "credentials").exists()
    r = client.post("/ui/update/credentials", data={"user": "zach", "token": TOKEN}, headers=HX)
    assert r.status_code == 200 and "Checking for updates…" in r.text and TOKEN not in r.text
    assert "credentials-sent" in r.headers["HX-Trigger"]
    assert (root / "update" / "credentials").exists()
    (root / "update" / "request").unlink()
    write_status(root, credentials="zach")
    html = client.get("/settings").text
    assert "Updates download as <strong>zach</strong>" in html and "Remove token" in html and TOKEN not in html


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_update_sh_saves_and_clears_a_token(tmp_path):
    bin_dir, data = tmp_path / "bin", tmp_path / "data"
    bin_dir.mkdir(); data.mkdir()
    for tool in ("docker", "curl", "systemctl"):
        (bin_dir / tool).write_text("#!/bin/sh\nexit 1\n" if tool == "systemctl" else "#!/bin/sh\n")
        (bin_dir / tool).chmod(0o755)
    seed, remote, clone = tmp_path / "seed", tmp_path / "remote.git", tmp_path / "clone"
    seed.mkdir()
    shutil.copy(ROOT / "update.sh", seed / "update.sh")
    (seed / ".gitignore").write_text("compose.yaml\n")
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-qm", "first", cwd=seed)
    git("clone", "-q", "--bare", str(seed), str(remote), cwd=tmp_path)
    git("clone", "-q", str(remote), str(clone), cwd=tmp_path)
    (clone / "compose.yaml").write_text(f"    environment:\n      SHEPHERD_DATA_DIR: {data}\n")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    creds = clone / ".git" / "gpu-shepherd-credentials"

    r = subprocess.run([str(clone / "update.sh"), "--set-token", "zach"], input=TOKEN + "\n", env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert creds.read_text() == f"https://zach:{TOKEN}@github.com\n" and (creds.stat().st_mode & 0o777) == 0o600
    status = (data / "update" / "status.json").read_text()
    assert json.loads(status)["credentials"] == "zach"
    assert TOKEN not in r.stdout + r.stderr + status
    assert "doesn't use HTTPS" in r.stderr                # a local remote: the token isn't used for it

    r = subprocess.run([str(clone / "update.sh"), "--set-token", "zach"], input="not a token\n", env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1 and creds.exists()             # a bad token keeps the old one

    # from the dashboard: the file in the data folder is taken out at once and never logged
    (data / "update" / "credentials").write_text(f"other\n{TOKEN}\n")
    (data / "update" / "request").write_text("credentials\n")
    r = subprocess.run([str(clone / "update.sh"), "--from-dashboard"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (data / "update" / "log.txt").read_text()
    assert not (data / "update" / "credentials").exists() and creds.read_text().startswith("https://other:")
    log = (data / "update" / "log.txt").read_text()
    assert "Saved the GitHub token for other" in log and TOKEN not in log

    r = subprocess.run([str(clone / "update.sh"), "--clear-token"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and not creds.exists()
    assert json.loads((data / "update" / "status.json").read_text())["credentials"] == ""
