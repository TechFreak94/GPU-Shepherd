"""Updates: what is running, whether a newer version is waiting, and asking the host to update.

GPU Shepherd can't rebuild its own container from the inside, so the real work is done by
update.sh on the host. The two sides talk through files in <data dir>/update/, which is the same
path inside and outside the container:

  status.json  written by update.sh after every check or update (see write_status there)
  request      written here: "check" or "update"; a systemd path unit installed by
               `update.sh --enable-dashboard-updates` runs update.sh when it appears
  log.txt      the output of the last run started from the dashboard
  changelog.md CHANGELOG.md of the newest version on the remote, saved at every check
  credentials  written here (0600) with a GitHub user name and token, or "clear"; update.sh moves
               it into the clone's .git folder at once and never writes the token anywhere else

The published image (compose.standalone.yaml, SHEPHERD_INSTALL=image) has no clone and no
update.sh: it is updated with `docker compose pull`, and the dashboard only says so.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, buildinfo, changelog

ACTIONS = ("check", "update")             # what the dashboard's buttons can ask for
BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
GITHUB_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
GITHUB_TOKEN = re.compile(r"^[A-Za-z0-9_]{20,255}$")     # ghp_..., github_pat_..., same check as update.sh
STALE_REQUEST = 120         # seconds: a request nobody picked up means the service isn't running
STUCK = 1800                # seconds: a check or update that reported nothing for this long has died
# What update.sh does, in order, by the "step" it writes to status.json. "recover" only happens
# when an update fails and it goes back to the version that worked.
STEPS = (("check", "Check for updates"),
         ("backup", "Keep the running version and back up the settings"),
         ("download", "Download the new version"),
         ("rebuild", "Rebuild GPU Shepherd"),
         ("start", "Start it and wait for the dashboard"))
RECOVER = ("recover", "Go back to the version that worked")


@dataclass
class UpdateStatus:
    running: dict = field(default_factory=dict)     # build info of this image: commit, branch, built
    checked: bool = False                           # update.sh has written a status at least once
    checked_at: str = ""
    state: str = ""                                 # idle / checking / updating / done / failed
    step: str = ""                                  # the current (or failed) step, see STEPS
    message: str = ""
    repo: str = ""                                  # the clone on the host
    upstream: str = ""
    branch: str = ""                                # the branch the clone follows
    branches: list[str] = field(default_factory=list)   # branches on the remote, newest first
    branch_info: list[dict] = field(default_factory=list)   # name, date, subject of each branch (newest first)
    github: str = ""                                # owner/name when the remote is on github.com
    behind: int = 0
    commits: list[dict] = field(default_factory=list)
    local_changes: list[str] = field(default_factory=list)
    current: str = ""                               # commit the clone is on
    previous: str = ""
    version: str = ""                               # version of the code in the clone
    latest_version: str = ""                        # version of the newest code on the remote
    releases: list = field(default_factory=list)    # changelog entries newer than what is running
    dashboard_updates: bool = False                 # the host service is installed
    credentials: str = ""                           # GitHub user whose token update.sh has saved
    pending: str = ""                               # a request waiting for the host ("check"/"update")
    pending_stale: bool = False
    stuck: bool = False                             # "checking"/"updating" but silent for STUCK seconds
    age: float = 0.0                                # seconds since update.sh last wrote the status
    image: bool = field(default_factory=lambda: os.environ.get("SHEPHERD_INSTALL") == "image")

    @property
    def available(self) -> bool:
        return self.behind > 0

    @property
    def running_version(self) -> str:
        return __version__

    @property
    def new_version(self) -> str:
        """The version an update would bring, when it is a different one."""
        v = self.latest_version
        return v if self.available and v and changelog.key(v) > changelog.key(__version__) else ""

    @property
    def rebuild_needed(self) -> bool:
        """The clone has moved on (e.g. a manual git pull) but this image was built from older code."""
        built = self.running.get("commit", "")
        return bool(built and self.current and built != self.current and not self.available)

    @property
    def busy(self) -> bool:
        working = self.state in ("checking", "updating") and not self.stuck
        return working or (bool(self.pending) and not self.pending_stale)

    @property
    def action(self) -> str:
        """What is running or waiting: "check" or "update"."""
        if self.pending:
            return "update" if self.pending == "update" else "check"   # saving a token checks next
        return "update" if self.state in ("updating", "done") or self.step not in ("", "check") else "check"

    @property
    def just_updated(self) -> bool:
        """An update finished in the last 15 minutes (worth a "reload the page" hint)."""
        return self.state == "done" and 0 <= self.age < 900

    @property
    def progress(self) -> list[tuple[str, str]]:
        """(label, done / active / failed / todo) for each step of what is running or just ran."""
        steps = STEPS[:1] if self.action == "check" else STEPS
        if self.pending and self.state not in ("checking", "updating"):
            return [(label, "todo") for _, label in steps]      # the host hasn't started yet
        if self.step == RECOVER[0]:
            steps = steps + (RECOVER,)
        names = [s[0] for s in steps]
        at = names.index(self.step) if self.step in names else -1
        out = []
        for i, (_, label) in enumerate(steps):
            if self.state == "done" or (self.state == "idle" and self.action == "check"):
                mark = "done"
            elif i < at:
                mark = "done"
            elif i == at:
                mark = "failed" if self.state == "failed" or self.stuck else "active"
            else:
                mark = "todo"
            out.append((label, mark))
        return out

    @property
    def branch_options(self) -> list[dict]:
        return branch_options(self)

    @property
    def command(self) -> str:
        if self.image:
            return "sudo docker compose pull && sudo docker compose up -d"
        return f"sudo {self.repo or '<your clone>'}/update.sh"

    @property
    def checked_ago(self) -> str:
        return ago(self.checked_at)


def seconds_since(stamp: str, now: float | None = None) -> float:
    """Seconds since an ISO time in UTC; -1 if it can't be read."""
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return -1
    return max((now or time.time()) - t, 0)


def ago(stamp: str, now: float | None = None) -> str:
    """'just now', '5 minutes ago', '3 hours ago', '2 days ago' for an ISO time in UTC."""
    s = seconds_since(stamp, now)
    if s < 0:
        return ""
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if s >= size:
            n = int(s // size)
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


def folder(root: Path) -> Path:
    return Path(root) / "update"


def status(root: Path, now: float | None = None) -> UpdateStatus:
    st = UpdateStatus(running=buildinfo.load())
    d = folder(root)
    try:
        raw = json.loads((d / "status.json").read_text())
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        st.checked = True
        for key in ("checked_at", "state", "step", "message", "repo", "upstream", "branch", "current", "previous",
                    "version", "latest_version", "credentials"):
            setattr(st, key, str(raw.get(key) or ""))
        st.behind = int(raw.get("behind") or 0)
        st.branches = [str(b) for b in raw.get("branches") or []]
        st.branch_info = [b for b in raw.get("branch_info") or [] if isinstance(b, dict) and b.get("name")]
        st.github = str(raw.get("github") or "")
        st.commits = [c for c in raw.get("commits") or [] if isinstance(c, dict)]
        st.local_changes = [str(f) for f in raw.get("local_changes") or []]
        st.dashboard_updates = raw.get("dashboard_updates") is True
        st.age = seconds_since(st.checked_at, now)
        st.stuck = st.state in ("checking", "updating") and st.age > STUCK
        if st.available:
            st.releases = changelog.newer(changelog.read(d / "changelog.md"), __version__)
    req = d / "request"
    try:
        st.pending = req.read_text().strip()
        st.pending_stale = (now or time.time()) - req.stat().st_mtime > STALE_REQUEST
    except OSError:
        pass
    return st


def request(root: Path, action: str, arg: str = "") -> None:
    """Ask the host to check or update; update.sh deletes the file when it starts. `arg` is the
    second line (the branch for "branch")."""
    if action not in ACTIONS + ("credentials", "branch"):
        raise ValueError(f"unknown update action {action!r}")
    d = folder(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "request.tmp"
    tmp.write_text(action + "\n" + (arg + "\n" if arg else ""))
    tmp.replace(d / "request")


# ---------------------------------------------------------------- the branch picker's labels
PR_TTL = 600            # seconds a pull request lookup is kept
PR_RETRY = 120          # after a failed lookup (offline, private repo, rate limit) wait this long
_prs: dict[str, tuple[float, dict]] = {}
_prs_lock = threading.Lock()


def _fetch_prs(repo: str) -> dict:
    """Branch name -> {number, title, state} for the repository's recent pull requests (public repos only)."""
    import httpx
    r = httpx.get(f"https://api.github.com/repos/{repo}/pulls", timeout=4,
                  params={"state": "all", "sort": "updated", "direction": "desc", "per_page": 100},
                  headers={"Accept": "application/vnd.github+json", "User-Agent": "gpu-shepherd"})
    r.raise_for_status()
    out: dict = {}
    for pr in r.json():
        head = pr.get("head") or {}
        if (head.get("repo") or {}).get("full_name", "").lower() != repo.lower():
            continue                                    # from a fork: the branch is not on this remote
        state = "merged" if pr.get("merged_at") else str(pr.get("state") or "")
        out.setdefault(str(head.get("ref")), {"number": pr.get("number"), "title": str(pr.get("title") or ""), "state": state})
    return out


def _refresh_prs(repo: str) -> None:
    try:
        found, ttl = _fetch_prs(repo), PR_TTL
    except Exception:                                   # offline, private repo, rate limited: keep what we had
        found, ttl = _prs.get(repo, (0, {}))[1], PR_RETRY
    with _prs_lock:
        _prs[repo] = (time.time() + ttl, found)


def pull_requests(repo: str) -> dict:
    """Cached lookup. Never waits for GitHub: a stale or missing entry is refreshed in the background."""
    if not repo:
        return {}
    with _prs_lock:
        until, found = _prs.get(repo, (0.0, {}))
        if until > time.time():
            return found
        _prs[repo] = (time.time() + PR_RETRY, found)    # one refresh at a time
    threading.Thread(target=_refresh_prs, args=(repo,), daemon=True).start()
    return found


def _parse(iso: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _when(iso: str) -> str:
    d = _parse(iso)
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if d else ""


def branch_options(st: UpdateStatus, prs: dict | None = None) -> list[dict]:
    """The branch picker's entries: main pinned first (the stable one), then newest commit first.
    Each has name, label and detail; pull request number and title are added when GitHub knows them."""
    prs = pull_requests(st.github) if prs is None else prs
    info = {b["name"]: b for b in st.branch_info}
    names = list(st.branches)
    for n in info:
        if n not in names:
            names.append(n)
    out = []
    for n in names:
        b, pr = info.get(n, {}), prs.get(n)
        when = _when(str(b.get("date") or ""))
        subject = str(b.get("subject") or "")
        if n in ("main", "master"):
            title = "stable, the released version"
        elif pr:
            title = f"#{pr['number']} {pr['title']} ({pr['state']})"
        else:
            title = f"{subject} (no pull request)" if subject else ""
        title = title if len(title) <= 70 else title[:69] + "…"
        out.append({"name": n, "title": title, "when": when, "date": str(b.get("date") or ""),
                    "label": " · ".join(x for x in (n if n in ("main", "master") else "", title, when,
                                                    "" if n in ("main", "master") else n) if x)})
    stable = [o for o in out if o["name"] in ("main", "master")][:1]
    rest = sorted((o for o in out if o not in stable), key=lambda o: _parse(o["date"]) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return stable + rest


def request_branch(root: Path, name: str) -> None:
    """Ask the host to follow another branch of the repository (update.sh switches the clone)."""
    if not BRANCH_NAME.match(name) or ".." in name or "//" in name or name.endswith(".lock"):
        raise ValueError("That isn't a usable branch name.")
    request(root, "branch", name)


def request_credentials(root: Path, user: str = "", token: str = "", clear: bool = False) -> None:
    """Hand a GitHub user name and token (or "forget it") to update.sh, which stores it on the host.

    The file is readable only by its owner and lives in the data folder for the few seconds until
    the update service picks it up; it is never logged or kept in the settings."""
    if not clear:
        if not GITHUB_USER.match(user):
            raise ValueError("That doesn't look like a GitHub user name.")
        if not GITHUB_TOKEN.match(token):
            raise ValueError("That doesn't look like a GitHub access token (it starts with github_pat_ or ghp_).")
    d = folder(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "credentials.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("clear\n" if clear else f"{user}\n{token}\n")
    tmp.replace(d / "credentials")
    request(root, "credentials")


def live_log(root: Path, lines: int = 15, now: float | None = None) -> str:
    """The end of the log while the updater is writing it (a log older than 10 minutes is a past run's)."""
    try:
        if (now or time.time()) - (folder(root) / "log.txt").stat().st_mtime > 600:
            return ""
    except OSError:
        return ""
    return log_text(root, lines)


def log_text(root: Path, lines: int = 0) -> str:
    """The log of the last run started from the dashboard; with lines, only its end."""
    try:
        text = (folder(root) / "log.txt").read_text(errors="replace")[-20000:]
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:]) if lines else text
