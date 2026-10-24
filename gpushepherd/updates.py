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
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, buildinfo, changelog

ACTIONS = ("check", "update")             # what the dashboard's buttons can ask for
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
    def command(self) -> str:
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
        for key in ("checked_at", "state", "step", "message", "repo", "upstream", "current", "previous",
                    "version", "latest_version", "credentials"):
            setattr(st, key, str(raw.get(key) or ""))
        st.behind = int(raw.get("behind") or 0)
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


def request(root: Path, action: str) -> None:
    """Ask the host to check or update; update.sh deletes the file when it starts."""
    if action not in ACTIONS + ("credentials",):
        raise ValueError(f"unknown update action {action!r}")
    d = folder(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "request.tmp"
    tmp.write_text(action + "\n")
    tmp.replace(d / "request")


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
