"""Updates: what is running, whether a newer version is waiting, and asking the host to update.

GPU Shepherd can't rebuild its own container from the inside, so the real work is done by
update.sh on the host. The two sides talk through files in <data dir>/update/, which is the same
path inside and outside the container:

  status.json  written by update.sh after every check or update (see write_status there)
  request      written here: "check" or "update"; a systemd path unit installed by
               `update.sh --enable-dashboard-updates` runs update.sh when it appears
  log.txt      the output of the last run started from the dashboard
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import buildinfo

ACTIONS = ("check", "update")
STALE_REQUEST = 120         # seconds: a request nobody picked up means the service isn't running


@dataclass
class UpdateStatus:
    running: dict = field(default_factory=dict)     # build info of this image: commit, branch, built
    checked: bool = False                           # update.sh has written a status at least once
    checked_at: str = ""
    state: str = ""                                 # idle / updating / done / failed
    message: str = ""
    repo: str = ""                                  # the clone on the host
    upstream: str = ""
    behind: int = 0
    commits: list[dict] = field(default_factory=list)
    local_changes: list[str] = field(default_factory=list)
    current: str = ""                               # commit the clone is on
    previous: str = ""
    dashboard_updates: bool = False                 # the host service is installed
    pending: str = ""                               # a request waiting for the host ("check"/"update")
    pending_stale: bool = False

    @property
    def available(self) -> bool:
        return self.behind > 0

    @property
    def rebuild_needed(self) -> bool:
        """The clone has moved on (e.g. a manual git pull) but this image was built from older code."""
        built = self.running.get("commit", "")
        return bool(built and self.current and built != self.current and not self.available)

    @property
    def busy(self) -> bool:
        return self.state == "updating" or (bool(self.pending) and not self.pending_stale)

    @property
    def command(self) -> str:
        return f"sudo {self.repo or '<your clone>'}/update.sh"

    @property
    def checked_ago(self) -> str:
        return ago(self.checked_at)


def ago(stamp: str, now: float | None = None) -> str:
    """'just now', '5 minutes ago', '3 hours ago', '2 days ago' for an ISO time in UTC."""
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return ""
    s = max((now or time.time()) - t, 0)
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
        for key in ("checked_at", "state", "message", "repo", "upstream", "current", "previous"):
            setattr(st, key, str(raw.get(key) or ""))
        st.behind = int(raw.get("behind") or 0)
        st.commits = [c for c in raw.get("commits") or [] if isinstance(c, dict)]
        st.local_changes = [str(f) for f in raw.get("local_changes") or []]
        st.dashboard_updates = raw.get("dashboard_updates") is True
    req = d / "request"
    try:
        st.pending = req.read_text().strip()
        st.pending_stale = (now or time.time()) - req.stat().st_mtime > STALE_REQUEST
    except OSError:
        pass
    return st


def request(root: Path, action: str) -> None:
    """Ask the host to check or update; update.sh deletes the file when it starts."""
    if action not in ACTIONS:
        raise ValueError(f"unknown update action {action!r}")
    d = folder(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "request.tmp"
    tmp.write_text(action + "\n")
    tmp.replace(d / "request")


def log_text(root: Path) -> str:
    try:
        return (folder(root) / "log.txt").read_text(errors="replace")[-20000:]
    except OSError:
        return ""
