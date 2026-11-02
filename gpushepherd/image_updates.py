"""Image updates: newer releases of the apps' Docker images, and bringing them in.

Apps are installed from tags that move with each release (open-webui:main, librechat:latest,
mongo:8.0), so a new release shows up as the same tag pointing at a different image in the
registry. A check asks the registry, through Docker so its logins and mirrors apply, which image
the tag points at now (its digest) and compares that with the image on this machine. Images built
on this machine have no registry digest and are never reported.

Each instance chooses how it is updated (Instance.auto_update):
  manual  the dashboard card says an update is there; its Update button downloads it and restarts
          the app (waiting for running work, like any stop)
  auto    GPU Shepherd does the same by itself after the check that found it

Either way a stopped app only gets the download and runs the new version at its next start, and
a restart that would move other instances is left to the user (the card says to restart).

<data>/state/images.yaml keeps what the checks found:
  checked  when the last check ran (seconds since the epoch)
  images   image -> {id: local image ID, local: [its registry digests], remote: digest the tag
           points at now, error: why it could not be checked}
  updated  instance -> {at: when, image: the image ID it moved off}

The checks and the comparison are here; downloading and restarting are in the service.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Instance, load_yaml

RECENT = 24 * 3600          # seconds an automatic update stays mentioned on the card


@dataclass
class Status:
    state: str              # current | available | restart (downloaded, still running the old one) | unknown
    text: str = ""          # what the card says
    error: str = ""         # why it could not be checked
    checked: float = 0.0
    updated: float = 0.0    # when it was last updated, if recently

    @property
    def available(self) -> bool:
        return self.state == "available"


def status(inst: Instance, running_image_id: str, running: bool, rec: dict | None, updated: float = 0.0,
           now: float | None = None) -> Status:
    """How up to date an instance is. `rec` is images.yaml's entry for its image; running_image_id
    the image ID its container was created from."""
    now = time.time() if now is None else now
    recent = updated if updated and now - updated < RECENT else 0.0
    if not rec:
        return Status("unknown", updated=recent)
    checked = float(rec.get("at") or 0)
    local_id, remote = rec.get("id") or "", rec.get("remote") or ""
    if rec.get("error"):
        return Status("unknown", error=rec["error"], checked=checked, updated=recent)
    if remote and local_id and remote not in (rec.get("local") or []):
        return Status("available", "A newer version is available", checked=checked, updated=recent)
    if running and running_image_id and local_id and running_image_id != local_id:
        return Status("restart", "New version downloaded: restart to use it", checked=checked, updated=recent)
    return Status("current" if remote else "unknown", checked=checked, updated=recent)


def ago(ts: float, now: float | None = None) -> str:
    """5 minutes ago, 3 hours ago, 2 days ago."""
    secs = max(int((time.time() if now is None else now) - ts), 0)
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if secs >= size:
            n = secs // size
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


def digests(repo_digests: list[str]) -> list[str]:
    """["ghcr.io/x/y@sha256:..."] -> ["sha256:..."]"""
    return sorted({d.rpartition("@")[2] for d in repo_digests if "@" in d})


class Records:
    """<data>/state/images.yaml (see the module docstring)."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "state" / "images.yaml"
        self._lock = threading.Lock()

    def read(self) -> dict:
        try:
            raw = load_yaml(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        raw = raw if isinstance(raw, dict) else {}
        return {"checked": float(raw.get("checked") or 0), "images": dict(raw.get("images") or {}),
                "updated": dict(raw.get("updated") or {})}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def checked(self, images: dict[str, dict], at: float | None = None) -> None:
        """Save a check's findings: image -> {id, local, remote, error}."""
        at = time.time() if at is None else at
        with self._lock:
            data = self.read()
            data["checked"] = at
            data["images"] = {ref: rec | {"at": at} for ref, rec in images.items()}
            self._write(data)

    def downloaded(self, ref: str, image_id: str, local: list[str]) -> None:
        """The image was just pulled: it is what the registry has now."""
        with self._lock:
            data = self.read()
            rec = dict(data["images"].get(ref) or {})
            rec.update(id=image_id, local=local, error="")
            if local:
                rec["remote"] = local[0] if rec.get("remote") not in local else rec["remote"]
            data["images"][ref] = rec
            self._write(data)

    def updated(self, iid: str, old_id: str, at: float | None = None) -> None:
        with self._lock:
            data = self.read()
            data["updated"][iid] = {"at": time.time() if at is None else at, "image": old_id}
            self._write(data)
