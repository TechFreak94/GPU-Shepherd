"""Downloads suggested models into a model server's models folder.

GPU Shepherd sees the models folder read-only (if at all), so it doesn't write there itself: each
download runs in a small helper container with the folder mounted, using curl. A download resumes
where it stopped, so starting it again after a failure or a restart doesn't begin from zero.

Progress comes from the files' sizes when GPU Shepherd can see the folder (it is mounted at the
same path), and otherwise from whether the helper is still running.
"""
from __future__ import annotations

import logging
import os
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path

from .backend import Backend
from .model_library import Suggestion

log = logging.getLogger("gpushepherd")
IMAGE = "curlimages/curl:8.11.1"
KIND = "download"
FOLDER = "/models"           # the models folder inside the helper


@dataclass
class Status:
    state: str               # ready | downloading | starting | failed | missing
    done_gb: float = 0.0
    total_gb: float = 0.0
    error: str = ""

    @property
    def pct(self) -> int | None:
        return min(round(100 * self.done_gb / self.total_gb), 99) if self.total_gb and self.done_gb else None

    @property
    def text(self) -> str:
        if self.state == "ready":
            return "Downloaded"
        if self.state == "starting":
            return "Starting the download…"
        if self.state == "downloading":
            if self.pct is None:
                return "Downloading…"
            return f"Downloading: {self.done_gb:.1f} of {self.total_gb:.1f} GB ({self.pct}%)"
        if self.state == "failed":
            return f"Download failed: {self.error}" if self.error else "Download failed"
        return f"Not downloaded ({self.total_gb:.1f} GB)"


def script(s: Suggestion) -> str:
    """Fetch each missing file to <file>.part and rename it once complete."""
    lines = ["set -e", f"mkdir -p {shlex.quote(f'{FOLDER}/{s.key}')}"]
    for f in s.files:
        dest = shlex.quote(f"{FOLDER}/{s.path(f)}")
        part = shlex.quote(f"{FOLDER}/{s.path(f)}.part")
        lines.append(f"if [ ! -s {dest} ]; then echo {shlex.quote('Downloading ' + f.name)}; "
                     f"curl -fsSL --retry 5 --retry-delay 10 -C - -o {part} {shlex.quote(f.url)}; "
                     f"mv {part} {dest}; fi")
    lines.append(f"echo {shlex.quote('Downloaded ' + s.name)}")
    return "\n".join(lines)


class Downloads:
    def __init__(self, backend: Backend, prefix: str = "shepherd-"):
        self.backend = backend
        self.prefix = prefix
        self._starting: dict[str, str] = {}      # container name -> error while pulling the image ("" = busy)
        self._lock = threading.Lock()

    def name(self, s: Suggestion) -> str:
        return f"{self.prefix}download-{s.key}"

    def status(self, s: Suggestion, models_dir: str) -> Status:
        total = sum(f.gb for f in s.files)
        root = Path(models_dir) if models_dir else None
        visible = bool(root and root.is_dir())
        if visible and all(_size(root / s.path(f)) for f in s.files):
            return Status("ready", total, total)
        name = self.name(s)
        with self._lock:
            starting = self._starting.get(name)
        if starting == "":
            return Status("starting", 0, total)
        if starting:
            return Status("failed", 0, total, starting)
        helper = self.backend.helpers(KIND).get(name)
        done = sum(_size(root / s.path(f)) or _size(root / (s.path(f) + ".part")) for f in s.files) / 1e9 \
            if visible else 0.0
        if helper and helper.running:
            return Status("downloading", round(done, 2), total)
        if helper and helper.exit_code == 0:
            return Status("ready", total, total)
        if helper and helper.exit_code is not None:
            return Status("failed", round(done, 2), total, self._error(name))
        return Status("missing", round(done, 2), total)

    def start(self, s: Suggestion, models_dir: str) -> str:
        """Start downloading (or resume). The helper image is fetched first, in the background."""
        if not models_dir.startswith("/"):
            raise ValueError("the model server has no models folder to download into")
        name = self.name(s)
        with self._lock:
            if self._starting.get(name) == "":
                return f"{s.name} is already starting to download."
            self._starting[name] = ""
        helper = self.backend.helpers(KIND).get(name)
        if helper and helper.running:
            with self._lock:
                self._starting.pop(name, None)
            return f"{s.name} is already downloading."

        def run():
            try:
                self.backend.ensure_image(IMAGE, lambda line: log.info("download %s: %s", s.key, line))
                self.backend.run_helper(name, IMAGE, script(s), [(models_dir, FOLDER, False)],
                                        _owner(models_dir), KIND)
            except Exception as err:  # noqa: BLE001 - shown as the download's status
                log.warning("could not start downloading %s: %s", s.key, err)
                with self._lock:
                    self._starting[name] = str(err) or err.__class__.__name__
                return
            with self._lock:
                self._starting.pop(name, None)
            log.info("downloading %s into %s", s.key, models_dir)

        threading.Thread(target=run, daemon=True, name=f"download-{s.key}").start()
        return f"Downloading {s.name} ({s.download_gb:.1f} GB) into {models_dir}/{s.key}."

    def cancel(self, s: Suggestion) -> str:
        """Stop a download; what it fetched so far is kept, so starting it again resumes."""
        self.backend.remove(self.name(s))
        with self._lock:
            self._starting.pop(self.name(s), None)
        return f"Stopped downloading {s.name}. Download it again to carry on where it stopped."

    def _error(self, name: str) -> str:
        lines = [x for x in self.backend.logs(name, tail=20).splitlines() if x.strip()]
        return lines[-1].strip()[:200] if lines else ""


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _owner(models_dir: str) -> str | None:
    """Download as the models folder's owner, so the files are the user's, not root's. When GPU
    Shepherd can't see the folder, as root (the folder is the user's to fix up)."""
    try:
        st = os.stat(models_dir)
        return f"{st.st_uid}:{st.st_gid}"
    except OSError:
        return "0:0"
