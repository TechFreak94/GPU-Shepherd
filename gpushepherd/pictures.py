"""ComfyUI's pictures, for Storage > Pictures: what it made (its output folder, where pictures from
Open WebUI and SillyTavern land too) and what was given to it (its input folder: pictures uploaded
to edit, or to start from).

Pure looking: no deleting here. service.py moves what the user picks to the trash, where it can be
put back until the trash is emptied. Any app whose container has a ComfyUI/output or ComfyUI/input
folder counts, so a ComfyUI added by hand is found too.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Instance

FOLDERS = {"output": "Made", "input": "Uploaded"}       # ComfyUI's folder name: what the page calls it
PICTURE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VIDEO_EXT = (".mp4", ".webm", ".mov")
PER_PAGE = 48
DAY = 86400


@dataclass
class Picture:
    name: str                # relative to the folder, with / between subfolders
    size: int
    modified: float

    @property
    def video(self) -> bool:
        return self.name.lower().endswith(VIDEO_EXT)

    @property
    def age_days(self) -> int:
        return max(0, int((time.time() - self.modified) // DAY))


def folders(inst: Instance) -> dict[str, str]:
    """An app's ComfyUI picture folders on this machine: {"output"/"input": host path}."""
    out = {}
    for v in inst.volumes:
        for kind in FOLDERS:
            if v.container.rstrip("/").endswith(f"ComfyUI/{kind}"):
                out[kind] = v.host
    return out


def scan(folder: str | Path) -> list[Picture]:
    """The pictures and videos in a folder and its subfolders, newest first. Hidden folders (the
    trash's among them) are left out."""
    root, out = Path(folder), []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            if f.startswith(".") or not f.lower().endswith(PICTURE_EXT + VIDEO_EXT):
                continue
            p = Path(d) / f
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(Picture(p.relative_to(root).as_posix(), st.st_size, st.st_mtime))
    return sorted(out, key=lambda x: (-x.modified, x.name))


def path(folder: str | Path, name: str) -> Path | None:
    """One picture's path, or None when the name is not a picture inside the folder (no ../ tricks)."""
    if not name or name.startswith("/") or "\\" in name or "\0" in name:
        return None
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    p = Path(folder).joinpath(*parts)
    if not name.lower().endswith(PICTURE_EXT + VIDEO_EXT) or p.is_symlink() or not p.is_file():
        return None
    return p
