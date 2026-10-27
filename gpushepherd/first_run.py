"""First start without install.sh: creating a preset, as install.sh does on its first run.

compose.standalone.yaml sets these (nothing happens when SHEPHERD_PRESET is unset, as in a clone
installed with install.sh, which does this itself):

  SHEPHERD_PRESET           a folder name in presets/ ("starter"), a preset folder, or "none"
  SHEPHERD_MODELS_DIR       the models folder, mounted at the same path (default <data dir>/models)
  SHEPHERD_DOWNLOAD_MODELS  "1": also download the preset's default model groups (downloads.txt)

It runs once: only when the data folder has no instances, and never again after that (a marker
file), so deleting every instance doesn't bring the preset back. The files are filled in the same
way as install.sh does (presets/README.md): fresh API keys and passwords, this machine's folders,
autostart off. Downloads run in the same helper containers as the Models page (downloads.py), so a
suggested model shows its progress there.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import model_library
from .config import Store
from .downloads import Downloads

log = logging.getLogger("gpushepherd")
PRESETS = Path(__file__).resolve().parent / "presets"
MARKER = ".first-run-done"
MODELS_IN_CONTAINER = "/models"


@dataclass
class Created:
    preset: Path
    name: str
    models_dir: str


def preset_dir(name: str) -> Path | None:
    d = Path(name) if "/" in name else PRESETS / name
    return d if (d / "instances").is_dir() else None


def preset_field(d: Path, key: str) -> str:
    try:
        for line in (d / "preset.yaml").read_text().splitlines():
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def create_preset(store: Store, preset: str | None = None, models_dir: str | None = None) -> Created | None:
    """Create the preset's instances in an empty data folder, once. Returns what it created."""
    preset = preset if preset is not None else os.environ.get("SHEPHERD_PRESET", "")
    root = Path(store.root)
    marker = root / MARKER
    if not preset or marker.exists():
        return None
    has_instances = any((root / "instances").glob("*.yaml"))
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{preset}\n")
    if preset == "none" or has_instances:
        return None
    d = preset_dir(preset)
    if not d:
        log.warning("SHEPHERD_PRESET: no preset named %r in %s; starting empty", preset, PRESETS)
        return None
    models = (models_dir or os.environ.get("SHEPHERD_MODELS_DIR") or str(root / "models")).rstrip("/")
    owner = os.stat(root)
    llm_key = "sk-" + secrets.token_hex(20)
    subs = [("/srv/models", models), ("/opt/gpu-shepherd", str(root)),
            (re.compile(r"sk-demo-0*"), llm_key), ("REPLACE_WITH_LLM_API_KEY", llm_key),
            ("change-me-hermes-api-key", secrets.token_hex(24)), ("change-me-please", secrets.token_hex(8)),
            ('HERMES_UID: "1000"', f'HERMES_UID: "{owner.st_uid}"'),
            ('HERMES_GID: "1000"', f'HERMES_GID: "{owner.st_gid}"'),
            (re.compile(r"^autostart: true", re.M), "autostart: false")]

    def fill(src: Path, dest: Path, mode: int | None) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text()
        except UnicodeDecodeError:              # not a text file: copied as it is
            shutil.copy2(src, dest)
            return
        for old, new in subs:
            text = old.sub(new, text) if isinstance(old, re.Pattern) else text.replace(old, new)
        dest.write_text(text)
        os.chmod(dest, mode if mode is not None else src.stat().st_mode & 0o777)

    for part in ("instances", "catalogs"):
        for f in sorted((d / part).glob("*.yaml")):
            fill(f, root / part / f.name, 0o600)
    for f in sorted(p for p in (d / "appdata").rglob("*") if p.is_file()):
        fill(f, root / f.relative_to(d), 0o600 if f.suffix == ".yaml" else None)
    if (d / "NOTES.txt").is_file():         # may hold a password
        fill(d / "NOTES.txt", root / "PRESET_NOTES.txt", 0o600)
    # every host folder the instances mount inside the data and models folders, so Docker doesn't
    # make them root-only
    Path(models).mkdir(parents=True, exist_ok=True)
    for f in (root / "instances").glob("*.yaml"):
        for host in re.findall(r"host: (/[^,}\n]*)", f.read_text()):
            host = host.strip()
            if (host.startswith(f"{root}/") or host == models or host.startswith(f"{models}/")) \
                    and not os.path.exists(host):
                os.makedirs(host, exist_ok=True)
                os.chown(host, owner.st_uid, owner.st_gid)
    name = preset_field(d, "name") or d.name
    log.warning("First start: created the %r preset (nothing starts until you press Start). Notes: %s",
                name, root / "PRESET_NOTES.txt")
    return Created(d, name, models)


@dataclass(frozen=True)
class Bundle:
    """Files of a preset's download group that aren't a suggested model, in the shape downloads.py
    takes (key, name, files, path, download_gb)."""
    key: str
    name: str
    files: tuple[model_library.File, ...]

    def path(self, f: model_library.File) -> str:
        return f.name

    @property
    def download_gb(self) -> float:
        return round(sum(f.gb for f in self.files), 1)


def downloads(preset: Path) -> list:
    """The preset's default download groups: suggested models where a file is one (so the Models
    page shows their progress), otherwise one Bundle per group."""
    groups: dict[str, float] = {}
    files: dict[str, list[tuple[str, str]]] = {}
    try:
        lines = (preset / "downloads.txt").read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "group" and parts[2] == "default":
            groups[parts[1]] = float(re.sub(r"[^0-9.]", "", parts[3]) or 0)
        elif len(parts) >= 4 and parts[0] == "file":
            files.setdefault(parts[1], []).append((parts[2], parts[3]))
    out: list = []
    for group, size in groups.items():
        rest = []
        for rel, url in files.get(group, []):
            s = next((s for s in model_library.SUGGESTIONS if any(s.path(f) == rel for f in s.files)), None)
            if s:
                if s not in out:
                    out.append(s)
            else:
                rest.append((rel, url))
        if rest:
            gb = size / len(rest)
            out.append(Bundle(f"preset-{group}", f"{preset.name} {group} models",
                              tuple(model_library.File(rel, url, gb) for rel, url in rest)))
    return out


def download(created: Created, dl: Downloads, want: bool | None = None) -> list[str]:
    """Start the preset's default downloads if SHEPHERD_DOWNLOAD_MODELS=1 and there is room."""
    if not (want if want is not None else os.environ.get("SHEPHERD_DOWNLOAD_MODELS", "") in ("1", "yes", "true")):
        return []
    items = downloads(created.preset)
    need = sum(x.download_gb for x in items)
    free = shutil.disk_usage(created.models_dir).free / 1e9
    if need > free:
        log.warning("Not downloading the preset's models: they need %.0f GB, %s has %.0f GB free",
                    need, created.models_dir, free)
        return []
    started = []
    for x in items:
        try:
            log.info("%s", dl.start(x, created.models_dir))
            started.append(x.name)
        except ValueError as err:
            log.warning("not downloading %s: %s", x.name, err)
    return started
