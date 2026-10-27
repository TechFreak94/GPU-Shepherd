"""The trash: what a delete removes is moved here first, so it can be put back.

    trash/<entry>/manifest.yaml    what was deleted, when, and where each item came from
    trash/<entry>/files/<n>-<name> the items themselves

Deleting an instance moves its YAML, its model list, its generated files and its app data folder
(appdata/<folder>, unless another instance still uses it) into one entry. Everything lives in the
data folder, so moving is a rename: instant, and the space is only freed when the entry is
emptied. Model files are never put here by a delete. Entries older than the Trash setting's
number of days are emptied by themselves (0 keeps them until emptied by hand).
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("gpushepherd")
DAY = 86400


@dataclass
class Item:
    original: str            # where it was
    stored: str              # path inside the entry ("" = could not be moved, left where it was)


@dataclass
class Entry:
    id: str
    title: str
    deleted: float           # unix time
    instances: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    size: int = 0            # bytes, filled in by Trash.entries()

    @property
    def age_days(self) -> int:
        return int((time.time() - self.deleted) // DAY)


class Trash:
    def __init__(self, data_dir: Path):
        self.data = Path(data_dir)
        self.root = self.data / "trash"

    # ------------------------------------------------------------------ putting things in
    def put(self, title: str, instances: list[str], paths: list[Path]) -> Entry:
        """Move `paths` (inside the data folder) into a new entry. A path that can't be moved
        (another disk, no permission) stays where it is and is listed as such."""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        eid = _free(self.root, f"{stamp}-{(instances or ['files'])[0]}")
        files = self.root / eid / "files"
        files.mkdir(parents=True)
        entry = Entry(eid, title, time.time(), list(instances))
        for n, path in enumerate(p for p in paths if p.exists() or p.is_symlink()):
            dest = files / f"{n}-{path.name}"
            try:
                os.rename(path, dest)
                entry.items.append(Item(str(path), str(dest.relative_to(self.root / eid))))
            except OSError as err:
                log.warning("trash: left %s where it is (%s)", path, err)
                entry.items.append(Item(str(path), ""))
        self._save(entry)
        return entry

    # ------------------------------------------------------------------ looking
    def entries(self, sizes: bool = True) -> list[Entry]:
        out = []
        if not self.root.is_dir():
            return out
        for d in self.root.iterdir():
            e = self._load(d.name)
            if e:
                e.size = tree_size(d / "files") if sizes else 0
                out.append(e)
        return sorted(out, key=lambda e: e.deleted, reverse=True)

    def entry(self, eid: str) -> Entry:
        e = self._load(eid) if "/" not in eid and not eid.startswith(".") else None
        if not e:
            raise KeyError(f"nothing called '{eid}' in the trash")
        return e

    # ------------------------------------------------------------------ getting rid of / back
    def restore(self, eid: str) -> Entry:
        """Put every item back where it was. Refuses (moving nothing) when something now stands
        in one of those places, such as a new instance with the same name."""
        e = self.entry(eid)
        base = self.root / eid
        clashes = [i.original for i in e.items if i.stored and Path(i.original).exists()]
        if clashes:
            raise FileExistsError("something else is there now: " + ", ".join(clashes))
        for i in e.items:
            if i.stored:
                Path(i.original).parent.mkdir(parents=True, exist_ok=True)
                os.rename(base / i.stored, i.original)
        shutil.rmtree(base)
        return e

    def purge(self, eid: str) -> Entry:
        e = self.entry(eid)
        shutil.rmtree(self.root / eid)
        return e

    def purge_older(self, days: int) -> list[Entry]:
        """Empty entries deleted more than `days` days ago (0 = never)."""
        if days <= 0:
            return []
        gone = []
        for e in self.entries(sizes=False):
            if e.age_days >= days:
                try:
                    gone.append(self.purge(e.id))
                except OSError as err:
                    log.warning("trash: could not empty %s: %s", e.id, err)
        if gone:
            log.info("trash: emptied %d entr%s older than %d days", len(gone), "y" if len(gone) == 1 else "ies", days)
        return gone

    # ------------------------------------------------------------------ files
    def _save(self, e: Entry) -> None:
        data = {"title": e.title, "deleted": e.deleted, "instances": e.instances,
                "items": [{"original": i.original, "stored": i.stored} for i in e.items]}
        (self.root / e.id / "manifest.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _load(self, eid: str) -> Entry | None:
        try:
            raw = yaml.safe_load((self.root / eid / "manifest.yaml").read_text(encoding="utf-8")) or {}
            return Entry(eid, str(raw.get("title", eid)), float(raw.get("deleted", 0)),
                         [str(x) for x in raw.get("instances") or []],
                         [Item(str(i["original"]), str(i.get("stored", ""))) for i in raw.get("items") or []])
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
            return None


def tree_size(path: Path) -> int:
    """Bytes used by a file or folder (not following symlinks); 0 if it isn't there."""
    total = 0
    try:
        if not path.is_dir() or path.is_symlink():
            return path.lstat().st_size
        for dirpath, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.lstat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _free(root: Path, name: str) -> str:
    n, out = 1, name
    while (root / out).exists():
        n += 1
        out = f"{name}-{n}"
    return out
