"""Moving a folder the apps use (a models folder, an app's data folder) to another place, such as a
new disk, without editing anything by hand.

A move, run as a background change by service.py:
  1. checks the new place: its parent folder exists, it is new or empty, and the disk it is on has
     room (and says when it is the same disk, which usually means the new disk isn't mounted);
  2. stops the apps using the folder (the normal way: running work finishes first);
  3. copies everything, then compares the two copies (every name, and every file's size);
  4. only then points the apps at the new place (instances/<id>.yaml, and the trash's records);
  5. starts again what was running.
The old copy stays where it was until it is deleted from Settings > Storage, and "Move back"
points the apps at it again. Anything that fails before step 4 leaves the apps as they were and
removes the unfinished copy.

GPU Shepherd usually can't see the new disk (only its data folder is mounted into its container),
so the checking, copying and deleting run in a helper container (HelperMover). Demo mode and the
tests use LocalMover, which does the same with Python. Moves are listed in <data>/state/moves.yaml.
"""
from __future__ import annotations

import os
import posixpath
import re
import shlex
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .config import Instance, load_yaml

KIND = "move"                          # label of the helper containers
SYSTEM = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/run", "/sbin", "/sys", "/usr", "/var/run")


@dataclass
class Movable:
    label: str                         # "Models", "App data: Open WebUI"
    path: str                          # on this machine
    used_by: list[str]                 # instance ids
    home: str = ""                     # where it was before GPU Shepherd moved it ("" = it is still there)


@dataclass
class Check:
    """What the new place looks like. error = the move can't go ahead."""
    size: int = 0                      # bytes in the folder to move
    free: int = 0                      # bytes free where it goes
    disk: str = ""                     # the disk the new place is on, e.g. /dev/sdb1
    same_disk: bool = False            # the same disk as the folder now
    exists: bool = False               # the new place is an empty folder already
    windows: bool = False              # a Windows drive on WSL2 (/mnt/c ...): slow, no Linux owners
    error: str = ""

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.windows:
            out.append("This is a Windows drive. It works, but models load much more slowly from it than from "
                       "Linux's own disk, and files there keep no Linux owners or permissions.")
        if self.same_disk:
            out.append("The new place is on the same disk as the folder now, so no space is freed. If it should "
                       "be on another disk, check that disk is mounted (it may not be, after a restart).")
        return out


@dataclass
class Record:
    id: str
    label: str
    source: str
    dest: str
    used_by: list[str] = field(default_factory=list)
    size: int = 0
    status: str = "copying"            # copying | done | failed | cancelled | moved-back
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    leftover: str = ""                 # a copy that can be deleted: the old one, or an unfinished one
    job: int = 0


class History:
    """<data>/state/moves.yaml: every move, newest first."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "state" / "moves.yaml"
        self._lock = threading.Lock()

    def all(self) -> list[Record]:
        try:
            raw = load_yaml(self.path.read_text(encoding="utf-8")) or []
        except (OSError, yaml.YAMLError):
            return []
        out = []
        for r in raw if isinstance(raw, list) else []:
            try:
                out.append(Record(**r))
            except TypeError:
                continue
        return sorted(out, key=lambda r: r.started, reverse=True)

    def get(self, mid: str) -> Record:
        r = next((r for r in self.all() if r.id == mid), None)
        if not r:
            raise KeyError(f"no move called '{mid}'")
        return r

    def put(self, rec: Record) -> Record:
        with self._lock:
            rows = [r for r in self.all() if r.id != rec.id] + [rec]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump([asdict(r) for r in sorted(rows, key=lambda r: r.started)],
                                          sort_keys=False), encoding="utf-8")
            os.replace(tmp, self.path)
        return rec

    def new_id(self, source: str) -> str:
        base = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + (Path(source).name or "root")
        ids, n, out = {r.id for r in self.all()}, 1, base
        while out in ids:
            n += 1
            out = f"{base}-{n}"
        return out


# --------------------------------------------------------------------------- what can move

def homes(records: list[Record]) -> dict[str, tuple[str, str]]:
    """Folders finished moves put somewhere else: {where it is now: (where it first was, label)}.
    Moves one after another (disk 1 to disk 2 to disk 3) lead back to the first place."""
    first: dict[str, tuple[str, str]] = {}
    for r in sorted(records, key=lambda r: r.started):
        if r.status == "done":                      # a move undone with "Move back" left things as they were
            first[r.dest] = first.pop(r.source, (r.source, r.label))
    return {now: was for now, was in first.items() if was[0] != now}


def movable(instances: dict[str, Instance], appdata: str, model_folders: list[str],
            moved: dict[str, tuple[str, str]] | None = None) -> list[Movable]:
    """Models folders, each app data folder (appdata/<folder>), and folders moved out of those, with
    the instances using them and, for moved ones, where they were first (`home`)."""
    moved = moved or {}
    out = [Movable("Models", m, []) for m in model_folders]
    tops: dict[str, list[str]] = {}
    for inst in instances.values():
        for v in inst.volumes:
            p = Path(v.host)
            if p.is_relative_to(appdata) and p != Path(appdata):
                tops.setdefault(str(Path(appdata) / p.relative_to(appdata).parts[0]), [])
    for now in moved:
        if any(inside(v.host, now) for i in instances.values() for v in i.volumes):
            tops.setdefault(now, [])
    for top in sorted(tops):
        if not any(Path(top).is_relative_to(m.path) for m in out):
            label = moved[top][1].split(":")[0] if top in moved else "App data"
            out.append(Movable(label, top, []))
    for m in out:
        m.used_by = sorted(i.id for i in instances.values() if any(inside(v.host, m.path) for v in i.volumes))
        m.home = moved[m.path][0] if m.path in moved else ""
        names = [instances[i].name for i in m.used_by]
        if m.label == "App data" and names:
            m.label = f"App data: {', '.join(names)}"
    return out


def inside(path: str, folder: str) -> bool:
    return Path(path) == Path(folder) or Path(path).is_relative_to(folder)


def relocate(path: str, old: str, new: str) -> str:
    """`path` with the folder `old` it is in (or is) replaced by `new`; other paths unchanged."""
    if not inside(path, old):
        return path
    rest = Path(path).relative_to(old)
    return str(Path(new) / rest) if str(rest) != "." else new


def windows_drive(path: str) -> bool:
    """A Windows drive as WSL2 mounts it (/mnt/c, /mnt/d ...). Copies there keep no owners: a
    `cp -a` would fail trying to set them."""
    from . import host
    return host.is_wsl() and re.match(r"^/mnt/[a-z](/|$)", posixpath.normpath(path)) is not None


def check_paths(source: str, dest: str, data_dir: str) -> str:
    """A problem with the two paths themselves (before looking at the disk), or ""."""
    dest = dest.strip()
    if not dest.startswith("/"):
        return "The new place must be a full path, starting with /"
    if posixpath.normpath(dest) != dest.rstrip("/") or "//" in dest:
        return "Write the new place without .. or // in it"
    dest = dest.rstrip("/")
    if not dest or dest == "/":
        return "Pick a folder for the new place, not /"
    if any(inside(dest, s) for s in SYSTEM):
        return f"{dest} is a system folder"
    if inside(data_dir, source):
        return "The folder holds GPU Shepherd's data folder, which can't be moved from here"
    if inside(dest, source):
        return "The new place can't be inside the folder being moved"
    if inside(source, dest):
        return "The new place can't be a folder the one being moved is in"
    if inside(dest, data_dir) and not inside(dest, str(Path(data_dir) / "appdata")) \
            and not inside(dest, str(Path(data_dir) / "models")):
        return "Inside the data folder, only appdata/ and models/ are for app files"
    return ""


# --------------------------------------------------------------------------- doing it

class Mover:
    """Checks, copies and deletes. start() returns at once; poll() says how it is going."""

    def check(self, source: str, dest: str) -> Check: ...
    def start(self, mid: str, source: str, dest: str) -> None: ...
    def poll(self, mid: str) -> tuple[bool, int | None, int]:
        """(still copying, exit code once finished (0 = copied and compared), bytes copied so far)"""
        ...
    def stop(self, mid: str) -> None: ...
    def remove(self, path: str) -> None: ...


# The helper sees the whole machine read-only at /host for the check, and for the copy the folder
# read-only at /src and the new place's parent folder at /dst. Only busybox commands.
CHECK = r"""S=/host{src}; P=/host{parent}; D=/host{dst}
[ -d "$S" ] || {{ echo "ERROR ${{S#/host}} does not exist"; exit 0; }}
[ -d "$P" ] || {{ echo "ERROR ${{P#/host}} does not exist (is the disk mounted?)"; exit 0; }}
if [ -e "$D" ] || [ -L "$D" ]; then
  [ -d "$D" ] && [ ! -L "$D" ] && [ -z "$(ls -A "$D")" ] || {{ echo "ERROR ${{D#/host}} already exists and is not an empty folder"; exit 0; }}
  echo EXISTS
fi
echo "SIZE $(du -sk "$S" | cut -f1)"
df -Pk "$P" | tail -n 1 | while read -r fs total used free rest; do echo "FREE $free"; echo "DISK $fs"; done
[ "$(stat -c %d "$S")" = "$(stat -c %d "$P")" ] && echo SAME
true
"""

COPY = r"""S=/src; D=/dst/{name}
mkdir -p "$D" || exit 3
(cd "$S" && cp {keep} . "$D/") & pid=$!
while kill -0 $pid 2>/dev/null; do echo "COPIED $(du -sk "$D" | cut -f1)"; sleep 5; done
wait $pid || {{ echo "ERROR the copy failed"; exit 4; }}
echo "COPIED $(du -sk "$D" | cut -f1)"
list() {{ cd "$1" && {{ find . -type f -exec stat -c '%n %s' {{}} + ; find . ! -type f -exec stat -c '%n %F' {{}} + ; }} | sort; }}
list "$S" > /tmp/a; list "$D" > /tmp/b
cmp -s /tmp/a /tmp/b || {{ echo "ERROR the copy is not the same as the folder"; diff /tmp/a /tmp/b | head -n 5; exit 5; }}
echo "VERIFIED $(wc -l < /tmp/a) items"
"""


def parse_check(out: str) -> Check:
    c = Check()
    for line in out.splitlines():
        key, _, val = line.strip().partition(" ")
        try:
            if key == "ERROR":
                c.error = val
            elif key == "EXISTS":
                c.exists = True
            elif key == "SIZE":
                c.size = int(val) * 1024
            elif key == "FREE":
                c.free = int(val) * 1024
            elif key == "DISK":
                c.disk = val
            elif key == "SAME":
                c.same_disk = True
        except ValueError:
            continue
    if not c.error and not c.disk:
        c.error = "could not look at the new place"
    return c


class HelperMover(Mover):
    def __init__(self, backend, image: str):
        self.backend, self.image = backend, image

    def name(self, mid: str) -> str:
        return f"gpu-shepherd-move-{mid}"[:120]

    def check(self, source: str, dest: str) -> Check:
        q = shlex.quote
        script = CHECK.format(src=q(source), parent=q(str(Path(dest).parent)), dst=q(dest))
        return parse_check(self.backend.run_task(self.image, script, [("/", "/host", True)], "0:0"))

    def start(self, mid: str, source: str, dest: str) -> None:
        self.backend.run_helper(self.name(mid), self.image, COPY.format(name=shlex.quote(Path(dest).name),
                                                                         keep="-dR" if windows_drive(dest) else "-a"),
                                [(source, "/src", True), (str(Path(dest).parent), "/dst", False)], "0:0", KIND)

    def poll(self, mid: str) -> tuple[bool, int | None, int]:
        h = self.backend.helpers(KIND).get(self.name(mid))
        if h is None:
            return False, None, 0
        copied = 0
        for line in reversed(self.backend.logs(self.name(mid), tail=20).splitlines()):
            if line.startswith("COPIED "):
                try:
                    copied = int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    pass
                break
        return h.running, h.exit_code, copied

    def stop(self, mid: str) -> None:
        self.backend.remove(self.name(mid))

    def remove(self, path: str) -> None:
        p = Path(_removable(path))
        self.backend.run_task(self.image, f"rm -rf /p/{shlex.quote(p.name)}", [(str(p.parent), "/p", False)], "0:0")


def _removable(path: str) -> str:
    """Only a folder of its own, never / or a system folder, is deleted."""
    p = Path(path)
    if not p.is_absolute() or p.name in ("", ".", "..") or p.parent == p or any(inside(path, s) for s in SYSTEM):
        raise ValueError(f"won't delete {path}")
    return path


class LocalMover(Mover):
    """The same with Python, for paths GPU Shepherd can see (demo mode and tests)."""

    def __init__(self):
        self.runs: dict[str, dict] = {}

    def check(self, source: str, dest: str) -> Check:
        s, d = Path(source), Path(dest)
        if not s.is_dir():
            return Check(error=f"{source} does not exist")
        if not d.parent.is_dir():
            return Check(error=f"{d.parent} does not exist (is the disk mounted?)")
        c = Check(disk=str(d.parent))
        if d.exists() or d.is_symlink():
            if not d.is_dir() or d.is_symlink() or any(d.iterdir()):
                return Check(error=f"{dest} already exists and is not an empty folder")
            c.exists = True
        c.size = sum(f.stat().st_size for f in s.rglob("*") if f.is_file() and not f.is_symlink())
        c.free = shutil.disk_usage(d.parent).free
        c.same_disk = os.stat(s).st_dev == os.stat(d.parent).st_dev
        return c

    def start(self, mid: str, source: str, dest: str) -> None:
        run = self.runs[mid] = {"running": True, "code": None}

        def copy():
            try:
                shutil.copytree(source, dest, symlinks=True, dirs_exist_ok=True)
                run["code"] = 0 if _listing(Path(source)) == _listing(Path(dest)) else 5
            except OSError:
                run["code"] = 4
            run["running"] = False
        threading.Thread(target=copy, daemon=True).start()

    def poll(self, mid: str) -> tuple[bool, int | None, int]:
        run = self.runs.get(mid)
        return (run["running"], run["code"], 0) if run else (False, None, 0)

    def stop(self, mid: str) -> None:
        self.runs.pop(mid, None)

    def remove(self, path: str) -> None:
        p = Path(_removable(path))
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)


def _listing(root: Path) -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() and not p.is_symlink() else -1)
                  for p in root.rglob("*"))
