"""The machine's disks: which drives are there, mounting the ones that have no mount point yet,
and listing folders for the "where to?" picker (Settings > Storage).

GPU Shepherd runs in a container and only sees its own data folder, so the host is looked at (and
a newly attached drive mounted) from a short-lived helper container (HelperHost): the host's own
`lsblk` and `mount` run through `nsenter` in a privileged helper; listing folders only needs the
whole machine mounted read-only. Demo mode and the tests use LocalHost.

Rules for mounting, because a drive can hold someone's data:
  * only a drive that already has a filesystem GPU Shepherd can use is mounted; it is never
    formatted, partitioned or written to, and an empty drive is only reported;
  * encrypted, RAID/LVM member, swap and boot/recovery partitions are left alone;
  * the mount point is a new folder under /mnt (named after the drive's label), and with
    `persist` an /etc/fstab line (by UUID, `nofail`) brings it back after a restart.
What GPU Shepherd mounted (and what failed) is kept in <data>/state/drives.yaml.
"""
from __future__ import annotations

import json
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

from .config import load_yaml

MOUNT_ROOT = "/mnt"
# Filesystems that mean "this is not a plain data drive": GPU Shepherd leaves them alone.
SKIP_FS = {"swap": "swap space", "LVM2_member": "part of an LVM volume", "crypto_LUKS": "encrypted (unlock it first)",
           "linux_raid_member": "part of a RAID", "zfs_member": "part of a ZFS pool", "BitLocker": "encrypted with BitLocker",
           "bcache": "part of a bcache", "ceph_bluestore": "used by Ceph", "isw_raid_member": "part of a RAID",
           "ddf_raid_member": "part of a RAID", "squashfs": "a read-only image", "iso9660": "a disc image"}
# Partition types of boot, recovery and reserved partitions (GPT type GUIDs, lower case).
SKIP_PARTTYPE = {"c12a7328-f81f-11d2-ba4b-00a0c93ec93b": "boot partition (EFI)",
                 "21686148-6449-6e6f-744e-656564454649": "boot partition (BIOS)",
                 "e3c9e316-0b5c-4db8-817d-f92df00215ae": "reserved by Windows",
                 "de94bba4-06d1-4d40-a16a-bfd50179d6ac": "recovery partition",
                 "0657fd6d-a4ab-43c4-84e5-0933c84b4f4f": "swap space"}
SYSTEM_MOUNTS = ("/", "/boot", "/boot/efi", "/home", "/usr", "/var", "/opt", "/srv", "/tmp", "/efi", "[SWAP]")
HIDDEN_ROOT = ("bin", "boot", "dev", "etc", "lib", "lib64", "lost+found", "proc", "run", "sbin", "snap", "sys", "usr")


@dataclass
class Drive:
    """One partition (or a whole disk with no partitions). state:
    mounted    in use at `mountpoint`
    ready      has a filesystem and no mount point: can be mounted
    empty      nothing on it (no filesystem): GPU Shepherd never formats a drive
    unsupported  a filesystem GPU Shepherd won't mount, `note` says why
    failed     mounting it was tried and did not work, `note` says why"""
    device: str                         # /dev/sdb1
    name: str                           # what to call it: the label, else the model
    size: int = 0
    fstype: str = ""
    label: str = ""
    uuid: str = ""
    model: str = ""
    mountpoint: str = ""
    free: int | None = None
    removable: bool = False
    state: str = "ready"
    note: str = ""
    system: bool = False                # holds the operating system: shown, but not for moving things to
    by_us: bool = False                 # GPU Shepherd mounted it


# --------------------------------------------------------------------------- looking at the disks

LSBLK_COLUMNS = "NAME,KNAME,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINT,RM,RO,MODEL,PARTTYPE,FSAVAIL"


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _flag(v) -> bool:
    return v in (True, 1, "1", "true")


def parse_lsblk(text: str) -> list[Drive]:
    """The drives in `lsblk -J -b` output: partitions, and disks that have none. Disks that only
    hold partitions, loop devices, optical drives and RAM disks are not drives."""
    try:
        tree = json.loads(text).get("blockdevices") or []
    except (ValueError, AttributeError):
        return []
    out: list[Drive] = []

    def walk(dev: dict, parent_model: str, removable: bool) -> None:
        kids = dev.get("children") or []
        model = (dev.get("model") or parent_model or "").strip()
        removable = removable or _flag(dev.get("rm"))
        kind = dev.get("type") or ""
        if kind in ("loop", "rom", "ram", "zram") or str(dev.get("kname", "")).startswith(("ram", "zram")):
            return
        has_fs = bool(dev.get("fstype"))
        if kids and not has_fs:
            for k in kids:
                walk(k, model, removable)
            return
        if kind not in ("part", "disk", "lvm", "crypt", "md", "dm"):
            return
        out.append(_drive(dev, model, removable))
        for k in kids:
            walk(k, model, removable)

    for d in tree:
        walk(d, "", False)
    return out


def _drive(dev: dict, model: str, removable: bool) -> Drive:
    fs = dev.get("fstype") or ""
    mp = dev.get("mountpoint") or ""
    if isinstance(dev.get("mountpoints"), list):
        mp = next((m for m in dev["mountpoints"] if m), mp)
    d = Drive(device=str(dev.get("kname") and "/dev/" + dev["kname"] or dev.get("name") or ""),
              name=(dev.get("label") or "").strip() or model or str(dev.get("name") or "").lstrip("├─└─│ "),
              size=_int(dev.get("size")) or 0, fstype=fs, label=(dev.get("label") or "").strip(),
              uuid=dev.get("uuid") or "", model=model, mountpoint=mp, free=_int(dev.get("fsavail")),
              removable=removable)
    if mp:
        d.state = "mounted"
        d.system = mp in SYSTEM_MOUNTS or any(mp.startswith(p + "/") for p in ("/boot", "/usr", "/var/lib", "/etc"))
        if mp == "[SWAP]":
            d.state, d.note = "unsupported", "swap space"
        return d
    parttype = (dev.get("parttype") or "").lower()
    if _flag(dev.get("ro")):
        d.state, d.note = "unsupported", "read-only"
    elif not fs:
        d.state, d.note = "empty", "Nothing on it yet. GPU Shepherd never formats a drive: set it up with your system's disk tool, then it shows here."
    elif fs in SKIP_FS:
        d.state, d.note = "unsupported", SKIP_FS[fs]
    elif parttype in SKIP_PARTTYPE:
        d.state, d.note = "unsupported", SKIP_PARTTYPE[parttype]
    elif not d.uuid:
        d.state, d.note = "unsupported", "has no ID to find it by"
    return d


def mount_name(drive: Drive, taken: set[str]) -> str:
    """Where to mount a drive: /mnt/<label>, or /mnt/disk-<first of its ID> without a label."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", drive.label).strip("-.")[:32]
    base = f"{MOUNT_ROOT}/{slug or 'disk-' + (drive.uuid.replace('-', '')[:8] or 'new')}"
    out, n = base, 2
    while out in taken:
        out, n = f"{base}-{n}", n + 1
    return out


# --------------------------------------------------------------------------- folders

@dataclass
class Folder:
    name: str
    mount: bool = False                 # another disk is mounted here


@dataclass
class Listing:
    path: str
    folders: list[Folder] = field(default_factory=list)
    free: int = 0
    total: int = 0
    disk: str = ""
    error: str = ""


def clean_path(path: str) -> str:
    """`path` as a plain absolute path, or "/" when it is empty or odd."""
    p = posixpath.normpath("/" + (path or "").strip().lstrip("/"))
    return "/" if p.startswith("//") else p


def parent_of(path: str) -> str:
    return posixpath.dirname(path.rstrip("/")) or "/"


# --------------------------------------------------------------------------- state kept

class Memory:
    """<data>/state/drives.yaml: the drives GPU Shepherd mounted, and the ones it could not."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "state" / "drives.yaml"
        self._lock = threading.Lock()

    def read(self) -> dict:
        try:
            raw = load_yaml(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {"mounted": {}, "failed": {}}
        raw = raw if isinstance(raw, dict) else {}
        return {"mounted": dict(raw.get("mounted") or {}), "failed": dict(raw.get("failed") or {})}

    def update(self, uuid: str, mounted: str | None = None, failed: str | None = None) -> None:
        with self._lock:
            data = self.read()
            data["mounted"].pop(uuid, None)
            data["failed"].pop(uuid, None)
            if mounted:
                data["mounted"][uuid] = {"at": mounted, "when": int(time.time())}
            if failed:
                data["failed"][uuid] = failed
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            os.replace(tmp, self.path)


# --------------------------------------------------------------------------- the host

class Host:
    """Looks at the host's disks and folders. All of it is read-only except mount()."""

    def scan(self) -> list[Drive]: ...
    def mount(self, drive: Drive, base: str, persist: bool) -> str:
        """Mount a drive at `base` (or base-2 ... when taken). Returns where it is now mounted."""
        ...
    def browse(self, path: str) -> Listing: ...


# On the host (through nsenter): the disks as JSON.
SCAN = "lsblk -J -b -o {cols} 2>/dev/null || lsblk -J -b -o {cols}"
# On the host: mount a drive found by its ID. Prints MOUNTED <where> or ERROR <why>.
MOUNT = r"""dev=/dev/disk/by-uuid/{uuid}; base={base}
[ -b "$dev" ] || {{ echo "ERROR the drive is not there any more"; exit 0; }}
now=$(lsblk -no MOUNTPOINT "$dev" 2>/dev/null | head -n 1)
if [ -n "$now" ]; then echo "MOUNTED $now"; exit 0; fi
fs=$(lsblk -no FSTYPE "$dev" 2>/dev/null | head -n 1)
[ -n "$fs" ] || {{ echo "ERROR there is no filesystem on it"; exit 0; }}
d=$base; n=2
while grep -q " $d " /proc/mounts || {{ [ -e "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; }}; do d="$base-$n"; n=$((n+1)); done
mkdir -p "$d" || {{ echo "ERROR could not make $d"; exit 0; }}
out=$(mount -o nosuid,nodev "$dev" "$d" 2>&1) || {{ rmdir "$d" 2>/dev/null; echo "ERROR $out" | head -n 3; exit 0; }}
if [ {persist} = 1 ] && ! grep -q "UUID={uuid}" /etc/fstab 2>/dev/null; then
  if printf 'UUID=%s %s auto defaults,nofail,nosuid,nodev 0 0\n' {uuid} "$d" >> /etc/fstab 2>/dev/null; then
    systemctl daemon-reload >/dev/null 2>&1 || true
  else echo "NOTE it will not be mounted again after a restart (could not write /etc/fstab)"; fi
fi
echo "MOUNTED $d"
"""
# In a helper with the whole machine mounted read-only at /host: the folders in one folder.
LIST = r"""P=/host{path}
[ -d "$P" ] || {{ echo "ERROR $(printf %s "${{P#/host}}") is not a folder"; exit 0; }}
df -Pk "$P" | tail -n 1 | while read -r fs total used free pct mnt; do echo "DISK $fs"; echo "TOTAL $total"; echo "FREE $free"; done
dev=$(stat -c %d "$P"); n=0
for d in "$P"/*/; do
  [ -d "$d" ] || continue
  b=${{d%/}}; [ -L "$b" ] && continue
  name=${{b##*/}}; m=0; [ "$(stat -c %d "$b" 2>/dev/null)" != "$dev" ] && m=1
  printf 'DIR\t%s\t%s\n' "$name" "$m"
  n=$((n+1)); [ $n -ge 500 ] && break
done
true
"""


def parse_listing(path: str, out: str) -> Listing:
    li = Listing(path)
    for line in out.splitlines():
        if line.startswith("DIR\t"):
            parts = line.split("\t")
            if len(parts) == 3 and parts[1]:
                li.folders.append(Folder(parts[1], parts[2] == "1"))
            continue
        key, _, val = line.strip().partition(" ")
        if key == "ERROR":
            li.error = val
        elif key == "DISK":
            li.disk = val
        elif key == "FREE":
            li.free = (_int(val) or 0) * 1024
        elif key == "TOTAL":
            li.total = (_int(val) or 0) * 1024
    if li.path == "/":
        li.folders = [f for f in li.folders if f.name not in HIDDEN_ROOT]
    li.folders.sort(key=lambda f: f.name.lower())
    return li


class HelperHost(Host):
    def __init__(self, backend, image: str):
        self.backend, self.image = backend, image
        self.note = ""                  # something to tell about the last mount

    def _host(self, script: str) -> str:
        """Run a script in the host's own namespaces (a privileged helper sharing its process view)."""
        wrapped = ("command -v nsenter >/dev/null || { echo 'ERROR this helper has no nsenter'; exit 0; }\n"
                   f"nsenter -t 1 -m -u -i -n -p -- sh -c {shlex.quote(script)}")
        return self.backend.run_task(self.image, wrapped, [], "0:0", host=True)

    def scan(self) -> list[Drive]:
        out = self._host(SCAN.format(cols=LSBLK_COLUMNS))
        return parse_lsblk(out[out.find("{"):] if "{" in out else out)

    def mount(self, drive: Drive, base: str, persist: bool) -> str:
        if not re.fullmatch(r"[0-9A-Za-z-]+", drive.uuid):
            raise ValueError("the drive has no usable ID")
        out = self._host(MOUNT.format(uuid=drive.uuid, base=shlex.quote(base), persist=1 if persist else 0))
        where, note = "", ""
        for line in out.splitlines():
            key, _, val = line.strip().partition(" ")
            if key == "MOUNTED":
                where = val
            elif key == "ERROR":
                raise ValueError(val)
            elif key == "NOTE":
                note = val
        if not where:
            raise ValueError("could not tell whether it was mounted")
        self.note = note
        return where

    def browse(self, path: str) -> Listing:
        path = clean_path(path)
        out = self.backend.run_task(self.image, LIST.format(path=shlex.quote(path)), [("/", "/host", True)], "0:0")
        return parse_listing(path, out)


class LocalHost(Host):
    """The same for paths GPU Shepherd can see itself. In demo mode `drives` stands in for the
    machine's disks: mounting one only changes its state here."""

    def __init__(self, drives: list[Drive] | None = None):
        self.drives = list(drives or [])
        self.note = ""

    def scan(self) -> list[Drive]:
        return [Drive(**asdict(d)) for d in self.drives]

    def mount(self, drive: Drive, base: str, persist: bool) -> str:
        mine = next((d for d in self.drives if d.uuid == drive.uuid), None)
        if not mine:
            raise ValueError("the drive is not there any more")
        taken = {d.mountpoint for d in self.drives}
        where, n = base, 2
        while where in taken:
            where, n = f"{base}-{n}", n + 1
        mine.mountpoint, mine.state, mine.free = where, "mounted", mine.free if mine.free is not None else mine.size
        return where

    def browse(self, path: str) -> Listing:
        path = clean_path(path)
        p = Path(path)
        if not p.is_dir():
            return Listing(path, error=f"{path} is not a folder")
        try:
            names = sorted(c for c in os.listdir(p) if (p / c).is_dir() and not (p / c).is_symlink())
            usage = shutil.disk_usage(p)
            dev = os.stat(p).st_dev
            li = Listing(path, [Folder(n, os.stat(p / n).st_dev != dev) for n in names[:500]],
                         usage.free, usage.total, str(p))
        except OSError as err:
            return Listing(path, error=str(err))
        if path == "/":
            li.folders = [f for f in li.folders if f.name not in HIDDEN_ROOT]
        return li


def proc_signature(path: str = "/proc/partitions") -> str:
    """The block devices the kernel knows right now: it changes when a drive is attached. Cheap to
    read, so GPU Shepherd looks often and only asks the host for details when it changed."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def demo_drives() -> list[Drive]:
    """Disks for demo mode: a system disk, a data drive nobody has mounted, and an empty new one."""
    gb = 10**9
    return [Drive("/dev/nvme0n1p2", "System SSD", 500 * gb, "ext4", "", "demo-0001", "Demo NVMe", "/", 310 * gb,
                  state="mounted", system=True),
            Drive("/dev/sdb1", "Data SSD", 1000 * gb, "ext4", "Data SSD", "demo-0002", "Demo SATA SSD", "", None),
            Drive("/dev/sdc", "Demo USB disk", 2000 * gb, "", "", "", "Demo USB disk", "", None, removable=True,
                  state="empty", note="Nothing on it yet. GPU Shepherd never formats a drive: set it up with your "
                  "system's disk tool, then it shows here.")]
