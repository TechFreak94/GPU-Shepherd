"""What takes disk space, for Settings > Storage: free space per disk, the big folders, app images
nothing uses any more, and model files worth a look (not in any model list, unfinished
downloads, the same file twice), plus what deletes made before the trash existed left behind.

Pure scanning: no deleting here. service.py acts on what a scan found, and only on that.
Folders GPU Shepherd can't see (not mounted into its container at the same path) are listed
as such, without sizes.
"""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import Instance
from .trash import tree_size

MODEL_EXT = (".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin")
LOW_GB = 5.0                 # free space under this on a disk GPU Shepherd uses is worth a banner
MAX_DEPTH = 6                # how deep a models folder is searched


@dataclass
class Disk:
    path: str                # the first folder found on it
    total: int
    free: int
    folders: list[str] = field(default_factory=list)   # GPU Shepherd's folders on this disk

    @property
    def low(self) -> bool:
        return self.free < LOW_GB * 1e9


@dataclass
class Folder:
    label: str
    path: str
    size: int | None         # None = not visible


@dataclass
class ModelFile:
    path: str                # on this machine
    size: int
    folder: str              # the models folder it is in
    why: str                 # unused | partial | duplicate


@dataclass
class Scan:
    disks: list[Disk] = field(default_factory=list)
    folders: list[Folder] = field(default_factory=list)
    files: list[ModelFile] = field(default_factory=list)
    leftover_folders: list[Folder] = field(default_factory=list)    # appdata/<x> no instance uses
    hidden: list[str] = field(default_factory=list)                  # folders GPU Shepherd can't see

    def by_why(self, why: str) -> list[ModelFile]:
        return [f for f in self.files if f.why == why]


def model_folders(instances: dict[str, Instance], llm_mount: str) -> list[str]:
    """Folders on this machine that hold models: what model servers see at `llm_mount` and folders
    mounted at a path with "models" in it (ComfyUI's), outermost first, without folders inside others."""
    found = set()
    for inst in instances.values():
        for v in inst.volumes:
            if v.container == llm_mount or "models" in v.container.lower().rsplit("/", 1)[-1]:
                found.add(v.host.rstrip("/") or "/")
    out = []
    for p in sorted(found, key=len):
        if not any(Path(p).is_relative_to(o) for o in out):
            out.append(p)
    return out


def used_model_files(instances: dict[str, Instance], catalogs: dict[str, list[str]]) -> set[str]:
    """Host paths of every file a model list uses (model, vision projector, MTP draft)."""
    used = set()
    for iid, files in catalogs.items():
        inst = instances.get(iid)
        if not inst:
            continue
        for f in files:
            for v in inst.volumes:
                inside = v.container.rstrip("/") + "/"
                if f.startswith(inside):
                    used.add(str(Path(v.host) / f[len(inside):]))
    return used


def scan(data_dir: Path, instances: dict[str, Instance], catalogs: dict[str, list[str]], llm_mount: str,
         downloading: set[str] = frozenset()) -> Scan:
    """`catalogs`: instance id -> file paths (inside its container) its model list uses.
    `downloading`: models-folder paths of files being downloaded right now (left alone)."""
    s = Scan()
    data_dir = Path(data_dir)
    models = model_folders(instances, llm_mount)
    visible = [m for m in models if Path(m).is_dir()]
    s.hidden = [m for m in models if m not in visible]

    # disks, by device
    by_dev: dict[int, Disk] = {}
    for p in [str(data_dir)] + visible:
        try:
            dev = os.stat(p).st_dev
            if dev not in by_dev:
                u = shutil.disk_usage(p)
                by_dev[dev] = Disk(p, u.total, u.free)
            by_dev[dev].folders.append(p)
        except OSError:
            continue
    s.disks = list(by_dev.values())

    # big folders
    s.folders = [Folder("Trash", str(data_dir / "trash"), tree_size(data_dir / "trash")),
                 Folder("App data", str(data_dir / "appdata"), tree_size(data_dir / "appdata")),
                 Folder("Update backups", str(data_dir / "update" / "backups"),
                        tree_size(data_dir / "update" / "backups"))]
    s.folders += [Folder("Models", m, tree_size(Path(m)) if m in visible else None) for m in models]

    # model files
    used = used_model_files(instances, catalogs)
    by_key: dict[tuple[str, int], list[ModelFile]] = defaultdict(list)
    for m in visible:
        for path in _walk(Path(m)):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            name = str(path)
            if name.endswith(".part"):
                if name.removesuffix(".part") not in downloading:
                    s.files.append(ModelFile(name, size, m, "partial"))
                continue
            f = ModelFile(name, size, m, "unused")
            by_key[(path.name, size)].append(f)
            # only GGUF files are what model lists use; other formats belong to ComfyUI and such
            if path.suffix == ".gguf" and name not in used and name not in downloading:
                s.files.append(f)
    for group in by_key.values():
        if len(group) > 1:
            keep = next((f for f in group if f.path in used), group[0])    # a copy a model list uses
            for f in group:
                if f is not keep and f.path not in used:
                    s.files = [x for x in s.files if x.path != f.path]
                    s.files.append(ModelFile(f.path, f.size, f.folder, "duplicate"))

    # app data folders no instance uses
    appdata = data_dir / "appdata"
    in_use = {appdata / Path(v.host).relative_to(appdata).parts[0]
              for i in instances.values() for v in i.volumes
              if Path(v.host).is_relative_to(appdata) and Path(v.host) != appdata}
    if appdata.is_dir():
        for d in sorted(appdata.iterdir()):
            if d.is_dir() and d not in in_use:
                s.leftover_folders.append(Folder(d.name, str(d), tree_size(d)))
    return s


def _walk(root: Path):
    base = len(root.parts)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]          # skips .shepherd-trash
        if len(Path(dirpath).parts) - base >= MAX_DEPTH:
            dirs[:] = []
        for f in files:
            if f.endswith(MODEL_EXT) or f.endswith(tuple(e + ".part" for e in MODEL_EXT)):
                yield Path(dirpath) / f
