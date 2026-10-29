"""Memory checks before a start: will what starts fit in the system RAM that is free, and do a
model server's models fit the GPUs it gets? Warnings only, shown with the plan to confirm: the
numbers are estimates, and a start is never refused because of them.

RAM an instance needs, best guess first: the most it used while it ran before (Peaks, measured
by the dashboard), its memory limit, and for a model server what its models leave over once the
GPUs are full (plus what offload settings deliberately keep in RAM). Apps with none of these
count as 0.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import yaml

from .config import Catalog, CatalogModel, Instance

GB = 1e9
VRAM_USABLE = 0.9          # share of GPU memory left for model weights (the rest: context, buffers)
OVERHEAD = 1 * GB          # a model server's own RAM beside the weights
HEADROOM = 2 * GB          # RAM to leave for the system


class Peaks:
    """The most RAM each instance used while running, kept in <data>/state/memory.yaml."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "state" / "memory.yaml"
        self._lock = threading.Lock()
        self._data: dict[str, int] | None = None

    def get(self) -> dict[str, int]:
        with self._lock:
            return dict(self._load())

    def record(self, used: dict[str, int]) -> None:
        """Remember new highs; the file is only written when one grows by 5% or more."""
        with self._lock:
            data = self._load()
            grown = {i: b for i, b in used.items() if b > data.get(i, 0) * 1.05}
            if not grown:
                return
            data.update(grown)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(yaml.safe_dump(data), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass

    def forget(self, iid: str) -> None:
        with self._lock:
            self._load().pop(iid, None)

    def _load(self) -> dict[str, int]:
        if self._data is None:
            try:
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                self._data = {str(k): int(v) for k, v in raw.items()}
            except (OSError, yaml.YAMLError, ValueError, AttributeError):
                self._data = {}
        return self._data


def parse_size(text: str | None) -> int:
    """Docker's memory sizes ("4g", "512m") in bytes; 0 if empty or not understood."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([bkmg]?)b?\s*", (text or "").lower())
    if not m:
        return 0
    return int(float(m.group(1)) * {"": 1, "b": 1, "k": 2**10, "m": 2**20, "g": 2**30}[m.group(2)])


def host_path(inst: Instance, inside: str) -> Path | None:
    for v in inst.volumes:
        prefix = v.container.rstrip("/") + "/"
        if inside.startswith(prefix):
            return Path(v.host) / inside[len(prefix):]
    return None


def file_size(inst: Instance, inside: str) -> int:
    p = host_path(inst, inside) if inside else None
    try:
        return p.stat().st_size if p else 0
    except OSError:
        return 0


def model_size(inst: Instance, m: CatalogModel) -> int:
    return sum(file_size(inst, f) for f in (m.file, m.mmproj, m.mtp_draft) if f)


def loaded_together(cat: Catalog, sizes: dict[str, int]) -> list[CatalogModel]:
    """The models that can be loaded at once: the ones kept loaded, plus the biggest of the rest."""
    kept = [m for m in cat.models if m.keep_loaded]
    rest = [m for m in cat.models if not m.keep_loaded]
    return kept + ([max(rest, key=lambda m: sizes[m.id])] if rest else [])


def server_ram(inst: Instance, cat: Catalog, gpus: int, vram: int) -> int:
    """RAM a model server needs on `gpus` GPUs with `vram` bytes between them (0 = sizes unknown)."""
    sizes = {m.id: model_size(inst, m) for m in cat.models}
    models = loaded_together(cat, sizes)
    total = sum(sizes[m.id] for m in models)
    if not total:
        return 0
    kept_in_ram = 0
    for m in models:
        rule = m.offload_for(max(gpus, 1))
        if not gpus or (rule and rule.gpu_layers == 0):
            kept_in_ram += sizes[m.id]
        elif rule and (rule.cpu_moe or rule.gpu_layers):
            kept_in_ram += sizes[m.id] // 2
    return int(max(total - VRAM_USABLE * vram, kept_in_ram, 0) + OVERHEAD)


def ram_needed(inst: Instance, cat: Catalog | None, gpus: int, vram: int, peak: int) -> int:
    est = server_ram(inst, cat, gpus, vram) if cat is not None else 0
    return max(peak, parse_size(inst.memory_limit) if not est else 0, est)


def too_big(inst: Instance, cat: Catalog, gpus: int, vram: int) -> list[str]:
    """Models that won't fit the GPUs they get and have no setting to keep part of them in RAM.
    llama.cpp then keeps what doesn't fit in RAM by itself, which only a MoE model runs well with."""
    out = []
    for m in cat.models:
        size = model_size(inst, m)
        if size and vram and size > VRAM_USABLE * vram and not m.offload_for(gpus) and m.role == "chat":
            out.append(f"{m.id} ({size / GB:.1f} GB) is bigger than the GPU memory {inst.name} gets with "
                       f"{gpus} GPU{'s' if gpus != 1 else ''} ({vram / GB:.1f} GB): llama.cpp keeps the rest in "
                       f"system RAM, so it answers slowly unless it is a mixture-of-experts model.")
    return out


def gb(n: float) -> str:
    return f"{n / GB:.1f} GB"
