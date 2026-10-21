"""GPU discovery through nvidia-smi (the NVIDIA container runtime provides it inside the app's
container when the 'utility' capability is requested).

For development and demos, SHEPHERD_FAKE_GPUS="2x8192:NVIDIA GeForce RTX 3070" fakes the GPUs.
"""
from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class Gpu:
    index: int
    name: str
    mem_used: float      # MiB
    mem_total: float     # MiB
    util: float          # percent

    @property
    def mem_pct(self) -> int:
        return round(100 * self.mem_used / self.mem_total) if self.mem_total else 0


_cache: tuple[float, list[Gpu], str] = (0.0, [], "")
_lock = threading.Lock()


def query(max_age: float = 2.0) -> tuple[list[Gpu], str]:
    """Returns (gpus, error). Cached briefly so a busy dashboard does not hammer nvidia-smi."""
    global _cache
    with _lock:
        if time.time() - _cache[0] < max_age:
            return _cache[1], _cache[2]
        gpus, err = _query()
        _cache = (time.time(), gpus, err)
        return gpus, err


def _query() -> tuple[list[Gpu], str]:
    fake = os.environ.get("SHEPHERD_FAKE_GPUS", "")
    if fake:
        return _fake(fake), ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except FileNotFoundError:
        return [], "nvidia-smi not found - give GPU Shepherd's container the 'utility' GPU capability"
    except (subprocess.SubprocessError, OSError) as err:
        return [], f"nvidia-smi failed: {err}"
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), parts[1], _num(parts[2]), _num(parts[3]), _num(parts[4])))
        except ValueError:
            continue
    return gpus, ""


def _num(text: str) -> float:
    return float(text) if text not in ("", "[N/A]", "N/A") else 0.0


def _fake(spec: str) -> list[Gpu]:
    count, _, rest = spec.partition("x")
    mem, _, name = rest.partition(":")
    return [Gpu(i, name or "Fake GPU", round(random.uniform(0.05, 0.2) * float(mem or 8192)),
                float(mem or 8192), 0.0) for i in range(int(count or 1))]


def physical_cores() -> int:
    """Physical CPU cores (best llama.cpp thread count for CPU-offloaded work)."""
    pairs = set()
    phys = core = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if core is not None:
                        pairs.add((phys, core))
                    phys = core = None
    except OSError:
        pass
    return len(pairs) or max(1, (os.cpu_count() or 2) // 2)
