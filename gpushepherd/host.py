"""Host CPU and memory, read from /proc.

/proc/stat, /proc/meminfo and /proc/loadavg are not namespaced, so inside GPU Shepherd's own
container they still describe the whole machine. CPU use needs two samples; the previous
sample is kept between calls, so the dashboard's refresh interval is the measuring window.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

PROC = os.environ.get("SHEPHERD_PROC", "/proc")
_wsl: bool | None = None


def is_wsl() -> bool:
    """Whether the machine is WSL2. The kernel is shared with every container, so its name in
    /proc/version ("...-microsoft-standard-WSL2") says so from inside GPU Shepherd's container
    too. SHEPHERD_WSL=1 or 0 overrides it (tests, demos)."""
    global _wsl
    forced = os.environ.get("SHEPHERD_WSL", "")
    if forced:
        return forced == "1"
    if _wsl is None:
        try:
            _wsl = "microsoft" in _read("version").lower()
        except OSError:
            _wsl = False
    return _wsl


@dataclass
class Host:
    cpu_pct: float = 0.0          # whole machine, 0-100
    cpu_threads: int = 0
    cpu_cores: int = 0
    cpu_model: str = ""
    load: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mem_total: int = 0            # bytes
    mem_available: int = 0
    swap_total: int = 0
    swap_used: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def mem_used(self) -> int:
        return max(self.mem_total - self.mem_available, 0)

    @property
    def mem_pct(self) -> int:
        return round(100 * self.mem_used / self.mem_total) if self.mem_total else 0


_prev: tuple[float, int, int] | None = None     # (time, busy jiffies, total jiffies)
_cpu_info: tuple[str, int, int] | None = None
_lock = threading.Lock()


def _read(name: str) -> str:
    with open(os.path.join(PROC, name), encoding="utf-8") as fh:
        return fh.read()


def _cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies since boot, summed over all CPUs."""
    first = _read("stat").splitlines()[0].split()
    vals = [int(v) for v in first[1:9]]          # user nice system idle iowait irq softirq steal
    idle = vals[3] + vals[4]
    total = sum(vals)
    return total - idle, total


def _cpu_details() -> tuple[str, int, int]:
    """(model name, physical cores, hardware threads)"""
    global _cpu_info
    if _cpu_info is None:
        model, pairs, phys, core, threads = "", set(), None, None, 0
        try:
            for line in _read("cpuinfo").splitlines():
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key == "processor":
                    threads += 1
                elif key == "model name" and not model:
                    model = val
                elif key == "physical id":
                    phys = val
                elif key == "core id":
                    core = val
                elif not line.strip():
                    if core is not None:
                        pairs.add((phys, core))
                    phys = core = None
            if core is not None:
                pairs.add((phys, core))
        except OSError:
            pass
        _cpu_info = (" ".join(model.split()), len(pairs), threads)
    return _cpu_info


def _meminfo() -> dict[str, int]:
    out = {}
    for line in _read("meminfo").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * 1024     # values are in kB
    return out


def ram_gb() -> float:
    """Installed system RAM in GB (0 if unknown), without query()'s CPU sample."""
    try:
        return _meminfo().get("MemTotal", 0) / 2**30
    except OSError:
        return 0.0


def mem_available() -> tuple[int, int]:
    """(total, available) system RAM in bytes, (0, 0) if unknown."""
    try:
        mem = _meminfo()
        return mem.get("MemTotal", 0), mem.get("MemAvailable", mem.get("MemFree", 0))
    except OSError:
        return 0, 0


def query() -> Host:
    global _prev
    h = Host()
    h.cpu_model, h.cpu_cores, h.cpu_threads = _cpu_details()
    h.cpu_threads = h.cpu_threads or os.cpu_count() or 0
    with _lock:
        try:
            busy, total = _cpu_times()
            prev = _prev
            if prev is None or time.time() - prev[0] > 60:
                # no usable earlier sample: take a short one now
                time.sleep(0.2)
                prev = (0.0, busy, total)
                busy, total = _cpu_times()
            if total > prev[2]:
                h.cpu_pct = round(100 * (busy - prev[1]) / (total - prev[2]), 1)
            _prev = (time.time(), busy, total)
        except (OSError, ValueError, IndexError) as err:
            h.errors.append(f"CPU use unavailable: {err}")
    try:
        h.load = tuple(float(x) for x in _read("loadavg").split()[:3])
    except (OSError, ValueError):
        pass
    try:
        mem = _meminfo()
        h.mem_total = mem.get("MemTotal", 0)
        h.mem_available = mem.get("MemAvailable", mem.get("MemFree", 0))
        h.swap_total = mem.get("SwapTotal", 0)
        h.swap_used = max(h.swap_total - mem.get("SwapFree", 0), 0)
    except OSError as err:
        h.errors.append(f"memory use unavailable: {err}")
    h.cpu_cores = h.cpu_cores or h.cpu_threads
    return h
