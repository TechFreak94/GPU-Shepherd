"""GPU discovery: NVIDIA GPUs through nvidia-smi (the NVIDIA container runtime provides it inside
the app's container when the 'utility' capability is requested), AMD GPUs from the amdgpu driver's
files in /sys (no AMD tool needed; /sys shows the host's devices inside a container too).

NVIDIA GPUs keep nvidia-smi's numbers; AMD GPUs are numbered from config.AMD_BASE (100, 101...) in
the order of their PCI addresses, which don't change between boots. An app given an AMD GPU gets
its render node (/dev/dri/renderD128...), which is all Vulkan and ROCm need from /dev/dri.

For development and demos, SHEPHERD_FAKE_GPUS="2x8192:NVIDIA GeForce RTX 3070" fakes the GPUs,
and SHEPHERD_FAKE_AMD_GPUS="2x8192:AMD Radeon RX 580" the AMD ones.
"""
from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass

from .config import AMD_BASE, gpu_label, gpu_vendor


@dataclass
class Gpu:
    index: int
    name: str
    mem_used: float      # MiB
    mem_total: float     # MiB
    util: float          # percent
    bus_id: str = ""     # PCI address as lspci -D prints it, e.g. 0000:01:00.0
    uuid: str = ""       # GPU-xxxxxxxx-..., which CUDA_VISIBLE_DEVICES also accepts
    render: str = ""     # AMD: its render node, e.g. /dev/dri/renderD128

    @property
    def vendor(self) -> str:
        return gpu_vendor(self.index)

    @property
    def label(self) -> str:
        return gpu_label(self.index)

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
    """NVIDIA GPUs, then AMD ones. A machine with AMD GPUs only needs no nvidia-smi."""
    nvidia, err = _nvidia()
    amd = amd_gpus()
    if amd and not nvidia and err.startswith("nvidia-smi not found"):
        err = ""
    return nvidia + amd, err


def _nvidia() -> tuple[list[Gpu], str]:
    fake = os.environ.get("SHEPHERD_FAKE_GPUS", "")
    if fake:
        return _fake(fake), ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,pci.bus_id,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except FileNotFoundError:
        return [], "nvidia-smi not found - give GPU Shepherd's container the 'utility' GPU capability"
    except subprocess.CalledProcessError as err:
        return [], explain_failure(f"{err.stdout or ''}\n{err.stderr or ''}", err.returncode)
    except (subprocess.SubprocessError, OSError) as err:
        return [], f"nvidia-smi failed: {err}"
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), parts[1], _num(parts[2]), _num(parts[3]), _num(parts[4]),
                            _bus(parts[5]) if len(parts) > 5 else "",
                            parts[6] if len(parts) > 6 and parts[6].startswith("GPU-") else ""))
        except ValueError:
            continue
    return gpus, ""


def explain_failure(output: str, code: int) -> str:
    """What a failed nvidia-smi means, and what to do about it, from what it printed."""
    said = " ".join(output.split())[:300]
    low = said.lower()
    if "unknown error" in low:
        return ("GPU Shepherd's container lost access to the GPUs (NVML: Unknown Error). That happens "
                "when the host reloads systemd while containers run. Running apps lost them too: "
                "restart the containers on the host (docker restart $(docker ps -q)) or reboot.")
    if "version mismatch" in low:
        return ("The NVIDIA driver on the host was updated but the old one is still loaded "
                "(Driver/library version mismatch). Reboot the machine.")
    if "couldn't communicate" in low or "no devices were found" in low:
        return (f"The NVIDIA driver isn't working on the host ({said}). Run nvidia-smi on the host; "
                "after a kernel update, reinstalling the driver or rebooting usually fixes it.")
    return f"nvidia-smi failed (exit status {code}): {said or 'no output'}"


def _num(text: str) -> float:
    return float(text) if text not in ("", "[N/A]", "N/A") else 0.0


def _bus(text: str) -> str:
    """nvidia-smi's 00000000:01:00.0 -> 0000:01:00.0."""
    b = text.strip().lower()
    return b[4:] if len(b) == 16 and b[8] == ":" else b


def _fake(spec: str, base: int = 0) -> list[Gpu]:
    count, _, rest = spec.partition("x")
    mem, _, name = rest.partition(":")
    amd = base >= AMD_BASE
    return [Gpu(base + i, name or "Fake GPU", round(random.uniform(0.05, 0.2) * float(mem or 8192)),
                float(mem or 8192), 0.0, f"0000:{i + (0x41 if amd else 1):02x}:00.0",
                "" if amd else f"GPU-fake-{i}", render=f"/dev/dri/renderD{128 + i}" if amd else "")
            for i in range(int(count or 1))]


def render_nodes(indices: list[int]) -> list[str] | None:
    """The AMD GPUs' render nodes, in the same order; None if any is unknown."""
    by_index = {g.index: g.render for g in query()[0]}
    nodes = [by_index.get(i, "") for i in indices]
    return nodes if all(nodes) else None


DRM = "/sys/class/drm"
AMD_VENDOR = "0x1002"


def amd_gpus(drm: str = DRM) -> list[Gpu]:
    """The machine's AMD GPUs that the amdgpu driver runs, numbered from AMD_BASE in the order of
    their PCI addresses. Empty if there are none or /sys can't be read."""
    fake = os.environ.get("SHEPHERD_FAKE_AMD_GPUS", "")
    if fake:
        return _fake(fake, AMD_BASE)
    if os.environ.get("SHEPHERD_FAKE_GPUS"):      # faked GPUs: none of the real ones
        return []
    found = []
    try:
        cards = [p for p in os.listdir(drm) if p.startswith("card") and p[4:].isdigit()]
    except OSError:
        return []
    for card in cards:
        dev = os.path.join(drm, card, "device")
        if _read(os.path.join(dev, "vendor")) != AMD_VENDOR:
            continue
        total = _read(os.path.join(dev, "mem_info_vram_total"))
        if not total.isdigit():                  # not driven by amdgpu (radeon, or no driver)
            continue
        try:
            render = next((f"/dev/dri/{x}" for x in sorted(os.listdir(os.path.join(dev, "drm")))
                           if x.startswith("renderD")), "")
        except OSError:
            render = ""
        used = _read(os.path.join(dev, "mem_info_vram_used"))
        busy = _read(os.path.join(dev, "gpu_busy_percent"))
        name = _read(os.path.join(dev, "product_name")) or f"AMD GPU {_read(os.path.join(dev, 'device'))}"
        bus = os.path.basename(os.path.realpath(dev))
        found.append((bus if bus.count(":") == 2 else "", int(card[4:]), name, used, total, busy, render))
    found.sort(key=lambda x: (x[0] or "~", x[1]))
    return [Gpu(AMD_BASE + n, name, int(used) / 2**20 if used.isdigit() else 0.0, int(total) / 2**20,
                float(busy) if busy.isdigit() else 0.0, bus, render=render)
            for n, (bus, _, name, used, total, busy, render) in enumerate(found)]


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def cuda_names(indices: list[int]) -> list[str] | None:
    """The GPUs' UUIDs, in the same order, for CUDA_VISIBLE_DEVICES; None if any is unknown."""
    by_index = {g.index: g.uuid for g in query()[0]}
    names = [by_index.get(i, "") for i in indices]
    return names if all(names) else None


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
