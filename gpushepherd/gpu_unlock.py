"""The experimental tensor unlock: gpu-unlock (https://github.com/Zagitta/gpu-unlock) on the host.

gpu-unlock is a kernel module that lifts GeForce cards' half-rate FP16/BF16/TF32 tensor cap at
boot. GPU Shepherd can't install a kernel module from inside its container, so gpu-unlock.sh on
the host does the work, and the two sides talk through files in <data dir>/gpu-unlock/ (the same
path inside and outside the container), like updates.py and update.sh:

  status.json  written by gpu-unlock.sh after every action and at boot (see write_status there)
  request      written here: line 1 "check", "install", "set" or "uninstall", line 2 the cards
               for "set" (PCI addresses, comma separated, or "none"); a systemd path unit
               installed with dashboard updates (or `gpu-unlock.sh --install-service`) runs it
  log.txt      the output of the last action asked for here

Every change takes effect at the next reboot. The safety state gpu-unlock.sh keeps ("state"):
off, armed (set for the next boot), trying (the first boot with it, checked after 10 minutes),
on (checked and working). A failed check, a boot that dies before it, or a boot without the
unlock (GRUB's recovery mode) turns it back off.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

ACTIONS = ("check", "install", "set", "uninstall")
PCI = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
STALE_REQUEST = 120         # seconds: a request nobody picked up means the host service isn't there
STUCK = 3600                # seconds: an install that reported nothing for this long has died
URL = "https://github.com/Zagitta/gpu-unlock"


@dataclass
class Card:
    pci: str
    device: str = ""
    name: str = ""
    family: str = ""            # GA104, AD102, TU104... empty: gpu-unlock doesn't support it
    tier: str = ""              # stable (RTX 30) / experimental (RTX 40, RTX 20) / "" unsupported
    unlocked: bool = False      # unlocked on this boot
    active: bool = False        # this boot's kernel line asks for it
    configured: bool = False    # the next boot's kernel line asks for it
    note: str = ""              # gpu_unlock's last kernel message about it

    @property
    def supported(self) -> bool:
        return bool(self.family)

    @property
    def reboot_needed(self) -> bool:
        return self.active != self.configured

    @property
    def badge(self) -> str:
        """Short state for the GPU tile: "" when there is nothing to say."""
        if self.unlocked and self.configured:
            return "Tensor unlock on"
        if self.unlocked:
            return "Tensor unlock on until reboot"
        if self.configured:
            return "Tensor unlock after reboot"
        if self.active:
            return "Tensor unlock failed"
        return ""


@dataclass
class UnlockStatus:
    checked: bool = False               # gpu-unlock.sh has written a status at least once
    checked_at: str = ""
    run: str = ""                       # idle / working / done / failed: the last action asked for
    message: str = ""
    installed: bool = False
    installed_version: str = ""
    version: str = ""                   # the version gpu-unlock.sh installs (its pin)
    commit: str = ""
    source: bool = False                # the pinned source is downloaded, so supported cards are known
    bootloader: str = ""
    blocker: str = ""                   # why it can't be installed here
    service: bool = False               # the host service for dashboard requests is installed
    state: str = "off"
    note: str = ""
    repo: str = ""                      # the clone on the host
    active: list[str] = field(default_factory=list)
    configured: list[str] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    pending: str = ""
    pending_stale: bool = False
    age: float = 0.0

    @property
    def busy(self) -> bool:
        working = self.run == "working" and self.age < STUCK
        return working or (bool(self.pending) and not self.pending_stale)

    @property
    def stuck(self) -> bool:
        return self.run == "working" and self.age >= STUCK

    @property
    def supported(self) -> list[Card]:
        return [c for c in self.cards if c.supported]

    @property
    def reboot_needed(self) -> bool:
        return any(c.reboot_needed for c in self.cards)

    @property
    def command(self) -> str:
        return f"sudo {self.repo or '<your clone>'}/gpu-unlock.sh"

    def card(self, pci: str) -> Card | None:
        pci = normalize_pci(pci)
        return next((c for c in self.cards if c.pci == pci), None)


def normalize_pci(bus_id: str) -> str:
    """nvidia-smi's "00000000:01:00.0" (or "0000:01:00.0", any case) -> "0000:01:00.0"."""
    b = bus_id.strip().lower()
    if re.match(r"^[0-9a-f]{8}:", b):
        b = b[4:]
    return b


def folder(root: Path) -> Path:
    return Path(root) / "gpu-unlock"


def _targets(text: str) -> list[str]:
    return [t for t in (text or "").split(",") if PCI.match(t)]


def status(root: Path, now: float | None = None) -> UnlockStatus:
    from .updates import seconds_since
    st = UnlockStatus()
    d = folder(root)
    try:
        raw = json.loads((d / "status.json").read_text())
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        st.checked = True
        for key in ("checked_at", "run", "message", "installed_version", "version", "commit", "bootloader",
                    "blocker", "state", "note", "repo"):
            setattr(st, key, str(raw.get(key) or ""))
        st.state = st.state or "off"
        for key in ("installed", "source", "service"):
            setattr(st, key, raw.get(key) is True)
        st.active = _targets(str(raw.get("active") or ""))
        st.configured = _targets(str(raw.get("configured") or ""))
        for c in raw.get("cards") or []:
            if not isinstance(c, dict) or not PCI.match(str(c.get("pci") or "")):
                continue
            pci = str(c["pci"])
            st.cards.append(Card(pci=pci, device=str(c.get("device") or ""), name=str(c.get("name") or ""),
                                 family=str(c.get("family") or ""), tier=str(c.get("tier") or ""),
                                 unlocked=c.get("unlocked") is True, active=pci in st.active,
                                 configured=pci in st.configured, note=str(c.get("note") or "")))
        st.age = seconds_since(st.checked_at, now)
    req = d / "request"
    try:
        st.pending = req.read_text().split("\n", 1)[0].strip()
        st.pending_stale = (now or time.time()) - req.stat().st_mtime > STALE_REQUEST
    except OSError:
        pass
    return st


def request(root: Path, action: str, cards: list[str] | None = None) -> None:
    """Ask the host for an action; gpu-unlock.sh deletes the file when it starts."""
    if action not in ACTIONS:
        raise ValueError(f"unknown gpu-unlock action {action!r}")
    arg = ""
    if action == "set":
        cards = [normalize_pci(c) for c in cards or []]
        bad = [c for c in cards if not PCI.match(c)]
        if bad:
            raise ValueError(f"{bad[0]} is not a PCI address like 0000:01:00.0")
        arg = ",".join(dict.fromkeys(cards)) or "none"
    d = folder(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "request.tmp"
    tmp.write_text(action + "\n" + (arg + "\n" if arg else ""))
    tmp.replace(d / "request")


def log_text(root: Path) -> str:
    try:
        return (folder(root) / "log.txt").read_text(errors="replace")[-20000:]
    except OSError:
        return ""


def badges(st: UnlockStatus, bus_ids: dict[int, str]) -> dict[int, Card]:
    """GPU index (nvidia-smi) -> its card, for the GPU tiles; only cards with something to show."""
    out = {}
    for index, bus in bus_ids.items():
        c = st.card(bus) if bus else None
        if c and c.badge:
            out[index] = c
    return out
