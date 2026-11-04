"""gpu-unlock.sh on a pretend machine (GPU_UNLOCK_TEST_ROOT): the boot line it writes and its
safety state machine (armed -> trying -> on, and every way back to off)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "gpu-unlock.sh"
CARDS = ("0000:01:00.0", "0000:02:00.0")
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

STUBS = {
    "update-grub": "exit 0",
    "systemctl": "exit 0",
    "journalctl": 'cat "$GPU_UNLOCK_TEST_ROOT/klog" 2>/dev/null; exit 0',
    "modinfo": '[ -f "$GPU_UNLOCK_TEST_ROOT/installed" ] || exit 1; [ "$1" = -F ] && echo 2.0.0; exit 0',
    "nvidia-smi": 'cat "$GPU_UNLOCK_TEST_ROOT/smi" 2>/dev/null; exit 0',
    "lspci": 'echo "01:00.0 VGA compatible controller: NVIDIA Corporation GA104 [GeForce RTX 3070] (rev a1)"',
    "mokutil": 'echo "SecureBoot disabled"',
}


class Machine:
    def __init__(self, root: Path):
        self.root = root
        bindir = root / "bin"
        bindir.mkdir()
        for name, body in STUBS.items():
            p = bindir / name
            p.write_text("#!/bin/bash\n" + body + "\n")
            p.chmod(0o755)
        for i, pci in enumerate(CARDS):
            d = root / "sys/bus/pci/devices" / pci
            d.mkdir(parents=True)
            (d / "vendor").write_text("0x10de\n")
            (d / "class").write_text("0x030000\n")
            (d / "device").write_text("0x2484\n")
        audio = root / "sys/bus/pci/devices/0000:01:00.1"     # the card's sound device is not a card
        audio.mkdir(parents=True)
        (audio / "vendor").write_text("0x10de\n")
        (audio / "class").write_text("0x040300\n")
        (audio / "device").write_text("0x228b\n")
        recipes = root / "var/lib/gpu-shepherd-unlock/src/kernel/gpu_unlock/recipes.c"
        recipes.parent.mkdir(parents=True)
        recipes.write_text("#define GA104(id) { id, 0x174, &ga10x }\n\tGA104(0x2482), GA104(0x2484),\n")
        (root / "etc/default").mkdir(parents=True)
        (root / "etc/default/grub").write_text('GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"\n')
        (root / "etc/systemd/system").mkdir(parents=True)
        (root / "run").mkdir()
        (root / "proc/sys/kernel/random").mkdir(parents=True)
        (root / "proc/version").write_text("Linux version 6.8.0\n")
        (root / "data").mkdir()
        (root / "compose.yaml").write_text(f"    environment:\n      SHEPHERD_DATA_DIR: {root / 'data'}\n")
        shutil.copy(SCRIPT, root / "gpu-unlock.sh")          # reads compose.yaml next to itself
        self.boot("quiet splash")
        self.env = {**os.environ, "GPU_UNLOCK_TEST_ROOT": str(root), "PATH": f"{bindir}:{os.environ['PATH']}"}

    def boot(self, cmdline: str, klog: str = "", gpus: int = 2) -> None:
        (self.root / "proc/cmdline").write_text(cmdline + "\n")
        (self.root / "proc/sys/kernel/random/boot_id").write_text(os.urandom(8).hex() + "\n")
        (self.root / "klog").write_text(klog)
        (self.root / "smi").write_text("".join(f"GPU {i}: NVIDIA GeForce RTX 3070 (UUID: x)\n" for i in range(gpus)))

    def reboot(self, unlocked: tuple[str, ...] = CARDS, gpus: int = 2, recovery: bool = False) -> None:
        """Boot with whatever the boot line says now (recovery mode leaves it out)."""
        targets = "" if recovery else self.configured()
        klog = "".join(f"pci {c}: gpu_unlock: unlocked\n" for c in unlocked if c in targets.split(","))
        self.boot("quiet splash" + (f" gpu_unlock.targets={targets}" if targets else ""), klog, gpus)
        self.run("--boot")

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(["bash", str(self.root / "gpu-unlock.sh"), *args], env=self.env,
                           capture_output=True, text=True, timeout=60)
        if check:
            assert r.returncode == 0, r.stdout + r.stderr
        return r

    def configured(self) -> str:
        p = self.root / "etc/default/grub.d/99-gpu-shepherd-unlock.cfg"
        if not p.exists():
            return ""
        text = p.read_text()
        assert 'GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT:-} gpu_unlock.targets=' in text
        return text.split("gpu_unlock.targets=")[1].split('"')[0]

    def status(self) -> dict:
        return json.loads((self.root / "data/gpu-unlock/status.json").read_text())

    def state(self) -> str:
        return self.status()["state"]


@pytest.fixture
def m(tmp_path) -> Machine:
    machine = Machine(tmp_path)
    (tmp_path / "installed").touch()
    return machine


def test_status_lists_only_the_cards(m):
    m.run("--check")
    st = m.status()
    assert [c["pci"] for c in st["cards"]] == list(CARDS)
    assert st["cards"][0]["family"] == "GA104" and st["cards"][0]["tier"] == "stable"
    assert st["installed"] and st["state"] == "off" and st["blocker"] == ""
    assert "RTX 3070" in st["cards"][0]["name"]


def test_set_one_card_then_a_good_boot_confirms(m):
    m.run("--set", CARDS[1])
    assert m.configured() == CARDS[1] and m.state() == "armed"
    m.reboot()
    assert m.state() == "trying"
    st = m.status()
    assert st["active"] == CARDS[1]
    assert [c["unlocked"] for c in st["cards"]] == [False, True]
    m.run("--confirm")
    assert m.state() == "on"
    m.reboot()                                     # later boots stay on
    assert m.state() == "on" and m.configured() == CARDS[1]


def test_all_expands_to_the_supported_cards(m):
    m.run("--set", "all")
    assert m.configured() == ",".join(CARDS)


def test_rejects_unknown_or_malformed_cards(m):
    assert m.run("--set", "0000:09:00.0", check=False).returncode == 1
    assert m.run("--set", "0000:01:00.0;reboot", check=False).returncode == 1
    assert m.configured() == "" and m.status()["run"] == "failed"


def test_set_none_is_the_rollback(m):
    m.run("--set", "all")
    m.reboot()
    m.run("--confirm")
    m.run("--set", "none")
    assert m.configured() == "" and m.state() == "off"


def test_a_boot_that_dies_before_the_check_turns_it_off(m):
    m.run("--set", "all")
    m.reboot()                                     # trying, then the machine freezes before --confirm
    m.reboot()
    assert m.state() == "off" and m.configured() == ""
    assert "stopped before" in m.status()["note"]


def test_recovery_mode_turns_it_off(m):
    m.run("--set", "all")
    m.reboot(recovery=True)
    assert m.state() == "off" and m.configured() == ""
    m.run("--set", "all")
    m.reboot()
    m.run("--confirm")
    m.reboot(recovery=True)                        # also once it was confirmed
    assert m.state() == "off"


def test_lost_gpu_after_the_unlock_turns_it_off(m):
    m.run("--set", "all")
    m.reboot(gpus=1)
    m.run("--confirm")
    assert m.state() == "off" and m.configured() == ""
    assert "1 of 2" in m.status()["note"]


def test_a_card_that_did_not_unlock_is_reported(m):
    m.run("--set", "all")
    m.reboot(unlocked=(CARDS[0],))
    m.run("--confirm")
    assert m.state() == "on" and CARDS[1] in m.status()["note"]


def test_needs_the_module_and_refuses_secure_boot(tmp_path):
    m = Machine(tmp_path)
    assert m.run("--set", "all", check=False).returncode == 1
    (tmp_path / "installed").touch()
    (tmp_path / "bin/mokutil").write_text('#!/bin/bash\necho "SecureBoot enabled"\n')
    assert m.run("--set", "all", check=False).returncode == 1
    assert "Secure Boot" in m.status()["message"]
    assert m.configured() == ""


def test_dashboard_request(m):
    req = m.root / "data/gpu-unlock/request"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(f"set\n{CARDS[0]}\n")
    m.run("--from-dashboard")
    assert not req.exists() and m.configured() == CARDS[0]
    assert "requested from the dashboard" in (m.root / "data/gpu-unlock/log.txt").read_text()
    req.write_text("rm -rf\n")
    m.run("--from-dashboard")
    assert m.configured() == CARDS[0]
