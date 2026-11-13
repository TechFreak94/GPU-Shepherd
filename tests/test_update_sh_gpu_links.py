"""update.sh keeps NVIDIA GPUs in containers through a systemd reload: /dev/char links now, a udev
rule and a boot service (gpu-shepherd-nvidia-links) that make them again, on every run."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(tmp_path, *tools, wsl=False):
    machine, bin_dir, clone = tmp_path / "machine", tmp_path / "bin", tmp_path / "clone"
    for d in (machine / "run/systemd/system", machine / "proc", bin_dir, clone):
        d.mkdir(parents=True, exist_ok=True)
    (machine / "proc/version").write_text("Linux version 6.8 (Microsoft WSL)" if wsl else "Linux version 6.8")
    for tool in ("systemctl", "docker", *tools):
        (bin_dir / tool).write_text(f'#!/bin/sh\necho "{tool} $*" >> {tmp_path}/calls\n'
                                    + ('[ "$1" = is-enabled ] && exit 1\n' if tool == "systemctl" else "") + "exit 0\n")
        (bin_dir / tool).chmod(0o755)
    shutil.copy(ROOT / "update.sh", clone / "update.sh")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "UPDATE_TEST_ROOT": str(machine)}
    r = subprocess.run(["bash", str(clone / "update.sh"), "--gpu-links"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = (tmp_path / "calls").read_text() if (tmp_path / "calls").exists() else ""
    return machine, calls, r.stdout


def test_links_now_and_after_every_boot(tmp_path):
    machine, calls, out = run(tmp_path, "nvidia-ctk", "nvidia-smi")
    unit = machine / "etc/systemd/system/gpu-shepherd-nvidia-links.service"
    assert "create-dev-char-symlinks --create-all --load-kernel-modules" in unit.read_text()
    assert "Before=docker.service" in unit.read_text()
    assert (machine / "etc/udev/rules.d/71-nvidia-dev-char.rules").exists()
    first = calls.index("nvidia-ctk system create-dev-char-symlinks --create-all")
    assert first < calls.index("systemctl daemon-reload")          # never a reload without the links
    assert "systemctl enable --no-reload gpu-shepherd-nvidia-links.service" in calls
    assert "stay in containers" in out
    # the next run changes nothing, so it doesn't reload systemd again
    (tmp_path / "calls").unlink()
    _, calls, _ = run(tmp_path, "nvidia-ctk", "nvidia-smi")
    assert "create-dev-char-symlinks" in calls and "daemon-reload" not in calls


@pytest.mark.parametrize("tools,wsl", [((), False), (("nvidia-smi",), False), (("nvidia-ctk", "nvidia-smi"), True)])
def test_nothing_without_nvidia_or_on_wsl(tmp_path, tools, wsl):
    machine, calls, _ = run(tmp_path, *tools, wsl=wsl)
    assert not (machine / "etc/systemd/system/gpu-shepherd-nvidia-links.service").exists()
    assert not (machine / "etc/udev").exists()
    assert "nvidia-ctk" not in calls and "daemon-reload" not in calls
