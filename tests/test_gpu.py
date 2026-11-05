"""nvidia-smi failures: what the dashboard says, and that planning doesn't treat unreadable GPUs as gone."""
from __future__ import annotations

import shutil
from pathlib import Path

import gpushepherd.gpu as gpu_mod
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store
from gpushepherd.engine import Engine

EXAMPLE = Path(__file__).parent / "fixtures" / "stack"   # ComfyUI on GPU 0, elastic LLM, Hermes, Open WebUI


def fake_smi(tmp_path, monkeypatch, text: str, code: int) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    smi = bindir / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho '{text}'\nexit {code}\n")
    smi.chmod(0o755)
    monkeypatch.delenv("SHEPHERD_FAKE_GPUS", raising=False)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))


def test_lost_gpus_explained(tmp_path, monkeypatch):
    fake_smi(tmp_path, monkeypatch, "Failed to initialize NVML: Unknown Error", 255)
    gpus, err = gpu_mod.query()
    assert gpus == [] and "lost access to the GPUs" in err and "docker restart" in err


def test_other_failures_explained():
    assert "Reboot" in gpu_mod.explain_failure("Failed to initialize NVML: Driver/library version mismatch", 18)
    assert "isn't working on the host" in gpu_mod.explain_failure(
        "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", 9)
    assert gpu_mod.explain_failure("something odd\n", 3) == "nvidia-smi failed (exit status 3): something odd"
    assert gpu_mod.explain_failure("", 255).endswith("no output")


def test_unreadable_gpus_stop_nothing(tmp_path, monkeypatch):
    shutil.copytree(EXAMPLE, tmp_path / "data")
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    engine = Engine(Store(str(tmp_path / "data")), FakeBackend())
    engine.execute(engine.plan("start", "comfyui"), lambda _m: None)
    engine.execute(engine.plan("start", "llm"), lambda _m: None)
    running = engine.snapshot().allocations
    assert running == {"comfyui": [0], "llm": [1]}

    fake_smi(tmp_path, monkeypatch, "Failed to initialize NVML: Unknown Error", 255)
    p = engine.autostart_plan()                       # what a restart of GPU Shepherd does
    assert not p.error and not p.to_stop and not p.to_restart and p.to_start == ["hermes"]   # before: ComfyUI and the LLM were stopped
    p = engine.plan("restart", "llm")
    assert not p.error and p.allocations["llm"] == [1] and p.allocations["comfyui"] == [0]
    stop = engine.plan("stop", "llm")
    assert not stop.error and stop.to_stop == ["llm"]
