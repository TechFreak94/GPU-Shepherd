"""The experimental tensor unlock: reading gpu-unlock.sh's status, asking it for changes, the
Settings card, the GPU tile badges and the API."""
from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from gpushepherd import auth, gpu as gpu_mod, gpu_unlock
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

A, B = "0000:01:00.0", "0000:02:00.0"


def write_status(root, **over):
    raw = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": "done", "message": "",
           "installed": True, "installed_version": "2.0.0", "version": "2.0.0", "commit": "7eb51bc",
           "source": True, "bootloader": "grub", "blocker": "", "service": True, "active": "", "configured": "",
           "state": "off", "note": "", "repo": "/home/me/GPU-Shepherd",
           "cards": [{"pci": p, "device": "2484", "name": "NVIDIA Corporation GA104 [GeForce RTX 3070]",
                      "family": "GA104", "tier": "stable", "unlocked": False, "note": ""} for p in (A, B)]}
    raw.update(over)
    d = gpu_unlock.folder(root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps(raw))


def unlocked(*pcis):
    return [{"pci": p, "device": "2484", "name": "RTX 3070", "family": "GA104", "tier": "stable",
             "unlocked": p in pcis, "note": "unlocked" if p in pcis else ""} for p in (A, B)]


def pending(root):
    p = gpu_unlock.folder(root) / "request"
    return p.read_text() if p.exists() else ""


def test_status_without_a_file(tmp_path):
    st = gpu_unlock.status(tmp_path)
    assert not st.checked and not st.service and st.cards == [] and st.state == "off"


def test_status_and_card_states(tmp_path):
    write_status(tmp_path, active=A, configured=f"{A},{B}", state="on", cards=unlocked(A))
    st = gpu_unlock.status(tmp_path)
    a, b = st.cards
    assert a.unlocked and a.active and a.configured and not a.reboot_needed and a.badge == "Tensor unlock on"
    assert not b.unlocked and b.configured and b.reboot_needed and b.badge == "Tensor unlock after reboot"
    assert st.reboot_needed and [c.pci for c in st.supported] == [A, B]
    write_status(tmp_path, active=A, configured="", cards=unlocked(A))
    assert gpu_unlock.status(tmp_path).cards[0].badge == "Tensor unlock on until reboot"
    write_status(tmp_path, active=A, configured=A, cards=unlocked())
    assert gpu_unlock.status(tmp_path).cards[0].badge == "Tensor unlock after reboot"
    write_status(tmp_path, active=A, configured="", cards=unlocked())
    assert gpu_unlock.status(tmp_path).cards[0].badge == "Tensor unlock failed"


def test_bad_status_is_ignored(tmp_path):
    write_status(tmp_path, active="0000:01:00.0;x", cards=[{"pci": "../etc"}, "nonsense"])
    st = gpu_unlock.status(tmp_path)
    assert st.cards == [] and st.active == []


def test_badges_match_nvidia_smi_bus_ids(tmp_path):
    write_status(tmp_path, active=B, configured=B, cards=unlocked(B))
    st = gpu_unlock.status(tmp_path)
    marks = gpu_unlock.badges(st, {0: "0000:01:00.0", 1: "00000000:02:00.0".upper(), 2: ""})
    assert list(marks) == [1]


def test_normalize_and_request(tmp_path):
    assert gpu_unlock.normalize_pci("00000000:0A:00.0") == "0000:0a:00.0"
    gpu_unlock.request(tmp_path, "set", [A, A, B])
    assert pending(tmp_path) == f"set\n{A},{B}\n"
    gpu_unlock.request(tmp_path, "set", [])
    assert pending(tmp_path) == "set\nnone\n"
    with pytest.raises(ValueError):
        gpu_unlock.request(tmp_path, "set", ["0000:01:00.0\ninstall"])
    with pytest.raises(ValueError):
        gpu_unlock.request(tmp_path, "reboot")
    assert gpu_unlock.status(tmp_path).busy


def test_stale_request_is_not_busy(tmp_path):
    write_status(tmp_path)
    gpu_unlock.request(tmp_path, "check")
    old = time.time() - 600
    os.utime(gpu_unlock.folder(tmp_path) / "request", (old, old))
    assert not gpu_unlock.status(tmp_path).busy


# ---------------------------------------------------------------------- dashboard

@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app)
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303
    client.headers["HX-Request"] = "true"          # the dashboard's posts come from htmx
    return client, tmp_path


def test_settings_card_without_the_host_service(ui):
    client, root = ui
    html = client.get("/settings").text
    assert "Tensor unlock" in html and "--install-service" in html and "Save for the next boot" not in html
    r = client.post("/ui/unlock/install")
    assert r.status_code == 204 and "install-service" in r.headers["HX-Trigger"]
    assert pending(root) == ""


def test_install_check_set_and_remove(ui):
    client, root = ui
    write_status(root, installed=False, source=False, cards=unlocked())
    html = client.get("/ui/unlock/card").text
    assert "Install…" in html and "Save for the next boot" not in html
    r = client.post("/ui/unlock/install")
    assert r.status_code == 200 and pending(root) == "install\n"
    assert "Installing" in r.headers["HX-Trigger"]
    assert client.post("/ui/unlock/check").status_code == 204          # busy until the host takes it
    (gpu_unlock.folder(root) / "request").unlink()

    write_status(root, cards=unlocked())
    r = client.post("/ui/unlock/set", data={"cards": [B]})
    assert r.status_code == 200 and pending(root) == f"set\n{B}\n"
    (gpu_unlock.folder(root) / "request").unlink()

    write_status(root, configured=B, state="armed", cards=unlocked())
    html = client.get("/ui/unlock/card").text
    assert "Reboot the server to apply this" in html and "recovery mode" in html and "Turn off for every card" in html
    assert client.post("/ui/unlock/set", data={"cards": [B]}).status_code == 204     # nothing to change
    r = client.post("/ui/unlock/set")                                                # the rollback
    assert r.status_code == 200 and pending(root) == "set\nnone\n"
    (gpu_unlock.folder(root) / "request").unlink()

    r = client.post("/ui/unlock/set", data={"cards": ["0000:09:00.0"]})
    assert r.status_code == 204 and "not a card" in r.headers["HX-Trigger"]
    assert client.post("/ui/unlock/uninstall").status_code == 200 and pending(root) == "uninstall\n"


def test_blocker_and_unsupported(ui):
    client, root = ui
    write_status(root, installed=False, blocker="Secure Boot is on")
    assert "Secure Boot is on" in client.get("/ui/unlock/card").text
    r = client.post("/ui/unlock/install")
    assert r.status_code == 204 and "Secure Boot" in r.headers["HX-Trigger"]
    write_status(root, installed=False, cards=[{"pci": A, "name": "RTX 5090", "family": "", "tier": ""}])
    assert "not supported" in client.get("/ui/unlock/card").text
    assert client.post("/ui/unlock/install").status_code == 204


def test_gpu_tile_badges(ui):
    client, root = ui
    assert "Tensor unlock" not in client.get("/ui/dashboard").text
    write_status(root, active=A, configured=f"{A},{B}", state="on", cards=unlocked(A))
    html = client.get("/ui/dashboard").text
    assert "Tensor unlock on" in html and "Tensor unlock after reboot" in html


def test_log(ui):
    client, root = ui
    assert "Nothing yet" in client.get("/ui/unlock/log").text
    (gpu_unlock.folder(root)).mkdir(parents=True, exist_ok=True)
    (gpu_unlock.folder(root) / "log.txt").write_text("==> Building gpu-unlock")
    assert "Building gpu-unlock" in client.get("/ui/unlock/log").text


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    app = create_app(data_dir=str(tmp_path), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(tmp_path))}"})
    write_status(tmp_path, active=A, configured=A, state="on", cards=unlocked(A))
    got = client.get("/api/v1/gpu-unlock").json()
    assert got["state"] == "on" and got["cards"][0]["unlocked"] and got["cards"][0]["next_boot"]
    assert not got["reboot_needed"]
    r = client.post("/api/v1/gpu-unlock/set", json={"cards": [A, B]})
    assert r.status_code == 202 and pending(tmp_path) == f"set\n{A},{B}\n"
    assert client.post("/api/v1/gpu-unlock/set", json={"cards": "all"}).status_code == 400
    assert client.post("/api/v1/gpu-unlock/reboot").status_code == 400
