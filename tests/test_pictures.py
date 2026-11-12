"""Storage > Pictures: ComfyUI's made and uploaded pictures, looked through and moved to the trash."""
import os
import time

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, pictures
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store

HX = {"HX-Request": "true"}


def login(client):
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 303


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    data, out, inp = tmp_path / "data", tmp_path / "comfy" / "output", tmp_path / "comfy" / "input"
    data.mkdir()
    (out / "sub").mkdir(parents=True)
    inp.mkdir(parents=True)
    old = time.time() - 40 * 86400
    for name, age in (("OpenWebUI_00001_.png", old), ("OpenWebUI_00002_.png", None), ("sub/clip.mp4", None)):
        (out / name).write_bytes(b"p" * 100)
        if age:
            os.utime(out / name, (age, age))
    (out / "notes.txt").write_text("not a picture")
    (out / ".hidden").mkdir()
    (out / ".hidden" / "x.png").write_bytes(b"h")
    (inp / "upload.jpg").write_bytes(b"u" * 50)
    app = create_app(data_dir=str(data), backend=FakeBackend(), demo=True, autostart=False)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(data))}"})
    comfy = {"name": "ComfyUI", "image": "x/comfy:1",
             "volumes": [{"host": str(out), "container": "/root/ComfyUI/output"},
                         {"host": str(inp), "container": "/root/ComfyUI/input"}]}
    assert client.put("/api/v1/instances/comfyui", json=comfy).status_code == 201
    return client, out, inp


def test_lists_pictures_newest_first(stack):
    client, out, inp = stack
    apps = client.get("/api/v1/pictures").json()
    assert [(f["kind"], f["count"], f["size"]) for f in apps[0]["folders"]] == [("output", 3, 300), ("input", 1, 50)]
    page = client.get("/api/v1/pictures/comfyui/output").json()
    names = [p["name"] for p in page["pictures"]]
    assert names[-1] == "OpenWebUI_00001_.png" and set(names) == {"OpenWebUI_00001_.png", "OpenWebUI_00002_.png", "sub/clip.mp4"}
    assert next(p for p in page["pictures"] if p["name"] == "sub/clip.mp4")["video"]
    assert page["pictures"][-1]["age_days"] == 40


def test_file_names_stay_inside_the_folder(stack, tmp_path):
    client, out, inp = stack
    (tmp_path / "secret.png").write_bytes(b"s")
    login(client)
    r = client.get("/storage/pictures/comfyui/output/file", params={"name": "OpenWebUI_00002_.png"})
    assert r.status_code == 200 and r.content == b"p" * 100
    assert "default-src 'none'" in r.headers["content-security-policy"]
    for bad in ("../../secret.png", "/etc/passwd", "notes.txt", ".hidden/x.png", "sub/../OpenWebUI_00002_.png", ""):
        assert client.get("/storage/pictures/comfyui/output/file", params={"name": bad}).status_code == 404, bad
    assert client.get("/storage/pictures/comfyui/models/file", params={"name": "a.png"}).status_code == 404
    assert pictures.path(out, "../input/upload.jpg") is None


def test_picked_and_old_pictures_go_to_the_trash_and_come_back(stack):
    client, out, inp = stack
    r = client.post("/api/v1/pictures/comfyui/output/trash", json={"older_than": 30})
    assert r.status_code == 200 and "Moved 1 picture" in r.json()["message"]
    assert not (out / "OpenWebUI_00001_.png").exists() and (out / "OpenWebUI_00002_.png").exists()
    r = client.post("/api/v1/pictures/comfyui/input/trash", json={"names": ["upload.jpg"]})
    assert r.status_code == 200 and not (inp / "upload.jpg").exists()
    assert client.post("/api/v1/pictures/comfyui/input/trash", json={"names": ["../x.png"]}).status_code == 404
    assert client.post("/api/v1/pictures/comfyui/input/trash", json={}).status_code == 400
    assert client.get("/api/v1/pictures").json()[0]["folders"][0]["count"] == 2     # the trash's own folder is hidden
    trash = client.get("/api/v1/trash").json()
    assert {t["title"] for t in trash} == {"OpenWebUI_00001_.png from ComfyUI", "upload.jpg from ComfyUI"}
    for t in trash:
        assert client.post(f"/api/v1/trash/{t['id']}/restore").status_code == 200
    assert (out / "OpenWebUI_00001_.png").exists() and (inp / "upload.jpg").exists()


def test_pictures_page(stack):
    client, out, inp = stack
    login(client)
    r = client.get("/storage/pictures")
    assert r.status_code == 200 and "OpenWebUI_00002_.png" in r.text and 'loading="lazy"' in r.text
    assert "upload.jpg" in client.get("/storage/pictures?app=comfyui&folder=input").text
    assert "Look through and clear out" in client.get("/storage").text
    r = client.post("/storage/pictures/comfyui/output/trash", data={"name": "OpenWebUI_00002_.png"}, headers=HX)
    assert r.status_code == 204 and r.headers["HX-Redirect"].startswith("/storage/pictures?app=comfyui&folder=output&msg=Moved")
    assert not (out / "OpenWebUI_00002_.png").exists()
