"""Settings > Storage: moving a models or app data folder to another place (moves.py)."""
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, moves
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store
from gpushepherd.trash import Trash


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    data, models, disk2 = tmp_path / "data", tmp_path / "data" / "models", tmp_path / "disk2"
    (models / "qwen").mkdir(parents=True)
    disk2.mkdir()
    (models / "qwen" / "qwen.gguf").write_bytes(b"q" * 4000)
    (models / "comfyui" / "checkpoints").mkdir(parents=True)
    (models / "comfyui" / "checkpoints" / "sd.safetensors").write_bytes(b"s" * 100)
    os.symlink("qwen/qwen.gguf", models / "link.gguf")
    backend = FakeBackend()
    app = create_app(data_dir=str(data), backend=backend, demo=True, autostart=False)
    app.state.svc.move_interval = 0.01
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(data))}"})
    llm = {"name": "LLM", "image": "ghcr.io/mostlygeek/llama-swap:cuda", "kind": "llama-swap",
           "volumes": [{"host": str(models), "container": "/models", "read_only": True}]}
    assert client.put("/api/v1/instances/llm", json=llm).status_code == 201
    comfy = {"name": "ComfyUI", "image": "x/comfy:1",
             "volumes": [{"host": str(models / "comfyui"), "container": "/root/ComfyUI/models"},
                         {"host": str(data / "appdata" / "comfyui" / "output"), "container": "/out"}]}
    assert client.put("/api/v1/instances/comfyui", json=comfy).status_code == 201
    (data / "appdata" / "comfyui" / "output").mkdir(parents=True)
    return app, client, backend, data, models, disk2


HX = {"HX-Request": "true"}


def wait(app):
    for _ in range(500):
        if not app.state.jobs.busy:
            return app.state.jobs.recent()[0]
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def hosts(client, iid):
    return [v["host"] for v in client.get(f"/api/v1/instances/{iid}").json()["volumes"]]


def test_movable_folders_are_listed(stack):
    app, client, backend, data, models, disk2 = stack
    rows = {m["path"]: m for m in client.get("/api/v1/storage").json()["movable"]}
    assert rows[str(models)]["label"] == "Models" and rows[str(models)]["used_by"] == ["comfyui", "llm"]
    assert rows[str(models)]["size"] > 4000
    app_data = rows[str(data / "appdata" / "comfyui")]
    assert app_data["label"] == "App data: ComfyUI" and app_data["used_by"] == ["comfyui"]
    assert len(rows) == 2                     # comfyui's models are inside the models folder


def test_check_paths():
    ok = moves.check_paths("/opt/gs/models", "/mnt/disk2/models", "/opt/gs")
    assert ok == ""
    assert "full path" in moves.check_paths("/opt/gs/models", "disk2", "/opt/gs")
    assert ".." in moves.check_paths("/opt/gs/models", "/mnt/../etc/x", "/opt/gs")
    assert "system" in moves.check_paths("/opt/gs/models", "/etc/models", "/opt/gs")
    assert "inside" in moves.check_paths("/opt/gs/models", "/opt/gs/models/new", "/opt/gs")
    assert "is in" in moves.check_paths("/opt/gs/models", "/opt", "/opt/gs")
    assert "data folder" in moves.check_paths("/srv/x", "/opt/gs/state", "/opt/gs")
    assert "data folder" in moves.check_paths("/opt", "/mnt/opt", "/opt/gs")
    assert moves.relocate("/a/b/c", "/a/b", "/z") == "/z/c"
    assert moves.relocate("/a/b", "/a/b", "/z") == "/z" and moves.relocate("/a/bc", "/a/b", "/z") == "/a/bc"


def test_move_models_to_another_disk_and_back(stack):
    app, client, backend, data, models, disk2 = stack
    assert client.post("/api/v1/instances/llm/start", json={"confirm": True}).status_code == 202
    wait(app)
    new = disk2 / "models"
    body = {"source": str(models), "dest": str(new)}
    r = client.post("/api/v1/storage/move", json=body)
    assert r.status_code == 409 and r.json()["needs_confirm"]
    assert r.json()["stops"] == ["LLM"] and r.json()["size"] >= 4100 and not r.json()["same_disk"] is None
    r = client.post("/api/v1/storage/move", json=body | {"confirm": True})
    assert r.status_code == 202, r.text
    job = wait(app)
    assert job.status == "done", job.lines
    assert (new / "qwen" / "qwen.gguf").read_bytes() == b"q" * 4000
    assert os.readlink(new / "link.gguf") == "qwen/qwen.gguf"
    assert (models / "qwen" / "qwen.gguf").exists()                   # the old copy is kept
    assert hosts(client, "llm") == [str(new)]
    assert hosts(client, "comfyui") == [str(new / "comfyui"), str(data / "appdata" / "comfyui" / "output")]
    assert backend.states()["llm"].running                            # started again
    assert any("Stopping LLM" in line for line in job.lines)
    [rec] = client.get("/api/v1/moves").json()
    assert rec["status"] == "done" and rec["leftover"] == str(models)
    # move back: the apps use the old copy again, the new one is left over
    assert "go back" in client.post(f"/api/v1/moves/{rec['id']}/back").json()["message"]
    assert wait(app).status == "done"
    assert hosts(client, "llm") == [str(models)]
    [rec] = client.get("/api/v1/moves").json()
    assert rec["status"] == "moved-back" and rec["leftover"] == str(new)
    assert client.post(f"/api/v1/moves/{rec['id']}/back").status_code == 409
    assert "Deleted" in client.delete(f"/api/v1/moves/{rec['id']}/leftover").json()["message"]
    assert not new.exists() and models.is_dir()


def test_old_copy_is_deleted_only_when_unused(stack):
    app, client, backend, data, models, disk2 = stack
    src = data / "appdata" / "comfyui"
    (src / "output" / "img.png").write_bytes(b"i" * 10)
    r = client.post("/api/v1/storage/move", json={"source": str(src), "dest": str(disk2 / "comfyui"), "confirm": True})
    assert r.status_code == 202 and wait(app).status == "done"
    assert hosts(client, "comfyui")[1] == str(disk2 / "comfyui" / "output")
    [rec] = client.get("/api/v1/moves").json()
    client.delete(f"/api/v1/moves/{rec['id']}/leftover")
    assert not src.exists() and (disk2 / "comfyui" / "output" / "img.png").exists()
    assert client.delete(f"/api/v1/moves/{rec['id']}/leftover").status_code == 409


def test_what_is_refused(stack):
    app, client, backend, data, models, disk2 = stack
    move = lambda src, dest: client.post("/api/v1/storage/move", json={"source": src, "dest": dest})   # noqa: E731
    assert move("/etc", str(disk2 / "x")).status_code == 404                  # not a folder the apps use
    assert move(str(models), "/nowhere/models").json()["error"].startswith("Can't move there")
    (disk2 / "full").mkdir()
    (disk2 / "full" / "f").write_text("x")
    r = move(str(models), str(disk2 / "full"))
    assert r.status_code == 409 and "not an empty folder" in r.json()["error"] and "full/models" in r.json()["error"]
    (disk2 / "empty").mkdir()
    assert move(str(models), str(disk2 / "empty")).json()["needs_confirm"]   # an empty folder is fine
    # a folder an instance mounts as a whole can't have a part of it moved
    other = {"name": "All", "image": "x/y:1", "volumes": [{"host": str(data), "container": "/all"}]}
    client.put("/api/v1/instances/all", json=other)
    assert "uses a folder" in move(str(models), str(disk2 / "m")).json()["error"]


def test_failed_copy_leaves_the_apps_where_they_were(stack, monkeypatch):
    app, client, backend, data, models, disk2 = stack
    svc = app.state.svc
    monkeypatch.setattr(moves, "_listing", lambda root: [str(root)])          # the copies never match
    r = client.post("/api/v1/storage/move", json={"source": str(models), "dest": str(disk2 / "m"), "confirm": True})
    assert r.status_code == 202
    job = wait(app)
    assert job.status == "failed" and "not the same" in job.error
    assert hosts(client, "llm") == [str(models)] and not (disk2 / "m").exists()
    [rec] = svc.move_history()
    assert rec["status"] == "failed" and rec["leftover"] == ""


def test_trash_records_follow_a_moved_folder(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.gguf").write_text("a")
    trash = Trash(tmp_path / "data")
    e = trash.put_files("a", str(models), [str(models / "a.gguf")])
    assert trash.relocate(str(models), "/mnt/new") == 1
    [item] = trash.entry(e.id).items
    assert item.folder == "/mnt/new" and item.original == "/mnt/new/a.gguf"
    assert item.stored.startswith("/mnt/new/.shepherd-trash/")


def test_helper_scripts(tmp_path):
    """The helper's scripts, run with sh here (the paths it mounts swapped for real ones)."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a" / "b").mkdir(parents=True)
    (src / "a" / "b" / "m.gguf").write_bytes(b"m" * 3000)
    os.symlink("a/b/m.gguf", src / "link")
    dst.mkdir()
    check = moves.CHECK.format(src=str(src), parent=str(dst), dst=str(dst / "models")).replace("/host", "")
    c = moves.parse_check(subprocess.run(["sh", "-c", check], capture_output=True, text=True).stdout)
    assert not c.error and c.size >= 3000 and c.free > 0 and c.disk and c.same_disk
    copy = moves.COPY.format(name="models").replace("S=/src; D=/dst/", f"S={src}; D={dst}/")
    p = subprocess.run(["sh", "-c", copy], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "VERIFIED" in p.stdout and (dst / "models" / "a" / "b" / "m.gguf").read_bytes() == b"m" * 3000
    assert os.readlink(dst / "models" / "link") == "a/b/m.gguf"
    again = moves.parse_check(subprocess.run(["sh", "-c", check], capture_output=True, text=True).stdout)
    assert "not an empty folder" in again.error


def test_helper_mover_uses_containers():
    backend = FakeBackend()
    m = moves.HelperMover(backend, "img")
    c = m.check("/opt/gs/models", "/mnt/my disk/models")
    assert c.error                                               # the fake helper answers nothing
    script, volumes = backend.tasks[-1]
    assert volumes == [("/", "/host", True)] and "'/mnt/my disk/models'" in script
    m.start("x", "/opt/gs/models", "/mnt/my disk/models")
    assert "helper gpu-shepherd-move-x" in backend.calls
    m.remove("/opt/gs/models")
    script, volumes = backend.tasks[-1]
    assert script == "rm -rf /p/models" and volumes == [("/opt/gs", "/p", False)]
    for bad in ("/", "/etc", "relative", "/usr/lib/x"):
        with pytest.raises(ValueError):
            m.remove(bad)


def test_storage_page_moves(stack):
    app, client, backend, data, models, disk2 = stack
    page = TestClient(app)
    page.post("/login", data={"user": "admin", "password": "correct-horse"})
    html = page.get("/storage").text
    assert "Move to another disk" in html and f'value="{models}"' in html
    r = page.post("/storage/move/check", headers=HX, data={"source": str(models), "dest": str(disk2 / "m")})
    assert r.status_code == 200 and "Move Models?" in r.text and r.headers["HX-Retarget"] == "#modal-body"
    r = page.post("/storage/move", headers=HX, data={"source": str(models), "dest": str(disk2 / "m")})
    assert r.status_code == 204 and r.headers["HX-Redirect"].startswith("/?msg=Moving")
    wait(app)
    html = page.get("/storage").text
    assert "Move back" in html and "Delete old copy" in html
