"""Settings > Storage: moving a models or app data folder to another place (moves.py)."""
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, drives, moves
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
    assert r.json()["stops"] == ["LLM"] and r.json()["size"] >= 4100 and r.json()["same_disk"] is not None
    assert not any("can't see" in w for w in r.json()["warnings"])     # GPU Shepherd sees disk2
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


def test_moving_back_home_after_the_old_copy_is_deleted(stack):
    """Once the old copy is deleted, "Move back" can't just switch over: the folder is copied home."""
    app, client, backend, data, models, disk2 = stack
    src = data / "appdata" / "comfyui"
    (src / "output" / "img.png").write_bytes(b"i" * 10)
    rows = {m["path"]: m for m in client.get("/api/v1/storage").json()["movable"]}
    assert rows[str(src)]["home"] == ""
    assert client.post("/api/v1/storage/move", json={"source": str(src), "home": True}).status_code == 409
    there, further = disk2 / "comfyui", disk2 / "again"
    client.post("/api/v1/storage/move", json={"source": str(src), "dest": str(there), "confirm": True})
    assert wait(app).status == "done"
    # the old copy is still at home: Move back is the way
    r = client.post("/api/v1/storage/move", json={"source": str(there), "home": True})
    assert r.status_code == 409 and "Move back" in r.json()["error"]
    [rec] = client.get("/api/v1/moves").json()
    client.delete(f"/api/v1/moves/{rec['id']}/leftover")
    # moved again, then home: back to the first place, which is listed for the moved folder
    client.post("/api/v1/storage/move", json={"source": str(there), "dest": str(further), "confirm": True})
    assert wait(app).status == "done"
    rows = {m["path"]: m for m in client.get("/api/v1/storage").json()["movable"]}
    assert rows[str(further)]["home"] == str(src) and rows[str(further)]["label"] == "App data: ComfyUI"
    r = client.post("/api/v1/storage/move", json={"source": str(further), "home": True})
    assert r.status_code == 409 and r.json()["needs_confirm"] and r.json()["dest"] == str(src)
    r = client.post("/api/v1/storage/move", json={"source": str(further), "home": True, "confirm": True})
    assert r.status_code == 202 and wait(app).status == "done"
    assert hosts(client, "comfyui")[1] == str(src / "output")
    assert (src / "output" / "img.png").read_bytes() == b"i" * 10
    rows = {m["path"]: m for m in client.get("/api/v1/storage").json()["movable"]}
    assert rows[str(src)]["home"] == "" and str(further) not in rows


def test_storage_page_moves_home(stack):
    app, client, backend, data, models, disk2 = stack
    page = TestClient(app)
    page.post("/login", data={"user": "admin", "password": "correct-horse"})
    client.post("/api/v1/storage/move", json={"source": str(models), "dest": str(disk2 / "m"), "confirm": True})
    assert wait(app).status == "done"
    [rec] = client.get("/api/v1/moves").json()
    client.delete(f"/api/v1/moves/{rec['id']}/leftover")
    html = page.get("/storage").text
    assert f"Move back to {models}" in html
    r = page.post("/storage/move/check", headers=HX, data={"source": str(disk2 / "m"), "home": "1"})
    assert r.status_code == 200 and f'"dest": "{models}"' in r.text
    assert "the install drive" in page.get("/storage/pick", params={"source": str(disk2 / "m")}).text


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
    for keep, name in (("-dR", "plain"), ("-a", "models")):    # -dR: to a Windows drive on WSL2
        copy = moves.COPY.format(name=name, keep=keep).replace("S=/src; D=/dst/", f"S={src}; D={dst}/")
        p = subprocess.run(["sh", "-c", copy], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "VERIFIED" in p.stdout and (dst / name / "a" / "b" / "m.gguf").read_bytes() == b"m" * 3000
        assert os.readlink(dst / name / "link") == "a/b/m.gguf"
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


# --------------------------------------------------------------------------- drives and the folder picker

LSBLK = """{"blockdevices": [
 {"name": "sda", "kname": "sda", "size": 500107862016, "type": "disk", "fstype": null, "model": "System SSD", "rm": false, "ro": false,
  "children": [
   {"name": "sda1", "kname": "sda1", "size": 536870912, "type": "part", "fstype": "vfat", "uuid": "AAAA-1111", "mountpoint": "/boot/efi",
    "parttype": "c12a7328-f81f-11d2-ba4b-00a0c93ec93b", "rm": false, "ro": false},
   {"name": "sda2", "kname": "sda2", "size": 499000000000, "type": "part", "fstype": "ext4", "uuid": "root-uuid", "mountpoint": "/", "fsavail": 300000000000}]},
 {"name": "sdb", "kname": "sdb", "size": 1000204886016, "type": "disk", "fstype": null, "model": "Fast SSD", "rm": false, "ro": false,
  "children": [
   {"name": "sdb1", "kname": "sdb1", "size": 1000000000000, "type": "part", "fstype": "ext4", "label": "Models Disk", "uuid": "bbbb-2222", "mountpoint": null},
   {"name": "sdb2", "kname": "sdb2", "size": 1000000, "type": "part", "fstype": "crypto_LUKS", "uuid": "cccc-3333", "mountpoint": null}]},
 {"name": "sdc", "kname": "sdc", "size": 2000000000000, "type": "disk", "fstype": null, "model": "New Disk", "rm": true, "ro": false},
 {"name": "loop0", "kname": "loop0", "size": 1000, "type": "loop", "fstype": "squashfs", "mountpoint": "/snap/x"}]}"""


def test_lsblk_is_read_into_drives():
    found = {d.device: d for d in drives.parse_lsblk(LSBLK)}
    assert set(found) == {"/dev/sda1", "/dev/sda2", "/dev/sdb1", "/dev/sdb2", "/dev/sdc"}      # no loop device, no bare parent disks
    assert found["/dev/sda2"].state == "mounted" and found["/dev/sda2"].system
    assert found["/dev/sdb1"].state == "ready" and found["/dev/sdb1"].name == "Models Disk"
    assert found["/dev/sdb2"].state == "unsupported" and "encrypted" in found["/dev/sdb2"].note
    assert found["/dev/sdc"].state == "empty" and found["/dev/sdc"].removable and "never formats" in found["/dev/sdc"].note


def test_mount_names_are_new_folders_under_mnt():
    d = drives.Drive("/dev/sdb1", "x", label="Models Disk", uuid="bbbb-2222")
    assert drives.mount_name(d, set()) == "/mnt/Models-Disk"
    assert drives.mount_name(d, {"/mnt/Models-Disk"}) == "/mnt/Models-Disk-2"
    assert drives.mount_name(drives.Drive("/dev/sdb1", "x", uuid="bbbb-2222"), set()) == "/mnt/disk-bbbb2222"
    assert drives.mount_name(drives.Drive("/dev/sdb1", "x", label="../../etc", uuid="u"), set()).startswith("/mnt/etc")


class FakeHost(drives.Host):
    def __init__(self):
        self.found = drives.parse_lsblk(LSBLK)
        self.mounts = []

    def scan(self):
        self.scans = getattr(self, "scans", 0) + 1
        return [drives.Drive(**d.__dict__) for d in self.found]

    def signature(self):                   # as /proc/partitions: changes when a drive comes or goes
        return ",".join(d.device for d in self.found)

    def mount(self, drive, base, persist):
        self.mounts.append((drive.device, base, persist))
        mine = next(d for d in self.found if d.device == drive.device)
        mine.state, mine.mountpoint = "mounted", base
        return base

    def browse(self, path):
        return drives.Listing(path, [drives.Folder("models", False), drives.Folder("disk2", True)], 10, 20, "/dev/sda2")


@pytest.fixture
def host_stack(stack):
    app, client, backend, data, models, disk2 = stack
    app.state.svc.host_fs = FakeHost()
    page = TestClient(app)
    page.post("/login", data={"user": "admin", "password": "correct-horse"})
    return app.state.svc, page, models


def attach_later(svc, device="/dev/sdb1"):
    """Start with `device` unplugged and scan, then plug it in: only then is it newly attached."""
    host = svc.host_fs
    drive = next(d for d in host.found if d.device == device)
    host.found.remove(drive)
    svc.scan_drives()
    host.found.append(drive)


def test_drives_already_there_at_start_are_not_mounted(host_stack, stack):
    svc, _, _ = host_stack
    by = {d["device"]: d for d in stack[1].get("/api/v1/storage/drives").json()["drives"]}
    assert by["/dev/sdb1"]["state"] == "ready" and not svc.host_fs.mounts       # Mount does it by hand


def test_mounting_never_reloads_systemd():
    # a daemon-reload takes the GPUs away from running containers (NVML "Unknown Error")
    assert "daemon-reload" not in drives.MOUNT and "systemctl" not in drives.MOUNT


def test_a_new_drive_with_data_is_mounted_and_others_are_left_alone(host_stack, stack):
    svc, _, _ = host_stack
    client = stack[1]
    attach_later(svc)
    r = client.get("/api/v1/storage/drives").json()
    by = {d["device"]: d for d in r["drives"]}
    assert svc.host_fs.mounts == [("/dev/sdb1", "/mnt/Models-Disk", True)]
    assert by["/dev/sdb1"]["state"] == "mounted" and by["/dev/sdb1"]["by_us"]
    assert by["/dev/sdc"]["state"] == "empty" and by["/dev/sdb2"]["state"] == "unsupported"      # never touched
    client.get("/api/v1/storage/drives")
    assert len(svc.host_fs.mounts) == 1                                                       # not mounted twice


def test_automatic_mounting_can_be_turned_off(host_stack, stack):
    svc, _, _ = host_stack
    client = stack[1]
    assert client.put("/api/v1/storage/drives/auto", json={"enabled": False}).status_code == 200
    by = {d["device"]: d for d in client.get("/api/v1/storage/drives").json()["drives"]}
    assert by["/dev/sdb1"]["state"] == "ready" and not svc.host_fs.mounts
    assert client.post("/api/v1/storage/drives/mount", json={"device": "/dev/sdb1"}).status_code == 200
    assert svc.host_fs.mounts[0][0] == "/dev/sdb1"
    assert client.post("/api/v1/storage/drives/mount", json={"device": "/dev/sdc"}).status_code == 409       # empty: no
    assert client.post("/api/v1/storage/drives/mount", json={"device": "/dev/sdz"}).status_code == 404


def test_a_failed_mount_is_reported_and_not_retried(host_stack, stack):
    svc, _, _ = host_stack
    client = stack[1]

    def boom(drive, base, persist):
        raise ValueError("wrong fs type")
    svc.host_fs.mount = boom
    attach_later(svc)
    by = {d["device"]: d for d in client.get("/api/v1/storage/drives").json()["drives"]}
    assert by["/dev/sdb1"]["state"] == "failed" and "wrong fs type" in by["/dev/sdb1"]["note"]
    svc.host_fs.mount = lambda *a: pytest.fail("tried again by itself")
    assert client.get("/api/v1/storage/drives").json()["drives"]


def test_the_picker_dialog_and_tree(host_stack):
    svc, client, models = host_stack
    attach_later(svc)                                   # mounted by itself, so it shows as a place
    r = client.get("/storage/pick", params={"source": str(models)}, headers=HX)
    assert r.status_code == 200 and "Where should Models go?" in r.text
    assert "Models Disk" in r.text and "/mnt/Models-Disk" in r.text and "This machine" in r.text
    assert "System SSD" in r.text and 'hx-get="/storage/tree?root=1&path=/mnt/Models-Disk' in r.text
    t = client.get("/storage/tree", params={"path": "/mnt/x", "source": str(models)}, headers=HX).text
    assert 'value="/mnt/x/disk2"' in t and "another disk" in t
    assert client.get("/storage/pick", params={"source": "/nope"}, headers=HX).status_code == 204       # a toast


def test_picking_a_folder_gives_the_new_place(host_stack):
    svc, client, models = host_stack
    ok = client.get("/storage/pick/info", params={"source": str(models), "pick": "/mnt/disk2", "name": "models"}, headers=HX).text
    assert 'data-pick-file="/mnt/disk2/models"' in ok and "/dev/sda2" in ok
    bad = client.get("/storage/pick/info", params={"source": str(models), "pick": "/etc", "name": "models"}, headers=HX).text
    assert "system folder" in bad and "data-pick-file" not in bad
    inside = client.get("/storage/pick/info", params={"source": str(models), "pick": str(models), "name": "new"}, headers=HX).text
    assert "inside" in inside
    slash = client.get("/storage/pick/info", params={"source": str(models), "pick": "/mnt/d", "name": "a/b"}, headers=HX).text
    assert "just a folder name" in slash


def test_browse_api_and_local_listing(stack, tmp_path):
    app, client, *_ = stack
    (tmp_path / "disk2" / "sub").mkdir()
    r = client.get("/api/v1/storage/browse", params={"path": str(tmp_path / "disk2")}).json()
    assert [f["name"] for f in r["folders"]] == ["sub"] and r["free"] > 0
    assert client.get("/api/v1/storage/browse", params={"path": "/nonexistent-x"}).status_code == 404
    root = client.get("/api/v1/storage/browse", params={"path": "/"}).json()
    assert not {"proc", "etc", "sys"} & {f["name"] for f in root["folders"]}


def test_drive_scripts_and_listing_parse(stack):
    """The listing script, and the host script for a drive, run in the right places and with the right limits."""
    app, client, backend, data, models, disk2 = stack
    h = drives.HelperHost(backend, "img")
    h.browse("/mnt/some disk")
    script, vols = backend.tasks[-1]
    assert vols == [("/", "/host", True)] and "P=/host'/mnt/some disk'" in script
    with pytest.raises(ValueError):
        h.mount(drives.Drive("/dev/x", "x", uuid="a;rm -rf /"), "/mnt/x", True)
    parsed = drives.parse_listing("/", "DISK /dev/sda2\nTOTAL 100\nFREE 40\nDIR\tmnt\t0\nDIR\tetc\t0\nDIR\tdata\t1\n")
    assert [f.name for f in parsed.folders] == ["data", "mnt"] and parsed.free == 40 * 1024 and parsed.folders[0].mount


def test_check_warns_when_gpu_shepherd_cant_see_the_new_place():
    assert any("can't see the new place" in w for w in moves.Check(hidden=True).warnings)
    assert moves.Check().warnings == []


def test_partitions_are_grouped_under_their_drive():
    groups = {g["device"]: g for g in drives.group_by_disk(drives.parse_lsblk(LSBLK))}
    assert [d.device for d in groups["/dev/sdb"]["parts"]] == ["/dev/sdb1", "/dev/sdb2"]
    assert groups["/dev/sdb"]["name"] == "Fast SSD" and groups["/dev/sdb"]["size"] == 1000204886016
    assert len(groups["/dev/sda"]["parts"]) == 2 and len(groups["/dev/sdc"]["parts"]) == 1


def test_drives_are_scanned_once_while_nothing_changes(host_stack):
    """A scan runs a privileged helper on the host, and the Storage page asks every few seconds."""
    svc, page, models = host_stack
    for _ in range(5):
        assert page.get("/storage/drives").status_code == 200
    assert svc.host_fs.scans == 1
    attach_later(svc)                                    # a drive goes (scanned there) and comes back
    assert page.get("/storage/drives").status_code == 200
    assert svc.host_fs.scans == 3
