"""Settings > Storage: what takes space, and cleaning it up through the trash."""
import os

import pytest
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, storage
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Instance, Store, VolumeMap
from gpushepherd.service import HelperFileOps


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    (models / "qwen").mkdir(parents=True)
    (models / "qwen" / "qwen.gguf").write_bytes(b"q" * 4000)            # in the model list
    (models / "old.gguf").write_bytes(b"o" * 3000)                      # in no model list
    (models / "copy").mkdir()
    (models / "copy" / "qwen.gguf").write_bytes(b"q" * 4000)            # the same file again
    (models / "half.gguf.part").write_bytes(b"h" * 500)                  # an unfinished download
    (models / "comfyui" / "checkpoints").mkdir(parents=True)
    (models / "comfyui" / "checkpoints" / "sd.safetensors").write_bytes(b"s" * 100)   # ComfyUI's
    backend = FakeBackend()
    app = create_app(data_dir=str(data), backend=backend, demo=True, autostart=False)
    client = TestClient(app, headers={"Authorization": f"Bearer {auth.api_token(Store(data))}"})
    llm = {"name": "LLM", "image": "ghcr.io/mostlygeek/llama-swap:cuda", "kind": "llama-swap",
           "volumes": [{"host": str(models), "container": "/models", "read_only": True}]}
    assert client.put("/api/v1/instances/llm", json=llm).status_code == 201
    r = client.put("/api/v1/instances/llm/models/qwen", json={"id": "qwen", "file": "/models/qwen/qwen.gguf"})
    assert r.status_code in (200, 201), r.text
    comfy = {"name": "ComfyUI", "image": "x/comfy:1",
             "volumes": [{"host": str(models / "comfyui"), "container": "/root/ComfyUI/models"},
                         {"host": str(data / "appdata" / "comfyui"), "container": "/out"}]}
    assert client.put("/api/v1/instances/comfyui", json=comfy).status_code == 201
    return app, client, backend, data, models


def test_scan_finds_what_is_worth_a_look(stack):
    app, client, backend, data, models = stack
    (data / "appdata" / "gone-app").mkdir(parents=True)
    (data / "appdata" / "gone-app" / "db").write_bytes(b"d" * 200)
    (data / "appdata" / "comfyui").mkdir(parents=True)
    s = client.get("/api/v1/storage").json()
    files = {os.path.relpath(f["path"], models): f["why"] for f in s["files"]}
    assert files == {"old.gguf": "unused", "copy/qwen.gguf": "duplicate", "half.gguf.part": "partial"}
    assert [f["label"] for f in s["leftover_folders"]] == ["gone-app"]
    assert [f["path"] for f in s["folders"] if f["label"] == "Models"] == [str(models)]    # comfyui is inside it
    assert s["disks"] and s["disks"][0]["free"] > 0 and s["hidden"] == []
    assert s["trash"] == [] and s["trash_days"] == 30


def test_files_and_folders_go_to_the_trash_and_come_back(stack):
    app, client, backend, data, models = stack
    old, part = str(models / "old.gguf"), str(models / "half.gguf.part")
    r = client.post("/api/v1/storage/files", json={"paths": [old, part]})
    assert r.status_code == 200 and "Moved 2 file(s)" in r.json()["message"]
    assert not os.path.exists(old) and (models / ".shepherd-trash").is_dir()
    s = client.get("/api/v1/storage").json()
    assert [f["why"] for f in s["files"]] == ["duplicate"]                 # the trash folder isn't scanned
    [entry] = s["trash"]
    assert entry["size"] == 3500 and entry["title"].startswith("2 model files")
    r = client.post("/api/v1/storage/files", json={"paths": ["/etc/passwd"]})
    assert r.status_code == 404                                          # only what the scan listed
    assert "Put" in client.post(f"/api/v1/trash/{entry['id']}/restore").json()["message"]
    assert os.path.exists(old) and os.path.exists(part)
    assert not (models / ".shepherd-trash" / entry["id"]).exists()
    # a leftover app data folder, then empty the trash
    (data / "appdata" / "gone-app").mkdir(parents=True)
    assert "Moved" in client.post("/api/v1/storage/leftovers", json={"name": "gone-app"}).json()["message"]
    assert client.post("/api/v1/storage/leftovers", json={"name": "comfyui"}).status_code == 404
    client.post("/api/v1/storage/files", json={"paths": [old]})
    assert len(client.get("/api/v1/trash").json()) == 2
    assert "freeing" in client.delete("/api/v1/trash").json()["message"]
    assert client.get("/api/v1/trash").json() == [] and not os.path.exists(old)
    assert not list((models / ".shepherd-trash").iterdir())


def test_unused_images_and_orphan_containers(stack):
    app, client, backend, data, models = stack
    backend.present = {"ghcr.io/mostlygeek/llama-swap:cuda": 5 * 10**9, "curlimages/curl:8.11.1": 20 * 10**6,
                       "someone/else:1": 10**9}
    rows = {i["image"]: i["used_by"] for i in client.get("/api/v1/storage").json()["images"]}
    assert rows == {"ghcr.io/mostlygeek/llama-swap:cuda": ["LLM"], "curlimages/curl:8.11.1": []}   # not other images
    msg = client.post("/api/v1/storage/images").json()["message"]
    assert "Removed 1 unused image(s), freeing 20 MB" in msg
    assert set(backend.present) == {"ghcr.io/mostlygeek/llama-swap:cuda", "someone/else:1"}
    from gpushepherd.backend import ContainerSpec
    backend.create_and_start(ContainerSpec(name="shepherd-ghost", image="x", instance_id="ghost", network="n", alias="g"))
    assert client.get("/api/v1/storage").json()["orphans"] == ["shepherd-ghost"]
    assert "Removed 1" in client.post("/api/v1/storage/orphans").json()["message"]
    assert "shepherd-ghost" not in backend.containers


def test_trash_days_setting(stack):
    app, client, *_ = stack
    r = client.patch("/api/v1/settings", json={"trash_days": "7"})
    assert r.status_code == 200 and r.json()["settings"]["trash_days"] == 7
    assert client.patch("/api/v1/settings", json={"trash_days": "soon"}).status_code == 400


def test_read_only_models_folders_use_a_helper(tmp_path, monkeypatch):
    backend = FakeBackend()
    ops = HelperFileOps(backend)
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    ops.move(str(tmp_path), [(str(tmp_path / "a b.gguf"), str(tmp_path / ".shepherd-trash" / "e1" / "0-a b.gguf"))])
    script, volumes = backend.tasks[-1]
    assert "mv -n '/f/a b.gguf' '/f/.shepherd-trash/e1/0-a b.gguf'" in script
    assert volumes == [(str(tmp_path), "/f", False)]
    ops.remove(str(tmp_path), [str(tmp_path / ".shepherd-trash" / "e1")])
    assert backend.tasks[-1][0].endswith("rm -rf /f/.shepherd-trash/e1")


def test_low_space_banner(stack, monkeypatch):
    app, client, backend, data, models = stack
    svc = app.state.svc
    assert svc.low_space() == []
    monkeypatch.setattr(storage, "LOW_GB", 10**9)
    low = svc.low_space()
    assert low and low[0]["path"] == str(data)


def test_model_folders_skip_nested_ones():
    vols = lambda *hosts: [VolumeMap(host=h, container="/models") for h in hosts]     # noqa: E731
    insts = {"a": Instance(id="a", name="A", image="x", volumes=vols("/srv/models")),
             "b": Instance(id="b", name="B", image="x", volumes=[VolumeMap(host="/srv/models/comfyui",
                                                                           container="/root/ComfyUI/models")]),
             "c": Instance(id="c", name="C", image="x", volumes=vols("/data/m"))}
    assert storage.model_folders(insts, "/models") == ["/data/m", "/srv/models"]
