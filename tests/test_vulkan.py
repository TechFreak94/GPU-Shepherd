"""The Vulkan Llama Swap: AMD GPUs passed in as devices, outside the GPU planner."""
import pytest
from test_app import stack  # noqa: F401 - the app fixture
import yaml

from gpushepherd import gpu as gpu_mod, instance_templates as tpl
from gpushepherd.backend import ContainerSpec, FakeBackend
from gpushepherd.config import Catalog, CatalogModel, ConfigError, Instance, Store
from gpushepherd.engine import Engine


def machine(**kw):
    base = dict(gpu_mem_gb=[24.0], instances={}, servers={}, model_sizes={})
    return tpl.Machine(**(base | kw))


def test_amd_gpus_from_sysfs(tmp_path, monkeypatch):
    monkeypatch.delenv("SHEPHERD_FAKE_AMD_GPUS", raising=False)
    files = {"card0": {"vendor": "0x10de"},                                   # NVIDIA: skipped
             "card1": {"vendor": "0x1002", "device": "0x67df", "mem_info_vram_total": str(8 * 2**30),
                       "mem_info_vram_used": str(2**30), "gpu_busy_percent": "12"},
             "card2": {"vendor": "0x1002", "product_name": "Radeon RX 580", "mem_info_vram_total": str(8 * 2**30)},
             "card3": {"vendor": "0x1002"},                                    # no amdgpu driver: skipped
             "card1-DP-1": {}}                                                 # a connector, not a card
    for card, values in files.items():
        dev = tmp_path / card / "device"
        dev.mkdir(parents=True)
        for name, value in values.items():
            (dev / name).write_text(value + "\n")
    gpus = gpu_mod.amd_gpus(str(tmp_path))
    assert [(g.index, g.name, g.mem_total, g.mem_used, g.util) for g in gpus] == [
        (0, "AMD GPU 0x67df", 8192, 1024, 12.0), (1, "Radeon RX 580", 8192, 0, 0.0)]
    assert gpu_mod.amd_gpus(str(tmp_path / "missing")) == []
    monkeypatch.setenv("SHEPHERD_FAKE_AMD_GPUS", "3x8192:AMD Radeon RX 580")
    assert len(gpu_mod.amd_gpus()) == 3 and gpu_mod.device_gpu_mem_gb(2) == [8.0, 8.0]
    assert gpu_mod.device_gpu_mem_gb(4) == [gpu_mod.ASSUMED_GPU_GB] * 4   # more than found: assumed


def test_vulkan_server_installs_with_devices_and_no_planned_gpu():
    t = tpl.get("llm-server-vulkan")
    m = machine(device_gpu_mem_gb=[8.0, 8.0, 8.0])
    f = tpl.defaults(t, m, "/srv/models", "/a")
    assert f["device_gpus"] == 3 and "device_gpus" in tpl.answer_keys(t) and "gpu" not in tpl.answer_keys(t)
    inst, models = tpl.build(t, f | {"chat_file": "chat.gguf"}, m)
    assert inst.kind == "llama-swap" and inst.image == tpl.LLAMA_SWAP_VULKAN_IMAGE
    assert inst.devices == ["/dev/dri"] and inst.device_gpus == 3 and inst.gpu.mode == "none"
    assert [x.kv_cache for x in models] == ["f16"] and models[0].ctx_size == 16384
    with pytest.raises(ConfigError, match="GPUs"):
        tpl.build(t, f | {"device_gpus": "0"}, m)
    # the GPUs it gets are the AMD ones (8 GB each when unknown), not the NVIDIA one
    assert tpl.gpu_mem_gb(t, m, {"device_gpus": "2"}) == [8.0, 8.0]
    assert tpl.gpu_mem_gb(t, machine(), {}) == [8.0] and tpl.gpu_mem_gb(tpl.get("llm-server"), m) == [24.0]


def test_the_two_llama_swaps_tell_their_instances_apart():
    vk, cuda = tpl.get("llm-server-vulkan"), tpl.get("llm-server")
    mine = Instance(id="v", name="V", image="x", kind="llama-swap", app="llm-server-vulkan")
    by_hand = Instance(id="h", name="H", image="x", kind="llama-swap")
    old = Instance(id="o", name="O", image="x", kind="llama-swap", app="llm-chat")
    assert tpl.matches(vk, mine) and not tpl.matches(cuda, mine)
    assert tpl.matches(cuda, by_hand) and not tpl.matches(vk, by_hand)
    assert tpl.matches(cuda, old) and not tpl.matches(vk, old)


def test_devices_must_be_device_paths():
    assert Instance(id="a", name="A", image="x", devices=[" /dev/dri/ ", "", "/dev/dri"]).devices == ["/dev/dri"]
    for bad in ("/etc/passwd", "/dev/../etc", "dri"):
        with pytest.raises(ValueError):
            Instance(id="a", name="A", image="x", devices=[bad])


def test_engine_passes_the_devices_and_splits_over_their_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x8192:Test GPU")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    store = Store(str(tmp_path / "data"))
    inst = Instance(id="vk", name="VK", image=tpl.LLAMA_SWAP_VULKAN_IMAGE, kind="llama-swap",
                    devices=["/dev/dri"], device_gpus=2)
    store.save_instance(inst)
    store.save_catalog(Catalog(instance_id="vk", llama_server="/app/llama-server",
                               models=[CatalogModel(id="m", file="/models/m.gguf", split_mode="layer")]))
    engine = Engine(store, FakeBackend())
    spec = engine.build_spec(inst, [])
    assert spec.devices == ["/dev/dri"] and spec.gpus == [] and "CUDA_DEVICE_ORDER" not in spec.env
    cmd = yaml.safe_load((store.generated_dir("vk") / "llama-swap.yaml").read_text())["models"]["m"]["cmd"]
    assert "--n-gpu-layers 0" not in cmd and "for 2 GPU(s)" in (store.generated_dir("vk") / "llama-swap.yaml").read_text()
    # no devices: a llama-swap without GPUs runs on the CPU, as before
    engine.build_spec(inst.model_copy(update={"devices": []}), [])
    assert "--n-gpu-layers 0" in (store.generated_dir("vk") / "llama-swap.yaml").read_text()


def test_docker_gets_the_devices():
    from test_docker_backend import make_backend, spec
    b = make_backend()
    b.create_and_start(spec(gpus=[], devices=["/dev/dri"]))
    assert b.calls[1][2]["host_config"]["Devices"] == [
        {"PathOnHost": "/dev/dri", "PathInContainer": "/dev/dri", "CgroupPermissions": "rwm"}]
    b = make_backend()
    b.create_and_start(spec())
    assert not b.calls[1][2]["host_config"].get("Devices")
    assert ContainerSpec(name="n", image="i", instance_id="i", network="n", alias="a").devices == []


def test_install_edit_and_start_from_the_dashboard(stack, monkeypatch):  # noqa: F811 - the fixture
    from test_app import HX, act, login
    app, client, backend, data = stack
    monkeypatch.setenv("SHEPHERD_FAKE_AMD_GPUS", "2x8192:AMD Radeon RX 580")
    login(client)
    assert "Every AMD GPU, through Vulkan" in client.get("/apps").text
    form = client.get("/apps/llm-server-vulkan").text
    assert 'name="device_gpus"' in form and "AMD Radeon RX 580 (8 GB)" in form and 'name="gpu"' not in form
    r = client.post("/instances/guided", headers=HX, data={
        "template": "llm-server-vulkan", "name": "Vulkan", "id": "vk", "port": "8090", "device_gpus": "2",
        "models_dir": "/srv/models", "chat_file": "/models/x.gguf", "action": "create"})
    assert r.status_code == 204, r.text
    inst = yaml.safe_load((data / "instances" / "vk.yaml").read_text())
    assert inst["devices"] == ["/dev/dri"] and inst["device_gpus"] == 2 and inst["app"] == "llm-server-vulkan"

    # editing keeps the devices (the full form has them)
    edit = client.get("/instances/vk/edit").text
    assert 'name="devices" value="/dev/dri"' in edit and 'name="device_gpus"' in edit

    act(client, app, "vk", "start")
    assert backend.containers["shepherd-vk"].devices == ["/dev/dri"] and backend.containers["shepherd-vk"].gpus == []
    assert "2 through /dev/dri" in client.get("/ui/dashboard", headers=HX).text
