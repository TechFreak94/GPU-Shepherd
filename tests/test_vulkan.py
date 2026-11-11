"""AMD GPUs: found in /sys, planned apart from NVIDIA ones, and given to apps as render nodes
(the Vulkan Llama Swap)."""
import pytest
import yaml
from test_app import stack  # noqa: F401 - the app fixture

from gpushepherd import gpu as gpu_mod, instance_templates as tpl
from gpushepherd.backend import FakeBackend
from gpushepherd.config import AMD_BASE, Catalog, CatalogModel, ConfigError, GpuPolicy, Instance, Store
from gpushepherd.engine import Engine
from gpushepherd.scheduler import gpu_list, gpu_text, plan


def machine(**kw):
    base = dict(gpu_mem_gb=[24.0], instances={}, servers={}, model_sizes={})
    return tpl.Machine(**(base | kw))


def _card(root, name, bus, values, render=None):
    real = root / "devices" / bus
    real.mkdir(parents=True)
    for key, value in values.items():
        (real / key).write_text(value + "\n")
    if render:
        (real / "drm" / render).mkdir(parents=True)
    (root / "drm" / name).mkdir(parents=True)
    (root / "drm" / name / "device").symlink_to(real)


def test_amd_gpus_from_sysfs(tmp_path, monkeypatch):
    monkeypatch.delenv("SHEPHERD_FAKE_AMD_GPUS", raising=False)
    monkeypatch.delenv("SHEPHERD_FAKE_GPUS", raising=False)
    vram = str(8 * 2**30)
    _card(tmp_path, "card0", "0000:01:00.0", {"vendor": "0x10de"})                        # NVIDIA: skipped
    _card(tmp_path, "card1", "0000:0b:00.0", {"vendor": "0x1002", "device": "0x67df", "mem_info_vram_total": vram,
                                              "mem_info_vram_used": str(2**30), "gpu_busy_percent": "12"}, "renderD129")
    _card(tmp_path, "card2", "0000:05:00.0", {"vendor": "0x1002", "product_name": "Radeon RX 580",
                                              "mem_info_vram_total": vram}, "renderD130")
    _card(tmp_path, "card3", "0000:06:00.0", {"vendor": "0x1002"})                        # no amdgpu: skipped
    (tmp_path / "drm" / "card1-DP-1").mkdir()                                            # a connector
    gpus = gpu_mod.amd_gpus(str(tmp_path / "drm"))
    # numbered from AMD_BASE by PCI address, not by card number
    assert [(g.index, g.name, g.mem_total, g.mem_used, g.util, g.bus_id, g.render) for g in gpus] == [
        (100, "Radeon RX 580", 8192, 0, 0.0, "0000:05:00.0", "/dev/dri/renderD130"),
        (101, "AMD GPU 0x67df", 8192, 1024, 12.0, "0000:0b:00.0", "/dev/dri/renderD129")]
    assert gpus[0].vendor == "amd" and gpus[1].label == "AMD GPU 1"
    assert gpu_mod.amd_gpus(str(tmp_path / "missing")) == []


def test_amd_only_machine_needs_no_nvidia_smi(monkeypatch):
    monkeypatch.delenv("SHEPHERD_FAKE_GPUS", raising=False)
    monkeypatch.setattr(gpu_mod, "_nvidia", lambda: ([], "nvidia-smi not found - give ..."))
    monkeypatch.setenv("SHEPHERD_FAKE_AMD_GPUS", "2x8192:AMD Radeon RX 580")
    gpus, err = gpu_mod._query()
    assert [g.index for g in gpus] == [100, 101] and err == ""
    monkeypatch.delenv("SHEPHERD_FAKE_AMD_GPUS")
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x8192:x")       # faked NVIDIA: no real AMD read
    assert gpu_mod.amd_gpus() == []


def test_gpu_names():
    assert gpu_text([0, 1]) == "GPUs 0,1" and gpu_text([100]) == "AMD GPU 0"
    assert gpu_text([100, 101, 102]) == "AMD GPUs 0-2" and gpu_text([1, 100]) == "GPU 1 and AMD GPU 0"
    assert gpu_list([0, 100, 101]) == "0,amd0,amd1"


def test_policy_gpus_must_be_of_its_kind():
    assert GpuPolicy(mode="fixed", gpus=[101], vendor="amd").gpus == [101]
    with pytest.raises(ValueError, match="AMD GPU 0 is not an NVIDIA GPU"):
        GpuPolicy(mode="fixed", gpus=[100])
    with pytest.raises(ValueError, match="GPU 0 is not an AMD GPU"):
        GpuPolicy(mode="fixed", gpus=[0], vendor="amd")


def _inst(iid, vendor="nvidia", **gpu):
    return Instance(id=iid, name=iid, image="x", gpu=GpuPolicy(mode="elastic", vendor=vendor, **gpu))


def test_each_kind_of_gpu_is_planned_apart():
    gpus = [0, AMD_BASE, AMD_BASE + 1]
    insts = {"cuda": _inst("cuda", priority=90), "vk": _inst("vk", "amd"), "vk2": _inst("vk2", "amd", priority=80),
             "web": Instance(id="web", name="web", image="x")}
    p = plan(gpus, insts, {}, "start", "vk", extra_wanted={"cuda", "web"})
    assert p.allocations == {"cuda": [0], "vk": [100, 101], "web": []}      # NVIDIA apps never get AMD GPUs
    p = plan(gpus, insts, {"cuda": [0], "vk": [100, 101]}, "start", "vk2")
    assert sorted(p.allocations["vk2"] + p.allocations["vk"]) == [100, 101] and p.allocations["cuda"] == [0]
    # no AMD GPU: an AMD app can't start (or falls back to the CPU), whatever NVIDIA GPUs are free
    p = plan([0, 1], insts, {}, "start", "vk")
    assert "needs at least 1 GPU(s); only 0 of this machine's 0 AMD GPUs" in p.error
    insts["vk"] = _inst("vk", "amd", cpu_fallback=True)
    assert plan([0, 1], insts, {}, "start", "vk").on_cpu == ["vk"]


def test_vulkan_server_installs_on_the_amd_gpus():
    t = tpl.get("llm-server-vulkan")
    m = machine(amd_gpu_mem_gb=[8.0, 8.0, 8.0])
    f = tpl.defaults(t, m, "/srv/models", "/a")
    inst, models = tpl.build(t, f | {"chat_file": "chat.gguf"}, m)
    assert inst.kind == "llama-swap" and inst.image == tpl.LLAMA_SWAP_VULKAN_IMAGE and inst.devices == []
    assert (inst.gpu.mode, inst.gpu.vendor, inst.gpu.max_gpus, inst.gpu.cpu_fallback) == ("elastic", "amd", 0, True)
    assert [x.kv_cache for x in models] == ["f16"] and models[0].ctx_size == 16384
    assert tpl.gpu_mem_gb(t, m) == [8.0, 8.0, 8.0] and tpl.gpu_mem_gb(tpl.get("llm-server"), m) == [24.0]
    # its GPU choices list only AMD GPUs, and pinning one checks its kind
    gpus = [gpu_mod.Gpu(0, "RTX", 0, 24576, 0), gpu_mod.Gpu(100, "RX 580", 0, 8192, 0),
            gpu_mod.Gpu(101, "RX 580", 0, 8192, 0)]
    choices = tpl.gpu_choices(gpus, t)
    assert "pin:100" in choices and "pin:0" not in choices and "Always AMD GPU 0" in choices["pin:100"]
    assert "pin:100" not in tpl.gpu_choices(gpus, tpl.get("comfyui"))
    assert tpl.build(t, f | {"gpu": "pin:101"}, m)[0].gpu.gpus == [101]
    with pytest.raises(ConfigError, match="not an AMD GPU"):
        tpl.build(t, f | {"gpu": "pin:0"}, m)


def test_the_two_llama_swaps_tell_their_instances_apart():
    vk, cuda = tpl.get("llm-server-vulkan"), tpl.get("llm-server")
    mine = Instance(id="v", name="V", image="x", kind="llama-swap", app="llm-server-vulkan")
    by_hand = Instance(id="h", name="H", image="x", kind="llama-swap")
    old = Instance(id="o", name="O", image="x", kind="llama-swap", app="llm-chat")
    assert tpl.matches(vk, mine) and not tpl.matches(cuda, mine)
    assert tpl.matches(cuda, by_hand) and not tpl.matches(vk, by_hand)
    assert tpl.matches(cuda, old) and not tpl.matches(vk, old)


def test_devices_must_be_device_paths():
    assert Instance(id="a", name="A", image="x", devices=[" /dev/kfd/ ", "", "/dev/kfd"]).devices == ["/dev/kfd"]
    for bad in ("/etc/passwd", "/dev/../etc", "kfd"):
        with pytest.raises(ValueError):
            Instance(id="a", name="A", image="x", devices=[bad])


def test_new_fields_dont_restart_existing_apps(tmp_path):
    engine = Engine(Store(str(tmp_path)), FakeBackend())
    inst = Instance(id="a", name="A", image="x")
    before = engine.config_hash(inst)
    assert engine.config_hash(inst.model_copy(update={"gpu": GpuPolicy(vendor="amd")})) == before
    assert engine.config_hash(inst.model_copy(update={"devices": ["/dev/kfd"]})) != before


def _engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "1x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_FAKE_AMD_GPUS", "3x8192:AMD Radeon RX 580")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    store = Store(str(tmp_path / "data"))
    inst = Instance(id="vk", name="VK", image=tpl.LLAMA_SWAP_VULKAN_IMAGE, kind="llama-swap",
                    gpu=GpuPolicy(mode="elastic", vendor="amd"), devices=["/dev/kfd"])
    store.save_instance(inst)
    return store, inst, Engine(store, FakeBackend())


def test_engine_gives_an_amd_app_its_render_nodes(tmp_path, monkeypatch):
    store, inst, engine = _engine(tmp_path, monkeypatch)
    store.save_catalog(Catalog(instance_id="vk", llama_server="/app/llama-server", fit_gpus=True,
                               models=[CatalogModel(id="big", file="/models/big.gguf"),
                                       CatalogModel(id="small", file="/models/small.gguf", gpus=1)]))
    spec = engine.build_spec(inst, [101, 102])
    assert spec.devices == ["/dev/kfd", "/dev/dri/renderD129", "/dev/dri/renderD130"] and spec.gpus == [101, 102]
    assert "CUDA_DEVICE_ORDER" not in spec.env
    cfg = yaml.safe_load((store.generated_dir("vk") / "llama-swap.yaml").read_text())["models"]
    assert "--n-gpu-layers 0" not in cfg["big"]["cmd"] and "env" not in cfg["big"]
    assert cfg["small"]["env"][0].startswith("GGML_VK_VISIBLE_DEVICES=")     # Vulkan's way to pick its card
    # the plan puts it on the AMD GPUs, and the NVIDIA one stays free
    assert engine.plan("start", "vk").allocations["vk"] == [100, 101, 102]


def test_an_amd_gpu_without_a_render_node_is_refused(tmp_path, monkeypatch):
    store, inst, engine = _engine(tmp_path, monkeypatch)
    monkeypatch.setattr(gpu_mod, "render_nodes", lambda _ids: None)
    with pytest.raises(RuntimeError, match="render node"):
        engine.build_spec(inst, [100])


def test_docker_gets_amd_gpus_as_devices_and_nvidia_ones_as_a_request():
    from test_docker_backend import make_backend, spec
    b = make_backend()
    b.create_and_start(spec(gpus=[100], devices=["/dev/dri/renderD128"]))
    hc = b.calls[1][2]["host_config"]
    assert hc["Devices"] == [{"PathOnHost": "/dev/dri/renderD128", "PathInContainer": "/dev/dri/renderD128",
                              "CgroupPermissions": "rwm"}]
    assert not hc.get("DeviceRequests")
    assert b.calls[1][2]["labels"]["gpushepherd.gpus"] == "100"      # still remembered, for the planner
    b = make_backend()
    b.create_and_start(spec())
    assert not b.calls[1][2]["host_config"].get("Devices")


def test_install_and_run_on_the_dashboard(stack, monkeypatch):  # noqa: F811 - the fixture
    from test_app import HX, act, login
    app, client, backend, data = stack
    monkeypatch.setenv("SHEPHERD_FAKE_AMD_GPUS", "2x8192:AMD Radeon RX 580")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    login(client)
    assert "Every free AMD GPU" in client.get("/apps").text
    form = client.get("/apps/llm-server-vulkan").text
    assert "This machine has 2 AMD GPUs (AMD Radeon RX 580, 8 GB" in form and "Always AMD GPU 1" in form
    assert "Always GPU 0" not in form
    r = client.post("/instances/guided", headers=HX, data={
        "template": "llm-server-vulkan", "name": "Vulkan", "id": "vk", "port": "8090", "gpu": "all",
        "models_dir": "/srv/models", "chat_file": "/models/x.gguf", "action": "create"})
    assert r.status_code == 204, r.text
    inst = yaml.safe_load((data / "instances" / "vk.yaml").read_text())
    assert inst["gpu"]["vendor"] == "amd" and inst["app"] == "llm-server-vulkan"

    edit = client.get("/instances/vk/edit").text      # the full form keeps the kind of GPU
    assert '<option value="amd" selected' in edit and "AMD GPU 1</label>" in edit

    act(client, app, "vk", "start")
    spec = backend.containers["shepherd-vk"]
    assert spec.gpus == [100, 101] and spec.devices == ["/dev/dri/renderD128", "/dev/dri/renderD129"]
    dash = client.get("/ui/dashboard", headers=HX).text
    assert "<strong>AMD GPU 1</strong>" in dash and "AMD GPUs 0,1" in dash
