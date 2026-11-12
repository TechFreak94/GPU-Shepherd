"""Every preset in gpushepherd/presets must load cleanly and describe itself for the installer."""
import re
import shutil
from pathlib import Path

import pytest
import yaml

import gpushepherd
from gpushepherd.config import Store

PRESETS = Path(gpushepherd.__file__).parent / "presets"
NAMES = sorted(p.name for p in PRESETS.iterdir() if (p / "instances").is_dir())


def test_presets_exist():
    assert "starter" in NAMES


@pytest.mark.parametrize("name", NAMES)
def test_preset_loads(name, tmp_path):
    preset = PRESETS / name
    for part in ("instances", "catalogs"):
        if (preset / part).is_dir():
            shutil.copytree(preset / part, tmp_path / part)
    store = Store(tmp_path)
    instances, problems = store.instances()
    assert not problems and instances
    for iid, inst in instances.items():
        if inst.kind == "llama-swap":
            assert store.catalog(iid).models
    host_ports = [p.host for i in instances.values() for p in i.ports]
    assert len(host_ports) == len(set(host_ports))
    # paths use the placeholders the installer fills in
    for inst in instances.values():
        for vol in inst.volumes:
            assert vol.host.startswith(("/srv/models", "/opt/gpu-shepherd/")), vol.host


@pytest.mark.parametrize("name", NAMES)
def test_preset_description(name):
    text = (PRESETS / name / "preset.yaml").read_text()
    fields = dict(re.findall(r"^(\w+): *(.+)$", text, re.MULTILINE))
    assert {"name", "description", "needs"} <= fields.keys()
    if (PRESETS / name / "downloads.txt").exists():
        assert fields["download_gb"].isdigit()


@pytest.mark.parametrize("name", NAMES)
def test_preset_downloads_are_well_formed(name):
    path = PRESETS / name / "downloads.txt"
    if not path.exists():
        pytest.skip("no downloads")
    groups, used = {}, set()
    for line in path.read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0] == "group":
            assert words[2] in ("default", "optional") and len(words) >= 5, line
            groups[words[1]] = words[2]
        else:
            assert words[0] == "file" and len(words) >= 4, line
            assert words[1] in groups, line
            assert not words[2].startswith("/") and ".." not in words[2], line
            assert all(u.startswith("https://") for u in words[3:]), line
            used.add(words[1])
    assert used == set(groups)
    assert "default" in groups.values()


def test_starter_open_webui_makes_pictures_as_the_app_center_sets_it_up():
    from gpushepherd.connectors import comfyui
    env = yaml.safe_load((Path(__file__).parent.parent / "gpushepherd/presets/starter/instances/webui.yaml")
                         .read_text())["env"]
    want = comfyui.openwebui_env("http://comfyui:8188", "sd_xl_base_1.0.safetensors")
    assert {k: env.get(k) for k in want} == want
