import pytest
from pydantic import ValidationError

from gpushepherd.config import (Catalog, CatalogModel, ConfigError, GpuPolicy, Instance, Store, format_env,
                                format_ports, format_volumes, parse_env, parse_ports, parse_volumes)


def test_parse_and_format_round_trip():
    ports = parse_ports("8188:8188\n# comment\n\n5353:53/udp")
    assert [(p.host, p.container, p.protocol) for p in ports] == [(8188, 8188, "tcp"), (5353, 53, "udp")]
    assert format_ports(ports) == "8188:8188\n5353:53/udp"
    vols = parse_volumes("/srv/models:/models:ro\n/srv/out:/out")
    assert vols[0].read_only and not vols[1].read_only
    assert format_volumes(vols) == "/srv/models:/models:ro\n/srv/out:/out"
    env = parse_env("A=1\nB=x=y")
    assert env == {"A": "1", "B": "x=y"} and format_env(env) == "A=1\nB=x=y"


@pytest.mark.parametrize("text", ["8188", "a:b", "1:2:3"])
def test_bad_ports(text):
    with pytest.raises(ConfigError):
        parse_ports(text)


def test_relative_volume_rejected():
    with pytest.raises(ValidationError):
        parse_volumes("models:/models")


def test_instance_validation():
    with pytest.raises(ValidationError):
        Instance(id="Bad Name", name="x", image="img")
    with pytest.raises(ValidationError):
        Instance(id="ok", name="x", image="has space")
    with pytest.raises(ValidationError):
        GpuPolicy(mode="fixed", gpus=[])
    with pytest.raises(ValidationError):
        GpuPolicy(mode="elastic", min_gpus=2, max_gpus=1)


def test_store_round_trip_and_port_clash(tmp_path):
    store = Store(tmp_path)
    a = Instance(id="a", name="A", image="img:1", ports=parse_ports("8000:80"))
    store.save_instance(a)
    assert store.instance("a") == a
    with pytest.raises(ConfigError, match="already used"):
        store.save_instance(Instance(id="b", name="B", image="img:1", ports=parse_ports("8000:81")))
    (tmp_path / "instances" / "broken.yaml").write_text("id: [unclosed")
    found, problems = store.instances()
    assert list(found) == ["a"] and "broken.yaml" in problems[0]


def test_store_sees_hand_edits_and_hands_out_fresh_copies(tmp_path):
    store = Store(tmp_path)
    store.save_instance(Instance(id="a", name="A", image="img:1", env={"X": "1"}))
    first = store.instance("a")
    first.env["X"] = "changed"                    # a caller changing what it got back...
    assert store.instance("a").env == {"X": "1"}  # ...does not change the next read
    path = tmp_path / "instances" / "a.yaml"
    path.write_text(path.read_text().replace("name: A", "name: Edited by hand"))
    assert store.instance("a").name == "Edited by hand"


def test_settings_created_with_secret(tmp_path):
    s = Store(tmp_path).settings()
    assert len(s.session_secret) == 64
    assert oct((tmp_path / "settings.yaml").stat().st_mode)[-3:] == "600"


def test_catalog_consistency():
    m = CatalogModel(id="a", file="/m/a.gguf")
    with pytest.raises(ValidationError, match="unknown model"):
        Catalog(instance_id="x", models=[m], auto_aliases=[{"alias": "auto", "default_model": "zzz"}])
    with pytest.raises(ValidationError, match="used twice"):
        Catalog(instance_id="x", models=[m, CatalogModel(id="b", file="/m/b.gguf", aliases=["a"])])
    with pytest.raises(ValidationError):
        CatalogModel(id="a", file="relative.gguf")


def test_unreadable_settings_go_back_to_the_previous_copy(tmp_path):
    store = Store(tmp_path)
    store.save_settings(store.settings())
    store.save_settings(store.settings().model_copy(update={"network": "kept"}))   # leaves settings.yaml.prev
    store.settings_path.write_text("network: [oops\n")
    note = store.recover_settings()
    assert "went back" in note and "settings.yaml.broken" in note
    assert store.settings().network == "gpushepherd"          # the copy from before the last save
    assert (tmp_path / "settings.yaml.broken").read_text() == "network: [oops\n"
    assert store.recover_settings() == ""


def test_unreadable_settings_without_a_copy_say_what_to_do(tmp_path):
    store = Store(tmp_path)
    store.settings_path.write_text("network: [oops\n")
    with pytest.raises(ConfigError, match="restore-settings"):
        store.recover_settings()
