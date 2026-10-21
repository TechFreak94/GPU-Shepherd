"""Checks the Docker API requests DockerBackend builds, without a Docker daemon: the real
docker-py APIClient builds the payloads, and the network calls are intercepted."""
import docker
from docker import errors as derr

from gpushepherd.backend import LABEL_CONFIG, LABEL_GPUS, LABEL_INSTANCE, ContainerSpec, DockerBackend


def make_backend():
    b = DockerBackend.__new__(DockerBackend)
    b._errors = derr
    b.api = docker.APIClient(base_url="unix:///nonexistent.sock", version="1.44")
    b.calls = []
    b.api.create_container = lambda image, **kw: b.calls.append(("create", image, kw)) or {"Id": "abc"}
    b.api.start = lambda cid: b.calls.append(("start", cid))
    b.api.remove_container = lambda name, force: b.calls.append(("remove", name))
    return b


def spec(**kw):
    base = dict(name="shepherd-llm", image="ghcr.io/mostlygeek/llama-swap:cuda", instance_id="llm", network="gpushepherd",
                alias="llm", command=["--x"], env={"A": "1"}, ports=[("0.0.0.0", 8080, 8080, "tcp")],
                volumes=[("/srv/models", "/models", True), ("/opt/gs/generated/llm/llama-swap.yaml", "/app/config.yaml", True)],
                gpus=[0, 1], config_hash="cafe", shm_size="1g")
    base.update(kw)
    return ContainerSpec(**base)


def test_create_payload():
    b = make_backend()
    b.create_and_start(spec())
    assert [c[0] for c in b.calls] == ["remove", "create", "start"]
    _, image, kw = b.calls[1]
    hc = kw["host_config"]
    assert image == "ghcr.io/mostlygeek/llama-swap:cuda" and kw["name"] == "shepherd-llm"
    assert hc["DeviceRequests"] == [{"Driver": "", "Count": 0, "DeviceIDs": ["0", "1"], "Capabilities": [["gpu"]],
                                     "Options": {}}]
    assert hc["PortBindings"] == {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}]}
    assert "/srv/models:/models:ro" in hc["Binds"]
    assert hc["RestartPolicy"] == {"Name": "unless-stopped"} and hc["Init"] is True
    assert kw["labels"] == {LABEL_INSTANCE: "llm", LABEL_GPUS: "0,1", LABEL_CONFIG: "cafe"}
    assert kw["networking_config"]["EndpointsConfig"]["gpushepherd"]["Aliases"] == ["llm"]
    assert kw["environment"] == {"A": "1"} and kw["command"] == ["--x"]


def test_cpu_only_container_gets_no_gpu_request():
    b = make_backend()
    b.create_and_start(spec(gpus=[], name="shepherd-hermes", instance_id="hermes", alias="hermes"))
    hc = b.calls[1][2]["host_config"]
    assert not hc.get("DeviceRequests")
