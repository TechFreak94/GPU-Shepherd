"""Checks the Docker API requests DockerBackend builds, without a Docker daemon: the real
docker-py APIClient builds the payloads, and the network calls are intercepted."""
import docker
from docker import errors as derr

from gpushepherd.backend import LABEL_CONFIG, LABEL_GPUS, LABEL_INSTANCE, ContainerSpec, DockerBackend, bind_spec


def make_backend():
    b = DockerBackend.__new__(DockerBackend)
    b._errors = derr
    b.api = docker.APIClient(base_url="unix:///nonexistent.sock", version="1.44")
    b._selinux = False
    b._log_config = {"Type": "json-file", "Config": {"max-size": "10m", "max-file": "3"}}
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
    assert hc["LogConfig"] == {"Type": "json-file", "Config": {"max-size": "10m", "max-file": "3"}}


def test_log_limit_only_for_the_json_file_driver():
    b = make_backend()
    del b._log_config
    b.client = type("C", (), {"info": lambda self: {"LoggingDriver": "journald"}})()
    assert b.log_config is None
    b.create_and_start(spec())
    assert not b.calls[1][2]["host_config"].get("LogConfig")


def test_cpu_only_container_gets_no_gpu_request():
    b = make_backend()
    b.create_and_start(spec(gpus=[], name="shepherd-hermes", instance_id="hermes", alias="hermes"))
    hc = b.calls[1][2]["host_config"]
    assert not hc.get("DeviceRequests")


def test_stats_cpu_from_two_samples_and_memory_without_cache():
    b = make_backend()
    b._cpu_prev = {}
    samples = iter([(1_000, 100_000), (6_000, 200_000)])

    def stats(name, stream, one_shot):
        assert stream is False and one_shot is True
        total, system = next(samples)
        return {"cpu_stats": {"cpu_usage": {"total_usage": total}, "system_cpu_usage": system},
                "memory_stats": {"usage": 3 * 2**30, "limit": 64 * 2**30, "stats": {"inactive_file": 2**30}}}
    b.api.stats = stats
    first = b.stats(["shepherd-llm"])["shepherd-llm"]
    assert first.cpu_pct is None and first.mem_used == 2 * 2**30 and first.mem_limit == 64 * 2**30
    assert b.stats(["shepherd-llm"])["shepherd-llm"].cpu_pct == 5.0     # 5,000 of 100,000 jiffies


def test_stats_skip_containers_that_fail():
    b = make_backend()
    b._cpu_prev = {}
    b.api.stats = lambda *a, **kw: (_ for _ in ()).throw(derr.NotFound("gone"))
    assert b.stats(["shepherd-x"]) == {} and b.stats([]) == {}


def test_starts_without_docker_and_retries_on_use(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///nonexistent.sock")
    b = DockerBackend()                        # no exception: GPU Shepherd still starts
    assert b.ping().startswith("Docker not reachable")


def test_selinux_binds_get_z_except_system_folders():
    assert bind_spec("/srv/models", "/models", True) == "/srv/models:/models:ro"
    assert bind_spec("/srv/models", "/models", True, selinux=True) == "/srv/models:/models:ro,z"
    assert bind_spec("/opt/gs/appdata/x", "/data", False, selinux=True) == "/opt/gs/appdata/x:/data:rw,z"
    for system in ("/", "/home", "/home/", "/usr", "/etc", "/var/run/docker.sock", "/run/user/1000", "/dev/dri"):
        assert ",z" not in bind_spec(system, "/x", False, selinux=True), system


def test_selinux_docker_relabels_instance_mounts():
    b = make_backend()
    b._selinux = True
    b.create_and_start(spec())
    hc = b.calls[1][2]["host_config"]
    assert "/srv/models:/models:ro,z" in hc["Binds"]


def test_selinux_detected_from_docker_info():
    b = DockerBackend.__new__(DockerBackend)
    b.client = type("C", (), {"info": lambda self: {"SecurityOptions": ["name=seccomp,profile=builtin",
                                                                       "name=selinux"]}})()
    assert b.selinux is True
    b2 = DockerBackend.__new__(DockerBackend)
    b2.client = type("C", (), {"info": lambda self: {"SecurityOptions": ["name=seccomp,profile=builtin"]}})()
    assert b2.selinux is False
