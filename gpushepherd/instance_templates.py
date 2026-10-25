"""The App Store's apps: built-in templates for installing an app as an instance.

A template is a generic starting point (a chat or coding model server, ComfyUI, Open WebUI,
Hermes Agent) that asks a few questions and fills in everything else. What it creates is an
ordinary instance (and, for model servers, a first catalogue entry), so it can be edited field by
field afterwards. Every app can also be installed by hand: the full form, filled in from the
same defaults.

This module is pure: the app passes in what it knows about the machine (GPUs, existing
instances, folders) and gets pydantic objects back.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from .config import (Catalog, CatalogModel, ConfigError, GpuPolicy, Instance, OffloadRule, PortMap, VolumeMap,
                     MODEL_ID, SLUG)

LLAMA_SWAP_IMAGE = "ghcr.io/mostlygeek/llama-swap:cuda"
MODELS_IN_CONTAINER = "/models"

# how much bigger than the .gguf file a model is once loaded (KV cache, buffers), for "auto" offload
LOAD_OVERHEAD = 1.2


@dataclass(frozen=True)
class FirstModel:
    """Defaults for the model a model-server template adds to its catalogue."""
    alias: str
    ctx_size: int
    kv_cache: str = "q8_0"
    sampling: str = ""


@dataclass(frozen=True)
class Template:
    key: str
    title: str
    group: str
    blurb: str
    points: tuple[str, ...]
    default_id: str
    default_name: str
    description: str
    image: str
    container_port: int
    host_port: int
    gpu: str                                   # "one", "all" or "none" (see GPU_CHOICES)
    kind: str = "container"
    priority: int = 50
    open_path: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    appdata: tuple[tuple[str, str], ...] = ()  # (subfolder of the app data folder, path inside)
    models_mount: tuple[str, str, bool] | None = None   # (subfolder of the models folder, path inside, read-only)
    first_model: FirstModel | None = None
    connects: str = ""                         # how it is told about a model server: "openwebui" or "hermes"
    command: tuple[str, ...] = ()
    login: tuple[str, str] | None = None       # (user env var, password env var) of its own login
    data_owner: tuple[int, int] | None = None  # uid:gid the app runs as, for the folders made for it
    needs: str = ""
    docs: str = ""
    badge: str = ""                            # a few characters for its App Store icon


TEMPLATES: tuple[Template, ...] = (
    Template(
        key="llm-chat", badge="LLM", title="Chat model server", group="Model servers",
        blurb="An OpenAI-compatible endpoint that loads whichever model a client asks for.",
        points=("llama-swap + llama.cpp", "Uses every free GPU and gives them back when something else needs one",
                "Pick a .gguf file now or add models later"),
        default_id="llm", default_name="LLM server", description="llama-swap model server (chat)",
        kind="llama-swap", image=LLAMA_SWAP_IMAGE, container_port=8080, host_port=8080, gpu="all",
        open_path="/ui", models_mount=("", MODELS_IN_CONTAINER, True),
        first_model=FirstModel(alias="chat", ctx_size=32768, sampling="--temp 0.7 --top-p 0.95"),
        needs="An NVIDIA GPU; models larger than its memory can use system RAM",
        docs="https://github.com/mostlygeek/llama-swap"),
    Template(
        key="llm-coder", badge="</>", title="Coding model server", group="Model servers",
        blurb="A model server tuned for coding agents and editor plugins (Aider, Continue, OpenCode).",
        points=("llama-swap + llama.cpp", "Long context (64K) and steadier sampling for code",
                "Answers to the model name \"coder\""),
        default_id="coder", default_name="Coding LLM", description="llama-swap model server (coding)",
        kind="llama-swap", image=LLAMA_SWAP_IMAGE, container_port=8080, host_port=8081, gpu="all",
        open_path="/ui", models_mount=("", MODELS_IN_CONTAINER, True),
        first_model=FirstModel(alias="coder", ctx_size=65536, sampling="--temp 0.3 --top-p 0.95"),
        needs="An NVIDIA GPU; models larger than its memory can use system RAM",
        docs="https://github.com/mostlygeek/llama-swap"),
    Template(
        key="open-webui", badge="OW", title="Open WebUI", group="Chat and agents",
        blurb="A ChatGPT-style web chat for your model servers, with image generation through ComfyUI.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Optional: make images with ComfyUI"),
        default_id="webui", default_name="Chat UI", description="Open WebUI",
        image="ghcr.io/open-webui/open-webui:main", container_port=8080, host_port=3000, gpu="none",
        env={"ENABLE_OLLAMA_API": "false"},
        appdata=(("data", "/app/backend/data"),), connects="openwebui",
        needs="Nothing special", docs="https://docs.openwebui.com"),
    Template(
        key="hermes", badge="H", title="Hermes Agent", group="Chat and agents",
        blurb="An AI agent by Nous Research with memory, skills and scheduled tasks, thinking with your own model server.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Web dashboard with its own login; optional Home Assistant connection"),
        default_id="hermes", default_name="Hermes Agent", description="Hermes Agent",
        image="nousresearch/hermes-agent:latest", command=("gateway", "run"),
        container_port=9119, host_port=9119, gpu="none",
        env={"HERMES_DASHBOARD": "1"}, login=("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"),
        appdata=(("data", "/opt/data"),), data_owner=(10000, 10000), connects="hermes",
        needs="A model server whose model can call tools (Qwen3 models can)",
        docs="https://hermes-agent.nousresearch.com/docs/user-guide/docker"),
    Template(
        key="comfyui", badge="CUI", title="ComfyUI", group="Image and video",
        blurb="Node-based image and video generation with a web UI.",
        points=("Takes one GPU, whichever is free", "Outranks model servers when GPUs run short",
                "Models in the models folder's comfyui/ subfolder"),
        default_id="comfyui", default_name="ComfyUI", description="Image and video generation",
        image="yanwk/comfyui-boot:cu130-slim-v2", container_port=8188, host_port=8188, gpu="one", priority=60,
        env={"CLI_ARGS": ""},
        appdata=(("custom_nodes", "/root/ComfyUI/custom_nodes"), ("input", "/root/ComfyUI/input"),
                 ("output", "/root/ComfyUI/output"), ("user", "/root/ComfyUI/user"), ("cache", "/root/.cache")),
        models_mount=("comfyui", "/root/ComfyUI/models", False),
        needs="An NVIDIA GPU with 8+ GB and driver 580 or newer (CUDA 13 image)",
        docs="https://github.com/YanWenKun/ComfyUI-Docker"),
)

# the model server an app with a connection can install alongside itself
BUNDLED_SERVER = "llm-chat"
NEW_SERVER = "new"

GPU_CHOICES = {
    "one": "One GPU, whichever is free",
    "all": "Every free GPU (splits a model across them), at least one",
    "none": "No GPU",
}

OFFLOAD_CHOICES = {
    "auto": "Decide from the file size and your GPUs",
    "gpu": "Keep the whole model on the GPUs",
    "moe": "Keep MoE expert layers in system RAM (for models larger than your GPUs)",
}


def matches(t: Template, inst: Instance) -> bool:
    """Whether an instance was installed from this template (older ones by their image)."""
    if inst.app:
        return inst.app == t.key
    if t.kind == "llama-swap" or inst.kind == "llama-swap":
        if inst.kind != t.kind:
            return False
        known = {x.description: x.key for x in TEMPLATES if x.kind == "llama-swap"}
        return known.get(inst.description, BUNDLED_SERVER) == t.key     # other model servers count as chat
    return inst.image.rsplit(":", 1)[0] == t.image.rsplit(":", 1)[0]


def installed(instances: dict[str, Instance]) -> dict[str, list[Instance]]:
    """Template key -> the instances installed from it."""
    return {t.key: [i for i in instances.values() if matches(t, i)] for t in TEMPLATES}


def get(key: str) -> Template:
    for t in TEMPLATES:
        if t.key == key:
            return t
    raise ConfigError(f"no template '{key}'")


def groups() -> list[tuple[str, list[Template]]]:
    out: dict[str, list[Template]] = {}
    for t in TEMPLATES:
        out.setdefault(t.group, []).append(t)
    return list(out.items())


# --------------------------------------------------------------------------- suggestions

def free_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def free_port(preferred: int, taken: set[int]) -> int:
    port = preferred
    while port in taken and port < 65535:
        port += 1
    return port


def model_id_from_file(path: str) -> str:
    stem = Path(path).name.removesuffix(".gguf").lower()
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)            # split models: first part's name
    stem = re.sub(r"[^a-z0-9._:-]+", "-", stem).strip("-._:")
    return stem[:80] or "model"


def auto_offload(size_gb: float | None, gpu_mem_gb: list[float]) -> list[OffloadRule]:
    """Offload rules for a model of size_gb on a server that can have 1..len(gpu_mem_gb) GPUs.

    For each GPU count, the model stays on the GPUs if it fits in the smallest GPUs' combined memory;
    otherwise its MoE expert layers go to system RAM. Expert layers are where most of a MoE model's
    weight is, and llama.cpp keeps the rest on the GPUs, which is what makes large MoE models usable.
    Dense models ignore the setting; they need a layer count, which is easier to tune on the Models page.
    """
    if not size_gb or not gpu_mem_gb:
        return []
    need = size_gb * LOAD_OVERHEAD
    smallest = sorted(gpu_mem_gb)
    rules, prev = [], None
    for n in range(1, len(smallest) + 1):
        fits = sum(smallest[:n]) >= need
        moe = 0 if fits else 999
        if moe != prev:
            rules.append(OffloadRule(min_gpus=n, cpu_moe=moe))
            prev = moe
    if len(rules) == 1 and rules[0].cpu_moe == 0:
        return []                                  # fits everywhere: nothing to say
    return rules


# --------------------------------------------------------------------------- building

@dataclass
class Machine:
    """What the app knows when building an instance from a template."""
    gpu_mem_gb: list[float]
    instances: dict[str, Instance]
    servers: dict[str, dict]           # llama-swap instance id -> {"api_key", "default_model"}
    model_sizes: dict[str, float]      # container path of a .gguf file -> size in GB


def defaults(t: Template, m: Machine, models_dir: str, appdata_root: str) -> dict:
    """The guided form's starting values."""
    taken_ports = {p.host for i in m.instances.values() for p in i.ports}
    iid = free_id(t.default_id, set(m.instances))
    name = t.default_name if iid == t.default_id else f"{t.default_name} {iid.rsplit('-', 1)[-1]}"
    comfy = next((i.id for i in m.instances.values() if "comfyui" in (i.image + i.id).lower()), "")
    return {"name": name, "id": iid, "port": free_port(t.host_port, taken_ports), "gpu": t.gpu,
            "models_dir": models_dir, "appdata_dir": str(Path(appdata_root) / iid),
            "server": next(iter(m.servers), NEW_SERVER if t.connects else ""), "comfyui": comfy, "file": "",
            "ctx_size": t.first_model.ctx_size if t.first_model else 0, "offload": "auto", "autostart": False,
            "login_user": "admin" if t.login else "", "password": secrets.token_urlsafe(12) if t.login else "",
            "hass_url": "", "hass_token": ""}


def build(t: Template, a: dict, m: Machine) -> tuple[Instance, CatalogModel | None]:
    """Turn the guided form's answers into an instance and, for model servers, its first model."""
    iid = str(a.get("id", "")).strip()
    if not SLUG.match(iid):
        raise ConfigError("ID: use lowercase letters, digits and dashes (max 40 characters)")
    if iid in m.instances:
        raise ConfigError(f"an instance called '{iid}' already exists")
    port = str(a.get("port", "")).strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ConfigError("Port: enter a number from 1 to 65535")
    gpu = _gpu_policy(str(a.get("gpu") or t.gpu), t.priority)

    volumes = []
    if t.models_mount:
        models_dir = str(a.get("models_dir", "")).strip().rstrip("/")
        if not models_dir.startswith("/"):
            raise ConfigError("Models folder: enter an absolute path, e.g. /srv/models")
        sub, inside, ro = t.models_mount
        volumes.append(VolumeMap(host=f"{models_dir}/{sub}" if sub else models_dir, container=inside, read_only=ro))
    if t.appdata:
        appdata = str(a.get("appdata_dir", "")).strip().rstrip("/")
        if not appdata.startswith("/"):
            raise ConfigError("App data folder: enter an absolute path")
        volumes += [VolumeMap(host=f"{appdata}/{sub}", container=inside) for sub, inside in t.appdata]

    env = dict(t.env)
    if t.login:
        user, password = str(a.get("login_user", "")).strip(), str(a.get("password", "")).strip()
        if not user or len(password) < 8:
            raise ConfigError("Login: enter a user name and a password of at least 8 characters")
        env |= {t.login[0]: user, t.login[1]: password}
    if t.connects == "openwebui":
        env.update(_openwebui(a, m))
    elif t.connects == "hermes":
        env.update(_hermes(a, m))

    inst = Instance(id=iid, name=str(a.get("name", "")).strip() or iid, description=t.description, kind=t.kind,
                    image=t.image, command=list(t.command), env=env,
                    ports=[PortMap(host=int(port), container=t.container_port)], volumes=volumes, gpu=gpu,
                    autostart=bool(a.get("autostart")), open_path=t.open_path, app=t.key)
    return inst, _first_model(t, a, m)


def bundled_server(a: dict, m: Machine, appdata_root: str) -> tuple[Instance, Catalog] | None:
    """The chat model server an app installs alongside itself when asked to (server = "new"),
    built from the answers' models folder and model file, with its catalogue (and API key). Adds it
    to the machine, so the app's own build connects to it."""
    if str(a.get("server", "")) != NEW_SERVER:
        return None
    t = get(BUNDLED_SERVER)
    d = defaults(t, m, str(a.get("models_dir", "")), appdata_root)
    if str(a.get("port", "")) == str(d["port"]):
        taken = {p.host for i in m.instances.values() for p in i.ports} | {d["port"]}
        d["port"] = free_port(d["port"], taken)
    if str(a.get("id", "")) == d["id"]:
        d["id"] = free_id(d["id"], set(m.instances) | {d["id"]})
    d |= {"file": str(a.get("file", "")), "gpu": t.gpu}
    srv, model = build(t, d, m)
    cat = Catalog(instance_id=srv.id, models=[model] if model else [])
    m.instances[srv.id] = srv
    m.servers[srv.id] = {"api_key": cat.api_key, "name": srv.name,
                         "default_model": t.first_model.alias if model else ""}
    return srv, cat


def seed_files(t: Template, a: dict, m: Machine) -> dict[str, str]:
    """Files to write into a new app's data folder (subfolder/path -> text), if they don't exist yet."""
    server = str(a.get("server", "")).strip()
    if t.connects != "hermes" or not server or server not in m.servers:
        return {}
    info = m.servers[server]
    lines = ["# Written by GPU Shepherd when Hermes Agent was installed. Hermes owns this file now:",
             "# change the model with `hermes model` or in its dashboard.",
             "model:", "  provider: custom", f"  base_url: {_server_url(server, m)}",
             f"  api_key: {info['api_key']}"]
    if info.get("default_model"):
        lines.append(f"  default: {info['default_model']}")
    return {"data/config.yaml": "\n".join(lines) + "\n"}


def _gpu_policy(choice: str, priority: int) -> GpuPolicy:
    if choice == "none":
        return GpuPolicy(mode="none", priority=priority)
    if choice == "one":
        return GpuPolicy(mode="elastic", min_gpus=1, max_gpus=1, priority=priority)
    if choice == "all":
        return GpuPolicy(mode="elastic", min_gpus=1, max_gpus=0, priority=priority)
    if choice.startswith("pin:") and choice[4:].isdigit():
        return GpuPolicy(mode="fixed", gpus=[int(choice[4:])], priority=priority)
    raise ConfigError(f"unknown GPU choice '{choice}'")


def _server(a: dict, m: Machine) -> str:
    server = str(a.get("server", "")).strip()
    if server == NEW_SERVER:
        raise ConfigError("the new model server has not been set up yet")
    if server and (server not in m.servers or server not in m.instances):
        raise ConfigError(f"no model server '{server}'")
    return server


def _server_url(server: str, m: Machine) -> str:
    ports = m.instances[server].ports
    return f"http://{server}:{ports[0].container if ports else 8080}/v1"


def _hermes(a: dict, m: Machine) -> dict[str, str]:
    env: dict[str, str] = {}
    server = _server(a, m)
    if server:
        info = m.servers[server]
        env |= {"OPENAI_BASE_URL": _server_url(server, m), "OPENAI_API_KEY": info["api_key"]}
        if info.get("default_model"):
            env["HERMES_MODEL"] = info["default_model"]
    hass_url, hass_token = str(a.get("hass_url", "")).strip(), str(a.get("hass_token", "")).strip()
    if hass_url and not re.match(r"^https?://\S+$", hass_url):
        raise ConfigError("Home Assistant address: enter a URL like http://homeassistant.local:8123")
    if hass_token:
        env |= {"HASS_TOKEN": hass_token, "HASS_URL": hass_url or "http://homeassistant.local:8123"}
    elif hass_url:
        raise ConfigError("Home Assistant: also paste a long-lived access token, or clear the address")
    return env


def _openwebui(a: dict, m: Machine) -> dict[str, str]:
    env: dict[str, str] = {}
    server = _server(a, m)
    if server:
        info = m.servers[server]
        env |= {"OPENAI_API_BASE_URLS": _server_url(server, m), "OPENAI_API_KEYS": info["api_key"]}
        if info.get("default_model"):
            env["DEFAULT_MODELS"] = info["default_model"]
    comfy = str(a.get("comfyui", "")).strip()
    if comfy:
        if comfy not in m.instances:
            raise ConfigError(f"no instance '{comfy}'")
        ports = m.instances[comfy].ports
        port = ports[0].container if ports else 8188
        env |= {"ENABLE_IMAGE_GENERATION": "true", "IMAGE_GENERATION_ENGINE": "comfyui",
                "COMFYUI_BASE_URL": f"http://{comfy}:{port}"}
    return env


def _first_model(t: Template, a: dict, m: Machine) -> CatalogModel | None:
    file = str(a.get("file", "")).strip()
    if not t.first_model or not file:
        return None
    if not file.startswith("/"):
        file = f"{MODELS_IN_CONTAINER}/{file}"
    ctx = str(a.get("ctx_size", "")).strip()
    if ctx and not ctx.isdigit():
        raise ConfigError("Context size: enter a whole number of tokens")
    choice = str(a.get("offload") or "auto")
    if choice == "moe":
        offload = [OffloadRule(min_gpus=1, cpu_moe=999)]
    elif choice == "auto":
        offload = auto_offload(m.model_sizes.get(file), m.gpu_mem_gb)
    else:
        offload = []
    mid = model_id_from_file(file)
    if not MODEL_ID.match(mid):
        mid = "model"
    fm = t.first_model
    return CatalogModel(id=mid, name=Path(file).name.removesuffix(".gguf"), file=file,
                        ctx_size=int(ctx or fm.ctx_size), kv_cache=fm.kv_cache, sampling=fm.sampling,
                        aliases=[fm.alias] if fm.alias != mid else [], offload=offload)
