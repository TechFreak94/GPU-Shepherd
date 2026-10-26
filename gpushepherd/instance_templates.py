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
class ModelSlot:
    """A model the install form offers to add to a model server's catalogue, under a fixed name
    that clients ask for. Its answers are <key>_file and <key>_ctx."""
    key: str
    label: str
    alias: str
    ctx_size: int
    sampling: str = ""
    kv_cache: str = "q8_0"
    hint: str = ""


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
    models: tuple[ModelSlot, ...] = ()        # model servers: the models picked while installing
    connects: str = ""                         # how it is told about a model server: "openwebui" or "hermes"
    command: tuple[str, ...] = ()
    login: tuple[str, str] | None = None       # (user env var, password env var) of its own login
    data_owner: tuple[int, int] | None = None  # uid:gid the app runs as, for the folders made for it
    needs: str = ""
    docs: str = ""
    badge: str = ""                            # a few characters for its App Store icon


TEMPLATES: tuple[Template, ...] = (
    Template(
        key="llm-server", badge="LLM", title="LLM server", group="Model servers",
        blurb="An OpenAI-compatible model server that loads whichever model a client asks for: chat, coding or both.",
        points=("llama-swap + llama.cpp", "Uses every free GPU and gives them back when something else needs one",
                "Pick a chat model and a coding model now, or add models later"),
        default_id="llm", default_name="LLM server", description="llama-swap model server",
        kind="llama-swap", image=LLAMA_SWAP_IMAGE, container_port=8080, host_port=8080, gpu="all",
        open_path="/ui", models_mount=("", MODELS_IN_CONTAINER, True),
        models=(ModelSlot(key="chat", label="Chat model", alias="chat", ctx_size=32768,
                          sampling="--temp 0.7 --top-p 0.95",
                          hint="For chat apps such as Open WebUI and Hermes Agent."),
                ModelSlot(key="coder", label="Coding model", alias="coder", ctx_size=65536,
                          sampling="--temp 0.3 --top-p 0.95",
                          hint="For coding agents and editor plugins (Aider, Continue, OpenCode): long context, steadier sampling.")),
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
        needs="An LLM server to chat with", docs="https://docs.openwebui.com"),
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

# the model server an app with a connection can install alongside itself, and the model slot it fills
BUNDLED_SERVER = "llm-server"
BUNDLED_SLOT = "chat"
# App Store keys from before there was one LLM server app
RENAMED = {"llm-chat": "llm-server", "llm-coder": "llm-server"}
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
    if inst.kind == "llama-swap" or t.kind == "llama-swap":
        return inst.kind == t.kind                 # every model server is an LLM server
    if inst.app:
        return RENAMED.get(inst.app, inst.app) == t.key
    return inst.image.rsplit(":", 1)[0] == t.image.rsplit(":", 1)[0]


def installed(instances: dict[str, Instance]) -> dict[str, list[Instance]]:
    """Template key -> the instances installed from it."""
    return {t.key: [i for i in instances.values() if matches(t, i)] for t in TEMPLATES}


def get(key: str) -> Template:
    key = RENAMED.get(key, key)
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
            "server": next(iter(m.servers), NEW_SERVER if t.connects else ""), "comfyui": comfy,
            **{f"{s.key}_file": "" for s in slots(t)}, **{f"{s.key}_ctx": s.ctx_size for s in slots(t)},
            "offload": "auto", "autostart": False,
            "login_user": "admin" if t.login else "", "password": secrets.token_urlsafe(12) if t.login else "",
            "hass_url": "", "hass_token": ""}


def answer_keys(t: Template) -> list[str]:
    """The install answers that matter for an app (defaults() returns every possible one)."""
    keys = ["name", "id", "port", "gpu"]
    if t.models_mount or slots(t):
        keys.append("models_dir")
    if t.appdata:
        keys.append("appdata_dir")
    if t.connects:
        keys.append("server")
    if t.connects == "openwebui":
        keys.append("comfyui")
    for s in slots(t):
        keys += [f"{s.key}_file", f"{s.key}_ctx"]
    if slots(t):
        keys.append("offload")
    if t.login:
        keys += ["login_user", "password"]
    if t.connects == "hermes":
        keys += ["hass_url", "hass_token"]
    return keys + ["autostart"]


def build(t: Template, a: dict, m: Machine) -> tuple[Instance, list[CatalogModel]]:
    """Turn the guided form's answers into an instance and, for model servers, the models picked."""
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
    return inst, _models(t, a, m)


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
    d |= {f"{BUNDLED_SLOT}_file": str(a.get(f"{BUNDLED_SLOT}_file", "")), "gpu": t.gpu}
    srv, models = build(t, d, m)
    cat = Catalog(instance_id=srv.id, models=models)
    m.instances[srv.id] = srv
    m.servers[srv.id] = {"api_key": cat.api_key, "name": srv.name, "default_model": BUNDLED_SLOT if models else ""}
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


def slots(t: Template) -> tuple[ModelSlot, ...]:
    """The model slots an app's form shows: a model server's own, or the one its bundled server fills."""
    if t.models:
        return t.models
    if t.connects:
        return tuple(x for x in get(BUNDLED_SERVER).models if x.key == BUNDLED_SLOT)
    return ()


def _models(t: Template, a: dict, m: Machine) -> list[CatalogModel]:
    """The catalogue entries for the model files picked in the form. The same file picked for two
    slots becomes one model that answers to both names."""
    choice = str(a.get("offload") or "auto")
    out: list[CatalogModel] = []
    for slot in t.models:
        file = str(a.get(f"{slot.key}_file", "")).strip()
        if not file:
            continue
        if not file.startswith("/"):
            file = f"{MODELS_IN_CONTAINER}/{file}"
        ctx = str(a.get(f"{slot.key}_ctx", "")).strip()
        if ctx and not ctx.isdigit():
            raise ConfigError(f"{slot.label}: enter the context size as a whole number of tokens")
        ctx_size = int(ctx or slot.ctx_size)
        same = next((x for x in out if x.file == file), None)
        if same:
            same.aliases.append(slot.alias)
            same.ctx_size = max(same.ctx_size, ctx_size)
            continue
        if choice == "moe":
            offload = [OffloadRule(min_gpus=1, cpu_moe=999)]
        elif choice == "auto":
            offload = auto_offload(m.model_sizes.get(file), m.gpu_mem_gb)
        else:
            offload = []
        mid = model_id_from_file(file)
        if not MODEL_ID.match(mid):
            mid = "model"
        mid = free_id(mid, {x.id for x in out})
        out.append(CatalogModel(id=mid, name=Path(file).name.removesuffix(".gguf"), file=file, ctx_size=ctx_size,
                                kv_cache=slot.kv_cache, sampling=slot.sampling, offload=offload,
                                aliases=[slot.alias] if slot.alias != mid else []))
    for x in out:
        x.aliases = [al for al in x.aliases if al != x.id]
    return out
