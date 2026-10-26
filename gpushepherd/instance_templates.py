"""The App Store's apps: built-in templates for installing an app as an instance.

A template is a generic starting point (an LLM server, ComfyUI, Open WebUI, LibreChat, Hermes
Agent...) that asks a few questions and fills in everything else. What it creates is an ordinary
instance (and, for model servers, the models picked), so it can be edited field by field
afterwards. Every app can also be installed by hand: the full form, filled in from the same
defaults.

An app that needs a helper container (LibreChat's database) lists it in `companions`: a hidden
template installed alongside as its own instance, which the app `requires`, so starting the app
starts it too.

This module is pure: the app passes in what it knows about the machine (GPUs, existing
instances, folders) and gets pydantic objects back.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from . import model_library
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
    role: str = "chat"                 # embed: answers /v1/embeddings, kept loaded beside the others


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
    files: tuple[tuple[str, str], ...] = ()    # (file in the app data folder, path inside): settings files
    models_mount: tuple[str, str, bool] | None = None   # (subfolder of the models folder, path inside, read-only)
    models: tuple[ModelSlot, ...] = ()        # model servers: the models picked while installing
    connects: str = ""                         # how it is told about a model server (see CONNECTS)
    command: tuple[str, ...] = ()
    login: tuple[str, str] | None = None       # (user env var, password env var) of its own login
    data_owner: tuple[int, int] | None = None  # uid:gid the app runs as, for the folders made for it
    needs: str = ""
    docs: str = ""
    badge: str = ""                            # a few characters for its App Store icon
    tag: str = ""                              # what kind of instance it is, on its dashboard card
    companions: tuple[str, ...] = ()           # hidden templates installed with it, which it requires
    hidden: bool = False                       # a companion: not offered in the App Store itself
    after_install: str = ""                    # what is left to do in the app itself once it runs


TEMPLATES: tuple[Template, ...] = (
    Template(
        key="llm-server", tag="Model Server", badge="LLM", title="LLM server", group="Model servers",
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
                          hint="For coding agents and editor plugins (Aider, Continue, OpenCode): long context, steadier sampling."),
                ModelSlot(key="embed", label="Embeddings model", alias="embed", ctx_size=8192, kv_cache="f16",
                          role="embed",
                          hint="For chatting with your documents (RAG) in Open WebUI, AnythingLLM and LibreChat. "
                               "It stays loaded beside the other models, on the CPU.")),
        needs="An NVIDIA GPU; models larger than its memory can use system RAM",
        docs="https://github.com/mostlygeek/llama-swap"),
    Template(
        key="open-webui", tag="Chat UI", badge="OW", title="Open WebUI", group="Chat and agents",
        blurb="A ChatGPT-style web chat for your model servers, with image generation through ComfyUI.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Optional: make images with ComfyUI"),
        default_id="webui", default_name="Open WebUI", description="Chat with your model servers",
        image="ghcr.io/open-webui/open-webui:main", container_port=8080, host_port=3000, gpu="none",
        env={"ENABLE_OLLAMA_API": "false"},
        appdata=(("data", "/app/backend/data"),), connects="openwebui",
        needs="An LLM server to chat with", docs="https://docs.openwebui.com"),
    Template(
        key="hermes", tag="Agent UI", badge="H", title="Hermes Agent", group="Chat and agents",
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
        key="librechat", tag="Chat UI", badge="LC", title="LibreChat", group="Chat and agents",
        blurb="A multi-user ChatGPT-style chat with agents, presets and conversation search, using your own model server.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Installs its MongoDB database alongside, which starts and stops with it"),
        default_id="librechat", default_name="LibreChat", description="LibreChat",
        image="librechat/librechat:latest", container_port=3080, host_port=3080, gpu="none",
        env={"HOST": "0.0.0.0", "PORT": "3080", "ALLOW_REGISTRATION": "true", "ALLOW_EMAIL_LOGIN": "true",
             "ALLOW_SOCIAL_LOGIN": "false", "ENDPOINTS": "custom,agents", "SEARCH": "false",
             # plain http on a LAN address: secure cookies would never come back and logins would loop
             "SESSION_COOKIE_SECURE": "false", "NO_INDEX": "true"},
        appdata=(("images", "/app/client/public/images"), ("uploads", "/app/uploads"), ("logs", "/app/logs")),
        files=(("librechat.yaml", "/app/librechat.yaml"),), data_owner=(1000, 1000),
        companions=("librechat-mongodb",), connects="librechat",
        after_install="Open it, sign up and log in: the first account becomes the admin. Then turn sign-ups "
                      "off by setting ALLOW_REGISTRATION=false under Edit, unless others should make accounts.",
        needs="An LLM server to chat with", docs="https://www.librechat.ai/docs"),
    Template(
        key="anythingllm", tag="Chat UI", badge="ALL", title="AnythingLLM", group="Chat and agents",
        blurb="Chat with your documents: workspaces, built-in RAG and agents, thinking with your own model server.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Embeds documents on the CPU and keeps vectors in its own folder"),
        default_id="anythingllm", default_name="AnythingLLM", description="AnythingLLM",
        image="mintplexlabs/anythingllm:latest", container_port=3001, host_port=3001, gpu="none",
        # its web scraper's Chromium without --cap-add SYS_ADMIN, as upstream documents
        env={"STORAGE_DIR": "/app/server/storage", "DISABLE_TELEMETRY": "true",
             "ANYTHINGLLM_CHROMIUM_ARGS": "--no-sandbox,--disable-setuid-sandbox"},
        appdata=(("storage", "/app/server/storage"),), files=(("server.env", "/app/server/.env"),),
        data_owner=(1000, 1000), connects="anythingllm",
        after_install="Open it and make a workspace to chat in; the model server is already set up. "
                      "It has no login until you set a password under Settings, Security.",
        needs="An LLM server to chat with", docs="https://docs.anythingllm.com/installation-docker/overview"),
    Template(
        key="comfyui", tag="Image Server", badge="CUI", title="ComfyUI", group="Image and video",
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

    # companions: installed with an app, never on their own
    Template(
        key="librechat-mongodb", tag="Database", badge="DB", title="LibreChat database", group="Companions",
        blurb="MongoDB for LibreChat's accounts and conversations.", points=(), hidden=True,
        default_id="mongodb", default_name="database", description="MongoDB for LibreChat",
        image="mongo:8.0", command=("mongod", "--noauth", "--bind_ip_all"), container_port=27017, host_port=0,
        gpu="none", appdata=(("mongodb", "/data/db"),), docs="https://www.librechat.ai/docs/local/docker"),
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


OTHER_TAG = "App"        # an instance that isn't from the App Store


def tag_for(inst: Instance) -> str:
    """The kind tag on an instance's dashboard card: from its App Store app, else generic."""
    return next((t.tag for t in TEMPLATES if matches(t, inst)), OTHER_TAG)


def app_name(inst: Instance) -> str:
    """Its name, with the app's real name where it only repeats the kind tag.

    Open WebUI used to be installed as "Chat UI" (then "Chat UI 2"...), which said the same as
    its tag; those show as "Open WebUI" ("Open WebUI 2"). Any other name is the user's choice.
    """
    t = next((t for t in TEMPLATES if matches(t, inst)), None)
    if t is None or t.title == t.tag:
        return inst.name
    if inst.name.casefold() == t.tag.casefold():
        return t.title
    head, _, suffix = inst.name.rpartition(" ")
    if head.casefold() == t.tag.casefold() and suffix == inst.id.rsplit("-", 1)[-1]:
        return f"{t.title} {suffix}"
    return inst.name


def installed(instances: dict[str, Instance]) -> dict[str, list[Instance]]:
    """Template key -> the instances installed from it (App Store apps only, not companions)."""
    return {t.key: [i for i in instances.values() if matches(t, i)] for t in TEMPLATES if not t.hidden}


def get(key: str) -> Template:
    key = RENAMED.get(key, key)
    for t in TEMPLATES:
        if t.key == key:
            return t
    raise ConfigError(f"no template '{key}'")


def groups() -> list[tuple[str, list[Template]]]:
    out: dict[str, list[Template]] = {}
    for t in TEMPLATES:
        if not t.hidden:
            out.setdefault(t.group, []).append(t)
    return list(out.items())


def companion_id(app_id: str, c: Template) -> str:
    """The ID of an app's companion, e.g. librechat-mongodb."""
    return f"{app_id}-{c.default_id}"


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
    ram_gb: float = 0.0                # system RAM (0 = unknown)


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
        volumes += [VolumeMap(host=f"{appdata}/{sub}", container=inside) for sub, inside in t.appdata + t.files]
    requires = []
    for c in map(get, t.companions):
        cid = companion_id(iid, c)
        if not SLUG.match(cid):
            raise ConfigError(f"ID: too long for its {c.default_name} ({cid}); use a shorter one")
        if cid in m.instances:
            raise ConfigError(f"an instance called '{cid}' already exists (it would be {t.title}'s {c.default_name})")
        requires.append(cid)

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
    elif t.connects == "librechat":
        env.update(_librechat(a, m, requires))

    inst = Instance(id=iid, name=str(a.get("name", "")).strip() or iid, description=t.description, kind=t.kind,
                    image=t.image, command=list(t.command), env=env,
                    ports=[PortMap(host=int(port), container=t.container_port)], volumes=volumes, gpu=gpu,
                    autostart=bool(a.get("autostart")), open_path=t.open_path, app=t.key, requires=requires)
    return inst, _models(t, a, m)


def appdata_dir(t: Template, inst: Instance) -> str:
    """The app data folder an instance of this app was installed with ("" if it has none)."""
    for sub, inside in t.appdata + t.files:
        v = next((v for v in inst.volumes if v.container == inside), None)
        if v and v.host.endswith("/" + sub):
            return v.host.removesuffix("/" + sub)
    return ""


def companions(t: Template, inst: Instance) -> list[Instance]:
    """The helper instances an app installs with itself, for the ones it requires. They keep their
    data in the app's own data folder and use no GPU; nothing outside the instances' network
    reaches them."""
    out = []
    appdata = appdata_dir(t, inst)
    for c in map(get, t.companions):
        cid = companion_id(inst.id, c)
        if cid not in inst.requires or not appdata:
            continue
        out.append(Instance(
            id=cid, name=f"{inst.name} {c.default_name}", description=c.description, image=c.image,
            command=list(c.command), env=dict(c.env), gpu=_gpu_policy("none", c.priority), app=c.key,
            volumes=[VolumeMap(host=f"{appdata}/{sub}", container=inside) for sub, inside in c.appdata + c.files]))
    return out


def bundled_server(a: dict, m: Machine, appdata_root: str) -> tuple[Instance, Catalog] | None:
    """The LLM server an app installs alongside itself when asked to (server = "new"),
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
    m.servers[srv.id] = {"api_key": cat.api_key, "name": srv.name, "default_model": BUNDLED_SLOT if models else "",
                         "ctx": models[0].ctx_size if models else 0}
    return srv, cat


def seed_files(t: Template, a: dict, m: Machine) -> dict[str, str]:
    """Files to write into a new app's data folder (path inside it -> text), if they don't exist yet.
    Every settings file the app mounts gets one, so Docker never makes a folder in its place."""
    server = str(a.get("server", "")).strip()
    info = m.servers.get(server) if server in m.instances else None
    out: dict[str, str] = {}
    if t.connects == "hermes" and info:
        lines = ["# Written by GPU Shepherd when Hermes Agent was installed. Hermes owns this file now:",
                 "# change the model with `hermes model` or in its dashboard.",
                 "model:", "  provider: custom", f"  base_url: {_server_url(server, m)}",
                 f"  api_key: {info['api_key']}"]
        if info.get("default_model"):
            lines.append(f"  default: {info['default_model']}")
        out["data/config.yaml"] = "\n".join(lines) + "\n"
    elif t.connects == "anythingllm":
        out["server.env"] = _anythingllm_env(_server_url(server, m) if info else "", info or {})
    elif t.connects == "librechat":
        out["librechat.yaml"] = LIBRECHAT_YAML
    for name, _ in t.files:
        out.setdefault(name, "")
    return out


# LibreChat's settings file: one custom endpoint whose address and key come from the instance's
# environment (LLM_BASE_URL, LLM_API_KEY), so pointing it at another model server is an Edit away.
# The version is the config schema LibreChat's release image reads (newer ones only log a notice).
LIBRECHAT_YAML = """\
# Written by GPU Shepherd when LibreChat was installed. LibreChat reads it when it starts, and it is
# yours to change: https://www.librechat.ai/docs/configuration/librechat_yaml
# The model server's address and API key are LLM_BASE_URL and LLM_API_KEY in the instance's
# environment (Edit on the GPU Shepherd dashboard).
version: 1.3.17
cache: true
endpoints:
  custom:
    - name: "Local models"
      apiKey: "${LLM_API_KEY}"
      baseURL: "${LLM_BASE_URL}"
      models:
        default: ["chat"]
        fetch: true
      titleConvo: true
      titleModel: "current_model"
      modelDisplayLabel: "Local models"
"""


def _anythingllm_env(url: str, info: dict) -> str:
    """AnythingLLM's own settings file (/app/server/.env). It rewrites the file when settings change
    in its web page, so the connection goes here rather than into the container's environment,
    which would win over the page on every start."""
    lines = ["# Written by GPU Shepherd when AnythingLLM was installed. AnythingLLM owns this file now:",
             "# change settings in its web page.",
             "VECTOR_DB='lancedb'"]
    if url and info.get("embed"):          # the model server's embeddings model, instead of AnythingLLM's own
        lines += ["EMBEDDING_ENGINE='generic-openai'", f"EMBEDDING_BASE_PATH='{url}'",
                  f"EMBEDDING_MODEL_PREF='{info['embed']}'", f"GENERIC_OPEN_AI_EMBEDDING_API_KEY='{info['api_key']}'",
                  "EMBEDDING_MODEL_MAX_CHUNK_LENGTH='8192'"]
    else:
        lines.append("EMBEDDING_ENGINE='native'")
    if url:
        lines += ["LLM_PROVIDER='generic-openai'", f"GENERIC_OPEN_AI_BASE_PATH='{url}'",
                  f"GENERIC_OPEN_AI_API_KEY='{info['api_key']}'",
                  f"GENERIC_OPEN_AI_MODEL_PREF='{info.get('default_model') or 'chat'}'",
                  f"GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT='{info.get('ctx') or 8192}'"]
    return "\n".join(lines) + "\n"


def app_env(t: Template, server: str, m: Machine) -> dict[str, str]:
    """The environment an app installed today gets from GPU Shepherd for a model server, leaving
    out what its install form asked for (logins, Home Assistant, ComfyUI) or made up (secrets).
    App updates compare an installed instance against this."""
    env = dict(t.env)
    a = {"server": server if server in m.servers and server in m.instances else ""}
    if t.connects == "openwebui":
        env |= _openwebui(a, m)
    elif t.connects == "hermes":
        env |= _hermes(a, m)
    elif t.connects == "librechat" and a["server"]:
        env |= {"LLM_BASE_URL": _server_url(server, m), "LLM_API_KEY": m.servers[server]["api_key"]}
    return env


def connected_server(inst: Instance, m: Machine, settings: str = "") -> str:
    """The model server an app instance talks to, found by its address in the instance's environment
    or in its own settings file (AnythingLLM keeps it there)."""
    for server in m.servers:
        if server in m.instances and any(_server_url(server, m) in v for v in [*inst.env.values(), settings]):
            return server
    return ""


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


def _librechat(a: dict, m: Machine, requires: list[str]) -> dict[str, str]:
    """Its database (the companion it requires), fresh secrets, and the model server for
    librechat.yaml's endpoint."""
    env = {"MONGO_URI": f"mongodb://{requires[0]}:27017/LibreChat",
           "CREDS_KEY": secrets.token_hex(32), "CREDS_IV": secrets.token_hex(16),
           "JWT_SECRET": secrets.token_hex(32), "JWT_REFRESH_SECRET": secrets.token_hex(32)}
    server = _server(a, m)
    if server:
        env |= {"LLM_BASE_URL": _server_url(server, m), "LLM_API_KEY": m.servers[server]["api_key"]}
    return env


def _openwebui(a: dict, m: Machine) -> dict[str, str]:
    env: dict[str, str] = {}
    server = _server(a, m)
    if server:
        info = m.servers[server]
        env |= {"OPENAI_API_BASE_URLS": _server_url(server, m), "OPENAI_API_KEYS": info["api_key"]}
        if info.get("default_model"):
            env["DEFAULT_MODELS"] = info["default_model"]
        if info.get("embed"):                  # documents are embedded by the model server, not on Open WebUI's CPU
            env |= {"RAG_EMBEDDING_ENGINE": "openai", "RAG_OPENAI_API_BASE_URL": _server_url(server, m),
                    "RAG_OPENAI_API_KEY": info["api_key"], "RAG_EMBEDDING_MODEL": info["embed"]}
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
            if same.role != slot.role:
                raise ConfigError(f"{slot.label}: an embeddings model can't also chat; pick another file")
            same.aliases.append(slot.alias)
            same.ctx_size = max(same.ctx_size, ctx_size)
            continue
        known = model_library.for_file(file, MODELS_IN_CONTAINER)
        if known and slot.role == known.role:     # a suggested model: its makers' settings
            x = model_library.catalog_model(known, m.gpu_mem_gb, MODELS_IN_CONTAINER, ctx_size=ctx_size)
            if choice == "gpu":
                x.offload = [r for r in x.offload if r.gpu_layers is not None]
            x.id = free_id(x.id, {y.id for y in out})
            x.aliases = [slot.alias] if slot.alias != x.id else []
            out.append(x)
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
        if slot.role == "embed":                  # on the CPU, so it never takes GPU memory from the chat model
            offload = [OffloadRule(min_gpus=1, gpu_layers=0)]
        out.append(CatalogModel(id=mid, name=Path(file).name.removesuffix(".gguf"), file=file, ctx_size=ctx_size,
                                kv_cache=slot.kv_cache, sampling=slot.sampling, offload=offload,
                                role=slot.role, keep_loaded=slot.role == "embed",
                                aliases=[slot.alias] if slot.alias != mid else []))
    for x in out:
        x.aliases = [al for al in x.aliases if al != x.id]
    return out
