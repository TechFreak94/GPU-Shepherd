"""The App Center's apps: built-in templates for installing an app as an instance.

A template is a generic starting point (a Llama Swap server, ComfyUI, Open WebUI, LibreChat, Hermes
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

import base64
import json
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from . import model_library
from .config import (Catalog, CatalogModel, ConfigError, GpuPolicy, Instance, OffloadRule, PortMap, VolumeMap,
                     MODEL_ID, SLUG, gpu_label, gpu_vendor)

LLAMA_SWAP_IMAGE = "ghcr.io/mostlygeek/llama-swap:cuda"
LLAMA_SWAP_VULKAN_IMAGE = "ghcr.io/mostlygeek/llama-swap:vulkan"
# what a client app needs: any of the App Center's model servers, which all speak the OpenAI API
NEEDS_SERVER = "A model server to chat with (Llama Swap, vLLM or Ollama)"
# what Speaches downloads first, and Open WebUI's voice uses
STT_MODEL = "Systran/faster-distil-whisper-large-v3"
TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
TTS_VOICE = "af_heart"
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
    gpu: str                                   # "one", "all" or "none" (see gpu_choices())
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
    login: tuple[str, str] | None = None       # (user env var or "" for a password only, password env var)
    data_owner: tuple[int, int] | None = None  # uid:gid the app runs as, for the folders made for it
    needs: str = ""
    docs: str = ""
    license: str = ""                          # the app's own license, credited in the App Center
    badge: str = ""                            # a few characters for its App Center icon
    tag: str = ""                              # what kind of instance it is, on its dashboard card
    companions: tuple[str, ...] = ()           # hidden templates installed with it, which it requires
    hidden: bool = False                       # a companion: not offered in the App Center itself
    after_install: str = ""                    # what is left to do in the app itself once it runs
    experimental: str = ""                     # why it is marked Experimental in the App Center ("" = it isn't)
    secrets: tuple[str, ...] = ()              # environment variables given a fresh random value when installed
    shares_env: tuple[str, ...] = ()           # a companion: variables it takes from its app (a shared password)
    user: str | None = None                    # run as this user (uid:gid) instead of the image's own
    shm_size: str | None = None                # shared memory, for apps that split work across GPUs
    engine: str = ""                           # "vllm": a model server with its own install answers (see VLLM_*)
    bundled_slot: str = "chat"                 # the model slot a Llama Swap installed alongside it fills
    cpu_ok: bool = False                       # a model server that keeps answering on the CPU (slower)
                                               # while an image app has the only GPU, instead of stopping
    vendor: str = "nvidia"                     # the kind of GPU it runs on: nvidia or amd (GpuPolicy.vendor)
    kv_cache: str = ""                         # every model's KV cache type, whatever a suggested model's own is


TEMPLATES: tuple[Template, ...] = (
    Template(
        key="llm-server", tag="Model Server", badge="LLM", title="Llama Swap", group="Model servers",
        blurb="An OpenAI-compatible model server that loads whichever model a client asks for: chat, coding or both.",
        points=("llama-swap + llama.cpp", "Uses every free GPU and gives them back when something else needs one",
                "Pick a chat model and a coding model now, or add models later"),
        default_id="llm", default_name="Llama Swap", description="llama-swap model server",
        kind="llama-swap", image=LLAMA_SWAP_IMAGE, container_port=8080, host_port=8080, gpu="all",
        open_path="/ui", models_mount=("", MODELS_IN_CONTAINER, True), cpu_ok=True,
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
        license="MIT (llama-swap and llama.cpp)", docs="https://github.com/mostlygeek/llama-swap"),
    Template(
        # llama.cpp's Vulkan build, set up for several older 8 GB AMD cards (Polaris: RX 470/480/570/580):
        # layer split over every card (Vulkan has no row split), contexts sized for 8 GB, and an f16
        # KV cache, which needs no flash attention (cards without matrix cores run it slowly, and a
        # quantised V cache can't do without it)
        key="llm-server-vulkan", tag="Model Server", badge="VK", title="Llama Swap (Vulkan)", group="Model servers",
        blurb="The Llama Swap model server on AMD GPUs through Vulkan, tuned for several 8 GB cards such as the RX 580.",
        points=("llama-swap + llama.cpp's Vulkan build: no ROCm needed, so older Polaris cards work",
                "Uses every free AMD GPU, splitting models across them layer by layer, and gives them back "
                "when another AMD app needs one",
                "Pick a chat model and a coding model now, or add models later"),
        default_id="llm-vulkan", default_name="Llama Swap (Vulkan)", description="llama-swap model server (Vulkan)",
        kind="llama-swap", image=LLAMA_SWAP_VULKAN_IMAGE, container_port=8080, host_port=8080, gpu="all",
        vendor="amd", kv_cache="f16", open_path="/ui", models_mount=("", MODELS_IN_CONTAINER, True), cpu_ok=True,
        models=(ModelSlot(key="chat", label="Chat model", alias="chat", ctx_size=16384, kv_cache="f16",
                          sampling="--temp 0.7 --top-p 0.95",
                          hint="For chat apps such as Open WebUI and Hermes Agent. 16K tokens of context "
                               "leaves room for the model on 8 GB cards."),
                ModelSlot(key="coder", label="Coding model", alias="coder", ctx_size=32768, kv_cache="f16",
                          sampling="--temp 0.3 --top-p 0.95",
                          hint="For coding agents and editor plugins (Aider, Continue, OpenCode)."),
                ModelSlot(key="embed", label="Embeddings model", alias="embed", ctx_size=8192, kv_cache="f16",
                          role="embed",
                          hint="For chatting with your documents (RAG) in Open WebUI, AnythingLLM and LibreChat. "
                               "It stays loaded beside the other models, on the CPU.")),
        after_install="It gets only the AMD GPUs GPU Shepherd gives it (each one's /dev/dri render node), so a "
                      "built-in graphics chip is left alone. Apps here can use it like the other Llama Swap.",
        experimental="Not yet tried on real AMD GPUs.",
        needs="AMD GPUs with the amdgpu driver (Vulkan: RX 400/500 series or newer); models larger than their "
              "memory can use system RAM",
        license="MIT (llama-swap and llama.cpp)", docs="https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan"),
    Template(
        key="ollama", tag="Model Server", badge="OL", title="Ollama", group="Model servers",
        blurb="Ollama's model server: models from Ollama's library, for the chat apps here and apps that speak Ollama's API.",
        points=("Uses every free GPU, gives them back when something else needs one, and keeps answering on "
                "the CPU (slower) meanwhile",
                "Models from Ollama's library, kept in the models folder's ollama/ subfolder",
                "Chat apps here can use it like the Llama Swap, through its OpenAI-compatible API"),
        default_id="ollama", default_name="Ollama", description="Ollama model server",
        image="ollama/ollama:latest", container_port=11434, host_port=11434, gpu="all",
        # Ollama's own default context is 4K on GPUs under 24 GB, too short for agents (Hermes,
        # OpenCode) and documents: the same 32K as the Llama Swap's chat model, with its 8-bit cache
        env={"OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_FLASH_ATTENTION": "1",
             "OLLAMA_KV_CACHE_TYPE": "q8_0"},
        models_mount=("ollama", "/root/.ollama/models", False), cpu_ok=True,
        after_install="Download a model once it runs: docker exec CONTAINER ollama pull qwen3.5:9b (CONTAINER is "
                      "shown on its dashboard card), then give it the name chat, which the apps here ask for first: "
                      "docker exec CONTAINER ollama cp qwen3.5:9b chat. Every model gets OLLAMA_CONTEXT_LENGTH "
                      "tokens of context (under Edit). Apps that list the server's models (Open WebUI, "
                      "LibreChat) show every model you pull. It answers at http://this-machine:PORT, and has no login or "
                      "API key, so keep that port off the internet. Home Assistant 2026.8 or newer can use the "
                      "Llama Swap through its llama.cpp integration instead.",
        needs="An NVIDIA GPU", license="MIT", docs="https://docs.ollama.com"),
    Template(
        key="vllm", tag="Model Server", badge="vL", title="vLLM", group="Model servers", engine="vllm",
        blurb="A fast OpenAI-compatible server for one Hugging Face model, built for many people chatting at once.",
        points=("One model, kept loaded, split across the GPUs you give it",
                "Downloads the model from Hugging Face into the models folder's huggingface/ subfolder",
                "Chat apps here can use it like the Llama Swap, as the model chat"),
        default_id="vllm", default_name="vLLM", description="vLLM model server",
        image="vllm/vllm-openai:latest", container_port=8000, host_port=8000, gpu="one",
        models_mount=("huggingface", "/root/.cache/huggingface", False), secrets=("VLLM_API_KEY",),
        shm_size="8g",   # its GPUs talk through shared memory when a model is split across them
        after_install="Its first start downloads the model, which can take a while: watch its log on the dashboard. "
                      "It answers at http://this-machine:PORT/v1 with the key VLLM_API_KEY (under Edit), as the model "
                      "chat. It takes the GPUs' memory for itself while it runs, and keeps the number of GPUs you "
                      "picked: to change it, install it again with another count rather than editing its GPUs.",
        needs="An NVIDIA GPU with driver 580 or newer, and a model that fits in the GPUs' memory (no system RAM)",
        license="Apache-2.0", docs="https://docs.vllm.ai"),
    Template(
        key="open-webui", tag="Chat UI", badge="OW", title="Open WebUI", group="Chat and agents",
        blurb="A ChatGPT-style web chat for your model servers, with image generation through ComfyUI.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Optional: make images with ComfyUI"),
        default_id="webui", default_name="Open WebUI", description="Chat with your model servers",
        image="ghcr.io/open-webui/open-webui:main", container_port=8080, host_port=3000, gpu="none",
        env={"ENABLE_OLLAMA_API": "false"},
        appdata=(("data", "/app/backend/data"),), connects="openwebui",
        needs=NEEDS_SERVER, license="Open WebUI License (BSD-3-Clause plus branding terms)",
        docs="https://docs.openwebui.com"),
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
        license="MIT", docs="https://hermes-agent.nousresearch.com/docs/user-guide/docker"),
    Template(
        key="opencode", tag="Agent UI", badge="OC", title="OpenCode", group="Chat and agents",
        blurb="An open-source AI coding agent in your browser: it reads, writes and runs code in a workspace folder, "
              "thinking with your own model server.",
        points=("No GPU needed", "Connects to a model server for you, API key included, preferring its coding model",
                "Its own login; projects live in the app data folder's workspace/"),
        default_id="opencode", default_name="OpenCode", description="AI coding agent",
        image="ghcr.io/anomalyco/opencode:latest", command=("web", "--hostname", "0.0.0.0", "--port", "4096"),
        container_port=4096, host_port=4096, gpu="none", bundled_slot="coder",
        login=("OPENCODE_SERVER_USERNAME", "OPENCODE_SERVER_PASSWORD"),
        appdata=(("config", "/root/.config/opencode"), ("data", "/root/.local/share/opencode"),
                 ("workspace", "/workspace")),
        connects="opencode",
        after_install="Open it and log in with the user name and password you chose, then open the folder "
                      "/workspace (the app data folder's workspace/ on this machine) or a project inside it. "
                      "The model server is already set up as the provider Local models; its settings are in "
                      "config/opencode.json in the app data folder. Its agent can run commands inside its "
                      "container, so keep its port off the internet.",
        needs="A model server whose model can call tools, ideally a coding model with a long context",
        license="MIT", docs="https://opencode.ai/docs"),
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
        needs=NEEDS_SERVER, license="MIT", docs="https://www.librechat.ai/docs"),
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
        needs=NEEDS_SERVER, license="MIT",
        docs="https://docs.anythingllm.com/installation-docker/overview"),
    Template(
        key="langflow", tag="Agent UI", badge="LF", title="Langflow", group="Chat and agents",
        blurb="Build chatbots, RAG pipelines and agents by wiring blocks together, running on your own model server.",
        points=("No GPU needed", "Its own login, and the model server's API key saved in it for you",
                "You type the model server's address into its OpenAI blocks (steps below)"),
        default_id="langflow", default_name="Langflow", description="Langflow",
        image="langflowai/langflow:latest", container_port=7860, host_port=7860, gpu="none",
        # its database lives inside the container unless told otherwise, and is lost with it
        env={"LANGFLOW_HOST": "0.0.0.0", "LANGFLOW_PORT": "7860", "LANGFLOW_CONFIG_DIR": "/app/data",
             "LANGFLOW_DATABASE_URL": "sqlite:////app/data/langflow.db", "LANGFLOW_AUTO_LOGIN": "false",
             "DO_NOT_TRACK": "true"},
        login=("LANGFLOW_SUPERUSER", "LANGFLOW_SUPERUSER_PASSWORD"),
        appdata=(("data", "/app/data"),), data_owner=(1000, 0), connects="langflow",
        experimental="Each flow picks its own model, so GPU Shepherd can't point Langflow's flows at the "
                     "model server for you: you type its address into them yourself.",
        after_install="Log in with the user name and password you chose. In a flow, use the OpenAI block: under "
                      "its advanced settings set OpenAI API Base to the model server's address (LLM_BASE_URL under "
                      "Edit on this instance), and type chat (or another name the server answers to) as the Model "
                      "Name. Its OpenAI API Key is already the server's key. For documents, the OpenAI Embeddings "
                      "block with the same address and the model embed, if the server has an embeddings model.",
        needs="A model server for its flows to use (Llama Swap, vLLM or Ollama)", license="MIT", docs="https://docs.langflow.org"),
    Template(
        key="sillytavern", tag="Chat UI", badge="ST", title="SillyTavern", group="Chat and agents",
        blurb="Chat and role-play with characters you make or import, with group chats, lore books and extensions.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Its own login for the web page"),
        default_id="sillytavern", default_name="SillyTavern", description="Character chat",
        image="ghcr.io/sillytavern/sillytavern:latest", container_port=8000, host_port=8000, gpu="none",
        # reachable from other machines only with a login: its default only lets this machine in
        env={"NODE_ENV": "production", "SILLYTAVERN_WHITELISTMODE": "false", "SILLYTAVERN_BASICAUTHMODE": "true"},
        login=("SILLYTAVERN_BASICAUTHUSER_USERNAME", "SILLYTAVERN_BASICAUTHUSER_PASSWORD"),
        appdata=(("config", "/home/node/app/config"), ("data", "/home/node/app/data"),
                 ("plugins", "/home/node/app/plugins"), ("extensions", "/home/node/app/public/scripts/extensions/third-party")),
        connects="sillytavern",
        after_install="Open it and log in with the user name and password you chose. It is already connected to the "
                      "model server (API Connections: Chat Completion, Custom), so pick a character and chat.",
        needs=NEEDS_SERVER, license="AGPL-3.0", docs="https://docs.sillytavern.app"),
    Template(
        key="open-notebook", tag="Research UI", badge="ON", title="Open Notebook", group="Chat and agents",
        blurb="A private NotebookLM: add PDFs, web pages and videos, then chat with them, take notes and make podcasts.",
        points=("No GPU needed", "Connects to a model server for you, API key included",
                "Installs its SurrealDB database alongside, which starts and stops with it"),
        default_id="notebook", default_name="Open Notebook", description="Research notebooks",
        image="lfnovo/open_notebook:v1-latest", container_port=8502, host_port=8502, gpu="none",
        env={"SURREAL_USER": "root", "SURREAL_NAMESPACE": "open_notebook", "SURREAL_DATABASE": "open_notebook"},
        secrets=("SURREAL_PASS", "OPEN_NOTEBOOK_ENCRYPTION_KEY"), login=("", "OPEN_NOTEBOOK_PASSWORD"),
        appdata=(("data", "/app/data"),), companions=("open-notebook-surrealdb",), connects="opennotebook",
        after_install="Open it and log in with the password you chose. Under Settings, Models, press Sync for "
                      "OpenAI Compatible, then pick chat as the default chat model (and embed as the embedding "
                      "model, if the server has one, to search your sources).",
        needs=NEEDS_SERVER, license="MIT", docs="https://github.com/lfnovo/open-notebook"),
    Template(
        key="invokeai", tag="Image UI", badge="INV", title="InvokeAI", group="Image and video",
        blurb="Image generation with a polished canvas: inpainting, layers and a model manager, in the browser.",
        points=("Takes one GPU, whichever is free", "Outranks model servers when GPUs run short",
                "Models in the models folder's invokeai/ subfolder, installed from its Model Manager"),
        default_id="invokeai", default_name="InvokeAI", description="Image generation",
        image="ghcr.io/invoke-ai/invokeai:latest", container_port=9090, host_port=9090, gpu="one", priority=60,
        env={"INVOKEAI_ROOT": "/invokeai", "INVOKEAI_HOST": "0.0.0.0", "INVOKEAI_PORT": "9090"},
        appdata=(("invokeai", "/invokeai"),), models_mount=("invokeai", "/invokeai/models", False),
        after_install="Open it and add models under Model Manager, Starter Models. It has no login: anyone "
                      "who can reach its port can use it.",
        needs="An NVIDIA GPU with 8+ GB (12+ GB for SDXL and Flux)", license="Apache-2.0", docs="https://invoke.ai"),
    Template(
        key="comfyui", tag="Image Server / Image UI", badge="CUI", title="ComfyUI", group="Image and video",
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
        license="GPL-3.0 (ComfyUI), Mulan PSL v2 (the image)", docs="https://github.com/YanWenKun/ComfyUI-Docker"),

    Template(
        key="speaches", tag="Speech Server", badge="SP", title="Speaches", group="Speech",
        blurb="Speech to text (Whisper) and text to speech (Kokoro) with an OpenAI-compatible API.",
        points=("Takes one GPU, whichever is free", "Gives Open WebUI a voice: pick it when installing Open WebUI",
                "Downloads its two models the first time it starts"),
        default_id="speaches", default_name="Speaches", description="Speech to text and text to speech",
        image="ghcr.io/speaches-ai/speaches:latest-cuda", container_port=8000, host_port=8000, gpu="one",
        env={"PRELOAD_MODELS": f'["{STT_MODEL}", "{TTS_MODEL}"]', "LOG_LEVEL": "info"}, secrets=("API_KEY",),
        appdata=(("hf-cache", "/home/ubuntu/.cache/huggingface/hub"),), data_owner=(1000, 1000),
        after_install="It answers at http://this-machine:PORT/v1 with the key API_KEY (under Edit). Open WebUI "
                      "installed after it can use it for voice; for an Open WebUI you already have, set Speech to "
                      "Text and Text to Speech to OpenAI with that address and key in its Admin Panel, Audio.",
        needs="An NVIDIA GPU with 4+ GB", license="MIT", docs="https://speaches.ai"),

    # companions: installed with an app, never on their own
    Template(
        key="librechat-mongodb", tag="Database", badge="DB", title="LibreChat database", group="Companions",
        blurb="MongoDB for LibreChat's accounts and conversations.", points=(), hidden=True,
        default_id="mongodb", default_name="database", description="MongoDB for LibreChat",
        image="mongo:8.0", command=("mongod", "--noauth", "--bind_ip_all"), container_port=27017, host_port=0,
        gpu="none", appdata=(("mongodb", "/data/db"),), license="SSPL v1 (MongoDB Community Server)",
        docs="https://www.librechat.ai/docs/local/docker"),
    Template(
        key="open-notebook-surrealdb", tag="Database", badge="DB", title="Open Notebook database", group="Companions",
        blurb="SurrealDB for Open Notebook's notebooks, sources and notes.", points=(), hidden=True,
        default_id="surrealdb", default_name="database", description="SurrealDB for Open Notebook",
        image="surrealdb/surrealdb:v2", command=("start", "--log", "info", "rocksdb:/mydata/mydatabase.db"),
        container_port=8000, host_port=0, gpu="none", appdata=(("surrealdb", "/mydata"),),
        shares_env=("SURREAL_USER", "SURREAL_PASS"), user="0:0",   # root, as upstream runs it, so it can write its folder
        license="BSL-1.1 (SurrealDB, Apache-2.0 after four years)", docs="https://github.com/lfnovo/open-notebook"),
)

# the model server an app with a connection can install alongside itself, and the model slot it fills
BUNDLED_SERVER = "llm-server"
BUNDLED_SLOT = "chat"
# App Center keys from before there was one Llama Swap app
RENAMED = {"llm-chat": "llm-server", "llm-coder": "llm-server"}
LLAMA_SWAP_APPS = {t.key for t in TEMPLATES if t.kind == "llama-swap"}
NEW_SERVER = "new"

GPU_CHOICES = {
    "one": "One GPU, whichever is free",
    "all": "Every free GPU (splits a model across them), at least one",
    "none": "No GPU",
}


def gpu_choices(gpus: list, t: Template | None = None) -> dict[str, str]:
    """The GPU answers an install offers on a machine with these GPUs (gpu.Gpu): GPU_CHOICES, plus
    "up:N" for each in-between count when there are 3 or more GPUs, and "pin:I" per GPU when there
    are 2 or more. vLLM splits its model across a fixed number of GPUs, so it gets "gpus:N"
    (2, 4 or 8: counts a model's layers split evenly into) instead of "up to" counts."""
    gpus = [g for g in gpus if gpu_vendor(g.index) == (t.vendor if t else "nvidia")]   # the kind it runs on
    pins = {f"pin:{g.index}": f"Always {gpu_label(g.index)} ({g.name}, {g.mem_total / 1024:.0f} GB)" for g in gpus} \
        if len(gpus) > 1 else {}
    if t is not None and t.engine == "vllm":
        out = {"one": GPU_CHOICES["one"]}
        out |= {f"gpus:{n}": f"{n} free GPUs (splits the model across them)" for n in (2, 4, 8) if n <= len(gpus)}
        return out | pins
    out = {"one": GPU_CHOICES["one"]}
    out |= {f"up:{n}": f"Up to {n} free GPUs (splits a model across them), at least one" for n in range(2, len(gpus))}
    out["all"] = GPU_CHOICES["all"]
    return out | pins | {"none": GPU_CHOICES["none"]}

OFFLOAD_CHOICES = {
    "auto": "Decide from the file size and your GPUs",
    "gpu": "Keep the whole model on the GPUs",
    "moe": "Keep MoE expert layers in system RAM (for models larger than your GPUs)",
}


def matches(t: Template, inst: Instance) -> bool:
    """Whether an instance was installed from this template (older ones by their image)."""
    if inst.kind == "llama-swap" or t.kind == "llama-swap":
        if inst.kind != t.kind:
            return False
        # one installed from the Vulkan Llama Swap is that; every other llama-swap instance (made by
        # hand, or from before there were two) is the Llama Swap
        app = RENAMED.get(inst.app, inst.app)
        return app == t.key if app in LLAMA_SWAP_APPS else t.key == BUNDLED_SERVER
    if inst.app:
        return RENAMED.get(inst.app, inst.app) == t.key
    return inst.image.rsplit(":", 1)[0] == t.image.rsplit(":", 1)[0]


OTHER_TAG = "App"        # an instance that isn't from the App Center


def tag_for(inst: Instance) -> str:
    """The kind tag on an instance's dashboard card: from its App Center app, else generic."""
    return next((t.tag for t in TEMPLATES if matches(t, inst)), OTHER_TAG)


# Default names an app used to be installed under; they show as its current title
LEGACY_NAMES = {"llm-server": ("llm server",)}


def app_name(inst: Instance) -> str:
    """Its name, with the app's real name where it only repeats the kind tag.

    Open WebUI used to be installed as "Chat UI" (then "Chat UI 2"...), which said the same as
    its tag; those show as "Open WebUI" ("Open WebUI 2"). Any other name is the user's choice.
    """
    t = next((t for t in TEMPLATES if matches(t, inst)), None)
    if t is None or t.title == t.tag:
        return inst.name
    if inst.name.casefold() in (t.tag.casefold(), *LEGACY_NAMES.get(t.key, ())):
        return t.title
    head, _, suffix = inst.name.rpartition(" ")
    if head.casefold() == t.tag.casefold() and suffix == inst.id.rsplit("-", 1)[-1]:
        return f"{t.title} {suffix}"
    return inst.name


def installed(instances: dict[str, Instance]) -> dict[str, list[Instance]]:
    """Template key -> the instances installed from it (App Center apps only, not companions)."""
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
    servers: dict[str, dict]           # model server instance id -> {"api_key", "default_model"} (see model_servers)
    model_sizes: dict[str, float]      # container path of a .gguf file -> size in GB
    ram_gb: float = 0.0                # system RAM (0 = unknown)
    amd_gpu_mem_gb: list[float] = field(default_factory=list)   # the AMD GPUs, for apps that run on them
    suggestions: tuple = ()            # suggested models besides the built-in ones (model_library.Suggestion)


def defaults(t: Template, m: Machine, models_dir: str, appdata_root: str) -> dict:
    """The guided form's starting values."""
    taken_ports = {p.host for i in m.instances.values() for p in i.ports}
    iid = free_id(t.default_id, set(m.instances))
    name = t.default_name if iid == t.default_id else f"{t.default_name} {iid.rsplit('-', 1)[-1]}"
    comfy = next((i.id for i in m.instances.values() if "comfyui" in (i.image + i.id).lower()), "")
    speech = next(iter(speech_servers(m)), "")
    return {"name": name, "id": iid, "port": free_port(t.host_port, taken_ports), "gpu": t.gpu,
            "models_dir": models_dir, "appdata_dir": str(Path(appdata_root) / iid),
            "server": next(iter(m.servers), NEW_SERVER if t.connects else ""), "comfyui": comfy, "speech": speech,
            **{f"{s.key}_file": "" for s in slots(t)}, **{f"{s.key}_ctx": s.ctx_size for s in slots(t)},
            "offload": "auto", "autostart": False,
            "login_user": "admin" if t.login and t.login[0] else "",
            "password": secrets.token_urlsafe(12) if t.login else "",
            "hass_url": "", "hass_token": "", "hf_model": VLLM_MODEL, "hf_token": "", "max_len": VLLM_MAX_LEN}


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
        keys += ["comfyui", "speech"]
    for s in slots(t):
        keys += [f"{s.key}_file", f"{s.key}_ctx"]
    if slots(t):
        keys.append("offload")
    if t.login:
        keys += ["login_user", "password"] if t.login[0] else ["password"]
    if t.engine == "vllm":
        keys += ["hf_model", "hf_token", "max_len"]
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
    gpu = _gpu_policy(str(a.get("gpu") or t.gpu), t.priority, t.vendor)
    if t.cpu_ok and gpu.mode == "elastic":
        # a chat server keeps answering (on the CPU, slower) while an image app has the only GPU
        gpu.cpu_fallback = True

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

    env = dict(t.env) | {k: secrets.token_urlsafe(24) for k in t.secrets}
    if t.login:
        user, password = str(a.get("login_user", "")).strip(), str(a.get("password", "")).strip()
        if t.login[0] and not user or len(password) < 8:
            raise ConfigError("Login: enter a user name and a password of at least 8 characters"
                              if t.login[0] else "Login: enter a password of at least 8 characters")
        env |= ({t.login[0]: user} if t.login[0] else {}) | {t.login[1]: password}
    if t.connects == "openwebui":
        env.update(_openwebui(a, m))
    elif t.connects == "hermes":
        env.update(_hermes(a, m))
    elif t.connects == "librechat":
        env.update(_librechat(a, m, requires))
    elif t.connects == "langflow":
        env.update(_langflow(a, m))
    elif t.connects in ("sillytavern", "opencode"):
        env.update(_server_env(a, m))
    elif t.connects == "opennotebook":
        env |= {"SURREAL_URL": f"ws://{requires[0]}:8000/rpc"} | _opennotebook(_server(a, m), m)
    command = list(t.command)
    if t.engine == "vllm":
        command, hf_env = _vllm(a, gpu)
        env |= hf_env

    inst = Instance(id=iid, name=str(a.get("name", "")).strip() or iid, description=t.description, kind=t.kind,
                    image=t.image, command=command, env=env,
                    ports=[PortMap(host=int(port), container=t.container_port)], volumes=volumes, gpu=gpu,
                    autostart=bool(a.get("autostart")), open_path=t.open_path, app=t.key, requires=requires,
                    shm_size=t.shm_size, user=t.user)
    return inst, _models(t, a, m)


def gpu_mem_gb(t: Template, m: Machine) -> list[float]:
    """The video memory (GB) of each GPU of the kind the app runs on, for fitting and offloading models."""
    return m.amd_gpu_mem_gb if t.vendor == "amd" else m.gpu_mem_gb


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
        shared = {k: inst.env[k] for k in c.shares_env if k in inst.env}
        out.append(Instance(
            id=cid, name=f"{inst.name} {c.default_name}", description=c.description, image=c.image,
            command=list(c.command), env=dict(c.env) | shared, gpu=_gpu_policy("none", c.priority), app=c.key,
            user=c.user, shm_size=c.shm_size,
            volumes=[VolumeMap(host=f"{appdata}/{sub}", container=inside) for sub, inside in c.appdata + c.files]))
    return out


def bundled_server(a: dict, m: Machine, appdata_root: str,
                   slot: str = BUNDLED_SLOT) -> tuple[Instance, Catalog] | None:
    """The Llama Swap an app installs alongside itself when asked to (server = "new"),
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
    d |= {f"{slot}_file": str(a.get(f"{slot}_file", "")), "gpu": t.gpu}
    srv, models = build(t, d, m)
    cat = Catalog(instance_id=srv.id, models=models, fit_gpus=True)
    m.instances[srv.id] = srv
    m.servers[srv.id] = {"api_key": cat.api_key, "name": srv.name, "default_model": slot if models else "",
                         "ctx": models[0].ctx_size if models else 0,
                         "models": {slot: models[0].ctx_size} if models else {}}
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
    elif t.connects == "sillytavern" and info:
        out |= _sillytavern_files(_server_url(server, m), info)
    elif t.connects == "opencode" and info:
        out["config/opencode.json"] = _opencode_config(info)
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
    elif t.connects == "langflow" and a["server"]:
        env |= _langflow_server(server, m)
    elif t.connects in ("sillytavern", "opencode") and a["server"]:
        env |= _server_env(a, m)
    elif t.connects == "opennotebook" and a["server"]:
        env |= _opennotebook(server, m)
    return env


def connected_server(inst: Instance, m: Machine, settings: str = "") -> str:
    """The model server an app instance talks to, found by its address in the instance's environment
    or in its own settings file (AnythingLLM keeps it there)."""
    for server in m.servers:
        if server in m.instances and any(_server_url(server, m) in v for v in [*inst.env.values(), settings]):
            return server
    return ""


def _gpu_policy(choice: str, priority: int, vendor: str = "nvidia") -> GpuPolicy:
    if choice == "none":
        return GpuPolicy(mode="none", priority=priority)
    if choice == "one":
        return GpuPolicy(mode="elastic", min_gpus=1, max_gpus=1, priority=priority, vendor=vendor)
    if choice == "all":
        return GpuPolicy(mode="elastic", min_gpus=1, max_gpus=0, priority=priority, vendor=vendor)
    if choice.startswith("gpus:") and choice[5:].isdigit() and int(choice[5:]) >= 1:
        return GpuPolicy(mode="elastic", min_gpus=int(choice[5:]), max_gpus=int(choice[5:]), priority=priority,
                         vendor=vendor)
    if choice.startswith("up:") and choice[3:].isdigit() and int(choice[3:]) >= 1:
        return GpuPolicy(mode="elastic", min_gpus=1, max_gpus=int(choice[3:]), priority=priority, vendor=vendor)
    if choice.startswith("pin:") and choice[4:].isdigit():
        try:
            return GpuPolicy(mode="fixed", gpus=[int(choice[4:])], priority=priority, vendor=vendor)
        except ValueError as err:
            raise ConfigError(f"GPU: {err}") from err
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


def _server_env(a: dict, m: Machine) -> dict[str, str]:
    """The model server's address and key, for apps whose own settings hold the connection: kept in
    the environment so they show under Edit and app updates can tell which server it uses."""
    server = _server(a, m)
    return {"LLM_BASE_URL": _server_url(server, m), "LLM_API_KEY": m.servers[server]["api_key"]} if server else {}


def _sillytavern_files(url: str, info: dict) -> dict[str, str]:
    """SillyTavern's first user's settings, connected to the model server: a Custom (OpenAI-compatible)
    chat completion source that connects when the page opens. SillyTavern fills in everything else
    from its defaults and moves the key into its own secrets format on its first start."""
    settings = {"main_api": "openai", "power_user": {"auto_connect": True},
                "oai_settings": {"chat_completion_source": "custom", "custom_url": url,
                                 "custom_model": info.get("default_model") or "chat"}}
    return {"data/default-user/settings.json": json.dumps(settings, indent=4) + "\n",
            "data/default-user/secrets.json": json.dumps({"api_key_custom": info["api_key"]}, indent=4) + "\n"}


OPENCODE_PROVIDER = "local"


def _opencode_config(info: dict) -> str:
    """OpenCode's own settings (its global opencode.json): the model server as an OpenAI-compatible
    provider whose address and key come from the instance's environment (LLM_BASE_URL, LLM_API_KEY),
    so pointing it at another server is an Edit away. It lists the server's chat models (OpenCode
    doesn't ask the server for them) and starts on the coding one, which also writes the session
    titles so the server never swaps models just for those."""
    models = dict(info.get("models") or {})
    default = info.get("default_model") or "chat"
    models.setdefault(default, info.get("ctx") or 0)
    pick = "coder" if "coder" in models else default
    entries = {}
    for name, ctx in models.items():
        entries[name] = {"name": name} | ({"limit": {"context": ctx, "output": min(ctx // 4, 16384)}} if ctx else {})
    config = {"$schema": "https://opencode.ai/config.json",
              "provider": {OPENCODE_PROVIDER: {"npm": "@ai-sdk/openai-compatible", "name": "Local models",
                                               "options": {"baseURL": "{env:LLM_BASE_URL}", "apiKey": "{env:LLM_API_KEY}"},
                                               "models": entries}},
              "model": f"{OPENCODE_PROVIDER}/{pick}", "small_model": f"{OPENCODE_PROVIDER}/{pick}",
              "autoupdate": False, "share": "disabled"}
    return json.dumps(config, indent=2) + "\n"


def _opennotebook(server: str, m: Machine) -> dict[str, str]:
    """Open Notebook's OpenAI Compatible provider, which it reads from the environment."""
    if not server:
        return {}
    return {"OPENAI_COMPATIBLE_BASE_URL": _server_url(server, m), "OPENAI_COMPATIBLE_API_KEY": m.servers[server]["api_key"]}


# vLLM: the model it serves when none is picked (small enough for one 8 GB GPU), and its context
VLLM_MODEL = "Qwen/Qwen3-8B-AWQ"
VLLM_MAX_LEN = 8192
VLLM_ALIAS = "chat"
HF_MODEL = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _vllm(a: dict, gpu: GpuPolicy) -> tuple[list[str], dict[str, str]]:
    """vLLM's arguments (the image runs `vllm serve`) and Hugging Face token. It splits the model
    across exactly the GPUs it gets, so the count is fixed when it is installed."""
    model = str(a.get("hf_model", "")).strip() or VLLM_MODEL
    if not HF_MODEL.match(model):
        raise ConfigError("Model: enter a Hugging Face model name like Qwen/Qwen3-8B-AWQ")
    length = str(a.get("max_len", "")).strip() or str(VLLM_MAX_LEN)
    if not length.isdigit() or int(length) < 512:
        raise ConfigError("Context: enter a whole number of tokens, 512 or more")
    if gpu.mode == "none" or (gpu.mode == "elastic" and gpu.min_gpus != gpu.max_gpus):
        raise ConfigError("GPUs: vLLM needs a set number of GPUs")
    count = len(gpu.gpus) if gpu.mode == "fixed" else gpu.min_gpus
    command = [model, "--served-model-name", VLLM_ALIAS, model, "--host", "0.0.0.0", "--port", "8000",
               "--tensor-parallel-size", str(count), "--max-model-len", length, "--gpu-memory-utilization", "0.90"]
    token = str(a.get("hf_token", "")).strip()
    return command, {"HF_TOKEN": token} if token else {}


MODEL_SERVER_TAG = "Model Server"


def is_model_server(t: Template) -> bool:
    """An App Center app other apps can connect to (Llama Swap, vLLM, Ollama)."""
    return t.tag == MODEL_SERVER_TAG


# Ollama has no API keys; apps that insist on one send this, which it ignores
OLLAMA_KEY = "ollama"


def ollama_server(inst: Instance) -> dict | None:
    """An Ollama instance as a model server apps can connect to (like Machine.servers' entries), or
    None. Its models are whatever was pulled into it, so apps ask for the name chat (its install
    steps say how to give a model that name) and the apps that list models show the rest."""
    if not matches(get("ollama"), inst):
        return None
    ctx = inst.env.get("OLLAMA_CONTEXT_LENGTH", "")
    ctx = int(ctx) if ctx.isdigit() else 0
    return {"api_key": OLLAMA_KEY, "name": inst.name, "default_model": VLLM_ALIAS, "ctx": ctx, "embed": "",
            "models": {VLLM_ALIAS: ctx}}


def vllm_server(inst: Instance) -> dict | None:
    """A vLLM instance as a model server apps can connect to (like Machine.servers' entries), or None."""
    if inst.app != "vllm" or not inst.env.get("VLLM_API_KEY"):
        return None
    args = inst.command
    ctx = next((int(args[i + 1]) for i, x in enumerate(args[:-1]) if x == "--max-model-len" and args[i + 1].isdigit()), 0)
    names = next((args[i + 1] for i, x in enumerate(args[:-1]) if x == "--served-model-name"), VLLM_ALIAS)
    return {"api_key": inst.env["VLLM_API_KEY"], "name": inst.name, "default_model": names, "ctx": ctx, "embed": "",
            "models": {names: ctx}}


def _langflow(a: dict, m: Machine) -> dict[str, str]:
    """A fresh key for the secrets it stores (a Fernet key), and the model server."""
    env = {"LANGFLOW_SECRET_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()}
    server = _server(a, m)
    if server:
        env |= _langflow_server(server, m)
    return env


def _langflow_server(server: str, m: Machine) -> dict[str, str]:
    """The server's key, which Langflow saves as its OPENAI_API_KEY variable, and the server's host
    on its allowlists: Langflow refuses plain-http internal addresses, and sending a key from its
    environment anywhere but OpenAI, unless the host is listed."""
    return {"OPENAI_API_KEY": m.servers[server]["api_key"], "LLM_BASE_URL": _server_url(server, m),
            "LANGFLOW_SSRF_ALLOWED_HOSTS": server, "LANGFLOW_PROVIDER_CREDENTIAL_ALLOWED_HOSTS": f"http://{server}"}


# How Open WebUI splits and searches documents when it uses a model server's embeddings model:
# bigger pieces than its default suit books, and more of them go to the chat model with a question.
# Open WebUI only reads these at its first start (then they are in its Admin Panel, Documents), so
# they aren't offered as app updates; the Books page sets the embeddings ones through its API.
OPENWEBUI_DOCS = {"CHUNK_SIZE": "1500", "CHUNK_OVERLAP": "200", "RAG_TOP_K": "8", "RAG_EMBEDDING_BATCH_SIZE": "16",
                  "RAG_EMBEDDING_QUERY_PREFIX": ""}
# What a question starts with for embeddings models that are trained with an instruction
QUERY_PREFIX = {"qwen3-embedding": "Instruct: Given a question, retrieve passages that answer it\nQuery: "}


def query_prefix(model_file: str) -> str:
    name = model_file.lower().rsplit("/", 1)[-1]
    return next((p for k, p in QUERY_PREFIX.items() if k in name), "")


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
                    "RAG_OPENAI_API_KEY": info["api_key"], "RAG_EMBEDDING_MODEL": info["embed"]} \
                | {k: v for k, v in OPENWEBUI_DOCS.items() if v}
            if info.get("embed_query_prefix"):
                env["RAG_EMBEDDING_QUERY_PREFIX"] = info["embed_query_prefix"]
    speech = str(a.get("speech", "")).strip()
    if speech:
        if speech not in speech_servers(m):
            raise ConfigError(f"no Speaches instance '{speech}'")
        ports = m.instances[speech].ports
        url = f"http://{speech}:{ports[0].container if ports else 8000}/v1"
        key = m.instances[speech].env.get("API_KEY") or "none"
        env |= {"AUDIO_STT_ENGINE": "openai", "AUDIO_STT_OPENAI_API_BASE_URL": url, "AUDIO_STT_OPENAI_API_KEY": key,
                "AUDIO_STT_MODEL": STT_MODEL, "AUDIO_TTS_ENGINE": "openai", "AUDIO_TTS_OPENAI_API_BASE_URL": url,
                "AUDIO_TTS_OPENAI_API_KEY": key, "AUDIO_TTS_MODEL": TTS_MODEL, "AUDIO_TTS_VOICE": TTS_VOICE}
    comfy = str(a.get("comfyui", "")).strip()
    if comfy:
        if comfy not in m.instances:
            raise ConfigError(f"no instance '{comfy}'")
        ports = m.instances[comfy].ports
        port = ports[0].container if ports else 8188
        env |= {"ENABLE_IMAGE_GENERATION": "true", "IMAGE_GENERATION_ENGINE": "comfyui",
                "COMFYUI_BASE_URL": f"http://{comfy}:{port}"}
    return env


def speech_servers(m: Machine) -> dict[str, str]:
    """Speaches instances an app can use for voice: id -> name."""
    return {i.id: i.name for i in m.instances.values() if matches(get("speaches"), i)}


def slots(t: Template) -> tuple[ModelSlot, ...]:
    """The model slots an app's form shows: a model server's own, or the one its bundled server fills."""
    if t.models:
        return t.models
    if t.connects:
        return tuple(x for x in get(BUNDLED_SERVER).models if x.key == t.bundled_slot)
    return ()


def _models(t: Template, a: dict, m: Machine) -> list[CatalogModel]:
    """The catalogue entries for the model files picked in the form. The same file picked for two
    slots becomes one model that answers to both names."""
    choice = str(a.get("offload") or "auto")
    mem = gpu_mem_gb(t, m)
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
        known = model_library.for_file(file, MODELS_IN_CONTAINER, m.suggestions)
        if known and slot.role == known.role:     # a suggested model: its makers' settings
            x = model_library.catalog_model(known, mem, MODELS_IN_CONTAINER, ctx_size=ctx_size)
            if t.kv_cache:
                x.kv_cache = t.kv_cache
            if choice == "gpu":
                x.offload = [r for r in x.offload if r.gpu_layers is not None]
            x.id = free_id(x.id, {y.id for y in out})
            x.aliases = [slot.alias] if slot.alias != x.id else []
            out.append(x)
            continue
        if choice == "moe":
            offload = [OffloadRule(min_gpus=1, cpu_moe=999)]
        elif choice == "auto":
            offload = auto_offload(m.model_sizes.get(file), mem)
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
                                role=slot.role, keep_loaded=slot.role == "embed", preload=slot.role == "embed",
                                aliases=[slot.alias] if slot.alias != mid else []))
    for x in out:
        x.aliases = [al for al in x.aliases if al != x.id]
    return out
