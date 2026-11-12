"""App updates: bring instances installed from the App Center up to what the app sets up today.

When an app's template changes (Open WebUI and AnythingLLM using the model server's embeddings
model, the Llama Swap's embeddings slot, a suggested model's settings), instances installed before
the change don't get it by themselves. This module compares each installed instance with what its
app would set up now and lists the differences as updates the user can apply or skip.

Updates only add what is missing or still at its default, so nothing the user set is changed:
an environment variable that is there keeps its value, a model setting that was filled in is left
alone. Applying an update never restarts anything; the dashboard shows the instance needs a restart.

This module is pure: the service passes in the machine, catalogues and settings files.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import instance_templates as tpl, model_library
from .connectors import comfyui
from .config import Catalog, CatalogModel, Instance

EMBED_MODEL = "qwen3-embedding-0.6b"           # the embeddings model a Llama Swap server without one is offered
ANYTHINGLLM_FILE = "server.env"


@dataclass
class Update:
    iid: str
    key: str                                   # stable for the same change, so skipping it sticks
    title: str
    lines: list[str]                           # what applying it changes
    manual: list[str] = field(default_factory=list)   # what is left to do in the app itself
    env: dict[str, str] = field(default_factory=dict)  # kind "env": variables to add
    model: CatalogModel | None = None          # kind "model": the model's new settings
    file_text: str = ""                        # kind "file": the settings file's new text
    kind: str = "env"                          # env | embed-model | model | file | images


def _key(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:8]}"


def for_instance(inst: Instance, m: tpl.Machine, catalog: Catalog | None = None,
                 settings_file: str | None = None) -> list[Update]:
    """The updates an instance can take. `catalog` is a model server's; `settings_file` is the text
    of an app's own settings file (AnythingLLM's server.env) when GPU Shepherd can read it."""
    t = next((t for t in tpl.TEMPLATES if tpl.matches(t, inst)), None)
    if t is None:
        return []
    out: list[Update] = []
    server = tpl.connected_server(inst, m, settings_file or "") if t.connects else ""
    want = tpl.app_env(t, server, m)
    # empty ones change nothing; Open WebUI's document settings only count at its first start
    missing = {k: v for k, v in want.items() if v and k not in inst.env and k not in tpl.OPENWEBUI_DOCS}
    if missing:
        u = Update(inst.id, _key("env", *sorted(missing)), "New settings for " + t.title,
                   [f"Adds {k}" + ("" if "KEY" in k else f"={v}") for k, v in sorted(missing.items())], env=missing)
        if "RAG_EMBEDDING_ENGINE" in missing:
            u.title = "Use the model server's embeddings model for documents"
            # Open WebUI keeps these in its database after the first start and ignores the variables then
            u.manual = ["Open WebUI saved its document settings when it first started, so also set them in Open WebUI: "
                        "Admin Panel, Settings, Documents. Embedding Model Engine: OpenAI; "
                        f"URL: {missing.get('RAG_OPENAI_API_BASE_URL', '')}; API key: the model server's key; "
                        f"Embedding Model: {missing.get('RAG_EMBEDDING_MODEL', '')}. Then re-upload documents "
                        "you added before, so they are embedded by the new model."]
        out.append(u)
    if t.connects == "openwebui":
        u = _images(inst, m)
        if u:
            out.append(u)
    if t.connects == "anythingllm" and server and settings_file is not None:
        u = _anythingllm(inst, server, m, settings_file)
        if u:
            out.append(u)
    if inst.kind == "llama-swap" and catalog is not None:
        out += _server(inst, catalog, m)
    return [u for u in out if u.key not in inst.skipped_updates]


def _images(inst: Instance, m: tpl.Machine) -> Update | None:
    """Open WebUI making and editing pictures with ComfyUI: finish the connection it has (an install
    from before the workflows only had the address), or offer one when a ComfyUI is installed."""
    url = inst.env.get("COMFYUI_BASE_URL", "")
    comfys = tpl.comfyui_instances(m)
    if url:
        comfy = next((c for c in comfys if tpl.comfyui_env(c, m)["COMFYUI_BASE_URL"] == url.rstrip("/")), "")
    else:
        comfy = comfys[0] if comfys and inst.env.get("ENABLE_IMAGE_GENERATION", "").lower() != "false" else ""
    if not comfy:
        return None
    want = tpl.comfyui_env(comfy, m)
    name = m.instances[comfy].name
    if comfy in m.comfy_workflows:        # the user's own workflows: they replace what was set before
        change = {k: v for k, v in want.items() if inst.env.get(k) != v and k != "COMFYUI_BASE_URL"}
        if not change:
            return None
        own = m.comfy_workflows[comfy]
        lines = [f"Makes pictures with your own workflow in {name}'s user/{comfyui.CUSTOM_DIR}/generate.json"
                 if "generate" in own else "Makes pictures with GPU Shepherd's checkpoint workflow"]
        lines.append("Edits them with your own edit.json" if "edit" in own else
                     "Turns editing pictures off (there is no edit.json beside it)")
        if change.get("IMAGE_GENERATION_MODEL"):
            lines.append(f"Model: {change['IMAGE_GENERATION_MODEL']}, at {want.get('IMAGE_SIZE', '')}")
        return Update(inst.id, _key("images", comfy, *sorted(f"{k}={v}" for k, v in change.items())),
                      f"Use your own {name} workflow for pictures", lines, env=change, kind="images")
    missing = {k: v for k, v in want.items() if k not in inst.env}
    if not missing:
        return None
    title = f"Make and edit pictures in chats with {name}" if not url else f"Finish connecting {name} for pictures"
    lines = [f"Turns on making pictures and editing them in chats with {name}, with its workflows and node "
             "mapping filled in"]
    if "IMAGE_GENERATION_MODEL" in missing:
        lines.append(f"Uses the checkpoint {missing['IMAGE_GENERATION_MODEL']} at {missing.get('IMAGE_SIZE', '')}")
    elif "IMAGE_GENERATION_MODEL" not in inst.env:
        lines.append(f"{name} has no checkpoint yet: pick one later in Open WebUI, Admin Panel, Settings, Images")
    return Update(inst.id, _key("images", comfy, *sorted(missing)), title, lines, env=missing, kind="images",
                  manual=[] if "IMAGE_GENERATION_MODEL" in missing or "IMAGE_GENERATION_MODEL" in inst.env else
                  ["Put a checkpoint in the models folder's comfyui/checkpoints/, then pick it in Open WebUI: "
                   "Admin Panel, Settings, Images. A model that isn't one checkpoint file needs a workflow of "
                   f"your own in {name}'s app data folder, user/{comfyui.CUSTOM_DIR}/ (see docs/advanced.md)."])


def _anythingllm(inst: Instance, server: str, m: tpl.Machine, text: str) -> Update | None:
    """Switch AnythingLLM from its built-in embedder to the model server's, while it still uses the
    built-in one GPU Shepherd set up (one picked in its web page is left alone)."""
    info = m.servers[server]
    if not info.get("embed") or not re.search(r"^EMBEDDING_ENGINE='native'\s*$", text, re.M):
        return None
    new = tpl._anythingllm_env(tpl._server_url(server, m), info)
    lines = [x for x in new.splitlines() if x.startswith(("EMBEDDING_", "GENERIC_OPEN_AI_EMBEDDING"))]
    body = re.sub(r"^EMBEDDING_ENGINE='native'\s*$", "\n".join(lines), text, count=1, flags=re.M)
    return Update(inst.id, _key("file", *lines), "Use the model server's embeddings model for documents",
                  [f"Sets {x.split('=', 1)[0]} in {ANYTHINGLLM_FILE}" for x in lines], kind="file", file_text=body,
                  manual=["Documents already in a workspace stay embedded by the old embedder: remove and add them "
                          "again so the new one embeds them."])


def _server(inst: Instance, cat: Catalog, m: tpl.Machine) -> list[Update]:
    out = []
    if not any(x.role == "embed" for x in cat.models):
        s = model_library.get(EMBED_MODEL)
        out.append(Update(inst.id, "embed-model", "Add an embeddings model",
                          [f"Adds {s.name} as \"embed\" ({s.download_gb:.1f} GB download). It runs on the CPU and "
                           "stays loaded beside the chat model, so chat apps can search your documents."],
                          kind="embed-model"))
    for x in cat.models:
        s = next((s for s in model_library.SUGGESTIONS if s.model.name == PurePosixPath(x.file).name), None)
        if not s or s.role != x.role:
            continue
        new = x.model_copy(deep=True)
        lines = []
        if not x.sampling and s.sampling:
            new.sampling = s.sampling
            lines.append(f"Sampling: {s.sampling}")
        if x.thinking == "auto" and s.thinking != "auto":
            new.thinking = s.thinking
            lines.append(f"Thinking: {s.thinking}")
        if not x.mtp and s.mtp and not s.draft:      # a separate draft file would have to be downloaded first
            new.mtp = s.mtp
            lines.append(f"MTP: {s.mtp} tokens guessed ahead")
        if not x.extra_args and s.extra_args:
            new.extra_args = s.extra_args
            lines.append(f"Extra flags: {s.extra_args}")
        if lines:
            out.append(Update(inst.id, _key("model", x.id, *lines), f"Suggested settings for {x.id}",
                              lines, kind="model", model=new))
    return out
