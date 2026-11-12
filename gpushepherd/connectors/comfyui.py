"""ComfyUI as Open WebUI's image engine: making images from a chat, and editing them in it.

Open WebUI sends ComfyUI a workflow in ComfyUI's API format, after writing the prompt, the model
(checkpoint) and the rest into the nodes its node map names. Its built-in workflow only works with
that map filled in, which Open WebUI's own settings page does but its environment variables don't,
so GPU Shepherd hands it both, plus a second workflow for editing: a picture from the chat (one it
made, or one the user attached) is uploaded to ComfyUI, scaled to about one megapixel and redrawn
from the prompt (image to image). With both on, a chat can make a picture, then change it in the
next message, and models that call tools get Open WebUI's generate and edit image tools.

Both workflows use only ComfyUI's own nodes and a single checkpoint file (Stable Diffusion 1.5 or
XL style, in the models folder's comfyui/checkpoints/), so nothing extra has to be installed.

Models that aren't one checkpoint file (a diffusion model with its own text encoder and VAE, and
the like) need a workflow of their own. Put it in ComfyUI's app data folder, user/open-webui/, in
ComfyUI's API format ("Export (API)") with Open WebUI's node map beside it:

    generate.json, generate-nodes.json    making pictures
    edit.json, edit-nodes.json            editing them (optional: without them, editing is off)

A node map is a list like GENERATE_NODES below. GPU Shepherd then uses these instead of its own,
taking the model, size and steps from the nodes the map names, and offers them as an app update.

Open WebUI reads these variables only until its image settings are saved, so for an Open WebUI that
already ran, GPU Shepherd also sends them through its API (connectors/openwebui.py, use_images).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CONTAINER_PORT = 8188
CHECKPOINT_EXT = (".safetensors", ".ckpt")
CUSTOM_DIR = "open-webui"            # in ComfyUI's user/ folder: workflows of the user's own

# text to image: Open WebUI's own default workflow, with a few more steps' worth of defaults
GENERATE = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 0, "steps": 25, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "Prompt", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "OpenWebUI", "images": ["8", 0]}},
}
GENERATE_NODES = [
    {"type": "prompt", "key": "text", "node_ids": ["6"]},
    {"type": "model", "key": "ckpt_name", "node_ids": ["4"]},
    {"type": "width", "key": "width", "node_ids": ["5"]},
    {"type": "height", "key": "height", "node_ids": ["5"]},
    {"type": "n", "key": "batch_size", "node_ids": ["5"]},
    {"type": "steps", "key": "steps", "node_ids": ["3"]},
    {"type": "seed", "key": "seed", "node_ids": ["3"]},
]

# image to image: the picture is scaled to about a megapixel (any shape), then partly redrawn.
# Open WebUI doesn't send a size or steps for edits, so neither is in the node map.
EDIT = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
    "3": {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 8}},
    "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "Prompt", "clip": ["1", 1]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark", "clip": ["1", 1]}},
    "7": {"class_type": "KSampler", "inputs": {
        "seed": 0, "steps": 25, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.6,
        "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "OpenWebUI-edit", "images": ["8", 0]}},
}
EDIT_NODES = [
    {"type": "prompt", "key": "text", "node_ids": ["5"]},
    {"type": "model", "key": "ckpt_name", "node_ids": ["1"]},
    {"type": "image", "key": "image", "node_ids": ["2"]},
    {"type": "seed", "key": "seed", "node_ids": ["7"]},
]


def base_url(iid: str, port: int | None = None) -> str:
    """ComfyUI's address from another container on GPU Shepherd's network."""
    return f"http://{iid}:{port or CONTAINER_PORT}"


def checkpoints(models_dir: str | Path) -> list[str]:
    """The checkpoint files in a ComfyUI models folder, as ComfyUI names them (relative to checkpoints/)."""
    root = Path(models_dir) / "checkpoints"
    try:
        return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                      if p.is_file() and p.suffix.lower() in CHECKPOINT_EXT)
    except OSError:
        return []


def pick_checkpoint(names: list[str]) -> str:
    """The checkpoint to make pictures with: an SDXL one if there is one (the Starter's), else the first."""
    return next((n for n in names if "xl" in n.lower()), names[0] if names else "")


def size(checkpoint: str) -> str:
    """The picture size a checkpoint is trained for: 1024 for SDXL-style ones, 512 for older ones."""
    return "512x512" if checkpoint and "xl" not in checkpoint.lower() else "1024x1024"


def custom(user_dir: str | Path) -> dict[str, tuple[dict, list]]:
    """The user's own workflows in ComfyUI's user folder: {"generate"/"edit": (workflow, node map)}.
    One that doesn't read, or whose map names a node or input its workflow doesn't have, is left out."""
    out = {}
    for kind in ("generate", "edit"):
        flow_file = Path(user_dir) / CUSTOM_DIR / f"{kind}.json"
        if not flow_file.is_file():
            continue
        try:
            flow = json.loads(flow_file.read_text(encoding="utf-8"))
            nodes = json.loads((flow_file.parent / f"{kind}-nodes.json").read_text(encoding="utf-8"))
            for n in nodes:
                for nid in n["node_ids"]:
                    if (n.get("key") or "text") not in flow[nid]["inputs"]:
                        raise KeyError(f"node {nid} has no input {n.get('key')}")
            if not any(n.get("type") == "prompt" for n in nodes):
                raise KeyError("no prompt node")
        except (OSError, ValueError, KeyError, TypeError) as err:
            log.warning("ignoring the ComfyUI workflow %s: %s", flow_file, err)
            continue
        out[kind] = (flow, nodes)
    return out


def _from_flow(flow: dict, nodes: list, kind: str):
    """The value a workflow's own node holds for one kind of mapped input (the model, width...)."""
    n = next((n for n in nodes if n.get("type") == kind and n.get("node_ids")), None)
    return flow[n["node_ids"][0]]["inputs"].get(n.get("key") or kind) if n else None


def settings(url: str, checkpoint: str = "", own: dict | None = None) -> dict:
    """Open WebUI's image settings (the names its API and environment use) for making and editing
    pictures with the ComfyUI at `url`: GPU Shepherd's checkpoint workflows, or the user's own (`own`,
    from custom()). Without a model the user picks one in Open WebUI (Admin Panel, Settings, Images
    lists ComfyUI's)."""
    own = own or {}
    # the prompt goes to ComfyUI as written: Open WebUI would otherwise have the chat model rewrite it first
    out = {"ENABLE_IMAGE_GENERATION": True, "IMAGE_GENERATION_ENGINE": "comfyui", "COMFYUI_BASE_URL": url,
           "ENABLE_IMAGE_PROMPT_GENERATION": False}
    if "generate" in own:
        flow, nodes = own["generate"]
        w, h = _from_flow(flow, nodes, "width"), _from_flow(flow, nodes, "height")
        out |= {"COMFYUI_WORKFLOW": json.dumps(flow, separators=(",", ":")), "COMFYUI_WORKFLOW_NODES": nodes,
                "IMAGE_SIZE": f"{w}x{h}" if w and h else "1024x1024",
                "IMAGE_STEPS": int(_from_flow(flow, nodes, "steps") or 25)}
        model = _from_flow(flow, nodes, "model") or ""
    else:
        out |= {"COMFYUI_WORKFLOW": json.dumps(GENERATE, separators=(",", ":")), "COMFYUI_WORKFLOW_NODES": GENERATE_NODES,
                "IMAGE_SIZE": size(checkpoint), "IMAGE_STEPS": 25}
        model = checkpoint
    if model:
        out["IMAGE_GENERATION_MODEL"] = model
    if "edit" in own or "generate" not in own:
        flow, nodes = own.get("edit", (EDIT, EDIT_NODES))
        out |= {"ENABLE_IMAGE_EDIT": True, "IMAGE_EDIT_ENGINE": "comfyui", "IMAGES_EDIT_COMFYUI_BASE_URL": url,
                "IMAGES_EDIT_COMFYUI_WORKFLOW": json.dumps(flow, separators=(",", ":")),
                "IMAGES_EDIT_COMFYUI_WORKFLOW_NODES": nodes}
        edit_model = (_from_flow(flow, nodes, "model") or "") if "edit" in own else checkpoint
        if edit_model:
            out["IMAGE_EDIT_MODEL"] = edit_model
    else:                                 # the built-in edit workflow wouldn't load the user's model
        out["ENABLE_IMAGE_EDIT"] = False
    return out


def openwebui_env(url: str, checkpoint: str = "", own: dict | None = None) -> dict[str, str]:
    """settings() as Open WebUI's environment variables."""
    return {k: (str(v).lower() if isinstance(v, bool) else json.dumps(v, separators=(",", ":"))
                if isinstance(v, list) else str(v))
            for k, v in settings(url, checkpoint, own).items()}


def from_env(env: dict[str, str]) -> dict:
    """The image settings in an Open WebUI's environment, typed as its API wants them."""
    keys = settings("x", "x") | {"ENABLE_IMAGE_EDIT": True}
    out: dict = {}
    for k, v in env.items():
        if k not in keys:
            continue
        if isinstance(keys[k], bool):
            out[k] = v.strip().lower() == "true"
        elif isinstance(keys[k], int):
            out[k] = int(v) if v.strip().isdigit() else keys[k]
        elif isinstance(keys[k], list):
            try:
                out[k] = json.loads(v)
            except ValueError:
                out[k] = []
        else:
            out[k] = v
    return out
