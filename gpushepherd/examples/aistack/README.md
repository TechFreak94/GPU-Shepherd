# Example: the "aistack" setup

Built for a box with two 8 GB GPUs (RTX 3070) and 64-128 GB of RAM:

| Instance | GPUs | What for |
|---|---|---|
| ComfyUI | GPU 0, always (priority 80) | images and video, Z-Image Turbo |
| ComfyUI (GPU 1) | GPU 1 (priority 40), off by default | a second image job at once; starting it stops the LLM server |
| LLM server (llama-swap) | elastic: both, or GPU 1 while ComfyUI runs | `chat` = Qwen3.6 35B-A3B, `coder` = Qwen3 Coder Next on 2 GPUs (Qwen3.6 on 1) |
| Hermes | none | home lab / home automation assistant, uses `chat` |
| Open WebUI | none | chat with the models and Hermes, images through ComfyUI |

`gpu-shepherd --demo` starts with these files. `sudo ./install.sh` offers to load them on the first run:
it fills in your models and data folders, generates the API keys and the Hermes dashboard password,
and offers to download the models with `download-models.sh`. To load them by hand, copy `instances/`, `catalogs/` and
`appdata/` into your data folder, then replace `/srv/models`, `/opt/gpu-shepherd` and the
`change-me` / `REPLACE_WITH_LLM_API_KEY` values (the key is on the LLM server's Models page).

## Models

`./download-models.sh /srv/models [groups]` downloads them (resumable, skips what you have):
the two LLMs (`chat`, `coder`), Z-Image Turbo for ComfyUI (`zimage`: Kijai's fp8 build, the fp8
Qwen3 4B text encoder and the VAE; `zimage-bf16` for the full-precision files), the coloring-book
LoRA (`lora`, trigger words `Coloring Book, ColoringBookAF`) and an anime upscaler (`upscaler`).
Run it with `--help` for sizes.

Z-Image runs on ComfyUI's own nodes, no custom nodes needed: Workflow > Browse Templates >
"Z-Image-Turbo: Text to Image", pick the fp8 files in the loaders, and start at 1024x1328,
8 steps, CFG 1. Both ComfyUIs share the models and custom nodes; each keeps its own output folder
(`appdata/comfyui/output` and `appdata/comfyui-2/output`).
