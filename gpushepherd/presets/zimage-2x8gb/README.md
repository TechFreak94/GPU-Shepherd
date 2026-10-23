# Preset: zimage-2x8gb

Zach's setup, which GPU Shepherd grew out of: two 8 GB GPUs (RTX 3070) and 64-128 GB of RAM, used
for Z-Image Turbo coloring-book pages, local chat and coding models, and a home-lab assistant.

`sudo ./install.sh --preset zimage-2x8gb` creates it; the download step can fetch only some groups
(`gpushepherd/presets/download-models.sh zimage-2x8gb` lists them).

| Instance | GPUs | What for |
|---|---|---|
| ComfyUI | GPU 0, always (priority 80) | Z-Image Turbo, coloring-book LoRA, upscaler |
| ComfyUI (GPU 1) | GPU 1 (priority 40) | a second image job at once; starting it stops the LLM server |
| LLM server (llama-swap) | elastic: both, or GPU 1 while ComfyUI runs | `chat` = Qwen3.6 35B-A3B, `coder` = Qwen3 Coder Next on 2 GPUs (Qwen3.6 on 1) |
| Hermes | none | home lab / home automation assistant, uses `chat` |
| Open WebUI | none | chat with the models and Hermes, images through ComfyUI |

Both ComfyUIs share the models and custom nodes; each keeps its own output folder
(`appdata/comfyui/output` and `appdata/comfyui-2/output`). `gpu-shepherd --demo --demo-preset
zimage-2x8gb` shows it with fake GPUs.

## Downloads (about 84 GB)

`chat`, `coder` (the two GGUFs), `zimage` (Kijai's fp8 Z-Image Turbo, the fp8 Qwen3 4B text encoder,
the VAE), `lora` (coloring book, trigger words `Coloring Book, ColoringBookAF`) and `upscaler`
(RealESRGAN x4plus anime 6B). `zimage-bf16` (the official full-precision files) is optional.

## Z-Image in ComfyUI

Workflow > Browse Templates > "Z-Image-Turbo: Text to Image". With the fp8 build, pick
`z-image-turbo_fp8_scaled_e4m3fn_KJ` (Load Diffusion Model) and `qwen_3_4b_fp8_mixed` (Load CLIP).
Keep 8 steps and CFG 1; 1024x1328 is about 8.5:11. For the LoRA, add LoraLoaderModelOnly after
Load Diffusion Model with `zimage/ColoringBook_Redmond_ZImage.safetensors`.
