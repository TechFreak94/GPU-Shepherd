# Preset: zimage-2x8gb

Zach's setup, which GPU Shepherd grew out of: two 8 GB GPUs (RTX 3070) and 64-128 GB of RAM, used
for Z-Image Turbo coloring-book pages, local chat and coding models, and a home-lab assistant.

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

## Coloring books

- `appdata/comfyui/COLORING_BOOK_SETTINGS.md`: the Z-Image template, the two fp8 dropdowns, canvas
  size (1024x1328), the LoRA, a prompt skeleton and complexity dials.
- `appdata/comfyui/tools/kdp-book.sh <folder of pages> book.pdf [--blank-backs]`: turns pages into a
  print-ready 8.5x11 in, 300 DPI PDF for KDP. It runs `kdp_compile.py` in a throwaway Python
  container, so nothing has to be installed on the machine.
