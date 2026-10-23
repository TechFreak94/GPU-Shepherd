# Preset: starter

A general setup for any machine with one or more NVIDIA GPUs of 8 GB or more and 16+ GB of RAM.

| Instance | GPUs | What for |
|---|---|---|
| LLM server (llama-swap) | elastic: every GPU nobody else needs | `chat` (and `coder`) = Qwen3 8B, which fits one 8 GB GPU |
| ComfyUI | any one GPU, outranks the LLM server | images and video |
| Open WebUI | none | chat with the models, images through ComfyUI |

With two or more GPUs, ComfyUI and the LLM server run side by side. With one GPU they take turns:
starting one stops the other, after you confirm.

Downloads (about 12 GB): Qwen3 8B, and Stable Diffusion XL base 1.0 for ComfyUI. ComfyUI's default
workflow uses it as is: pick `sd_xl_base_1.0.safetensors` in Load Checkpoint and set 1024x1024.

Add bigger models on the LLM server's Models page once you know what your machine can run.
