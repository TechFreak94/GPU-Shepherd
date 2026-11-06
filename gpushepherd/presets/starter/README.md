# Preset: starter

A general setup for any machine with one or more NVIDIA GPUs of 8 GB or more and 16+ GB of RAM.

| Instance | GPUs | What for |
|---|---|---|
| Llama Swap (llama-swap) | elastic: every GPU nobody else needs | `chat` (and `coder`) = Qwen3.5 9B, which fits one 8 GB GPU and reads images; `embed` = Qwen3 Embedding 0.6B on the CPU |
| ComfyUI | any one GPU, outranks the Llama Swap | images and video |
| Open WebUI | none | chat with the models and your documents, images through ComfyUI |

With two or more GPUs, ComfyUI and the Llama Swap run side by side. With one GPU they take turns:
starting one stops the other, after you confirm.

Downloads (about 15 GB): Qwen3.5 9B with its vision projector, Qwen3 Embedding 0.6B, and Stable
Diffusion XL base 1.0 for ComfyUI. ComfyUI's default workflow uses it as is: pick `sd_xl_base_1.0.safetensors` in Load Checkpoint and set 1024x1024.

Add bigger models on the Llama Swap's Models page: its suggested models say which ones fit your GPUs
and RAM, and download with one click.
