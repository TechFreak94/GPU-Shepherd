# Example: the "aistack" setup

Built for a box with two 8 GB GPUs (RTX 3070) and 64-128 GB of RAM:

| Instance | GPUs | What for |
|---|---|---|
| ComfyUI | GPU 0, always (priority 80) | images and video |
| LLM server (llama-swap) | elastic: both, or GPU 1 while ComfyUI runs | `chat` = Qwen3.6 35B-A3B, `coder` = Qwen3 Coder Next on 2 GPUs (Qwen3.6 on 1) |
| Hermes | none | home lab / home automation assistant, uses `chat` |
| Open WebUI | none | chat with the models and Hermes, images through ComfyUI |

`gpu-shepherd --demo` starts with these files. `sudo ./install.sh` offers to load them on the first run:
it fills in your models and data folders, generates the API keys and the Hermes dashboard password,
and prints which model files to download. To load them by hand, copy `instances/`, `catalogs/` and
`appdata/` into your data folder, then replace `/srv/models`, `/opt/gpu-shepherd` and the
`change-me` / `REPLACE_WITH_LLM_API_KEY` values (the key is on the LLM server's Models page).
