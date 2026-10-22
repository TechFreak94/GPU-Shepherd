# Presets

A preset is a ready-made set of instances that `install.sh` can create on the first run, so a new
machine goes from nothing to a working setup in one go. The installer lists the presets in this
folder and asks which one to use (or none); `--preset NAME` picks one up front, and
`--preset /path/to/folder` uses a preset of your own. `./install.sh --list-presets` lists them.

| Preset | For | What it creates |
|---|---|---|
| [`starter`](starter) | any machine with one or more 8+ GB NVIDIA GPUs | LLM server (Qwen3 8B), ComfyUI, Open WebUI |
| [`zimage-2x8gb`](zimage-2x8gb) | 2x 8 GB GPUs, 64+ GB RAM | a ComfyUI per GPU with Z-Image Turbo, coloring-book LoRA and KDP helper; LLM server with chat and coding models; Hermes; Open WebUI |

Nothing in a preset starts by itself: the installer turns autostart off until you have tried it.

## Making your own

Copy a preset folder and change it. Everything except `instances/` is optional:

```
my-preset/
  preset.yaml       name, description, needs, download_gb - one "key: value" per line, shown by the installer
  instances/*.yaml  one file per instance, the same format GPU Shepherd saves (Instances page > Edit)
  catalogs/*.yaml   model lists for llama-swap instances (Models page)
  appdata/...       starting files for the instances' folders: configs, scripts, notes
  downloads.txt     model files to download (see below)
  NOTES.txt         printed at the end of the install; HOST becomes this machine's address
```

When the installer creates a preset it fills in these placeholders, in every text file:

| Placeholder | Becomes |
|---|---|
| `/srv/models` | the models folder chosen during the install |
| `/opt/gpu-shepherd` | the settings folder |
| `sk-demo-000...` and `REPLACE_WITH_LLM_API_KEY` | a new API key for the LLM server (the same everywhere) |
| `change-me-hermes-api-key`, `change-me-please` | a new API key and a new password |
| `HERMES_UID/GID: "1000"` | the user who ran the installer |
| `autostart: true` | `autostart: false` |

It also creates every host folder the instances mount (inside the settings and models folders),
owned by you. Files that already exist are never overwritten.

The easiest way to make a preset from a machine you have already set up is to copy its
`/opt/gpu-shepherd/instances` and `catalogs` folders, then put the placeholders back in.

## downloads.txt

```
# group <name> <default|optional> <size> <description>
group chat  default  ~5GB  the chat model
# file <group> <path under the models folder> <url> [fallback url ...]
file chat  Qwen3-8B-UD-Q4_K_XL.gguf  https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-UD-Q4_K_XL.gguf
```

The installer offers to download the default groups at the end of the install. You can also run
`download-models.sh` yourself at any time: with only a preset it lists the groups, with a models
folder it downloads the default groups, and you can name groups to download only those:

```bash
gpushepherd/presets/download-models.sh zimage-2x8gb                       # list the groups
gpushepherd/presets/download-models.sh zimage-2x8gb /srv/models zimage lora
```

Downloads resume where they stopped and skip files you already have. Set `HF_TOKEN` for gated
Hugging Face models.
