# Instance and model options

Every option you can set in the editors, what it does, and the key it is saved under. The App
Store's automatic setup fills most of these in for you; open an instance's **Edit** page, or
choose *Manual configuration* (or *Customize every field*) while installing one, to change them.

Instances are saved to `instances/<id>.yaml` in the data folder, model servers' model lists to
`catalogs/<id>.yaml`. Changes take effect the next time the instance starts or restarts.

- [Instances](#instances): basics, GPUs, network and storage, command, behaviour, before stopping
- [Model servers](#model-servers): models, GPU and system RAM, auto names, server settings
- [Settings](#settings)
- [What App Store apps fill in](#what-app-store-apps-fill-in)

## Instances

### Basics

| Field | YAML | What it does |
|---|---|---|
| Name | `name` | Shown on the dashboard. |
| ID | `id` | Lowercase letters, digits and dashes, at most 40. Names the file and the container (`shepherd-<id>`), and is the host name other instances use to reach it (`http://<id>:<port>`). Can't be changed after creation. |
| Image | `image` | Any Docker image, e.g. `ghcr.io/org/app:tag`. Pulled when the instance starts. |
| Description | `description` | Free text. |
| - | `kind` | `container` for any app, `llama-swap` for a model server (adds the Models page). Set by the app or the *Model server, by hand* link. |
| - | `app` | The App Store app it was installed from (`llm-server`, `hermes`...), for the store's *installed* count and the tag on its dashboard card (*Model Server*, *Chat UI*, *Image UI*, *Agent UI*; *App* for anything else). Changing it needs no restart. |

### GPUs

| Field | YAML | What it does |
|---|---|---|
| GPU use | `gpu.mode` | `none`: no GPU. `fixed`: always exactly the GPUs listed. `elastic`: needs at least *Minimum GPUs*, takes every free GPU up to *Maximum GPUs*, and gives them back when something else needs one. |
| Use GPUs / Preferred GPUs | `gpu.gpus` | Fixed: the GPUs it always uses (by `nvidia-smi` index). Elastic: optional, the GPUs it takes first when it has a choice. |
| Minimum GPUs | `gpu.min_gpus` | Elastic only. Fewer than this and it can't start. |
| Maximum GPUs | `gpu.max_gpus` | Elastic only. `0` = no limit. `1` makes "one GPU, whichever is free". |
| Priority (0-100) | `gpu.priority` | When there aren't enough GPUs, higher priority keeps theirs and lower ones are moved or stopped (after you confirm). The instance you are starting always wins. |

Rule of thumb: apps that use one GPU (ComfyUI, speech models) are fixed or elastic with a maximum
of 1; LLM servers that split a model across GPUs are elastic with no maximum. The
[README](../README.md#how-gpu-sharing-works) walks through how a plan is made.

A container only sees the GPUs it was given. When its GPUs change it is re-created, so anything it
needs to keep must be in a mounted folder.

### Network and storage

| Field | YAML | What it does |
|---|---|---|
| Ports | `ports` | `HOST:CONTAINER`, one per line, `/udp` for UDP. Published on the address from Settings. A host port can belong to one instance only. |
| Folders | `volumes` | `/host/path:/container/path`, one per line, `:ro` for read-only. The host folder must exist. Model servers expect their models at `/models`. |

Every instance also joins the Docker network from Settings (`gpushepherd` by default), so
instances talk to each other by ID without publishing ports.

### Command and environment

| Field | YAML | What it does |
|---|---|---|
| Command | `command` | Replaces the image's default command. Quoted like a shell line; saved as a list. Empty = the image's default. |
| Environment | `env` | `NAME=value`, one per line. |

For model servers GPU Shepherd adds the llama-swap config and the API key itself; don't set a
command there.

### Behaviour

| Field | YAML | What it does |
|---|---|---|
| Start automatically | `autostart` | Started when GPU Shepherd starts, if the GPUs allow it. Instances created by templates and presets start with this off. |
| "Open" button port | `open_port` | Host port the Open button links to. Empty = the first port. |
| "Open" button path | `open_path` | Path after the address, e.g. `/ui`. |
| Shared memory | `shm_size` | Size of `/dev/shm`, e.g. `1g`. PyTorch apps with several workers often need more than Docker's 64 MB. |
| Run as user | `user` | `UID:GID` inside the container, e.g. `1000:1000`, so files it writes to your folders are yours. |
| Memory limit | `memory_limit` | Caps the container's RAM, e.g. `16g`. Empty = no limit. |

The last three are under *Advanced* in the editor.

### Before stopping

When a GPU swap, a restart or Stop would stop a running instance, GPU Shepherd first waits for the
work it is doing to finish, so a chat answer or an image isn't cut off half way. While it waits,
nothing has been stopped yet: the dashboard shows what it is waiting for and lets you go ahead at
once or call the change off.

| Field | YAML | What it does |
|---|---|---|
| Wait for running work | `drain.check` | How it tells whether the instance is busy (below). |
| Wait at most | `drain.timeout` | Seconds, default `600`. `0` = don't wait. Long video or batch jobs may need more. |
| If it is still busy then | `drain.on_timeout` | `stop`: stop it anyway. `cancel`: call the whole change off and leave everything as it was. |
| Container port to ask | `drain.port` | Which port inside the container to ask. Default: 8080 for llama-swap and llama.cpp, 8188 for ComfyUI, else the first port. |
| Path | `drain.path` | Custom check only: the URL path to ask, starting with `/`. |
| JSON field | `drain.key` | Custom check only: where the amount of work is in the answer. Dots go deeper (`queue.running`); empty = the whole answer. |

The checks:

| `check` | Busy means |
|---|---|
| `auto` (default) | Picks one of the checks below: `llama-swap` for model servers and llama-swap images, `comfyui` for images with "comfy" in the name, `llama-server` for llama.cpp images. Anything else isn't waited for. |
| `llama-swap` | llama-swap is answering requests (from its `/api/events`; on older versions, each loaded model's `/slots`). |
| `llama-server` | a llama.cpp server `/slots` entry is processing. |
| `comfyui` | prompts are running or queued (`/queue`). |
| `http` | `GET <path>` answers with a non-zero number, a non-empty list, or `true` at `<key>`. Use this for any app with a status endpoint. |
| `none` | never wait. |

It asks every 2 seconds. A check that gets no answer doesn't hold the change up: it is logged and
the stop goes ahead. GPU Shepherd isn't in the path of the requests, so it can't turn new work away
while it waits; work that arrives meanwhile is waited for too, up to the same timeout.

Changing these settings never asks for a restart.

Example, for an app whose `/api/status` answers `{"jobs": {"active": 2}}`:

```yaml
drain:
  check: http
  port: 7860
  path: /api/status
  key: jobs.active
  timeout: 1800
  on_timeout: cancel
```

## Model servers

A model server is a [llama-swap](https://github.com/mostlygeek/llama-swap) instance. It serves an
OpenAI-compatible API at `http://<server>:<port>/v1` and loads whichever model a client asks for,
unloading the previous one. Its **Models** page holds the model list. GPU Shepherd writes the
llama-swap config from that list, for the number of GPUs the server has each time it starts.

### Models

| Field | YAML (`models[]`) | What it does |
|---|---|---|
| Path inside the container | `file` | The .gguf file, e.g. `/models/model-Q4_K_M.gguf`. For split models, the first part (`…-00001-of-0000N.gguf`). The editor lists the files it can see. |
| Model ID | `id` | The name clients ask for. Letters, digits, `.`, `_`, `:` and `-`. |
| Display name, Description | `name`, `description` | Shown in the UI. |
| Other names | `aliases` | More names for the same model. Each name can be used once per server. |
| Context | `ctx_size` | Tokens, shared by all slots (`--ctx-size`). |
| Parallel slots | `parallel` | Requests served at once; each gets context ÷ slots (`--parallel`). |
| KV cache | `kv_cache` | `f16` (default), `q8_0` (half the memory) or `q4_0` (smallest) (`--cache-type-k/-v`). |
| Sampling flags | `sampling` | e.g. `--temp 0.6 --top-p 0.95 --top-k 20`. Use the model card's values. |
| Extra llama-server flags | `extra_args` | Anything else for `llama-server`, e.g. `--no-mmap`. |
| Vision projector | `mmproj` | For vision models, the `mmproj` file. It is kept in system RAM so the GPUs hold more of the model. |

Every model is started with `--jinja` (the model's own chat template).

### GPU and system RAM

For models bigger than the GPUs' memory, part of the model can stay in system RAM. A server can
have a different number of GPUs each time it starts, so this is set per GPU count (`offload`):

| Field | YAML (`offload[]`) | llama.cpp flag |
|---|---|---|
| Server has N GPUs | `min_gpus` | - |
| Layers on GPU | `gpu_layers` | `--n-gpu-layers`; empty = llama.cpp's default |
| MoE expert layers in RAM | `cpu_moe` | `--n-cpu-moe`; `999` = all of them |

A row covers its GPU count and every larger count until the next row. With no row that applies,
the whole model goes on the GPUs.

- **Mixture-of-experts models** (names like `35B-A3B`): keep every layer on the GPU and move expert
  layers to RAM. Raise `cpu_moe` until the model loads, then lower it for speed.
- **Dense models**: lower `gpu_layers` instead.

Example: 30 expert layers in RAM with one GPU, 14 with two or more.

```yaml
offload:
  - {min_gpus: 1, cpu_moe: 30}
  - {min_gpus: 2, cpu_moe: 14}
```

Keeping a model in RAM needs room for it plus the OS and your other containers: below 32 GB of RAM,
plan on models that fit in GPU memory.

### Auto names

An auto name (`auto_aliases`) is a model name that points at different models depending on how
many GPUs the server has right now. Point a client at it and it always gets the model that fits.

| Field | YAML | What it does |
|---|---|---|
| Name | `alias` | The name clients ask for, e.g. `coder`. |
| Default model | `default_model` | Used when no rule applies. |
| Rules | `rules` | `GPUs=model`, comma separated: `2=big-model` means "with 2 or more GPUs, use big-model". The rule with the highest count that is reached wins. |

```yaml
auto_aliases:
  - alias: coder
    default_model: small-coder
    rules: [{min_gpus: 2, model: big-coder}]
```

### Server settings

| Field | YAML | What it does |
|---|---|---|
| CPU threads | `threads` | `--threads` for every model. `0` = the number of physical cores, which is best when models spill into RAM. |
| Load timeout | `health_timeout` | Seconds llama-swap waits for a model to load before giving up. Raise it for very large models on slow disks. |
| llama-server path | `llama_server` | Where `llama-server` is inside the image. Empty = found on first start. |
| API key | `api_key` | Clients send it as `Authorization: Bearer <key>`. *Generate a new key* replaces it; update your clients afterwards. |

The bottom of the Models page shows the generated llama-swap config.

## Settings

Saved in `settings.yaml` (mode 600, it holds the password hash).

| Field | YAML | What it does |
|---|---|---|
| Password | `password_hash` | Change it on the Settings page. Lost it? See [CLI and scripts](cli.md#reset-the-password). |
| Publish ports on | `bind_address` | `0.0.0.0` = your whole network, `127.0.0.1` = this machine only. |
| Docker network | `network` | The network every instance joins. |
| Theme | - | System (follows your device), Light or Dark. Saved per browser in a `theme` cookie, not in `settings.yaml`. The theme button at the top of every page cycles through them. |
| - | `admin_user` | The login name, `admin` by default. YAML only. |
| - | `container_prefix` | Prefix for container names, `shepherd-` by default. YAML only; change it only with every instance stopped. |

## What App Store apps fill in

Each app in the App Store can be installed two ways. **Install** (automatic setup) asks only what
it can't work out. **Manual** opens the full form, filled in with the same defaults, so you can
change anything before it is saved. Either way the result is an ordinary instance (plus, for model
servers, the models you picked), so every field above can be changed afterwards.

| Template | GPUs | Fills in |
|---|---|---|
| LLM server | every free GPU (elastic, no maximum) | llama-swap image, models folder at `/models` (read-only), and the models you pick: a chat model named `chat` (32K context) and a coding model named `coder` (64K context, lower temperature). Either is optional, and one file can be both. |
| ComfyUI | one free GPU (elastic, maximum 1), priority 60 so it outranks model servers | models from the models folder's `comfyui/` subfolder; `custom_nodes`, `input`, `output`, `user` and cache in its app data folder |
| Open WebUI | none | connection to a model server you pick (address and API key), and image generation through a ComfyUI instance if you pick one |
| Hermes Agent | none | `gateway run` with its dashboard on port 9119 behind a user name and password you choose; the model server it thinks with (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `HERMES_MODEL`, plus a first `config.yaml` in its data folder); optionally Home Assistant (`HASS_URL`, `HASS_TOKEN`). Its data folder is made for the user Hermes runs as (10000). |

Open WebUI and Hermes Agent can also install a new LLM server alongside themselves
(*A new LLM server, installed with it*), with a chat model you pick from the models folder.
It is offered first when there is no model server yet, and the install page says so, with an
*Install an LLM server first* button: that installs the LLM server and then comes back to the app,
already connected to it.

Automatic setup also picks a free ID and port. The GPU question can pin an instance to one GPU instead (a fixed
policy).

For the models, *Decide from the file size* keeps the model on the GPUs for each GPU count
where it fits (file size plus 20% for the KV cache and buffers, against the smallest GPUs' combined
memory) and moves all MoE expert layers to RAM where it doesn't. Fine-tune it afterwards on the
model's edit page.

*Any Docker image* skips the questions and opens an empty form, for apps the App Store doesn't list.
