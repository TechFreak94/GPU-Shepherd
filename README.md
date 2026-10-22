# GPU Shepherd

A small web dashboard that runs your self-hosted AI containers and **shares the GPUs between them
automatically**. Start ComfyUI and your LLM server moves to the other GPU; stop it and the LLM
gets both. Everything is configured from the browser - no compose files to edit.

![Dashboard](docs/screenshots/dashboard.png)

> Status: early (0.1). It works on one Docker host with NVIDIA GPUs. Expect rough edges and
> breaking changes to the config format before 1.0.

## Why

Homelab GPUs are shared by tools that each assume they own the card: image generators, LLM
servers, speech models. Generic container managers (Dockge, Portainer) can start and stop them,
but they don't know about GPUs, so you end up stopping one thing by hand to start another, or
editing `CUDA_VISIBLE_DEVICES` and restarting. GPU Shepherd treats GPUs as a shared resource:
each instance says what it needs, and starting or stopping anything re-plans who gets which GPU.

## Features

- **Instances**: any Docker image - ports, folders, environment, command, autostart - edited in
  forms. One card per instance with Start / Stop / Restart / Open / Logs.
- **GPU policies** per instance:
  - *Fixed* - always these GPUs (e.g. ComfyUI on GPU 0).
  - *Elastic* - at least N GPUs, grows into every free GPU, shrinks when something else needs one
    (e.g. a llama.cpp server that splits a model across GPUs).
  - *Priority* - who keeps their GPUs when there are not enough to go around.
- **Plans you confirm**: if starting something means moving or stopping something else, you see
  exactly what will happen first.

  ![Confirmation](docs/screenshots/confirm.png)
- **Model servers** ([llama-swap](https://github.com/mostlygeek/llama-swap)): a model catalogue
  editor (pick .gguf files from disk, context, sampling, KV cache, vision projector), load and
  unload buttons, and **auto names** - a model name like `auto` that points at different models
  depending on how many GPUs the server currently has. Point an agent at `auto` and it always gets
  the model that fits.
- **GPU + system RAM**: per model, keep part of it in system RAM (MoE expert layers or whole
  layers) so models larger than your GPUs still run. Set it per GPU count: e.g. 30 expert layers
  in RAM when the server has one GPU, 14 when it has two.

  ![Models](docs/screenshots/models.png)
- **Plain files**: everything you change is saved as readable YAML - easy to back up, diff, or
  keep in git.
- **Small**: one Python process, server-rendered pages with htmx, no JavaScript build step.

## Quick start

### Ubuntu: guided install

On Ubuntu 26.04 (24.04 works too) with an NVIDIA GPU:

```bash
git clone <this repo> gpu-shepherd && cd gpu-shepherd
sudo ./install.sh
```

The installer first shows what the machine already has (driver, GPUs, Docker, NVIDIA Container
Toolkit, RAM). Then it asks a few questions, each with a default you can accept with Enter: where
your .gguf model files are, which port to use, and where to keep settings. If the NVIDIA driver is
missing it offers to install it (that needs a reboot; run the installer again afterwards). On the
first run it offers to load [the example setup](gpushepherd/examples/aistack) (a ComfyUI per GPU
with Z-Image Turbo, an LLM server with `chat` and `coder` models, Hermes and Open WebUI), filled in
with your folders and fresh API keys and passwords; nothing in it starts until you press Start. It
also offers to download the example's models (about 83 GB, resumable) once everything else is done. Finally it shows the plan, and
nothing is installed or changed until you confirm. Then it installs what is missing, checks that a
container can see the GPUs, adds you to the `docker` group (so `docker` works without sudo from your
next login; `--no-docker-group` skips this), starts GPU Shepherd and prints the address, the first
login, and the example's addresses. It is safe to re-run: it only does what is still missing, and keeps an existing
`compose.yaml`.

For scripted or advanced installs, every question has a flag, and `--yes` skips the rest:

```bash
sudo ./install.sh --yes --models-dir /srv/models --port 8090 --data-dir /opt/gpu-shepherd
```

`./install.sh --help` lists them all (`--example`, `--download-models`, `--install-driver`, `--skip-deps`, `--no-start`, ...).

RAM: with less than 32 GB installed, models have to fit in GPU memory, because keeping part of a
model in system RAM needs room for it plus the OS and your other containers. 64 GB is recommended
for the largest models (Qwen3 Coder Next is ~49 GB at Q4).

### By hand

Requirements: Linux, Docker with the Compose plugin, NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone <this repo> gpu-shepherd && cd gpu-shepherd
sudo mkdir -p /opt/gpu-shepherd
cp compose.example.yaml compose.yaml      # read the comments; adjust paths if you like
docker compose up -d --build
docker compose logs gpu-shepherd | grep "First start"   # the generated admin password
```

Open `http://<server>:8090`, log in as `admin`, then **Add instance** or **Add model server**.

The data folder must be mounted at the **same path** inside and outside the container (like
Dockge's stacks folder), because GPU Shepherd hands paths in it to Docker for the containers it
creates. To pick model files in the UI, mount your model folders the same way (read-only is fine).

### Try it without Docker or GPUs

```bash
pip install .
gpu-shepherd --demo --port 8090      # fake GPUs and containers, log in as admin / demo-password
```

The demo loads [the example stack](gpushepherd/examples/aistack), built for two 8 GB GPUs and
64-128 GB of RAM: ComfyUI pinned to GPU 0 (and a second one for GPU 1), an elastic llama-swap server with a `chat` and a `coder`
model, Hermes (home lab assistant) and Open WebUI. The installer offers to load it for real.

## How GPU sharing works

When you start, stop or restart an instance, GPU Shepherd plans the whole machine again:

1. The instance you are starting gets what it needs first.
2. Everyone else is served by priority. Fixed instances need exactly their GPUs; elastic ones
   only say *how many* they need, so they move aside instead of pushing a fixed one out.
3. Leftover GPUs go to elastic instances, highest priority first, up to their maximum.
4. Running instances keep the GPUs they already have when possible, to avoid needless restarts.
5. Anything whose GPUs change is restarted with the new ones; anything that no longer fits is
   stopped. You confirm before either happens.

Containers only ever see the GPUs they were given (Docker device requests), so applications need
no `CUDA_VISIBLE_DEVICES` tricks. Moving an instance means re-creating its container; its data
lives in the folders you mounted, so nothing is lost.

Example with two GPUs - ComfyUI *fixed GPU 0, priority 80*, LLM server *elastic 1+, priority 50*:

| Action | ComfyUI | LLM server |
|---|---|---|
| Start the LLM | - | GPUs 0,1 |
| Start ComfyUI | GPU 0 | restarted on GPU 1 |
| Stop ComfyUI | stopped | restarted on GPUs 0,1 |

With a single GPU the same settings make them take turns.

The example adds a second ComfyUI *fixed GPU 1, priority 40*, for two image jobs at once. Starting
it stops the LLM server (you are asked first); starting the LLM again stops it, since the LLM has the
higher priority.

## Coding agents

Coding tools (OpenCode, Aider, Continue, OpenHands, ...) don't need a GPU of their own: point them at
the model server's OpenAI-compatible endpoint and let GPU Shepherd decide what runs there. The
example stack has an auto name `coder` for this:

| Model server has | `coder` is | Memory |
|---|---|---|
| 1 GPU (ComfyUI has the other) | Qwen3.6 35B-A3B, 30 expert layers in RAM | ~22 GB file, mostly RAM |
| 2 GPUs | Qwen3 Coder Next 80B-A3B, 38 expert layers in RAM | ~49 GB file, ~40 GB of it in RAM |

So 64 GB of system RAM is enough for the larger one. For example with OpenCode or Aider:

```bash
export OPENAI_API_BASE=http://<server>:8080/v1
export OPENAI_API_KEY=<the model server's API key, shown on its Models page>
aider --model openai/coder
```

The expert-layer numbers are starting points for 8 GB cards: if a model fails to load, raise them on
its edit page (GPU and system RAM); if there is GPU memory left over, lower them for more speed.

## Configuration files

```
/opt/gpu-shepherd/
  settings.yaml          login, Docker network, publish address      (mode 600)
  instances/<id>.yaml    one per instance
  catalogs/<id>.yaml     model catalogue + API key per model server  (mode 600)
  generated/<id>/        files rendered for containers - do not edit
```

You can edit the YAML by hand; GPU Shepherd re-reads it on every page load and shows a warning
for files it cannot parse. Saving from the UI rewrites the file (comments are not kept) and leaves
the previous version next to it as `<file>.prev`.

## Security

GPU Shepherd controls Docker through its socket, which is root-equivalent on the host. It has a
login (scrypt-hashed password, `SameSite=strict` session cookie), refuses state-changing requests
that don't come from its own pages, and sends a strict Content-Security-Policy. Still: keep it on
a trusted network or behind a VPN/reverse proxy with authentication. Do not expose it to the
internet. The apps it starts (ComfyUI, for one) may have no login of their own.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # planner, config, llama-swap rendering, UI flows, Docker payloads
gpu-shepherd --demo                      # fake GPUs/containers, admin / demo-password
```

See [docs/architecture.md](docs/architecture.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- More model-server connectors: Ollama, vLLM, SGLang.
- Shared GPUs: let several small instances share one GPU with memory budgets.
- Import an existing `docker compose` project as instances.
- Scheduled actions (e.g. give the LLM every GPU overnight).
- Live log streaming, Prometheus metrics, multiple users.
- AMD (ROCm) and multi-host support.

## License

Apache-2.0 - see [LICENSE](LICENSE). Bundles [htmx](https://htmx.org) (Zero-Clause BSD,
`gpushepherd/static/htmx.LICENSE.txt`).
