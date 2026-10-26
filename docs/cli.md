# CLI and scripts

Everything GPU Shepherd does from the command line. Run the scripts from the folder you cloned.

- [`install.sh`](#installsh): install, or re-run to fill in what is missing
- [`update.sh`](#updatesh): update, roll back, restore settings, dashboard updates
- [`gpu-shepherd`](#gpu-shepherd): the app itself, and [commands](#commands) for everything the dashboard does
- [`download-models.sh`](#download-modelssh): a preset's model files
- [Environment variables](#environment-variables)

## install.sh

```bash
sudo ./install.sh                 # guided: shows what the machine has, asks, shows the plan, then installs
```

Without options it asks a few questions, each with a default (Enter accepts it), and changes
nothing until you confirm the plan. It is safe to re-run: it only does what is still missing and
keeps an existing `compose.yaml`. Every question also has an option, for scripted installs:

| Option | What it does |
|---|---|
| `--models-dir DIR` | Folder of .gguf files to pick from in the UI, mounted read-only. `""` = none. |
| `--port N` | Web UI port. Default `8090`. |
| `--data-dir DIR` | Where settings, instances and model lists are kept. Default `/opt/gpu-shepherd`. |
| `--install-driver` | Install the recommended NVIDIA driver if none works. Needs a reboot; run the installer again afterwards. |
| `--skip-deps` | Install nothing; only write `compose.yaml` and start GPU Shepherd. |
| `--preset NAME\|DIR` | On the first run, create a [preset](../gpushepherd/presets)'s instances: a folder name from `gpushepherd/presets`, or your own folder. |
| `--no-preset` | Don't offer a preset. |
| `--list-presets` | List the presets and exit. |
| `--download-models` | With a preset, download its models at the end (resumable). |
| `--no-download` | Don't offer the download. |
| `--no-docker-group` | Don't add you to the `docker` group (then `docker` needs sudo). |
| `--dashboard-updates` | Set up the service behind the dashboard's Update button (see [below](#updating-from-the-dashboard)). |
| `--no-dashboard-updates` | Don't offer it. |
| `--no-start` | Prepare everything but don't start GPU Shepherd. |
| `--update [OPTIONS]` | Run `./update.sh` with the options that follow, instead of installing. |
| `-y`, `--yes` | Ask nothing: defaults for everything no option sets. With `--yes`, a preset, the model download and dashboard updates only happen when their option is given. |
| `-h`, `--help` | List the options. |

Example:

```bash
sudo ./install.sh --yes --models-dir /srv/models --port 8090 --data-dir /opt/gpu-shepherd --preset starter
```

## update.sh

```bash
sudo ./update.sh                  # show what's new, ask, update the clone, rebuild and restart
```

Your settings, instances, model lists and password live in the data folder and are kept, and so is
`compose.yaml`. Instances keep running while GPU Shepherd restarts. The data folder and port are
read from `compose.yaml`.

| Option | What it does |
|---|---|
| `--check` | Only look. Exit status `10` means an update is available. Changes nothing. |
| `-y`, `--yes` | Don't ask before updating. |
| `--stash` | Set aside your own edits to files in the clone, and put them back afterwards. Without it, an update with local edits stops and lists them. |
| `--rebuild` | Rebuild and restart even if nothing new was downloaded. |
| `--rollback` | Go back to the version that ran before the last update. Uses the image saved then, so it needs no rebuild or network. |
| `--restore-settings [FILE]` | Put back the settings, instances and model lists from just before the last update (or from a backup `FILE`). The current ones are kept as `*.before-restore`. |
| `--set-token USER` | Save a GitHub user name and access token for updating from a private repository. Asks for the token without showing it (or reads it from standard input), then checks for updates. See [Private repository](updates.md#private-repository). |
| `--clear-token` | Forget that token. |
| `--enable-dashboard-updates` | Install the systemd service behind the dashboard's Update button. |
| `--disable-dashboard-updates` | Remove it. |
| `-q`, `--quiet` | Print less. |
| `-h`, `--help` | List the options. |

Updates only move forward along the branch you cloned, and never touch your models, app data or
the data folder beyond the backups below.

Installed before `update.sh` existed? Run `git pull && sudo ./update.sh --rebuild` once.

### When an update fails

Before every update, `update.sh` saves the running image as `gpu-shepherd:previous` and backs up
`settings.yaml`, `instances/` and `catalogs/` to `<data folder>/update/backups/` (the newest 10 are
kept). If the new version doesn't build, or doesn't answer within 90 seconds, it goes back by
itself: the clone returns to the old commit, the saved image starts again and the settings are
restored. The dashboard then says the update failed, with the log. By hand:

```bash
sudo ./update.sh --rollback             # previous version
sudo ./update.sh --restore-settings     # settings from just before the update
sudo docker compose logs gpu-shepherd   # why it failed
```

As a last resort, `git -C <clone> log` lists every version, and
`git -C <clone> reset --keep <commit>` followed by `sudo ./update.sh --rebuild` runs any of them.

### Updating from the dashboard

GPU Shepherd runs in a container and can't rebuild itself, so the dashboard's Update button is
handled by a small systemd service on the host (`--enable-dashboard-updates`). It checks once a
day; the dashboard then shows the changes and an Update button. The version number at the top of
every page opens the Updates window, with *Check for updates*, each step of a running update and
the changelog (see [Versions, changelog and updates](updates.md)). The two sides only exchange files in `<data folder>/update/`, so the container never
needs git, your credentials or more access to the host. Without the service, the dashboard shows
the command to run instead.

## gpu-shepherd

The app itself. Without a command it runs the web UI; with one it manages the GPU Shepherd that is
running, doing exactly what the same button on the dashboard does (both go through the
[JSON API](api.md)). In a normal install it runs in the `gpu-shepherd` container, so run it
through `docker exec`:

```bash
sudo docker exec gpu-shepherd gpu-shepherd status
sudo docker exec gpu-shepherd gpu-shepherd start comfyui
sudo docker exec gpu-shepherd gpu-shepherd --reset-password
```

A shorter way to type it: `alias gs='sudo docker exec -it gpu-shepherd gpu-shepherd'` (`-it` lets
it ask questions and follow a change as it runs). Options go before the command, and
`gpu-shepherd COMMAND -h` lists a command's own options.

| Option | What it does |
|---|---|
| `--host ADDR` | Address the web UI listens on. Default `0.0.0.0`. |
| `--port N` | Port. Default `8090`. Commands use it to find the app at `http://127.0.0.1:N`. |
| `--data-dir DIR` | Data folder. Default `/opt/gpu-shepherd`. Commands read the API token from it. |
| `--demo` | Fake GPUs and containers, no Docker needed; data goes to a temporary folder. Log in as `admin` / `demo-password`. The log prints the API token for commands. |
| `--demo-preset NAME` | The preset the demo starts with. Default `starter`. |
| `--reset-password` | Give the admin a new password, print it and exit. |
| `--url URL` | Commands: the GPU Shepherd to manage, e.g. `http://server:8090`. Default `http://127.0.0.1:<port>`, or `$SHEPHERD_URL`. |
| `--token TOKEN` | Commands: its API token. Default `$SHEPHERD_TOKEN`, or the `api-token` file in the data folder. |
| `--json` | Commands: print the API's JSON answer instead of text, for scripts. |
| `--version` | Print the version. |

### Commands

Everything on the dashboard, the App Store, the Models pages and Settings has a command. The
theme is the one exception: it is a choice saved in each browser.

| Command | What it does | On the dashboard |
|---|---|---|
| `status` | GPUs, CPU and RAM, every instance with its GPUs, and the change that is running. | Dashboard |
| `list` | Instances with their app, state, GPUs and ports. | Dashboard cards |
| `start ID`, `stop ID`, `restart ID` | Start, stop or restart, and follow the change's log. If it also stops or moves other instances, it shows the plan and asks; `-y` goes ahead without asking. `--no-wait` returns at once. A stop waits for running work like the dashboard does. | Start / Stop / Restart |
| `logs ID` | The container's log. | Logs |
| `show ID` | The instance's settings as YAML, the same as `instances/<id>.yaml`. | Edit |
| `set ID KEY=VALUE...` | Change settings, by the YAML keys in [Instance options](advanced.md#instances): `name=`, `image=`, `autostart=true`, `gpu.mode=elastic`, `gpu.gpus=0,1`, `gpu.priority=70`, `drain.timeout=300`, `open_port=`, ... `ports`, `volumes` and `env` take the editor's lines, separated by `;`: `ports=8188:8188;9000:9000/udp`, `volumes=/srv/x:/data:ro`, `env=A=1;B=2`. `command` takes a command line. An empty value clears an optional field. | Edit, Save |
| `apply FILE` | Create or replace an instance from a YAML or JSON file, such as `show`'s output (`-` reads standard input). `--id` gives the ID if the file has none. | Edit, *Any Docker image* |
| `delete ID` | Delete the instance and its container; folders on disk are kept. Asks first; `-y` doesn't. | Delete |
| `apps` | The App Store's apps and what is installed from each. | App Store |
| `apps show KEY` | An app, what it also installs and what is left to do afterwards, the answers its install form starts with, and the choices for each. | Install |
| `apps install KEY [KEY=VALUE...]` | Install an app. Only give what differs from `apps show`: `apps install comfyui gpu=pin:1 port=8189`, `apps install llm-server chat_file=Qwen3-8B-Q4_K_M.gguf`, `apps install open-webui server=new chat_file=...` (installs an LLM server with it), `apps install librechat` (and its database). | Install |
| `apps manual KEY [KEY=VALUE...]` | Print the instance an install would make, as YAML, without saving it: edit it and `apply` it. | Manual |
| `models list SERVER` | A model server's models, which are loaded, and its auto names. | Models |
| `models files SERVER` | The .gguf files the server can see. | Models › file list |
| `models show SERVER MODEL` | One model's settings. | Models › Edit |
| `models set SERVER MODEL KEY=VALUE...` | Add a model or change one, by the keys in [Models](advanced.md#models): `file=/models/x.gguf ctx_size=32768 kv_cache=q8_0 aliases=chat,coder sampling="--temp 0.6"`. `id=new-id` renames it (auto names follow). GPU and RAM: `offload.2.cpu_moe=999`, `offload.1.gpu_layers=30`, `offload.2=` removes the row for 2 GPUs. | Models › Add / Edit |
| `models delete SERVER MODEL` | Remove a model. | Models › Delete |
| `models load SERVER MODEL`, `models unload SERVER [MODEL]` | Load a model now; unload one, or all of them. Unloading refuses while the server is answering. | Load / Unload |
| `models auto-set SERVER ALIAS DEFAULT [N=MODEL...]` | Add or replace an auto name: `models auto-set llm coder small-coder 2=big-coder`. | Models › Auto names |
| `models auto-delete SERVER ALIAS` | Remove an auto name. | Models › Auto names |
| `models settings SERVER [KEY=VALUE...]` | Show or change `threads`, `health_timeout` and `llama_server`. `--new-api-key` gives the server a new API key. | Models › Server settings |
| `jobs` | Recent changes and how they ended. | Dashboard › last change |
| `jobs show N` | Follow a change's log. | Log |
| `jobs skip N`, `jobs cancel N` | While a change waits for running work: stop without waiting longer, or call it off. | Stop now / Cancel change |
| `settings`, `settings set KEY=VALUE...` | Show settings, or set `bind_address` and `network`. | Settings › Containers |
| `password` | Change the admin password (asks for the current and the new one). | Settings › Password |
| `token` | Print the API token. `--new` makes a new one; the old one stops working. Reads the data folder, so it works without the app running. | Settings › API and command line |
| `update` | Running version, and whether an update is waiting. | Version number |
| `update check`, `update install` | Ask the update service to check, or to update (asks first; `-y` doesn't). Needs [dashboard updates](#updating-from-the-dashboard); otherwise run `update.sh` on the host. | Check for updates / Update |
| `update log` | The log of the last check or update. | Update log |
| `update set-token USER`, `update clear-token` | Save or forget a GitHub token for a private repository (asks for it, or reads it from standard input). | Settings › Private repository |
| `changelog` | What changed in each version (`--limit N`, default 5). | Changelog |

Exit status: `0` when it worked, `1` when it didn't (the reason is printed), `130` when you pressed
Ctrl+C while following a change (the change carries on).

From another machine, give the address and the token from Settings:

```bash
export SHEPHERD_URL=http://server:8090 SHEPHERD_TOKEN=gs-...
gpu-shepherd status
```

### Reset the password

```bash
sudo docker exec gpu-shepherd gpu-shepherd --reset-password
```

It prints a new password and also saves it in `INITIAL_PASSWORD.txt` in the data folder. It works
at once, without a restart. Change it under Settings afterwards (that deletes the file).

On the first start, the generated password is in the same file and in
`sudo docker compose logs gpu-shepherd`.

### Run without Docker or GPUs

```bash
pip install .
gpu-shepherd --demo --port 8090
```

## download-models.sh

Downloads the model files a preset lists in its `downloads.txt`. The installer offers it; you can
also run it any time, as yourself rather than root so the files are yours:

```bash
gpushepherd/presets/download-models.sh starter                    # list the preset's groups
gpushepherd/presets/download-models.sh starter /srv/models        # download the default groups
gpushepherd/presets/download-models.sh starter /srv/models chat   # only the groups named
gpushepherd/presets/download-models.sh ~/my-preset /srv/models    # a preset folder of your own
```

Downloads resume where they stopped and skip files you already have, so run it again after a
failure. Set `HF_TOKEN=hf_...` if a Hugging Face download answers 401 or 403. The file format is in
the [presets README](../gpushepherd/presets/README.md#downloadstxt).

## Environment variables

Set these in `compose.yaml` under `environment:`.

| Variable | Default | What it does |
|---|---|---|
| `SHEPHERD_DATA_DIR` | `/opt/gpu-shepherd` | Data folder (same as `--data-dir`). |
| `SHEPHERD_HOST` / `SHEPHERD_PORT` | `0.0.0.0` / `8090` | Listen address and port (same as `--host`, `--port`). |
| `SHEPHERD_ADMIN_PASSWORD` | generated | The admin password set on the first start, instead of a generated one. |
| `SHEPHERD_HOST_DATA_DIR` | the data folder | Where the data folder is on the host, if it isn't mounted at the same path inside the container. Docker needs host paths for the folders it mounts. |
| `SHEPHERD_AUTOSTART` | `1` | `0` = don't start autostart instances when GPU Shepherd starts. |
| `SHEPHERD_CONNECT` | `auto` | How GPU Shepherd reaches its instances: `network` (by ID on the Docker network, used inside a container), `host` (published ports on 127.0.0.1, used when run directly on the host). |
| `SHEPHERD_URL` / `SHEPHERD_TOKEN` | - | For commands: the GPU Shepherd to manage and its API token (same as `--url`, `--token`). |
| `SHEPHERD_FAKE_GPUS` | - | Fake GPUs for testing, e.g. `2x8192:NVIDIA GeForce RTX 3070` (count x MiB:name). |
