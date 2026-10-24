# CLI and scripts

Everything GPU Shepherd does from the command line. Run the scripts from the folder you cloned.

- [`install.sh`](#installsh): install, or re-run to fill in what is missing
- [`update.sh`](#updatesh): update, roll back, restore settings, dashboard updates
- [`gpu-shepherd`](#gpu-shepherd): the app itself, demo mode, password reset
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
day; the dashboard then shows the changes and an Update button, and Settings > Updates has
*Check now*. The two sides only exchange files in `<data folder>/update/`, so the container never
needs git, your credentials or more access to the host. Without the service, the dashboard shows
the command to run instead.

## gpu-shepherd

The app itself. In a normal install it runs in the `gpu-shepherd` container, so run it through
`docker exec`:

```bash
sudo docker exec gpu-shepherd gpu-shepherd --reset-password
```

| Option | What it does |
|---|---|
| `--host ADDR` | Address to listen on. Default `0.0.0.0`. |
| `--port N` | Port. Default `8090`. |
| `--data-dir DIR` | Data folder. Default `/opt/gpu-shepherd`. |
| `--demo` | Fake GPUs and containers, no Docker needed; data goes to a temporary folder. Log in as `admin` / `demo-password`. |
| `--demo-preset NAME` | The preset the demo starts with. Default `starter`. |
| `--reset-password` | Give the admin a new password, print it and exit. |
| `--version` | Print the version. |

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
| `SHEPHERD_FAKE_GPUS` | - | Fake GPUs for testing, e.g. `2x8192:NVIDIA GeForce RTX 3070` (count x MiB:name). |
