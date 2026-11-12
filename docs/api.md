# JSON API

Everything the dashboard does is also available as JSON under `/api/v1`: for scripts, Home
Assistant, cron jobs, or your own tools. The dashboard's buttons, these endpoints and the
[`gpu-shepherd` commands](cli.md#commands) all run the same code, so they behave the same way and
give the same messages.

- [Authentication](#authentication)
- [Conventions](#conventions)
- [State](#state), [Instances](#instances), [Changes (jobs)](#changes-jobs), [App Center](#app-center),
  [Model servers](#model-servers), [Settings](#settings), [Tensor unlock](#tensor-unlock), [Updates](#updates)
- [Examples](#examples)

## Authentication

Send the API token in a header:

```
Authorization: Bearer gs-...
```

The token is made on the first start and kept in the `api-token` file in the data folder (mode
600). Find it under **Settings › API and command line**, or with
`sudo docker exec gpu-shepherd gpu-shepherd token`. *Make a new token* there (or `token --new`)
replaces it; the old one stops working at once.

Whoever has the token can do everything the admin can, so keep it like a password. A browser that
is logged in to the dashboard may also `GET` from the API (handy for a quick look), but changes
always need the token, so another web page can't make your browser change anything.

## Conventions

- Bodies are JSON (`Content-Type: application/json`). Field names are the YAML keys in
  [Instance and model options](advanced.md).
- Answers are JSON. A change answers `{"message": "..."}` with the text the dashboard shows.
- Errors answer `{"error": "..."}` with `400` (invalid input), `401` (no or wrong token), `403`
  (wrong current password), `404` (no such instance, model or app) or `409` (not possible right
  now: another change is running, a model is still answering, a plan needs confirming).
- `202` means the work runs in the background: a change (follow its job), a model load, or an
  update request.

## State

| Request | Answer |
|---|---|
| `GET /api/v1/state` | Version, GPUs (memory in MiB, utilisation, `users`), host CPU and RAM, each instance's `running`, `gpus` and GPU `policy`, the running change (`job`) and what it waits for (`waiting_for`), files that couldn't be read (`problems`). `GET /api/state` answers the same (the older address). |

## Instances

| Request | What it does |
|---|---|
| `GET /api/v1/instances` | Every instance: its settings plus `state` (`running`, `status`, `status_text`, `gpus`, `container`, `restart_needed`). |
| `GET /api/v1/instances/{id}` | One instance, the same way. |
| `PUT /api/v1/instances/{id}` | Create (`201`) or replace (`200`) an instance. The body is the whole instance, as `GET` returns it (`state` is ignored) or as in `instances/<id>.yaml`; `id` defaults to the one in the address and `name` to the ID. To change one field, `GET`, change it and `PUT` it back. |
| `DELETE /api/v1/instances/{id}` | Delete it: stop it (a running one waits for its work to finish, in the background) and move its settings, model list and app data folder, plus a helper only it uses (LibreChat's database), to the [trash](#storage-and-trash). `409` while another instance needs it. Model files are never touched. |
| `POST /api/v1/instances/{id}/start` | Start it. Also `/stop` and `/restart`. Body (optional): `{"confirm": true}`. |
| `GET /api/v1/instances/{id}/logs` | `{"logs": "..."}`, the container's log. |
| `GET /api/v1/instances/{id}/logins` | An app's [logins](advanced.md#logins): `where` it keeps them, `accounts` (`user`, `role`, and `password` where it can be read), `running`, `multi_user` and a `status` line. `404` for apps GPU Shepherd doesn't know the logins of. |
| `POST /api/v1/instances/{id}/logins` | Set a new password: `{"user": "...", "password": "..."}` (no `user` for Hermes Agent and single-user AnythingLLM; no `password` for a made-up one). Answers `message` and the `password` set. |

Starting can take GPUs from other instances, and stopping waits for running work, as on the
dashboard. The answer is one of:

- `202` `{"message", "plan": [...], "job": {...}}`: it is running; follow `job.id`.
- `200` `{"message", "noop": true}`: it is already in that state.
- `409` `{"error", "needs_confirm": true, "plan": [...]}`: it would also stop or move other
  instances (`plan` says which). Send it again with `{"confirm": true}` to go ahead.

## Changes (jobs)

GPU Shepherd makes one change at a time. Each start, stop, restart, delete and autostart is a job.

| Request | What it does |
|---|---|
| `GET /api/v1/jobs` | Recent jobs, newest first: `id`, `title`, `status` (`running`, `done`, `failed`, `cancelled`), `error`, `started`, `finished`, `seconds`, `waiting_for`. |
| `GET /api/v1/jobs/{id}` | One job, with its `log` (a list of lines). |
| `POST /api/v1/jobs/{id}/skip` | While it waits for running work: stop without waiting longer (*Stop now*). |
| `POST /api/v1/jobs/{id}/cancel` | Call the change off before anything is stopped (*Cancel change*). |

## App Center

| Request | What it does |
|---|---|
| `GET /api/v1/apps` | The apps: `key`, `title`, `group`, `blurb`, `needs`, `needs_llm_server`, `installed` (instance IDs), `also_installs` (helpers it comes with, e.g. LibreChat's database), `after_install` (the step left to do in the app itself), `experimental` (why it is marked Experimental, or empty), `license` and `docs` (its project page). |
| `GET /api/v1/apps/{key}` | One app, the `answers` its install form starts with, and the `choices` for `gpu`, `offload`, `server`, `comfyui`, `speech` and the model files. |
| `POST /api/v1/apps/{key}/install` | Install it (`201`). The body holds only the answers that differ from the defaults, e.g. `{"gpu": "pin:1", "port": 8189}` or `{"chat_file": "Qwen3-8B-Q4_K_M.gguf"}`. For apps that use a model server, `"server": "new"` installs a Llama Swap server with it. Answers `instance`, `models`, `llm_server` (the one installed with it), `companions` (helper instances installed with it) and `downloads` (suggested models it started downloading). |
| `POST /api/v1/apps/{key}/instance` | The instance an install would make, without saving it (*Manual*): change it and `PUT` it. |

The answers are the install form's fields: `name`, `id`, `port`, `gpu` (`one`, `all`, `up:N` (at most N),
`none` or `pin:N`), `models_dir`, `appdata_dir`, `server`, `comfyui`, `speech` (a Speaches instance, for Open WebUI), `chat_file` / `chat_ctx`,
`coder_file` / `coder_ctx` and `embed_file` / `embed_ctx` (model files, relative to the models
folder or a path inside the container; a [suggested model](advanced.md#suggested-models)'s file is
downloaded), `offload` (`auto`, `gpu`, `moe`), `login_user` / `password`, `hass_url` /
`hass_token` and `autostart`. `GET /api/v1/apps/{key}` lists the ones each app uses.

## Model servers

For instances of kind `llama-swap`.

| Request | What it does |
|---|---|
| `GET /api/v1/instances/{id}/models` | `models`, `auto_aliases`, `settings` (`threads`, `llama_server`, `health_timeout`, `api_key`), `loaded` (model → state), `loading`, and the .gguf `files` it can see. |
| `PUT /api/v1/instances/{id}/models/{model}` | Add (`201`) or replace (`200`) a model. The body is the model as in the catalogue (`file` is required); a different `id` in the body renames it, and auto names follow. |
| `DELETE /api/v1/instances/{id}/models/{model}` | Remove it (`409` while an auto name points at it). `?files=true` also deletes its files from disk (`409` while it is loaded; files another model uses are kept). |
| `GET /api/v1/instances/{id}/suggestions` | The [suggested models](advanced.md#suggested-models), best fit first: `key`, `name`, `uses`, `fits` and `fit` (how it fits this machine), `state` (`ready`, `downloading`, `starting`, `failed`, `missing`), `status`, `pct`, `in_catalog`, and its `files` with their URLs. |
| `POST /api/v1/instances/{id}/suggestions/{key}/add` | Add it with its makers' settings, downloading it if needed (`201`; `409` if the server has it). |
| `POST /api/v1/instances/{id}/suggestions/{key}/download` | Download or resume its files in the background (`202`). |
| `POST /api/v1/instances/{id}/suggestions/{key}/cancel` | Stop its download; what it fetched is kept. |
| `POST /api/v1/instances/{id}/suggestions/{key}/delete` | Delete the files of a downloaded suggestion the server doesn't have. |
| `POST /api/v1/instances/{id}/models/{model}/load` | Load it now, in the background (`202`). |
| `POST /api/v1/instances/{id}/models/{model}/unload` | Unload it; `_all` as the model unloads every model. `409` while the server is answering. |
| `PUT /api/v1/instances/{id}/auto-aliases` | Replace every auto name: `[{"alias": "coder", "default_model": "small", "rules": [{"min_gpus": 2, "model": "big"}]}]`. |
| `PATCH /api/v1/instances/{id}/server-settings` | Change any of `threads`, `health_timeout`, `llama_server`, `fit_gpus` (use only the GPUs each model needs) and `auto_tune` (apply suggested adjustments by itself); `"new_api_key": true` gives the server a new key. |
| `POST /api/v1/instances/{id}/models/from-link` | Add a model by the link of its `.gguf` file: `{"url": "https://.../Model-Q4_K_M.gguf", "name": "optional", "mmproj_url": "optional vision projector"}`. It is downloaded in the background like a suggested model and listed with them (`201`). `/suggestions/{key}/forget` removes a link from the list. |
| `GET /api/v1/instances/{id}/tuning` | [Suggested adjustments](advanced.md#tuning): each with a `key`, what it changes, why, and its `status` (new, applied or dismissed). |
| `POST /api/v1/instances/{id}/tuning` | Decide one: `{"key": "...", "action": "apply" \| "dismiss" \| "undo"}`. It stays listed either way. |

## Knowledge (documents) and skills

`/api/v1/books/...` is the same as `/api/v1/knowledge/...`, for older clients.

| Request | What it does |
|---|---|
| `GET /api/v1/knowledge` | The documents folder, its documents (status, words, why one couldn't be read), the reasons grouped with counts, where everything is kept (`storage`: path, files, size), the work running now, the Open WebUI apps documents can go to, and the drafted skills. |
| `PUT /api/v1/knowledge/folder` | `{"path": "..."}` picks the documents folder. |
| `POST /api/v1/knowledge/upload` | Add documents: a multipart form with one or more `file` fields; a scan follows (`202`). |
| `GET /api/v1/knowledge/files` | The document files already in the folder, `{path: size}`. |
| `PUT /api/v1/knowledge/files?path=Folder/Book.epub` | Upload one document (raw body, streamed to disk; folders in `path` are kept; a `.zip` is unpacked into a folder named after it). No scan; call `scan` when done. |
| `POST /api/v1/knowledge/scan`, `POST /api/v1/knowledge/stop` | Scan the folder for new documents (`202`), or stop the work running. |
| `POST /api/v1/knowledge/send` | Put the ready documents in an Open WebUI knowledge base: `{"instance": "webui", "name": "Documents", "split": false, "embeddings": "server" \| "keep"}` (`202`). |
| `POST /api/v1/knowledge/{key}/skill` | Draft an agent skill from a document with a model server's chat model: `{"server": "llm"}` (optional; `202`). |
| `POST /api/v1/knowledge/ocr` | Read the scanned PDFs (status `scanned`) with OCR, in the background (`202`). |
| `POST /api/v1/knowledge/delete` | Delete every duplicate or every document that can't be read, files and all: `{"status": "duplicate" \| "failed"}` (`202`). |
| `DELETE /api/v1/knowledge/{key}` | Delete one document: its files, its text, and its copies in a running Open WebUI. |
| `POST /api/v1/knowledge/clear` | Start over: `{"what": ["documents", "converted", "skills", "apps"]}` (all four if left out; `202`). |
| `GET`, `PUT`, `DELETE /api/v1/skills/{slug}` | Read a draft (`text`), save an edited one (`{"text": "..."}`), delete it. |
| `POST /api/v1/skills/{slug}/install` | Install it in Hermes Agent: `{"instance": "hermes"}`. |

## Settings

| Request | What it does |
|---|---|
| `GET /api/v1/settings` | `admin_user`, `bind_address`, `network`, `container_prefix`, `data_dir`, `host_data_dir`, `trash_days`, `image_check_hours`, `hf_suggestions`. |
| `PATCH /api/v1/settings` | Change `bind_address`, `network`, `trash_days`, `image_check_hours` and/or `hf_suggestions` (suggest models trending on Hugging Face) (the others are YAML only). |
| `POST /api/v1/settings/password` | `{"current": "...", "new": "..."}` (10+ characters). |
| `POST /api/v1/token` | Make a new API token and answer it as `token`; the one used for this request stops working. |

The theme is not here: it is saved per browser.

## Storage and trash

Deleting an instance or cleaning up a file moves it to the trash first, so it can be put back.

| Request | What it does |
|---|---|
| `GET /api/v1/storage` | Free space per `disks` (each named by its mount point), the `folders` and their sizes, app `images` and what uses them, `files` worth a look (`why`: `unused`, `partial` or `duplicate`), `leftover_folders`, `orphans` (containers without an instance) and the `trash`. |
| `POST /api/v1/storage/images` | Remove the app images no instance uses (they download again when needed). |
| `POST /api/v1/storage/orphans` | Remove containers left without an instance. Their data folders are not touched. |
| `POST /api/v1/storage/files` | `{"paths": [...]}`: move model files from `files` to the trash. |
| `POST /api/v1/storage/leftovers` | `{"name": "..."}`: move an unused app data folder from `leftover_folders` to the trash. |
| `GET /api/v1/storage/drives` | The machine's `drives` (device, label, size, filesystem, where mounted, whether it can be mounted and why not), an `error` if they can't be looked at, and `auto_mount`. |
| `POST /api/v1/storage/drives/mount` | `{"device": "/dev/sdb1"}`: mount a drive that holds data (never formatted or written to). |
| `PUT /api/v1/storage/drives/auto` | `{"enabled": true}`: mount newly attached drives by itself, or only when asked. |
| `GET /api/v1/storage/browse?path=/srv` | The folders in `path` and the room on its disk, for a folder picker. |
| `POST /api/v1/storage/move` | `{"source": "...", "dest": "...", "confirm": true}`: move a folder the storage report lists as movable to another place (a background job, `202`). Without `confirm` it only checks and answers `409` with what the move would do. The old copy is kept until deleted. `{"source": "...", "home": true}` (no `dest`) moves a moved folder back to where it first was, even after the old copy is deleted. |
| `GET /api/v1/moves` | Moves so far: `id`, `source`, `dest`, `status` (including `interrupted`) and `age_days`. |
| `POST /api/v1/moves/{id}/back` | Point the apps at the old copy again. `DELETE /api/v1/moves/{id}/leftover` deletes the old copy (to the trash). |
| `GET /api/v1/trash` | What is in it: `id`, `title`, `size`, `age_days`. |
| `POST /api/v1/trash/{id}/restore` | Put it back (`409` if something is in its place). |
| `DELETE /api/v1/trash/{id}` | Delete one entry for good. `DELETE /api/v1/trash` empties the trash. |

## App updates

Settings that an app's template gained after it was installed (the dashboard's *App updates* banner).

| Request | What it does |
|---|---|
| `GET /api/v1/app-updates` | A list of `instance`, `key`, `title`, `changes` and `afterwards` (what is left for you to do). |
| `POST /api/v1/app-updates/{id}/{key}/apply` | Apply it. `/skip` stops it being offered again. |

### App image updates

New releases of the apps' Docker images (checked every `image_check_hours`, 12 by default; `0` = only when asked).

| Request | What it does |
|---|---|
| `GET /api/v1/image-updates` | Newer releases of the apps' images: when `checked`, `every_hours`, and for each instance its `image`, `state`, `auto_update`, `error`, `checked` and `updated`. |
| `POST /api/v1/image-updates/check` | Look in the registries now. |
| `POST /api/v1/image-updates/{id}/update` | Download the new image and restart the app, waiting for running work (`202`, a job). |
| `PUT /api/v1/instances/{id}/auto-update` | `{"on": true}`: let GPU Shepherd bring in new releases of this app's image by itself. |

## Tensor unlock

Experimental; see [Tensor unlock](advanced.md#tensor-unlock). It needs the host service that comes with
dashboard updates (`service` below). Every change applies at the next reboot.

| Request | What it does |
|---|---|
| `GET /api/v1/gpu-unlock` | `installed`, `installed_version`, `version` (the one that gets installed), `state` (`off`, `armed`, `trying`, `on`), `note`, `run` and `message` (the last action), `busy`, `service`, `blocker` (why it can't be installed here), `reboot_needed`, `checked_at`, `command` (to run on the host) and the `cards`: `pci`, `name`, `family`, `tier` (`stable` or `experimental`), `supported`, `unlocked` (now), `next_boot`, `note`. |
| `POST /api/v1/gpu-unlock/check` | List this machine's cards and which ones are supported. Changes nothing (`202`). |
| `POST /api/v1/gpu-unlock/install`, `.../uninstall` | Build and install gpu-unlock (no card is unlocked yet), or turn it off and remove it (`202`). |
| `POST /api/v1/gpu-unlock/set` | `{"cards": ["0000:01:00.0"]}`: unlock these cards from the next boot; `[]` unlocks none (`202`). |

## Updates

Updates are done by `update.sh` on the host; these ask its service to act, like the dashboard's
buttons (see [Updating from the dashboard](cli.md#updating-from-the-dashboard)).

| Request | What it does |
|---|---|
| `GET /api/v1/updates` | `running_version`, `available`, `new_version`, `commits`, `checked_at`, `state`, `step`, `progress`, `busy`, `dashboard_updates`, `credentials_user`. |
| `POST /api/v1/updates/check` | Check now (`202`). |
| `POST /api/v1/updates/update` | Update now (`202`). GPU Shepherd restarts; instances keep running. |
| `POST /api/v1/updates/branch` | `{"branch": "feature-x"}`: follow that branch (one listed in `branches` of `GET /api/v1/updates`) and rebuild (`202`). |
| `GET /api/v1/updates/log` | The log of the last check or update. |
| `POST /api/v1/updates/credentials` | `{"user": "...", "token": "github_pat_..."}` to save a GitHub token for a private repository, or `{"clear": true}`. |
| `GET /api/v1/tailscale` | Tailscale on the host: `installed`, `state` (`Running`, `NeedsLogin`, `Stopped`...), `connected`, `name`, `ips`, `tailnet`, `https`, `https_available`, `login_url` (while logging in), `dashboard_urls`, `busy`, `run`, `message`, `service`, `wsl`. |
| `POST /api/v1/tailscale/{action}` | `refresh`, `install`, `up` (log in or connect), `down`, `https-on`, `https-off` or `logout` (`202`). Needs dashboard updates; the host does the work. |
| `GET /api/v1/changelog` | Releases, newest first: `version`, `date`, `items`, `running`, `new`. |

## Examples

```bash
TOKEN=$(sudo cat /opt/gpu-shepherd/api-token)
API=http://localhost:8090/api/v1
H="Authorization: Bearer $TOKEN"

curl -s -H "$H" $API/state
curl -s -H "$H" -X POST $API/instances/comfyui/start                       # 409 if it needs confirming
curl -s -H "$H" -X POST $API/instances/comfyui/start -d '{"confirm": true}'
curl -s -H "$H" -X POST $API/apps/comfyui/install -d '{"gpu": "pin:1", "port": 8189}'
curl -s -H "$H" -X PUT $API/instances/llm/models/qwen3-8b \
     -d '{"file": "/models/Qwen3-8B-Q4_K_M.gguf", "ctx_size": 32768, "aliases": ["chat"]}'
```

Home Assistant `rest_command`:

```yaml
rest_command:
  start_comfyui:
    url: http://server:8090/api/v1/instances/comfyui/start
    method: post
    headers:
      Authorization: !secret gpu_shepherd_token     # "Bearer gs-..."
    payload: '{"confirm": true}'
    content_type: application/json
```
