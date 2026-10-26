# Architecture

```
browser ── htmx ──────────> app.py (FastAPI routes, Jinja templates) ──┐
scripts ── JSON ──────────> api.py (/api/v1, API token) ───────────────┤
gpu-shepherd COMMAND ─ cli.py ─ JSON ─> api.py                         │
                                                                       v
                    service.py  Shepherd: every action and setting, checked; raises Problem
                      │
                      ├─ config.py     Store: pydantic models <-> YAML files in the data folder
                      ├─ engine.py     snapshot() of the world, plan(), execute(plan)
                      │     ├─ scheduler.py   pure GPU planner (no I/O)
                      │     ├─ gpu.py         nvidia-smi
                      │     ├─ backend.py     DockerBackend (docker SDK) / FakeBackend
                      │     └─ connectors/llamaswap.py   render config, talk to llama-swap
                      ├─ instance_templates.py   the App Store's apps and what they build (pure)
                      ├─ jobs.py       one change at a time, in a background thread, with a log
                      ├─ drain.py      wait for running work (chats, queued images) before a stop
                      ├─ auth.py       scrypt password hashing, first-run password, API token
                      ├─ updates.py    update status, steps and requests, shared with update.sh on the host
                      └─ changelog.py  reads CHANGELOG.md, compares versions
```

## Principles

- **The UI is the editor.** Anything a user needs to change should have a form. YAML is the
  storage format, readable and diffable, not the primary interface.
- **One implementation, three front ends.** Each action and setting is a method of
  `service.Shepherd`. The dashboard's routes (`app.py`), the JSON API (`api.py`) and the commands
  (`cli.py`, which only talks to the API) call it and differ only in how they show the result, so
  a button and its command can't drift apart. Commands go through the running app rather than
  editing files, because starts, stops and model loads depend on its job runner and its view of
  what is running.
- **Plan, show, then act.** `scheduler.plan()` is pure and fully unit-tested. The UI shows the
  plan when it touches anything besides the instance you clicked; `engine.execute()` carries it out.
- **Containers are disposable.** GPU assignments are fixed at container creation (Docker device
  requests), so a change of GPUs means re-creating the container. All state lives in mounted
  folders. Containers carry labels: `gpushepherd.instance`, `gpushepherd.gpus`,
  `gpushepherd.config` (a fingerprint used to show "restart to apply").
- **No build step.** Server-rendered HTML, htmx for partial updates, one small JS file, one CSS
  file. Contributors only need Python.

## Request flow: clicking "Start"

1. `POST /ui/instances/<id>/start` (htmx), or `POST /api/v1/instances/<id>/start` from a script or
   `gpu-shepherd start <id>`. Both call `Shepherd.change()`.
2. `engine.plan("start", id)` reads instances, running containers (with their GPU labels) and
   GPUs, then calls `scheduler.plan()`.
3. If the plan stops or restarts other instances and was not confirmed, the server returns
   `partials/confirm.html` into the modal. "Continue" re-posts with `confirm=1`. (The API answers
   `409` with the plan instead; the command shows it and asks.)
4. The job runner executes the plan: stop affected containers, then (re)create containers in
   priority order. For llama-swap instances the config is rendered from the catalogue for the
   number of GPUs assigned, written under `generated/`, and mounted read-only.
5. The dashboard polls `/ui/dashboard` every 4 seconds and shows the job's progress. Each poll
   also reads host CPU and memory from `/proc` (`host.py`; not namespaced, so it sees the whole
   machine from inside the app's container) and one-shot Docker stats for running containers.
   CPU figures are the change since the previous poll.

## Waiting for running work (`drain.py`)

`engine.execute()` pulls images, then calls `drain.wait_until_idle()` for every running instance it
is about to stop or restart, then stops them. Each instance's `drain` policy (`DrainPolicy` in
`config.py`) names a check: `llama-swap` (the in-flight snapshot llama-swap sends first on
`/api/events`, falling back to `/upstream/<model>/slots` of each loaded model), `llama-server`
(`/slots`), `comfyui` (`/queue`), `http` (a JSON field), or `none`; `auto` picks one from the kind
and image. The wait polls every 2 s until the instance reports no work, its timeout passes
(`on_timeout`: `stop` anyway or `cancel`), or the job's `skip` / `cancel` events are set by the
dashboard's *Stop now* / *Cancel change* buttons (`POST /ui/jobs/<id>/skip|cancel`). A check that
can't get an answer doesn't block. While it waits, `Job.waiting` holds what each instance is still
doing; the dashboard and `/api/v1/state` (`waiting_for`) show it. Cancelling raises `jobs.Cancelled`
before anything was stopped, and the job ends as `cancelled`. The policy is left out of the
config fingerprint, so changing it never asks for a restart.

## Updates

The app can't rebuild the container it runs in, so updating is `update.sh`'s job, on the host:
it fast-forwards the clone (as the clone's owner, so their git credentials work), then runs the
new copy of itself to `docker compose up -d --build` and wait for `/healthz`. The image records
the commit it was built from (`buildinfo.py`, run in a Dockerfile build stage, reads `.git`
without needing git) so the dashboard knows exactly what is running.

The app and `update.sh` share only files in `<data dir>/update/`: `status.json` (written by every
check or update and at each step of one: commits behind, their subjects, local edits, state, step,
the clone's and the remote's version), `request` (written by the app: `check` or `update`),
`log.txt`, `changelog.md` (the remote's newest `CHANGELOG.md`) and, for a few seconds,
`credentials` (a GitHub token from Settings, which `update.sh` moves into the clone's `.git`
folder and passes to git through its `store` credential helper). The dashboard's Updates dialog
polls while a check or update runs, so it shows each step and carries on across the restart. The optional systemd units from
`update.sh --enable-dashboard-updates` are a daily `--check` timer and a path unit that runs
`update.sh --from-dashboard` when `request` appears. A request no one picks up within two minutes
is reported on the dashboard as "the update service isn't running".

Recovery: before an update, `update.sh` tags the running image `gpu-shepherd:previous` and tars
the settings YAML into `update/backups/`. If the build fails or `/healthz` doesn't answer within
90 s, `recover()` resets the clone to the old commit, re-tags and starts the saved image with
`--no-build` (falling back to rebuilding the old code), restores the settings, and reports
`failed` with what it did. It never starts another update, so it can't loop.

## Adding a model-server connector

A connector needs: a `kind` value on `Instance`, a hook in `Engine.build_spec()` that adds its
mounts/environment, and a client with `models()`, `running()`, `load()`, `unload()` (see
`connectors/llamaswap.py`). The models page and dashboard card can then reuse the same templates.

## Testing

- `tests/test_scheduler.py` - planner behaviour, including the real two-GPU and one-GPU cases.
- `tests/test_app.py` - full UI flows through FastAPI's test client with `FakeBackend` and
  fake GPUs (`SHEPHERD_FAKE_GPUS=2x8192:Name`).
- `tests/test_updates.py` - build info, the update status and routes, and `update.sh` itself
  against a scratch git remote with a stand-in `docker`.
- `tests/test_release.py` - `scripts/release.py` (version bump and changelog entry for each merge)
  and that `CHANGELOG.md` matches the version.
- `tests/test_docker_backend.py` - the exact Docker API payloads, built by the real docker-py
  client with network calls intercepted (no daemon needed).
- `tests/test_drain.py` - each busy check against a small fake HTTP server, plus timeout, Stop now
  and cancel.
- `tests/test_instance_templates.py` - the App Store's apps, what they build and automatic offload.
- `tests/test_host.py` - host CPU and memory from a fake `/proc`.
