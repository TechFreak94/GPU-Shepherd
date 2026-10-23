# Architecture

```
browser ── htmx ──> app.py (FastAPI routes, Jinja templates)
                      │
                      ├─ config.py     Store: pydantic models <-> YAML files in the data folder
                      ├─ engine.py     snapshot() of the world, plan(), execute(plan)
                      │     ├─ scheduler.py   pure GPU planner (no I/O)
                      │     ├─ gpu.py         nvidia-smi
                      │     ├─ backend.py     DockerBackend (docker SDK) / FakeBackend
                      │     └─ connectors/llamaswap.py   render config, talk to llama-swap
                      ├─ jobs.py       one change at a time, in a background thread, with a log
                      ├─ auth.py       scrypt password hashing, first-run password
                      └─ updates.py    update status and requests, shared with update.sh on the host
```

## Principles

- **The UI is the editor.** Anything a user needs to change should have a form. YAML is the
  storage format, readable and diffable, not the primary interface.
- **Plan, show, then act.** `scheduler.plan()` is pure and fully unit-tested. The UI shows the
  plan when it touches anything besides the instance you clicked; `engine.execute()` carries it out.
- **Containers are disposable.** GPU assignments are fixed at container creation (Docker device
  requests), so a change of GPUs means re-creating the container. All state lives in mounted
  folders. Containers carry labels: `gpushepherd.instance`, `gpushepherd.gpus`,
  `gpushepherd.config` (a fingerprint used to show "restart to apply").
- **No build step.** Server-rendered HTML, htmx for partial updates, one small JS file, one CSS
  file. Contributors only need Python.

## Request flow: clicking "Start"

1. `POST /ui/instances/<id>/start` (htmx).
2. `engine.plan("start", id)` reads instances, running containers (with their GPU labels) and
   GPUs, then calls `scheduler.plan()`.
3. If the plan stops or restarts other instances and was not confirmed, the server returns
   `partials/confirm.html` into the modal. "Continue" re-posts with `confirm=1`.
4. The job runner executes the plan: stop affected containers, then (re)create containers in
   priority order. For llama-swap instances the config is rendered from the catalogue for the
   number of GPUs assigned, written under `generated/`, and mounted read-only.
5. The dashboard polls `/ui/dashboard` every 4 seconds and shows the job's progress. Each poll
   also reads host CPU and memory from `/proc` (`host.py`; not namespaced, so it sees the whole
   machine from inside the app's container) and one-shot Docker stats for running containers.
   CPU figures are the change since the previous poll.

## Updates

The app can't rebuild the container it runs in, so updating is `update.sh`'s job, on the host:
it fast-forwards the clone (as the clone's owner, so their git credentials work), then runs the
new copy of itself to `docker compose up -d --build` and wait for `/healthz`. The image records
the commit it was built from (`buildinfo.py`, run in a Dockerfile build stage, reads `.git`
without needing git) so the dashboard knows exactly what is running.

The app and `update.sh` share only files in `<data dir>/update/`: `status.json` (written by every
check or update: commits behind, their subjects, local edits, state), `request` (written by the
app: `check` or `update`) and `log.txt`. The optional systemd units from
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
- `tests/test_docker_backend.py` - the exact Docker API payloads, built by the real docker-py
  client with network calls intercepted (no daemon needed).
