---
name: gpu-shepherd
description: Manage the user's own GPU Shepherd (self-hosted AI apps sharing a machine's GPUs) through its MCP tools. Use when they ask what is running, which app has which GPU, to start, stop or swap apps, install an App Center app, add or load a model, read an app's log, or update apps or GPU Shepherd.
---

# Managing GPU Shepherd

GPU Shepherd runs self-hosted AI apps (Llama Swap, ComfyUI, Open WebUI, LibreChat, Hermes Agent...) as
Docker containers on one machine and shares its GPUs between them. You manage it through the
`gpu-shepherd` MCP tools, which call the user's own dashboard with their API token.

## How it works

- An **instance** is one installed app (one container), with an ID such as `llm` or `comfyui`.
- Each instance has a **GPU policy**: one GPU, all GPUs, up to N, a pinned GPU, or none (CPU only).
  Model servers can also spill a model into system RAM (offload) when GPU memory is short.
- **Starting** an instance can stop or move others to free its GPUs. A two-GPU box can run one app
  per GPU, or stop both for one app that needs both.
- **Stopping** waits for running work (a reply being written, a picture being made) before it stops.
- GPU Shepherd makes **one change at a time**. Each start, stop, restart and image update is a
  **job** with an ID and a log.
- A **model server** (kind `llama-swap`) holds a list of GGUF models and loads them on demand.
  Client apps (Open WebUI, LibreChat, Hermes...) use a model server for their chat models.

## Rules

1. **Look first.** Call `get_status` (or `list_instances`) before deciding anything. Use the IDs it
   shows; never guess an ID.
2. **Ask before every change.** Starting, stopping, restarting, installing, loading or unloading a
   model, and updates all change the user's machine. Say what you are about to do and get a yes,
   unless the user asked for exactly that change in this conversation.
3. **Plans need the user's OK.** When `start_instance`, `stop_instance` or `restart_instance`
   answers `needs_confirm`, nothing has happened yet. Show the user the `plan` and `warnings` in
   plain words ("this stops ComfyUI and moves Llama Swap to GPU 1"). Only call again with
   `confirm: true` after they agree. Never confirm on your own.
4. **Follow the job.** A change answers with a `job`. Check it with `get_job` until its status is
   `done`, `failed` or `cancelled`, and tell the user how it ended. If another change is running
   (status 409), wait for it rather than retrying in a loop.
5. **Losing work needs a clear yes.** `stop_now` drops the work a stop is waiting for; use it only
   when the user asks.
6. **Some things stay in the dashboard.** Passwords, the API token, reboot and shutdown, remote
   access, tensor unlock, deleting instances, models or files, the trash and moving storage are not
   available as tools. Tell the user where to do it (the dashboard's Settings or Storage page).
7. **Keep secrets out.** Don't repeat the API token, and don't paste app passwords or keys from
   logs into chat unless the user asks for them.

## Common tasks

**What is running and where?** `get_status`: each GPU's memory and `users`, each instance's
`running` and `gpus`, and the change running now.

**Swap apps on the GPUs.** `start_instance` with the app the user wants. Expect `needs_confirm` when
other apps must stop or move; show the plan, then confirm. Follow the job.

**Why won't an app start?** `get_instance` (status, `status_text`, `gpu_note`), then `get_logs`
(last 200 lines by default, `lines` for more), then the failed job with `get_job`.

**Install an app.** `list_apps`, then `get_app` for its default `answers` and the `choices`
(GPU, model server, model files). Call `install_app` with only the answers that differ, e.g.
`{"gpu": "pin:1"}` or `{"server": "new"}` to install a model server with it. Installing doesn't
start it: offer `start_instance` next, and pass on the app's `after_install` step if it has one.

**Add a model to a model server.** `list_suggested_models` shows models that fit this machine,
best first; `add_suggested_model` adds one and downloads it in the background. For a model the
user names, `add_model_from_link` takes its `.gguf` link. `list_models` shows what the server has
and what is loaded; `load_model` and `unload_model` load or free one.

**Who keeps the GPUs?** `get_priority` lists the GPU-using apps in priority order (first keeps
its GPUs when there are not enough). `set_priority_order` reorders them; nothing restarts. Ask first.

**Updates.** `list_app_updates` and `apply_app_update` bring an installed app up to its current
template. `list_image_updates`, `check_image_updates` and `update_app_image` handle new releases of
an app's image (it restarts the app). `get_dashboard_update`, `check_dashboard_update` and
`update_dashboard` update GPU Shepherd itself; the dashboard restarts and apps keep running.

**Disk space.** `get_storage` lists free space, folder sizes and files worth a look. Cleaning up is
done in the dashboard's Storage page.

## Reading answers

- Memory is in MiB. GPU indexes start at 0; AMD GPUs are numbered from 100.
- An error answer has `status` and `error`: 400 bad input, 401 wrong token, 404 no such thing,
  409 not possible right now (another change is running, a model is still answering, or a plan
  needs confirming).
- If every call answers 401, the token is wrong or was replaced: ask the user for the current one
  from Settings > API and command line.
