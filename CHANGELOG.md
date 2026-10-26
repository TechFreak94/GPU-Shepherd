# Changelog

What changed in each version of GPU Shepherd, newest first. Every pull request merged into `main`
gets its own version and entry here automatically (see [Versions and releases](docs/updates.md#versions-and-releases)).
The dashboard shows this list under the version number at the top of every page.

## 0.12.10 - 2026-10-26

- Dashboard screenshots in the README show the new card tags. (#24)

## 0.12.9 - 2026-10-26

- New: everything the dashboard does also works from the `gpu-shepherd` command, e.g. `gpu-shepherd start comfyui`, `apps install`, `models set`, `settings set` and `update check` (see docs/cli.md). (#23)
- New: a documented JSON API at `/api/v1` for scripts and Home Assistant, using an API token shown under Settings › API and command line (see docs/api.md). (#23)

## 0.12.8 - 2026-10-26

- Docs, code comments and screenshots brought up to date with the App Store, the LLM server app and theme selection. (#22)

## 0.12.7 - 2026-10-26

- Dashboard cards say what each instance is: Model Server, Chat UI, Image UI, Agent UI or App (#21)

## 0.12.6 - 2026-10-25

- Pick the dashboard theme yourself (System, Light or Dark) with the new button at the top of every page or under Settings › Appearance. It's remembered per browser. (#20)

## 0.12.5 - 2026-10-25

- Apps that need an LLM server say so when there isn't one, and offer to install it first (#19)
- Removed the dashboard's "Add model server" button (use the App Store) (#19)

## 0.12.4 - 2026-10-25

- The App Store has one LLM server app: pick its chat model and coding model while installing it (either or both, or one file for both) (#18)

## 0.12.3 - 2026-10-25

- New App Store: install model servers, Open WebUI, Hermes Agent and ComfyUI with automatic setup, or open every field with manual configuration (#17)
- Hermes Agent can be installed from the App Store, connected to your model server and optionally Home Assistant (#17)
- Open WebUI and Hermes Agent can install a chat model server alongside themselves (#17)

## 0.12.2 - 2026-10-25

- Every detail on an instance card has its own ⓘ tooltip again, and the Advanced details section is back. (#16)

## 0.12.1 - 2026-10-25

- Each merged change now bumps the last number of the version (0.12.0 → 0.12.1). Bigger milestones are labelled `minor`. (#15)

## 0.12.0 - 2026-10-24

- Remove the duplicate description line on instance cards (#14)

## 0.11.0 - 2026-10-24

- Click the version number at the top of any page to check for updates, see what's new and read the changelog. (#13)
- Checks and updates show each step live, keep going through the restart, and finish with a Reload button. (#13)
- Updates work from a private repository: save a GitHub access token in Settings > Private repository, or with `update.sh --set-token`. (#13)
- Every merged change now gets its own version number and changelog entry. (#13)

## 0.10.0 - 2026-10-24

- Instance cards are compact: state, GPUs, VRAM, CPU and RAM at a glance. Hover, tab to or tap a
  label (or the ⓘ next to the name) for the details behind it: GPU rule and priority, ports, image,
  folders, and how much of a model sits in system RAM. (#12)

## 0.9.0 - 2026-10-24

- A shorter README, plus guides for every option in the instance and model editors
  (`docs/advanced.md`) and for the command-line tools (`docs/cli.md`). (#11)

## 0.8.0 - 2026-10-24

- Before a GPU swap, restart or Stop stops a running instance, GPU Shepherd waits for its work to
  finish (a chat answer, the ComfyUI queue), so nothing is cut off half way. The dashboard shows what
  it is waiting for, with **Stop now** and **Cancel**. (#10)
- Per instance: how to tell it is busy, how long to wait at most, and whether to stop anyway or call
  the change off after that. (#10)

## 0.7.0 - 2026-10-23

- **Add instance** starts with templates for common apps: a chat or coding model server, ComfyUI and
  Open WebUI. A short guided form asks only what matters (name, GPUs, port, folders); the full
  editor is still there under Advanced. (#9)

## 0.6.0 - 2026-10-23

- GPU Shepherd can update itself: `sudo ./update.sh` shows what is new, updates the clone, rebuilds
  and restarts, and `--check` only looks. (#8)
- Optional **Update** button on the dashboard, through a small systemd service on the host
  (`update.sh --enable-dashboard-updates`); it also checks for updates once a day. (#8)
- If an update fails to build or start, GPU Shepherd goes back to the previous version and settings
  by itself. By hand: `update.sh --rollback` and `update.sh --restore-settings`. (#8)

## 0.5.0 - 2026-10-23

- Instance cards show more: uptime, GPUs and VRAM, CPU and RAM use, web addresses, image and, for
  model servers, the loaded model and how much of it sits in system RAM. (#6)
- New CPU and RAM panels show the whole machine and how much each instance uses. (#6)
- Buttons and links look clickable. (#6)

## 0.4.2 - 2026-10-23

- The first login password no longer uses look-alike characters, and spaces around it are ignored. (#5)
- `docker exec gpu-shepherd gpu-shepherd --reset-password` sets a new admin password. (#5)
- Login attempts are logged, and the login page explains when the browser drops the session cookie. (#5)

## 0.4.1 - 2026-10-22

- Removed a hardware-specific example preset; the generic starter is the preset that ships. (#4)

## 0.4.0 - 2026-10-22

- Install-time presets: `install.sh` can set up a ready-made group of instances and models. The
  generic **starter** preset is a model server, ComfyUI and Open WebUI. (#3)
- Large model downloads are opt-in. (#3)

## 0.3.0 - 2026-10-21

- `install.sh` guides you through the install step by step and offers the example setup on the first
  run; flags remain for scripted installs. (#2)
- It can download the example models and adds you to the `docker` group. Autostart is off unless you
  turn it on. (#2)

## 0.2.0 - 2026-10-21

- Models can have GPU/RAM offload settings per number of GPUs (`--n-gpu-layers`, `--n-cpu-moe`), so
  a large model can use system RAM on one GPU and run fully on two. (#1)
- A `coder` model alias in the example model server, and an install script for Ubuntu. (#1)

## 0.1.0 - 2026-10-21

- First version: a web dashboard that starts and stops Docker containers and shares the GPUs between
  them (fixed, elastic and priority policies), with a llama-swap connector for model servers.
