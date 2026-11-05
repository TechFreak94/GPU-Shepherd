# Changelog

What changed in each version of GPU Shepherd, newest first. Every pull request merged into `main`
gets its own version and entry here automatically (see [Versions and releases](docs/updates.md#versions-and-releases)).
The dashboard shows this list under the version number at the top of every page.

## 0.13.24 - 2026-11-05

- Books is now **Knowledge**, and books are **Documents** (ebooks, manuals, notes). Everything is kept in one folder, `<data>/knowledge`, and existing books move there automatically. The old page, command and API names still work. (#55)
- The Knowledge page shows where the documents, converted text and skill drafts are kept, how big they are, and the free space. (#55)
- Fewer documents that can't be read: sturdier EPUB reading, readme and licence files left out, short notes allowed, and Word (DOCX) files read. Documents that couldn't be read are tried again at the next scan. The reasons are grouped with counts, and the list can show only one status. (#55)
- Start over: delete the documents, converted text, skills and the copies sent to Open WebUI, chosen by checkbox. (#55)

## 0.13.23 - 2026-11-05

- Fixed: updating no longer cuts running apps (and the dashboard) off their GPUs ("nvidia-smi failed ... exit status 255"). (#54)
- When the GPUs can't be read, GPU Shepherd no longer stops running apps, and the dashboard says what went wrong and how to fix it. (#54)

## 0.13.22 - 2026-11-05

- Books: the table fits the page without sideways scrolling, and long lists scroll with the header kept in view. (#53)
- New experimental setting, Settings > Experimental > Tensor unlock. It installs gpu-unlock, which runs FP16/BF16 tensor math at full speed on GeForce RTX 20, 30 and 40 cards. Image generation and unquantized models get faster. (#53)
- You choose which cards to unlock. Every change applies at the next reboot, and each GPU on the dashboard shows whether it is unlocked. (#53)
- The unlock turns itself off after a boot that fails its check, and you can turn it off or remove it from Settings. (#53)

## 0.13.21 - 2026-11-04

- Books upload explains files the browser can't read (on Windows, paths over 260 characters) instead of saying GPU Shepherd is not reachable (#52)
- Zips with no books (music, videos, games) are left out before uploading, in one summary line (#52)
- PDFs: AES-encrypted ones can be read, and one broken page no longer loses the whole book (#52)
- Zip errors name the zip instead of a temporary .part file (#52)
- Books page table: long names and paths wrap, so every column stays visible (#52)

## 0.13.20 - 2026-11-04

- The "Follow branch" list in Settings now shows each branch's version, pull request number and title, and last update date and time, newest first with main at the top. (#51)
- Books can be uploaded from the browser up to 4 GB each, and zips up to 64 GB. (#51)

## 0.13.19 - 2026-11-04

- Bump actions/checkout from 4 to 7 (#50)

## 0.13.18 - 2026-11-04

- Storage: move a moved folder back to the install drive any time, even after its old copy is deleted (Settings > Storage, API, `storage move-home`). (#49)
- Books: load a whole library from the browser by dropping books, folders or zips, with progress, Stop and resume; `gpu-shepherd books upload` does the same from the command line. (#49)

## 0.13.17 - 2026-11-04

- Books: put ebooks in a folder, send them to an Open WebUI knowledge base to chat with them, or have your model server draft an agent skill from a book for Hermes Agent. (#48)
- Embeddings models now load when the LLM server starts, and "Used for: Embeddings" adds the right flag by itself. (#48)
- Settings › Updates can follow another branch, to try changes before they are released. (#48)

## 0.13.16 - 2026-11-04

- Bump docker/setup-buildx-action from 3 to 4 (#43)

## 0.13.15 - 2026-11-04

- Documentation now covers every App Center app and the newer features (tuning, adding a model from a link, app image updates, other disks). (#42)
- New contributing guide, security policy, code of conduct and issue and pull request templates. (#42)

## 0.13.14 - 2026-11-03

- The App Store is now called the App Center. (#41)
- New App Center apps: vLLM (a fast model server for one Hugging Face model, split across 1, 2, 4 or 8 GPUs), SillyTavern (character chat) and Open Notebook (a private NotebookLM with its own database). Each one connects to your model server for you. (#41)
- Chat apps can use a vLLM server as their model server. (#41)
- Logins shows and resets SillyTavern's and Open Notebook's passwords. (#41)

## 0.13.13 - 2026-11-03

- Fixed: after updating, the GPU panel and GPU apps could lose their GPUs ("nvidia-smi failed ... exit status 255") because mounting a drive reloaded systemd. Drives already attached when GPU Shepherd starts are no longer mounted by themselves. (#40)

## 0.13.12 - 2026-11-02

- Moving storage: pick the new place from your drives and a folder tree; newly attached drives with data are mounted automatically. (#39)
- Apps can now find and install new releases of their images, by hand or automatically per app, with a notice on the app's card. (#39)

## 0.13.11 - 2026-10-30

- Move a models folder or an app's data folder to another disk from Settings > Storage, with a verified copy and "Move back". (#38)
- ComfyUI cards show both the Image Server and Image UI tags. (#38)
- The dashboard refreshes use much less CPU. (#38)
- Fixed a rare dashboard error when a start or stop finished during a refresh. (#38)

## 0.13.10 - 2026-10-30

- Model names are now called tags, and the Models page explains how tags decide which model loads. (#37)
- The model editor explains slots, context, idle unload and GPU settings in plain language. (#37)
- Clearer wording on the app form, Settings, guided install and Storage pages. (#37)

## 0.13.9 - 2026-10-30

- MoE suggested models now let llama.cpp spread them across the GPUs itself, and the Models page shows why a model didn't start. (#36)
- Models added by hand are checked against this machine: too big to load at all is refused, other problems are warned about. (#36)
- Machines with 3 or more GPUs: installs can pick "Up to N free GPUs", and GPU lists are shortened ("GPUs 0-7"). `--demo-gpus N` tries a bigger machine in demo mode. (#36)
- Model servers can run each model on the fewest GPUs that hold it and leave the other GPUs free for other apps ("Use only the GPUs each model needs"; a per-model "GPUs it uses" setting). (#36)
- Suggested adjustments: GPU Shepherd learns how much GPU memory and speed each model really uses and suggests changes on the Models page. Nothing changes until you apply one; every decision stays listed to undo or apply later. "Apply suggested adjustments by itself" is optional. (#36)

## 0.13.8 - 2026-10-29

- The LLM server's Logins card explains its browser login: any user name, the API key as password. (#35)
- When a model can't start, the LLM server's log now shows llama-server's own error, and the Models page marks models whose file is missing. (#35)
- New suggested model: Muse Glimmer 30B, sized to run fully on two 8 GB GPUs. (#35)
- Add any model to the LLM server from the link of its .gguf file, on the Models page, the API or the CLI. (#35)
- Fixed: restarting the LLM server from its Models page showed an error (the restart itself worked). (#35)

## 0.13.7 - 2026-10-28

- App Store: Langflow, marked Experimental. (#34)
- App Store: InvokeAI, Ollama and Speaches. Open WebUI can now use Speaches for voice (pick it under *Voice with* when installing). (#34)
- Logins: see an app's accounts and set a new password from its dashboard card, the API or `gpu-shepherd logins`. (#34)
- Every app now shows its license. A new Credits dialog and ACKNOWLEDGEMENTS.md credit the open-source projects GPU Shepherd is built on and runs. (#34)

## 0.13.6 - 2026-10-27

- README: minimum system requirements. (#33)
- Docs: storage, trash and app updates, with new screenshots. (#33)

## 0.13.5 - 2026-10-27

- New: install with Docker Compose from a published image, no clone needed (`compose.standalone.yaml`). It sets up the Starter preset on the first start. (#32)
- New: Storage page, with a storage report, clean-ups that go through the trash, and a low-disk banner. (#32)
- New: deleting an instance stops it gracefully and moves it to a recoverable trash. (#32)
- New: warnings before a start when RAM or GPU memory may not be enough. (#32)
- New: optional Tailscale setup in `install.sh` (`--tailscale`, `--tailscale-https`) for remote access to the dashboard and apps. (#32)
- Improved: container logs are capped, old builds are removed after updates, and downloads can be tidied or discarded, with a free-space check first. (#32)

## 0.13.4 - 2026-10-26

- The installer now supports Debian 12/13, Fedora, RHEL/Rocky/Alma 9+ and Arch, as well as Ubuntu. WSL2 support is experimental. (#31)
- SELinux hosts: GPU Shepherd and its apps work when Docker uses SELinux. (#31)
- New guides: Linux distributions, Proxmox, TrueNAS SCALE and Unraid. (#31)
- GPU Shepherd stays up after a broken settings file, a bad model list or Docker being down. A failed update is now caught and rolled back. (#31)

## 0.13.3 - 2026-10-26

- Tag ComfyUI as Image Server (#30)

## 0.13.2 - 2026-10-26

- LibreChat's database now stops when LibreChat stops (and an instance's *Starts with* instances stop with it unless something else running needs them). (#29)

## 0.13.1 - 2026-10-26

- App updates: apps installed earlier can pick up settings added to them since, such as chatting with your documents through the model server's embeddings model. Review and apply them from the dashboard notice, the API or `gpu-shepherd app-updates` (#28)

## 0.13.0 - 2026-10-26

- Suggested models on the Models page and in the LLM server's install form: each says whether it fits your GPUs and RAM, and one click downloads it with its makers' settings (#27)
- New model settings: thinking on/off, MTP, bf16 and q5_1 KV cache, multi-GPU split, unload when idle, keep loaded, load at start (#27)
- Embeddings model slot for chatting with documents; Open WebUI and AnythingLLM use it (#27)
- Starter preset now uses Qwen3.5 9B (reads images) and Qwen3 Embedding (#27)

## 0.12.12 - 2026-10-26

- Open WebUI's dashboard card shows its name, Open WebUI, instead of "Chat UI" (also for instances installed before), with Chat UI as its tag. (#26)

## 0.12.11 - 2026-10-26

- New in the App Store: LibreChat (with its own MongoDB database, which starts with it) and AnythingLLM, both connected to your LLM server. (#25)
- New: an instance can list instances it *starts with* (`requires`); starting it starts them first. (#25)

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
