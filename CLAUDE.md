# GPU Shepherd: notes for Claude

A FastAPI + Jinja2 + htmx dashboard that runs self-hosted AI containers and shares the machine's
GPUs between them. Python 3.11+, no JavaScript build step. Read [CONTRIBUTING.md](CONTRIBUTING.md)
for the rules; [docs/architecture.md](docs/architecture.md) has the bigger picture. Read only the
modules a task touches: each one starts with a docstring that says what it owns.

## Where things live (`gpushepherd/`)

| Module | Owns |
| :- | :- |
| `service.py` | Every action and setting. The dashboard (`app.py`), the JSON API (`api.py`, `/api/v1`) and the CLI (`cli.py`) all call it, so new features go here first |
| `app.py`, `templates/`, `static/` | Web UI routes, pages and htmx partials (`templates/partials/`) |
| `scheduler.py` | Pure GPU planner (no I/O): which instance gets which GPUs |
| `engine.py` | Turns plans into container changes; the only caller of the backend |
| `backend.py` | Docker Engine backend, plus the in-memory fake used by tests and `--demo` |
| `drain.py`, `jobs.py` | Waiting for running work before a stop; one background change at a time |
| `config.py` | Pydantic models stored as YAML in the data directory (layout in its docstring) |
| `instance_templates.py` | App Center apps (Llama Swap, ComfyUI, Open WebUI, LibreChat, AnythingLLM, Hermes...) |
| `app_updates.py` | Bringing already-installed apps up to the current template |
| `model_library.py`, `hf_models.py`, `downloads.py` | Suggested GGUF models (built-in picks, plus trending Hugging Face ones that fit) and downloading them |
| `connectors/llamaswap.py` | Renders llama-swap's config from the model catalogue |
| `updates.py`, `changelog.py`, `buildinfo.py` | Self-update (with `update.sh` on the host), versions, changelog |
| `gpu.py`, `host.py`, `auth.py` | GPUs (nvidia-smi; AMD from /sys, numbered from 100), /proc CPU and RAM, passwords |
| `gpu_unlock.py` | Experimental tensor unlock: talks to `gpu-unlock.sh` on the host (Settings > Experimental) |
| `logins.py`, `credits.py` | Apps' accounts and password resets; open-source credits (keep `ACKNOWLEDGEMENTS.md` in step) |
| `presets/starter/` | The default example setup offered on first run |

Host-side scripts: `install.sh`, `update.sh`, `gpu-unlock.sh`. Release automation: `scripts/release.py`,
`.github/workflows/`.

## Working on it

- Test: `pip install -e ".[dev]" && pytest -q` (CI runs the same). Run a single file while iterating.
- Try the UI without Docker or GPUs: `gpu-shepherd --demo --port 8090` (admin / demo-password).
- Batch changes: commit to the shared batch branch instead of opening a pull request per change.
  Once 3 or more changes are waiting, list them for the maintainer; open the pull request only after they
  approve. Pull requests target `testing`, never `main`; `main` only gets promotions from `testing`
  (merge commit) and urgent fixes. See CONTRIBUTING.md#branches. (The branch name and the pending list are kept in the project's notes.)
- Don't edit the version or `CHANGELOG.md`. Put user-visible changes under `## Changelog` in the
  pull request description; merging releases it.
- Don't update docs or screenshots in a feature PR unless the change would leave them wrong.
  Screenshots only when asked.
- `AGENTS.md` holds these same notes for other coding agents (OpenCode, Aider, Cline...). Change both
  files together; `tests/test_agent_notes.py` checks they match.
- Keep everything generic: no personal models, paths, hardware, hostnames, IPs, emails or private workflows.
