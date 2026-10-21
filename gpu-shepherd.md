# GPU Shepherd (spin-off of the aistack control panel)

Open-source web dashboard that runs self-hosted AI containers and shares GPUs between them automatically.
Working name; easy to rename (package `gpushepherd`, CLI `gpu-shepherd`).

## Decisions (2026-10-21)
- Stack: Python, FastAPI + Jinja2 + htmx, no JavaScript build step
- License: Apache-2.0 (bundles htmx, 0BSD)
- Configuration: YAML files in a data folder (settings.yaml, instances/<id>.yaml, catalogs/<id>.yaml), edited through the UI
- Milestone 1 scope: working app + llama-swap model catalogue editor

## How it works
- Instances = any Docker image; GPU policy per instance: none / fixed GPUs / elastic (min-max) + priority
- Pure planner (scheduler.py) re-plans the whole machine on every start/stop; UI asks before moving or stopping other instances
- Containers get only their GPUs via Docker device requests; moving GPUs = recreating the container
- llama-swap connector renders the config from the catalogue; "auto" names resolve by GPU count (e.g. auto = Qwen3.6-35B-A3B on 1 GPU, Muse Glimmer on 2)

## Status: 0.1.0 (git repo + zip delivered)
- 39 tests: planner, config, llama-swap rendering, UI flows (fake backend), Docker API payloads
- Demo mode (`gpu-shepherd --demo`) verified in a headless browser
- Not yet run against real Docker/GPUs

## Next
- Run on the AI server (Ubuntu 26.04, 2x RTX 3070) and fix what breaks
- Make setup_aistack.sh install GPU Shepherd instead of its built-in panel
- Roadmap: Ollama/vLLM connectors, shared-GPU memory budgets, compose import, schedules
