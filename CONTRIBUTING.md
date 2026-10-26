# Contributing

Thanks for helping! GPU Shepherd aims to stay small and easy to read.

## Setup

```bash
pip install -e ".[dev]"
pytest
gpu-shepherd --demo --port 8090     # no Docker or GPU needed; log in as admin / demo-password
```

## Guidelines

- Keep the scheduler pure: no I/O in `scheduler.py`. New GPU behaviour comes with tests.
- New settings get a form in the UI, a pydantic field with validation, and sensible defaults so
  existing YAML files keep loading. Add them to [docs/advanced.md](docs/advanced.md); new script or
  command options go in [docs/cli.md](docs/cli.md).
- New actions and settings go in `service.py` and are used from there by the dashboard route, an
  `/api/v1` endpoint (`api.py`, listed in [docs/api.md](docs/api.md)) and a `gpu-shepherd` command
  (`cli.py`, in [docs/cli.md](docs/cli.md#commands)), so all three can do everything.
- No JavaScript build step. Prefer htmx attributes and server-rendered templates; keep
  `static/app.js` small and free of inline scripts (the CSP forbids them).
- App Store apps (`instance_templates.py`) are generic starting points for widely used apps.
  They must work on any NVIDIA machine: no personal models, paths or hardware assumptions.
- User-facing text: plain language, say what will happen and why.
- Security: every state-changing dashboard route is `POST`, requires a login, and must be called
  by htmx (the `HX-Request` header). `/api/` routes need the API token for every change instead.
  Validate anything that ends up in a Docker call.

## Pull requests

Run `pytest` and try your change in demo mode. Describe what you changed and how you tested it;
screenshots help for UI changes.

Don't edit the version or `CHANGELOG.md`: merging gives the pull request its own version and
changelog entry automatically. Put the entry under a `## Changelog` heading in the description
(one bullet per change users will notice; the title is used otherwise), and label the pull request
`minor`, `major` or `no release` when it isn't a normal `patch` step. Details:
[docs/updates.md](docs/updates.md#versions-and-releases).

By contributing you agree that your contributions are licensed under the Apache License 2.0.
