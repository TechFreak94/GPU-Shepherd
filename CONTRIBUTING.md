# Contributing

Thanks for helping! GPU Shepherd aims to stay small, generic and easy to read. This page covers
how to set up, what a good change looks like, and exactly what happens to a pull request from the
moment you open it to the moment it ships.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). To report a security
problem, don't open an issue: see [SECURITY.md](SECURITY.md).

## Ways to help

- **Report a bug or ask for a feature** with an [issue](../../issues/new/choose). Say what you did,
  what you expected and what happened, plus your version (click the version number at the top of
  the dashboard), distro, GPUs and driver. Never paste API keys, tokens, passwords or your
  `settings.yaml`; the log of a failed start is usually enough.
- **Improve the docs.** If something confused you, it confuses others.
- **Test on hardware we don't have.** Reports from other GPU generations, distros, Proxmox, Unraid,
  TrueNAS and WSL2 are especially valuable; see what is marked untested in the docs.
- **Add or update an App Center app**, a suggested model or a preset (see the rules below).
- **Fix something.** For anything bigger than a small fix, open an issue first so we agree on the
  direction before you spend time on it.

## Setup

```bash
pip install -e ".[dev]"
pytest
gpu-shepherd --demo --port 8090     # no Docker or GPU needed; log in as admin / demo-password
```

Demo mode uses fake GPUs and containers, so you can try every page and button. `pytest -q` is what
CI runs. Run a single file while you iterate. [docs/architecture.md](docs/architecture.md) explains
how the code fits together, and every module starts with a docstring that says what it owns.

## What a good change looks like

- Keep the scheduler pure: no I/O in `scheduler.py`. New GPU behaviour comes with tests.
- New settings get a form in the UI, a pydantic field with validation, and sensible defaults so
  existing YAML files keep loading. Add them to [docs/advanced.md](docs/advanced.md); new script or
  command options go in [docs/cli.md](docs/cli.md).
- New actions and settings go in `service.py` and are used from there by the dashboard route, an
  `/api/v1` endpoint (`api.py`, listed in [docs/api.md](docs/api.md)) and a `gpu-shepherd` command
  (`cli.py`, in [docs/cli.md](docs/cli.md#commands)), so all three can do everything.
- No JavaScript build step. Prefer htmx attributes and server-rendered templates; keep
  `static/app.js` small and free of inline scripts (the CSP forbids them).
- App Center apps (`instance_templates.py`) are generic starting points for widely used apps.
  They must work on any NVIDIA machine: no personal models, paths or hardware assumptions. A new
  app also needs its license and project page, an entry in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
  (a test keeps the two in step) and a test. Say how you tried it, and what you couldn't try
  (for example an image you couldn't download).
- User-facing text: plain language, say what will happen and why.
- Keep everything generic. Don't put your own hostnames, IP addresses, usernames, home folders,
  email addresses, API keys or tokens in code, tests, examples, docs or screenshots. Use
  `example.com`, `/srv/models`, `192.0.2.x` and the like.
- Don't add telemetry, network calls to third parties the user didn't ask for, or a new runtime
  dependency without discussing it in an issue first.
- Security: every state-changing dashboard route is `POST`, requires a login, and must be called
  by htmx (the `HX-Request` header). `/api/` routes need the API token for every change instead.
  Validate anything that ends up in a Docker call, a file path or a shell command. GPU Shepherd
  controls Docker, which is root-equivalent, so be strict.
- Docs and screenshots: update the docs a change would leave wrong. Don't retake screenshots unless
  the screen changed in a way a reader would notice; a reviewer will ask if one is needed.

## Pull requests

### Before you open one

1. Fork the repository, branch from `testing` and open the pull request against `testing` (see
   [Branches](#branches)).
2. Make one focused change. Unrelated fixes go in separate pull requests; they are reviewed and
   released separately.
3. Run `pytest -q`, and try the change in demo mode (`gpu-shepherd --demo`). Real GPUs are not
   required, but say what you could and couldn't test.
4. Sign off your commits (see [Sign-off](#sign-off-dco)).

### What goes in the description

The [pull request template](.github/pull_request_template.md) asks for:

- what changed and why, and how you tested it (screenshots help for UI changes);
- a **`## Changelog`** section: one bullet per change users will notice, written for people who
  use GPU Shepherd, not how it was done. Without one, the pull request title is used;
- whether it changes the config format, an API response or a command (a breaking change).

**Do not edit the version (`gpushepherd/__init__.py`) or `CHANGELOG.md`.** Merging a pull request
gives it its own version and changelog entry automatically, from the `## Changelog` section, so
two open pull requests never conflict over them. A maintainer sets the release label: `minor`,
`major` or `no release` when it isn't a normal `patch` step. Details:
[docs/updates.md](docs/updates.md#versions-and-releases).

### Sign-off (DCO)

Every commit in a pull request must carry a `Signed-off-by` line, added with `git commit -s`:

```
Signed-off-by: Your Name <you@example.com>
```

It says you wrote the change, or have the right to submit it, under the project's license (the
[Developer Certificate of Origin](https://developercertificate.org/)). You keep your copyright;
contributions are licensed under the [Apache License 2.0](LICENSE), as section 5 of that license
says. There is no CLA to sign. Use a name you are happy to have in the public history. GitHub's
`ID+username@users.noreply.github.com` address (Settings > Emails > *Keep my email addresses
private*) works as the email. Forgot? `git rebase --signoff testing` and force-push your branch.

Commits made by the maintainers and by automation are exempt.

### AI-assisted changes

Using an AI assistant is fine, and much of this project was written with one. You are still the
author: you must understand what you submit, have run it, and be able to answer questions about it.
Don't submit generated changes you haven't tested, and say so in the description if a tool wrote
most of it. Never paste secrets or private data into a tool.

### Branches

- **`testing`** is where development happens. Every pull request targets it, and merging into it
  does not release anything. People who want the newest changes early can follow it (Settings >
  Updates > branch, or `sudo ./update.sh --branch testing`); it may break.
- **`main`** only gets releases. A maintainer promotes `testing` to `main` with a pull request
  from `testing` to `main`, merged with a **merge commit** (not squashed, so the two branches keep
  the same history). That merge is one release: its `## Changelog` lists the changes it brings,
  each ending in its own pull request number, e.g. `- Faster starts (#61)`. A workflow fills this in
  (and replaces a placeholder title such as "testing") when the pull request is opened.
- After each release the Release workflow merges `main` back into `testing`, so `testing` has the
  new version number.
- Urgent fixes may go straight to `main` in their own pull request (squash-merged); the same step
  brings them into `testing`.

### Review and merge rules

- Every change reaches `testing` and `main` through a pull request. Nobody, maintainers included,
  pushes directly to either (the release bot's *Release X.Y.Z* commit, and its merge of `main`
  into `testing`, are the exceptions).
- The **Tests** check must pass, and a maintainer must approve. Pull requests from first-time
  contributors need a maintainer to start the checks.
- Code owners ([CODEOWNERS](.github/CODEOWNERS)) review changes to anything that touches how the
  project is built, released or installed or that handles credentials or Docker: `.github/`,
  `install.sh`, `update.sh`, `Dockerfile`, the compose files, `scripts/`, `auth.py`, `backend.py`
  and `updates.py`. Expect a closer look and more questions there.
- A maintainer will usually answer within a week. If you've heard nothing after that, a polite
  comment is fine. Small, focused, tested changes are reviewed fastest.
- Reviewers may ask for changes or decline a pull request that doesn't fit the project's scope.
  That's about the project, not about you.
- Resolve review comments by pushing fixes or answering them. Don't force-push over a review in
  progress unless asked; the pull request is squash-merged, so commit history within it doesn't
  matter.
- Pull requests into `testing` are **squash-merged** (promotions to `main` are merged, see
  [Branches](#branches)). The maintainer checks the `## Changelog` text before merge,
  because it is published word for word in the changelog and the GitHub release. Links, images
  and promotional text are removed.
- Merging into `main` releases: every pull request merged there (normally a promotion) becomes
  its own version, tag, GitHub release and container image.
- Stale pull requests with no reply for 30 days may be closed; reopen them any time.

### For maintainers

- Related changes may be batched on one shared branch and merged into `testing` as a single pull
  request, so a batch's `## Changelog` lists each change. Open it only once the maintainers have agreed it's
  ready.
- Review for: scope, tests, the three front ends staying in step (dashboard, API, command), docs,
  secrets or personal data in the diff, and new network calls or dependencies.
- Never merge a pull request whose checks you haven't seen pass, and never merge your own pull
  request without a second look at the diff.
- Urgent fixes may skip batching, but not review or tests.

## Releasing

Maintainers don't release by hand. The [Release workflow](.github/workflows/release.yml) does it
when a pull request merges into `main` (normally a promotion from `testing`). See [docs/updates.md](docs/updates.md#versions-and-releases).

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
