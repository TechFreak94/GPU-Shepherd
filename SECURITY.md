# Security policy

## Reporting a vulnerability

**Please don't open a public issue, discussion or pull request for a security problem.**

Report it privately with GitHub's
[private vulnerability reporting](https://github.com/TechFreak94/GPU-Shepherd/security/advisories/new)
(the repository's *Security* tab > *Report a vulnerability*). Only the maintainers can see the
report, and we can discuss it, prepare a fix and publish an advisory there. There is no email
address for reports; please don't look for one.

Helpful details: the version (top of the dashboard), how it is installed, what you did, what
happened, and why it matters. A minimal way to reproduce it is the most useful thing you can send.
Don't include real passwords, tokens or other people's data.

What to expect: we aim to acknowledge a report within 7 days, tell you whether we consider it a
vulnerability, and keep you updated until it is fixed. GPU Shepherd is maintained by volunteers, so
those are goals, not guarantees. We'll credit you in the advisory unless you prefer not to be named.
Please give us a reasonable time (90 days is typical) to release a fix before you publish details.

Good-faith research on your own installation is welcome. Don't test against machines you don't own,
and don't access, change or delete other people's data.

## Supported versions

GPU Shepherd is 0.x software. Only the **latest release** gets security fixes; update with
`sudo ./update.sh` or the dashboard's Update button (see [docs/updates.md](docs/updates.md)).

## What GPU Shepherd is, security-wise

Knowing the design helps decide what is and isn't a vulnerability.

- **It controls Docker.** It talks to the Docker socket, which is root-equivalent on the host.
  Anyone who can log in to the dashboard or use the API token can run containers and mount host
  folders. That is the product, not a bug; so the first line of defence is who can reach it.
- **Keep it off the internet.** Run it on a trusted network, or behind a VPN or an authenticating
  reverse proxy. Don't forward its port from your router. The apps it starts (ComfyUI, for one)
  may have no login of their own.
- **Dashboard login.** One admin account; the password is stored scrypt-hashed in `settings.yaml`
  (mode 600). The session cookie is `SameSite=strict`; state-changing routes are `POST`, need a
  login and the htmx header; a strict Content-Security-Policy forbids inline scripts. The initial
  password is printed once and kept in `INITIAL_PASSWORD.txt` until it is changed.
- **API and command line.** `/api/v1` needs the API token (in the data folder, mode 600) for every
  change. Anyone holding it has the same power as the admin login.
- **Helper containers.** A few jobs run in short-lived helper containers, some privileged: model
  downloads (`curl`), moving a folder to another disk, and looking at and mounting attached
  drives (`drives.py`, which only mounts drives that already hold data and never formats or writes
  to them). They carry the `gpushepherd.helper` label.
- **The update service.** `update.sh` runs on the host as root, fast-forwards the clone you
  installed from, rebuilds the image and restarts GPU Shepherd. An optional systemd unit lets the
  dashboard ask for it. A GitHub token saved for a private repository is kept in the clone's
  `.git` folder, mode 0600, and is never logged. So whoever controls the repository you update
  from controls the code that runs as root on your machine: install from the official repository
  or your own fork, and treat who has write access to it accordingly.
- **Release images.** Each merged change publishes a container image to `ghcr.io`. Pin
  `SHEPHERD_VERSION` to a release if you want updates only when you choose.
- **Apps and models.** Apps and models are downloaded from their own publishers' registries and
  sites and keep their own licenses and security properties. Report problems in them upstream.

In scope: the dashboard, API, command line, `install.sh`, `update.sh`, the release workflow, the
published image and the templates for the apps. Examples: bypassing login or CSRF protection,
command or path injection that reaches Docker or the host, leaking the API token or a password,
a malicious config or model-list file escaping its folder, a release or update path that runs
untrusted code.

Out of scope: needing the admin login or API token to do what they're for; exposing the dashboard
to the internet against the advice above; vulnerabilities in third-party apps or images; denial of
service by a logged-in admin; missing hardening that needs a stronger setup than a trusted network.

## For maintainers

Repository settings that keep the project safe (branch protection, required checks, secret
scanning with push protection, Dependabot, private vulnerability reporting, Actions permissions,
two-factor authentication) are listed in
[docs/maintainer-security.md](docs/maintainer-security.md).
