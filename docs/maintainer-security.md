# Maintaining the repository safely

For the people who own a GPU Shepherd repository (or a fork). Contributors don't need this page;
see [CONTRIBUTING.md](../CONTRIBUTING.md). The one thing to remember: **whoever can change `main`
controls the code that every user's `update.sh` rebuilds and runs as root, and the container image
published from it.** These settings protect that.

## Protect `main`

Settings > Rules > Rulesets (or Branches > branch protection) for `main`:

- Require a pull request before merging; block force pushes and deletion.
- Require the **Tests** status check to pass, with the branch up to date.
- Require conversation resolution. Squash merge only (Settings > General > Pull Requests; the
  release workflow expects one commit per pull request). Turn on *Automatically delete head
  branches*.
- Required approvals: with one maintainer, nobody else can approve your own pull request, so keep
  it at 0 and rely on the required checks, or add a second maintainer and require 1 approval plus
  *Require review from Code Owners* ([CODEOWNERS](../.github/CODEOWNERS)).
- Require signed commits only if every maintainer signs them; it is not needed for contributors.
- Protect tags `v*` so releases can't be moved (Rules > Tags).

## Releases and protected branches

The [Release workflow](../.github/workflows/release.yml) pushes the *Release X.Y.Z* commit and the
tag to `main` itself. A rule that requires pull requests stops the workflow's built-in token, so
either let it through or release another way:

1. Create a GitHub App (or a fine-grained token limited to this repository with *Contents:
   read and write*), add it to the ruleset's **bypass list**, and save its token as the repository
   secret `RELEASE_TOKEN`. The workflow uses it when present.
2. Merge a small change and check that its release appears (a `no release` label skips the
   release, so don't use it for this test). Test this before relying on it.

Keep `RELEASE_TOKEN` as narrow and short-lived as possible, and don't give it to workflows that run
pull request code.

## Actions

Settings > Actions > General:

- Workflow permissions: **read** by default (the workflows ask for what they need).
- *Require approval for all outside collaborators* before workflows run on their pull requests.
- Allow only actions you trust (GitHub-owned and verified creators), pinned by Dependabot.
- Never use `pull_request_target` with a checkout of pull request code, and never echo secrets.

## Secrets and vulnerabilities

Settings > Advanced Security (Code security):

- Turn on **Private vulnerability reporting** (the way [SECURITY.md](../SECURITY.md) asks to be
  contacted), **Dependabot alerts and security updates**, **secret scanning** and **push
  protection**.
- The repository's [Dependabot config](../.github/dependabot.yml) opens weekly update pull
  requests for the Actions, Python and Docker dependencies. Create the labels `dependencies` and
  `no release` first; each merge makes a release.
- Optional: CodeQL code scanning for Python.

## Packages and releases

- The container image is published to `ghcr.io/<owner>/gpu-shepherd`. Set the package's
  visibility to public when the repository is, and link it to the repository.
- Check what a release contains before you publish one by hand; the workflow's notes come from the
  `## Changelog` section of the pull request.

## Accounts

- Turn on two-factor authentication (a passkey or security key is best) for every account with
  write access, and give people the least role they need.
- Commit with a no-reply address (Settings > Emails > *Keep my email addresses private* and
  *Block command line pushes that expose my email*), so the public history carries no personal
  email address.
- Keep tokens in repository or organisation secrets, not in files.
