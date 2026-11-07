#!/usr/bin/env python3
"""Give a merged pull request its own version: bump gpushepherd/__init__.py and add a CHANGELOG.md entry.

Run by .github/workflows/release.yml after every push to main, with the pull request the pushed
commit came from (the JSON GitHub's "list pull requests for a commit" API returns). Standard
library only. How big the step is comes from the pull request's labels:

  major       1.2.3 -> 2.0.0   something that works differently or needs action after updating
  minor       1.2.3 -> 1.3.0   a bigger milestone worth marking
  patch       1.2.3 -> 1.2.4   (the default) every other change: features, improvements, fixes
  no release  no new version and no entry (e.g. a typo in a comment)

The entry is the bullet list under a "Changelog" heading in the pull request's description when
there is one, otherwise its title. Writes `version=`, `tag=` and `skip=` lines to stdout (for
$GITHUB_OUTPUT) and the release notes to --notes.

  python3 scripts/release.py --pr-json pr.json [--sha SHA] [--notes notes.md] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "gpushepherd" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"
VERSION_LINE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.M)
LEVELS = ("major", "minor", "patch")
SKIP_LABELS = {"no release", "no-release", "skip release", "skip-release"}


def read_version(text: str) -> str:
    m = VERSION_LINE.search(text)
    if not m:
        raise SystemExit(f"no __version__ line in {VERSION_FILE}")
    return ".".join(m.groups())


def bump(version: str, level: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def level_for(labels: list[str]) -> str | None:
    """The bump the labels ask for (the biggest wins), "patch" without any, None to skip."""
    names = {n.strip().lower() for n in labels}
    if names & SKIP_LABELS:
        return None
    for level in LEVELS:
        if level in names or f"release:{level}" in names:
            return level
    return "patch"


def entries(title: str, body: str | None, number: int | None) -> list[str]:
    """The changelog bullets for a pull request, each ending in (#number)."""
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    items: list[str] = []
    section = re.search(r"^#{1,4}\s*Changelog\s*$(.*?)(?=^#{1,4}\s|\Z)", body, flags=re.M | re.S | re.I)
    if section:
        for line in section.group(1).splitlines():
            m = re.match(r"^\s*[-*]\s+(.*\S)", line)
            if m:
                items.append(m.group(1))
            elif line.startswith(("  ", "\t")) and line.strip() and items:
                items[-1] += " " + line.strip()
    if not items:
        items = [title.strip().rstrip(".")]
    ref = f" (#{number})" if number else ""
    # a promotion from testing lists the pull requests it brings, each already ending in its own (#N)
    return [i if not ref or re.search(r"\(#\d+\)$", i) else i + ref for i in items]


def insert(changelog: str, version: str, date: str, items: list[str]) -> str:
    """The changelog with a new section on top (above the first "## " heading)."""
    section = f"## {version} - {date}\n\n" + "".join(f"- {i}\n" for i in items) + "\n"
    m = re.search(r"^## ", changelog, flags=re.M)
    if not m:
        return changelog.rstrip("\n") + "\n\n" + section
    return changelog[:m.start()] + section + changelog[m.start():]


def pick_pr(prs: list[dict], sha: str = "") -> dict | None:
    """The merged pull request that produced this commit (GitHub can list unrelated open ones too)."""
    merged = [p for p in prs if isinstance(p, dict) and p.get("merged_at")]
    for p in merged:
        if sha and p.get("merge_commit_sha") == sha:
            return p
    return merged[0] if merged else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pr-json", required=True, help="output of GET /repos/{repo}/commits/{sha}/pulls")
    ap.add_argument("--sha", default="")
    ap.add_argument("--subject", default="", help="commit subject, used when no pull request is found")
    ap.add_argument("--notes", default="", help="write the release notes (Markdown) here")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    a = ap.parse_args(argv)

    prs = json.loads(Path(a.pr_json).read_text() or "[]")
    pr = pick_pr(prs if isinstance(prs, list) else [], a.sha)
    if pr:
        number, title, body = pr.get("number"), pr.get("title") or "", pr.get("body")
        level = level_for([lb.get("name", "") for lb in pr.get("labels") or []])
    else:                                       # pushed straight to main
        number, title, body, level = None, a.subject, None, "patch"
    changelog = CHANGELOG.read_text()
    if level is None or not title.strip() or (number and re.search(rf"\(#{number}\)$", changelog, flags=re.M)):
        print("skip=true")                      # asked to skip, nothing to describe, or already released
        return 0
    old = read_version(VERSION_FILE.read_text())
    new = bump(old, level)
    items = entries(title, body, number)
    VERSION_FILE.write_text(VERSION_LINE.sub(f'__version__ = "{new}"', VERSION_FILE.read_text(), count=1))
    CHANGELOG.write_text(insert(changelog, new, a.date, items))
    if a.notes:
        Path(a.notes).write_text("".join(f"- {i}\n" for i in items))
    print(f"skip=false\nversion={new}\ntag=v{new}\nprevious={old}\nlevel={level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
