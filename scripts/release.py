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
there is one, otherwise its title. A promotion (testing -> main) without its own bullets lists those
of the pull requests it brings (--included-json); --fill-promotion writes them into the open
promotion pull request before it is merged (.github/workflows/promotion.yml). Writes `version=`, `tag=` and `skip=` lines to stdout (for
$GITHUB_OUTPUT) and the release notes to --notes.

  python3 scripts/release.py --pr-json pr.json [--included-json prs.json] [--sha SHA] [--notes notes.md]
  python3 scripts/release.py --pr-json open-pr.json --included-json prs.json --fill-promotion DIR
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
PROMOTION_HEAD = "testing"
# marks a Changelog section filled in from the pull requests a promotion brings (refreshed, not hand-written)
AUTO_MARK = "<!-- changelog: filled in from the pull requests merged into testing -->"


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


def changelog_items(body: str | None) -> list[str]:
    """The bullets under a "Changelog" heading in a pull request's description (comments ignored)."""
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
    return items


def entries(title: str, body: str | None, number: int | None) -> list[str]:
    """The changelog bullets for a pull request, each ending in (#number)."""
    items = changelog_items(body) or [title.strip().rstrip(".")]
    ref = f" (#{number})" if number else ""
    # a promotion from testing lists the pull requests it brings, each already ending in its own (#N)
    return [i if not ref or re.search(r"\(#\d+\)$", i) else i + ref for i in items]


def is_promotion(pr: dict) -> bool:
    """A pull request from testing into main (the release of everything waiting on testing)."""
    return (pr.get("head") or {}).get("ref") == PROMOTION_HEAD and (pr.get("base") or {}).get("ref") == "main"


def included_entries(included: list[dict]) -> list[str]:
    """The changelog of a promotion: every pull request it brings from testing, oldest first, each
    with its own bullets and number. Pull requests labelled "no release" are left out."""
    seen: set[int] = set()
    items: list[str] = []
    for p in sorted((p for p in included if isinstance(p, dict) and p.get("number")), key=lambda p: p["number"]):
        if p["number"] in seen or level_for([lb.get("name", "") for lb in p.get("labels") or []]) is None:
            continue
        seen.add(p["number"])
        items += entries(p.get("title") or "", p.get("body"), p["number"])
    return items


def promotion_items(pr: dict, included: list[dict]) -> list[str]:
    """What a promotion releases: its own hand-written Changelog bullets if it has any, otherwise
    (no section, an empty one, or the one this script filled in) those of the pull requests it brings."""
    own = changelog_items(pr.get("body"))
    if own and AUTO_MARK not in (pr.get("body") or ""):
        return entries(pr.get("title") or "", pr.get("body"), pr.get("number"))
    return included_entries(included) or entries(pr.get("title") or "", pr.get("body"), pr.get("number"))


GENERIC_TITLES = {"", "testing", "main", "testing to main", "merge testing", "merge testing into main", "promote testing"}


def promotion_body(pr: dict, included: list[dict]) -> tuple[str, str] | None:
    """A new (title, body) for an open promotion pull request, or None when it needs nothing.

    Fills its Changelog section with the bullets of the pull requests it brings (marked, so a later
    run refreshes it as more land on testing) and replaces a placeholder title such as "testing".
    A Changelog the maintainer wrote by hand is left alone."""
    title, body = (pr.get("title") or "").strip(), pr.get("body") or ""
    numbers = sorted({p["number"] for p in included if isinstance(p, dict) and p.get("number")})
    new_title = title
    if title.lower().rstrip(".") in GENERIC_TITLES and numbers:
        new_title = "Promote testing to main: " + ", ".join(f"#{n}" for n in numbers)
    new_body = body
    if not changelog_items(body) or AUTO_MARK in body:
        items = included_entries(included)
        if items:
            section = f"## Changelog\n\n{AUTO_MARK}\n" + "".join(f"- {i}\n" for i in items) + "\n"
            m = re.search(r"^#{1,4}\s*Changelog\s*$.*?(?=^#{1,4}\s|\Z)", body, flags=re.M | re.S | re.I)
            if m:
                new_body = body[:m.start()] + section + body[m.end():]
            else:
                new_body = body.rstrip("\n") + ("\n\n" if body.strip() else "") + section
    if (new_title, new_body) == (title, body):
        return None
    return new_title, new_body


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
    ap.add_argument("--included-json", default="",
                    help="for a promotion: JSON list of the pull requests merged into testing that it brings")
    ap.add_argument("--fill-promotion", metavar="DIR", default="",
                    help="don't release: --pr-json is one open promotion pull request; write a new title.txt "
                         "and body.md for it to DIR (and changed=true) when its title or Changelog is missing")
    a = ap.parse_args(argv)

    included = json.loads(Path(a.included_json).read_text() or "[]") if a.included_json else []
    included = included if isinstance(included, list) else []
    if a.fill_promotion:
        new = promotion_body(json.loads(Path(a.pr_json).read_text()), included)
        if new:
            out = Path(a.fill_promotion)
            (out / "title.txt").write_text(new[0])
            (out / "body.md").write_text(new[1])
        print(f"changed={'true' if new else 'false'}")
        return 0

    prs = json.loads(Path(a.pr_json).read_text() or "[]")
    pr = pick_pr(prs if isinstance(prs, list) else [], a.sha)
    if pr:
        number, title, body = pr.get("number"), pr.get("title") or "", pr.get("body")
        level = level_for([lb.get("name", "") for lb in pr.get("labels") or []])
    else:                                       # pushed straight to main
        number, title, body, level = None, a.subject, None, "patch"
    changelog = CHANGELOG.read_text()
    if pr and (included or is_promotion(pr)):
        items = promotion_items(pr, included)
    elif not pr and included:                   # a promotion GitHub didn't record as a merged pull request
        items = included_entries(included)
    else:
        items = entries(title, body, number)
    released = {line[2:] for line in changelog.splitlines() if line.startswith("- ")}
    if (level is None or not title.strip() or (number and re.search(rf"\(#{number}\)$", changelog, flags=re.M))
            or (items and all(re.search(r"\(#\d+\)$", i) and i in released for i in items))):
        print("skip=true")                      # asked to skip, nothing to describe, or already released
        return 0
    old = read_version(VERSION_FILE.read_text())
    new = bump(old, level)
    VERSION_FILE.write_text(VERSION_LINE.sub(f'__version__ = "{new}"', VERSION_FILE.read_text(), count=1))
    CHANGELOG.write_text(insert(changelog, new, a.date, items))
    if a.notes:
        Path(a.notes).write_text("".join(f"- {i}\n" for i in items))
    print(f"skip=false\nversion={new}\ntag=v{new}\nprevious={old}\nlevel={level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
