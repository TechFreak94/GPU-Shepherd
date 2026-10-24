"""CHANGELOG.md: reading it, comparing versions, and turning entries into safe HTML.

The changelog of the running version ships in the image. update.sh also saves the one from the
newest version on the remote (<data dir>/update/changelog.md), so the dashboard can show what an
update brings before it is installed. Only the small subset of Markdown the changelog uses is
understood: "## X.Y.Z - YYYY-MM-DD" headings, "- " bullets (continued on indented lines),
`code`, **bold** and [links](https://...).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from markupsafe import Markup, escape

HERE = Path(__file__).resolve().parent
HEADING = re.compile(r"^##\s+\[?v?(\d+\.\d+\.\d+)\]?(?:\s+-\s+(\d{4}-\d{2}-\d{2}))?\s*$")


@dataclass
class Release:
    version: str
    date: str = ""
    items: list[str] = field(default_factory=list)


def find() -> Path | None:
    """The changelog of the running version: in the image, or next to the package in a checkout."""
    for p in (os.environ.get("SHEPHERD_CHANGELOG", ""), "/usr/share/gpu-shepherd/CHANGELOG.md",
              HERE.parent / "CHANGELOG.md"):
        if p and Path(p).is_file():
            return Path(p)
    return None


def key(version: str) -> tuple[int, ...]:
    """Sortable form of "1.10.2"; anything unreadable sorts first."""
    try:
        return tuple(int(x) for x in version.strip().lstrip("v").split("."))
    except ValueError:
        return (-1,)


def parse(text: str) -> list[Release]:
    out: list[Release] = []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            out.append(Release(m.group(1), m.group(2) or ""))
        elif line.startswith("#"):
            if out and not line.startswith("###"):
                out.append(Release(""))          # some other section: stop collecting into the last release
        elif out and out[-1].version:
            if re.match(r"^[-*]\s+", line):
                out[-1].items.append(line[1:].strip())
            elif line.startswith((" ", "\t")) and line.strip() and out[-1].items:
                out[-1].items[-1] += " " + line.strip()
    return [r for r in out if r.version]


def read(path: Path | None) -> list[Release]:
    try:
        return parse(Path(path).read_text(errors="replace")) if path else []
    except OSError:
        return []


def newer(releases: list[Release], than: str) -> list[Release]:
    return [r for r in releases if key(r.version) > key(than)]


def inline(text: str) -> Markup:
    """One changelog line as HTML: escaped first, then `code`, **bold** and http(s) links."""
    s = str(escape(text))
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)\"]+)\)", r'<a class="ext" href="\2" target="_blank" rel="noopener">\1</a>', s)
    return Markup(s)
