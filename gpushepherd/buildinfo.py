"""Which commit this copy of GPU Shepherd was built from.

The Dockerfile runs this file on its own (standard library only, no git needed) against the build
context and saves the result in the image, so the dashboard can say exactly what is running and
whether an update is waiting. Run by hand:  python3 gpushepherd/buildinfo.py REPO_DIR OUT.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BUILD_INFO = Path(os.environ.get("SHEPHERD_BUILD_INFO", "/usr/share/gpu-shepherd/build.json"))


def _git_dir(repo: Path) -> Path | None:
    git = repo / ".git"
    if git.is_file():                    # worktree or submodule: "gitdir: <path>"
        text = git.read_text(errors="replace").strip()
        if text.startswith("gitdir:"):
            git = (repo / text[7:].strip()).resolve()
    return git if (git / "HEAD").is_file() else None


def _resolve(git: Path, ref: str) -> str:
    loose = git / ref
    if loose.is_file():
        return loose.read_text().strip()
    common = git / "commondir"           # worktrees keep shared refs in the main .git
    roots = [git] + ([(git / common.read_text().strip()).resolve()] if common.is_file() else [])
    for root in roots:
        if (root / ref).is_file():
            return (root / ref).read_text().strip()
        packed = root / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(errors="replace").splitlines():
                sha, _, name = line.partition(" ")
                if name == ref and len(sha) == 40:
                    return sha
    return ""


def read(repo: str | Path) -> dict:
    """{"commit": full sha, "branch": name or ""} for a git clone, or {} for anything else."""
    git = _git_dir(Path(repo))
    if not git:
        return {}
    head = (git / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        sha = _resolve(git, ref)
        branch = ref.removeprefix("refs/heads/")
    else:
        sha, branch = head, ""
    if len(sha) != 40:
        return {}
    return {"commit": sha, "branch": branch}


def load() -> dict:
    """The build info saved in the image ({} when running from a plain checkout or pip install)."""
    try:
        data = json.loads(BUILD_INFO.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 1
    info = read(argv[0]) | {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(argv[1]).write_text(json.dumps(info) + "\n")
    print(f"build info: {info.get('commit', 'not a git clone')[:12]} {info.get('branch', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
