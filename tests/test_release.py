"""scripts/release.py (run by the release workflow) and CHANGELOG.md."""
import importlib.util
import json
from pathlib import Path

import pytest

from gpushepherd import __version__, changelog

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_changelog_matches_the_version():
    """The newest changelog entry is the version in gpushepherd/__init__.py (the workflow keeps them together)."""
    releases = changelog.read(ROOT / "CHANGELOG.md")
    assert releases and releases[0].version == __version__
    assert all(r.items and r.date for r in releases)
    versions = [changelog.key(r.version) for r in releases]
    assert versions == sorted(versions, reverse=True)


def test_bump_and_levels():
    assert release.bump("0.7.3", "major") == "1.0.0"
    assert release.bump("0.7.3", "minor") == "0.8.0"
    assert release.bump("0.7.3", "patch") == "0.7.4"
    assert release.level_for([]) == "patch"
    assert release.level_for(["bug", "Minor"]) == "minor"
    assert release.level_for(["patch", "major"]) == "major"
    assert release.level_for(["release:patch"]) == "patch"
    assert release.level_for(["No Release"]) is None


def test_entries_from_title_or_changelog_section():
    assert release.entries("Add a thing.", "", 12) == ["Add a thing (#12)"]
    body = """Some intro.

## Changelog
<!-- one line per change users notice -->
- Faster starts
- Logs show the image,
  and the tag
* Fixed a crash (#12)

## Test plan
- not this
"""
    assert release.entries("Title", body, 12) == ["Faster starts (#12)", "Logs show the image, and the tag (#12)",
                                                  "Fixed a crash (#12)"]
    assert release.entries("Direct push", None, None) == ["Direct push"]
    promotion = "## Changelog\n- Llama Swap name (#60)\n- One-GPU support (#61)\n- A fix made on testing\n"
    assert release.entries("Promote testing", promotion, 62) == ["Llama Swap name (#60)", "One-GPU support (#61)",
                                                                 "A fix made on testing (#62)"]


def test_insert_goes_above_the_newest_version():
    text = "# Changelog\n\nIntro.\n\n## 0.1.0 - 2026-01-01\n\n- First\n"
    out = release.insert(text, "0.2.0", "2026-02-02", ["Second (#2)"])
    assert out == "# Changelog\n\nIntro.\n\n## 0.2.0 - 2026-02-02\n\n- Second (#2)\n\n## 0.1.0 - 2026-01-01\n\n- First\n"
    assert [r.version for r in changelog.parse(out)] == ["0.2.0", "0.1.0"]


def test_pick_pr_prefers_the_merged_one():
    prs = [{"number": 3, "merged_at": None}, {"number": 5, "merged_at": "x", "merge_commit_sha": "abc"}]
    assert release.pick_pr(prs, "abc")["number"] == 5
    assert release.pick_pr([{"number": 3, "merged_at": None}]) is None


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "v.py").write_text('"""doc"""\n\n__version__ = "0.7.0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.7.0 - 2026-10-23\n\n- Old (#9)\n")
    monkeypatch.setattr(release, "VERSION_FILE", tmp_path / "v.py")
    monkeypatch.setattr(release, "CHANGELOG", tmp_path / "CHANGELOG.md")
    return tmp_path


def run(repo, capsys, prs, *extra):
    (repo / "pr.json").write_text(json.dumps(prs))
    assert release.main(["--pr-json", str(repo / "pr.json"), "--date", "2026-10-24",
                         "--notes", str(repo / "notes.md"), *extra]) == 0
    return dict(line.split("=", 1) for line in capsys.readouterr().out.split())


def test_main_releases_a_merged_pr(repo, capsys):
    pr = {"number": 13, "merged_at": "x", "title": "Show versions", "body": "", "labels": []}
    out = run(repo, capsys, [pr])
    assert out == {"skip": "false", "version": "0.7.1", "tag": "v0.7.1", "previous": "0.7.0", "level": "patch"}
    assert '__version__ = "0.7.1"' in (repo / "v.py").read_text()
    assert changelog.read(repo / "CHANGELOG.md")[0].items == ["Show versions (#13)"]
    assert (repo / "notes.md").read_text() == "- Show versions (#13)\n"
    assert run(repo, capsys, [pr])["skip"] == "true"          # a re-run releases nothing twice


def test_main_skips_and_direct_pushes(repo, capsys):
    pr = {"number": 14, "merged_at": "x", "title": "Typo", "labels": [{"name": "no release"}]}
    assert run(repo, capsys, [pr])["skip"] == "true"
    assert '"0.7.0"' in (repo / "v.py").read_text()
    out = run(repo, capsys, [], "--subject", "Fix the README")
    assert out["version"] == "0.7.1" and changelog.read(repo / "CHANGELOG.md")[0].items == ["Fix the README"]
