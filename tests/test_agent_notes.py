"""AGENTS.md (read by OpenCode, Aider, Cline and other agents) mirrors CLAUDE.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _body(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").split("\n", 1)[1]


def test_agents_md_matches_claude_md():
    assert _body("AGENTS.md") == _body("CLAUDE.md"), "Change AGENTS.md and CLAUDE.md together"
