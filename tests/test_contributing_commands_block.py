"""Coverage: CONTRIBUTING.md claims its Commands block is CLAUDE.md's Commands block, verbatim.
Nothing enforced that claim before this test; it extracts the first fenced ```bash block from
each file and asserts the two are identical, so the two documents cannot drift apart silently.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_bash_block(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("```bash")
    end = lines.index("```", start + 1)
    return lines[start + 1 : end]


def test_contributing_commands_block_matches_claude_md():
    contributing_block = _first_bash_block(REPO_ROOT / "CONTRIBUTING.md")
    claude_md_block = _first_bash_block(REPO_ROOT / "CLAUDE.md")
    assert contributing_block == claude_md_block
