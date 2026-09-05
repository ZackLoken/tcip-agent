"""``environment.yml`` names the conda environment every process activates; nothing else may
carry a different name. Most of its copies cannot be derived at runtime (an activation key in
YAML, a bare argument in a JSON array, prose with no invocation to inspect), so this holds an
enumerated list of copy sites instead, checked against ``environment.yml``'s own name, and named
as the enumeration it is: a copy added somewhere new is not caught until it is added here too.
A separate test grepped the tracked tree for the name and asserts that grep's result equals this
enumeration, so a site missing from the list fails loudly instead of going unchecked.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment_name() -> str:
    data = yaml.safe_load((REPO_ROOT / "environment.yml").read_text(encoding="utf-8"))
    return data["name"]


def _line(path: Path, line_no: int) -> str:
    return path.read_text(encoding="utf-8").splitlines()[line_no - 1]


def test_ci_activate_environment_key_matches():
    line = _line(REPO_ROOT / ".github" / "workflows" / "ci.yml", 27)
    assert line.strip() == f"activate-environment: {_environment_name()}"


def test_ci_ray_exit_windows_activate_environment_key_matches():
    line = _line(REPO_ROOT / ".github" / "workflows" / "ci.yml", 154)
    assert line.strip() == f"activate-environment: {_environment_name()}"


def test_mcp_json_launch_argument_matches():
    data = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    args = data["mcpServers"]["tcip"]["args"]
    assert args[args.index("-n") + 1] == _environment_name()


def test_environment_win_lock_name_matches():
    data = yaml.safe_load((REPO_ROOT / "environment.win.lock.yml").read_text(encoding="utf-8"))
    assert data["name"] == _environment_name()


def _standalone_name_pattern() -> "re.Pattern[str]":
    """The name as its own token, never a hyphenated continuation of a longer identifier (an
    "X-tcip-agent-client-name" HTTP header names something else)."""
    return re.compile(r"(?<![\w-])" + re.escape(_environment_name()) + r"(?![\w-])")


def _sites_in(path: Path) -> "list[tuple[Path, int]]":
    """Every line of ``path`` carrying the standalone name, derived rather than pinned, for a
    document whose line numbers move with every edit above them."""
    pattern = _standalone_name_pattern()
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(path, n) for n, line in enumerate(lines, start=1) if pattern.search(line)]


_ARCHITECTURE_SITES = _sites_in(REPO_ROOT / "ARCHITECTURE.md")

# Enumerated, not derived, since none of these lines carries a structured key the name could be
# read from; ARCHITECTURE.md's sites are derived above because its line numbers shift under edits.
_PROSE_COPY_SITES = [
    (REPO_ROOT / "README.md", 73),
    (REPO_ROOT / "README.md", 95),
    (REPO_ROOT / "CONTRIBUTING.md", 14),
    (REPO_ROOT / "CONTRIBUTING.md", 26),
    (REPO_ROOT / "CLAUDE.md", 207),
    (REPO_ROOT / "environment.yml", 4),
    (REPO_ROOT / "environment.linux.lock.yml", 3),
    (REPO_ROOT / "environment.linux.lock.yml", 8),
    (REPO_ROOT / "packages" / "tcip-web" / "README.md", 44),
    (REPO_ROOT / "packages" / "tcip-web" / "README.md", 45),
    *_ARCHITECTURE_SITES,
    (REPO_ROOT / "scripts" / "distill_learnings.py", 8),
    (REPO_ROOT / "scripts" / "smoke_fence_e2e.py", 9),
    (REPO_ROOT / "scripts" / "smoke_phenology_e2e.py", 16),
    (REPO_ROOT / "scripts" / "smoke_terminal_e2e.py", 9),
    (REPO_ROOT / "scripts" / "watch_agent_chat.py", 13),
]


@pytest.mark.parametrize(
    "path,line_no", _PROSE_COPY_SITES,
    ids=[f"{p.name}:{n}" for p, n in _PROSE_COPY_SITES],
)
def test_enumerated_prose_copy_carries_the_environment_name(path, line_no):
    assert _environment_name() in _line(path, line_no)


# environment.yml itself, read structurally by _environment_name() above rather than as a copy.
_SOURCE_SITE = (REPO_ROOT / "environment.yml", 15)

# The structured sites, each checked by its own dedicated test above rather than by a literal
# line-content comparison.
_STRUCTURED_SITES = [
    (REPO_ROOT / ".github" / "workflows" / "ci.yml", 27),
    (REPO_ROOT / ".github" / "workflows" / "ci.yml", 154),
    (REPO_ROOT / ".mcp.json", 8),
]

# Out of scope: historical audit notes, the lock file (checked structurally above), the
# gitignore's unrelated build-artifact directory entry, and this file's own regex literal.
_EXCLUDED_PREFIXES = ("docs/audit/", "docs/decisions/")
_EXCLUDED_FILES = {
    "environment.win.lock.yml", ".gitignore",
    str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
}


def _tracked_files() -> "list[str]":
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()


def _name_occurrences() -> "set[tuple[str, int]]":
    # A real copy is the standalone token, never a hyphenated continuation of a longer
    # identifier (an "X-tcip-agent-client-name" HTTP header names something else).
    pattern = re.compile(r"(?<![\w-])" + re.escape(_environment_name()) + r"(?![\w-])")
    found: "set[tuple[str, int]]" = set()
    for rel in _tracked_files():
        if rel in _EXCLUDED_FILES or rel.startswith(_EXCLUDED_PREFIXES):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                found.add((rel, line_no))
    return found


def test_enumerated_copy_sites_match_every_occurrence_the_tree_carries():
    expected = {
        (str(path.relative_to(REPO_ROOT)).replace("\\", "/"), line_no)
        for path, line_no in [_SOURCE_SITE, *_STRUCTURED_SITES, *_PROSE_COPY_SITES]
    }
    actual = _name_occurrences()
    assert actual == expected, (
        f"missing from the enumeration: {actual - expected}; "
        f"enumerated but not found by the grep: {expected - actual}"
    )
