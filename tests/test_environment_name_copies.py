"""``environment.yml`` names the conda environment every process activates; nothing else may
carry a different name. Most of its copies cannot be derived at runtime (an activation key in
YAML, a bare argument in a JSON array, prose with no invocation to inspect), so this holds an
enumerated list of copy sites instead, checked against ``environment.yml``'s own name, and named
as the enumeration it is: a copy added somewhere new is not caught until it is added here too.
A site is a file and the number of lines in it that carry the name, never a line number, so an
edit above a copy moves nothing here; a copy left behind by a rename drops its file's count,
and a copy added anywhere raises a count or adds a file, which the tree-wide comparison at the
end reports by name.
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


def _standalone_name_pattern() -> "re.Pattern[str]":
    """The name as its own token, never a hyphenated continuation of a longer identifier (an
    "X-tcip-agent-client-name" HTTP header names something else)."""
    return re.compile(r"(?<![\w-])" + re.escape(_environment_name()) + r"(?![\w-])")


def _count_in(path: Path) -> int:
    pattern = _standalone_name_pattern()
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if pattern.search(line))


def test_ci_activate_environment_keys_match():
    """Every setup-miniconda step in the workflow activates the environment by its name; the
    workflow is parsed rather than read by line so a step added above one moves nothing."""
    data = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    values = [
        step["with"]["activate-environment"]
        for job in data["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step.get("with"), dict) and "activate-environment" in step["with"]
    ]
    assert values == [_environment_name()] * 2


def test_mcp_json_launch_argument_matches():
    data = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    args = data["mcpServers"]["tcip"]["args"]
    assert args[args.index("-n") + 1] == _environment_name()


def test_environment_win_lock_name_matches():
    data = yaml.safe_load((REPO_ROOT / "environment.win.lock.yml").read_text(encoding="utf-8"))
    assert data["name"] == _environment_name()


# Repo-relative file and the number of its lines carrying the name, the structured files above
# included so the tree-wide comparison is total; environment.yml's count is its key plus a prose line.
_EXPECTED_COUNTS = {
    ".github/workflows/ci.yml": 2,
    ".mcp.json": 1,
    "ARCHITECTURE.md": 3,
    "CLAUDE.md": 1,
    "CONTRIBUTING.md": 2,
    "README.md": 2,
    "environment.linux.lock.yml": 2,
    "environment.yml": 2,
    "packages/tcip-web/README.md": 2,
    "scripts/distill_learnings.py": 1,
    "tools/smoke_fence_e2e.py": 1,
    "tools/smoke_phenology_e2e.py": 1,
    "tools/smoke_terminal_e2e.py": 1,
}


@pytest.mark.parametrize("rel,count", _EXPECTED_COUNTS.items(), ids=list(_EXPECTED_COUNTS))
def test_enumerated_copy_file_carries_the_name_the_expected_number_of_times(rel, count):
    assert _count_in(REPO_ROOT / rel) == count


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


def _name_occurrences() -> "dict[str, int]":
    found: "dict[str, int]" = {}
    for rel in _tracked_files():
        if rel in _EXCLUDED_FILES or rel.startswith(_EXCLUDED_PREFIXES):
            continue
        try:
            count = _count_in(REPO_ROOT / rel)
        except (UnicodeDecodeError, OSError):
            continue
        if count:
            found[rel] = count
    return found


def test_enumerated_copy_files_match_every_occurrence_the_tree_carries():
    actual = _name_occurrences()
    differences = {
        rel: (actual.get(rel), _EXPECTED_COUNTS.get(rel))
        for rel in sorted(set(actual) | set(_EXPECTED_COUNTS))
        if actual.get(rel) != _EXPECTED_COUNTS.get(rel)
    }
    assert not differences, f"tree count versus enumerated count, per file: {differences}"
