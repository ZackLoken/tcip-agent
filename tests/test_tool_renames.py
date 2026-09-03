"""Guards each MCP tool rename named in the ruling: the new name registers, and the old name
survives nowhere the platform ships. CHANGELOG.md and everything under docs/ are project
history, never edited by a rename, and are excluded from every sweep in this module.

test_tool_manifest.py documents every retired tool name in its own removed set, the old name
from a landed rename among them; that literal set is the one legitimate place an old name
survives on purpose, so the whole file is excluded from the sweep rather than the one set.

run_hyperparameter_search's internal training-loop helper stays named ``_run_hpo_trial``, since
it is not the tool, and the HPO store names, ``pipelines/training/hpo.py``, and the ``hpo``
state directory are none of them the literal old token ``run_hpo`` either: an underscore or
another word sits against every one of them, so the whole-word sweep below would never have
flagged them, and every other place the old name appeared, tests included, was renamed by hand.

focus is common CSS/DOM vocabulary outside the tool (frontend ``.focus()`` calls, ``autoFocus``
props, ``:focus`` selectors), so its sweep is scoped to the Python tool surface, the tests, the
knowledge documents and ARCHITECTURE.md, and matches only the patterns that could plausibly
name the tool rather than every use of the English word.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

RENAMES = [
    ("calibrate_ordinal_regression_operating_point", "calibrate_scalar_operating_point"),
    ("cancel_hpo", "cancel_hyperparameter_search"),
    ("run_hpo", "run_hyperparameter_search"),
    ("claude_reports", "report_friction"),
]

_OWN_FILE = str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/")
_EXCLUDED_FILES = {_OWN_FILE, "tests/test_tool_manifest.py"}
_EXCLUDED_PREFIXES = ("docs/", ".claude/worktrees/", "packages/tcip-web/static/assets/")
# Extensions a text read would corrupt or fail on; skipped rather than reported.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".pyc", ".db", ".sqlite", ".onnx", ".pt", ".pth",
}

_FOCUS_SCOPE_PREFIXES = ("packages/tcip-mcp/", "tests/")
_FOCUS_SCOPE_FILES = {"ARCHITECTURE.md"}
_FOCUS_PATTERNS = [
    re.compile(r"\bdef focus\("),
    re.compile(r"\.focus\("),
    re.compile(r"[`\"']focus[`\"']"),
]


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line != "CHANGELOG.md"]


def _in_scope(rel: str) -> bool:
    if rel in _EXCLUDED_FILES:
        return False
    if any(rel.startswith(prefix) for prefix in _EXCLUDED_PREFIXES):
        return False
    if "node_modules" in rel.split("/"):
        return False
    return Path(rel).suffix.lower() not in _BINARY_SUFFIXES


def _read(rel: str) -> str | None:
    try:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _whole_word_sites(name: str, files: list[str]) -> list[str]:
    """Every file in ``files`` naming ``name`` as its own token, never a hyphenated or
    underscored continuation of a longer identifier."""
    pattern = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])")
    hits = []
    for rel in files:
        text = _read(rel)
        if text is not None and pattern.search(text):
            hits.append(rel)
    return hits


def _focus_sites(files: list[str]) -> list[str]:
    """Sites of the scoped focus patterns, restricted to the files the docstring above names."""
    hits = []
    for rel in files:
        if not (rel in _FOCUS_SCOPE_FILES or any(rel.startswith(p) for p in _FOCUS_SCOPE_PREFIXES)):
            continue
        text = _read(rel)
        if text is None:
            continue
        if any(p.search(text) for p in _FOCUS_PATTERNS):
            hits.append(rel)
    return hits


def _old_name_sites(old: str, files: list[str]) -> list[str]:
    if old == "focus":
        return _focus_sites(files)
    return _whole_word_sites(old, files)


@pytest.mark.parametrize("old,new", RENAMES, ids=[f"{o}->{n}" for o, n in RENAMES])
def test_new_tool_name_is_registered(old, new):
    from tcip_mcp.server import list_registered_tools

    assert new in set(list_registered_tools())


@pytest.mark.parametrize("old,new", RENAMES, ids=[f"{o}->{n}" for o, n in RENAMES])
def test_old_tool_name_survives_nowhere_tracked(old, new):
    files = [f for f in _tracked_files() if _in_scope(f)]
    sites = _old_name_sites(old, files)
    assert not sites, f"{old!r} still appears in: {sites}"
