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
props, ``:focus`` selectors, ``onFocus`` handlers, ``focus-`` Tailwind variants), so its
patterns match only what could plausibly name the tool: ``def focus(``, a bare ``focus(`` call,
``import focus``, or the quoted literal ``focus``. Each pattern carves out a DOM or CSS use by
the character immediately before the match rather than by scoping the check to a directory, so
it runs over the same tracked-file scope as every other rename.

A record a project stored under an old door name before that door was renamed is real history,
not a leftover to fix: a line trailing the marker ``# a stored value written before the
rename`` is skipped by every sweep below, the one way a fixture may reproduce a record's actual
historical shape without being read as a missed rename.

Every sweep checks a tracked path itself, not only the text inside it, and matches an old name
case-insensitively; the whole-word boundary still treats an underscore as a word character, so
a legitimately renamed file or identifier that merely contains the old token as a substring
(``test_tabulate_counts_bucket_regime.py`` before its own rename, say) is not itself proof the
sweep would have caught it.
A merge retires the absorbed door's own name outright: the surviving door serves its view under
an argument, so the merged-away name gets the identical whole-word, whole-tree sweep a rename's
old name gets, with no scoping needed.
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
    ("project_retrospective", "write_retrospective"),
    ("domain_knowledge", "serve_domain_knowledge"),
    ("focus", "focus_human_attention"),
    ("force_redraw_cal_holdout_split", "redraw_calibration_holdout"),
    ("make_splits", "draw_splits"),
    ("push_panel_data", "push_panel_event"),
    ("update_trait_spec_fields", "revise_trait_spec"),
    ("compute_phenology", "deliver_phenology_milestones"),
    ("tabulate_counts", "deliver_per_image_counts"),
    ("select_best_model", "rank_registered_models"),
    ("check_training_status", "monitor_training"),
    ("init_project", "initialize_project"),
    ("set_active_project", "activate_project"),
]

# A merge retires the absorbed door's name outright (the surviving door serves its view under an
# argument), so the whole-word sweep applies exactly as it does for a rename's old name.
MERGED = [
    ("list_training_runs", "list_experiments"),
    ("list_registered_models", "rank_registered_models"),
    ("stage_accepted_proposals", "stage_proposals"),
    ("export_predictions", "run_inference"),
]

# A demotion folded into a script with no single surviving door gets the same whole-tree sweep a
# merge's old name gets; there is no survivor tool name to check registration of, only "script".
RETIRED = [
    ("validate_data_quality", "script"),
]

_OWN_FILE = str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/")
_EXCLUDED_FILES = {_OWN_FILE, "tests/test_tool_manifest.py"}
_EXCLUDED_PREFIXES = ("docs/",)
# Extensions a text read would corrupt or fail on; skipped rather than reported.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".pyc", ".db", ".sqlite", ".onnx", ".pt", ".pth",
}

# Skips a line reproducing a real record's shape from before a door's own rename.
_HISTORICAL_VALUE_MARKER = "# a stored value written before the rename"

_FOCUS_PATTERNS = [
    re.compile(r"\bdef focus\("),
    re.compile(r"(?<![.\w:-])focus\("),
    re.compile(r"[`\"']focus[`\"']"),
    re.compile(r"\bimport focus\b"),
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
    """Every tracked path or line of file content in ``files`` naming ``name`` as its own token,
    case-insensitively, never a hyphenated or underscored continuation of a longer identifier. A
    line carrying ``_HISTORICAL_VALUE_MARKER`` is skipped."""
    pattern = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.IGNORECASE)
    hits = []
    for rel in files:
        if pattern.search(rel):
            hits.append(rel)
            continue
        text = _read(rel)
        if text is None:
            continue
        for line in text.splitlines():
            if _HISTORICAL_VALUE_MARKER in line:
                continue
            if pattern.search(line):
                hits.append(rel)
                break
    return hits


def _focus_sites(files: list[str]) -> list[str]:
    """Sites of the scoped focus patterns, over every tracked path and line of file content. A
    line carrying ``_HISTORICAL_VALUE_MARKER`` is skipped."""
    hits = []
    for rel in files:
        if any(p.search(rel) for p in _FOCUS_PATTERNS):
            hits.append(rel)
            continue
        text = _read(rel)
        if text is None:
            continue
        for line in text.splitlines():
            if _HISTORICAL_VALUE_MARKER in line:
                continue
            if any(p.search(line) for p in _FOCUS_PATTERNS):
                hits.append(rel)
                break
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


@pytest.mark.parametrize("old,new", MERGED, ids=[f"{o}->{n}" for o, n in MERGED])
def test_merge_survivor_is_registered(old, new):
    from tcip_mcp.server import list_registered_tools

    assert new in set(list_registered_tools())


@pytest.mark.parametrize("old,new", MERGED, ids=[f"{o}->{n}" for o, n in MERGED])
def test_merged_away_name_survives_nowhere_tracked(old, new):
    files = [f for f in _tracked_files() if _in_scope(f)]
    sites = _old_name_sites(old, files)
    assert not sites, f"{old!r} still appears in: {sites}"


@pytest.mark.parametrize("old,new", RETIRED, ids=[f"{o}->{n}" for o, n in RETIRED])
def test_retired_name_survives_nowhere_tracked(old, new):
    files = [f for f in _tracked_files() if _in_scope(f)]
    sites = _old_name_sites(old, files)
    assert not sites, f"{old!r} still appears in: {sites}"


# A demotion keeps its identifier as a library call or script name, so it is never swept for
# whole-tree absence the way a rename's or a merge's old name is; see _demoted_tool_table_sites.
DEMOTED = [
    "preflight_config",
    "read_annotations",
    "score_predictions",
    "triage_predictions",
    "compare_experiments",
    "overlay_reference_grid",
    "visualize",
]

def _demoted_tool_table_sites(name: str, files: list[str]) -> list[str]:
    """Every file whose Tools table still names ``name`` in its first cell, read through
    ``scripts.verify_skill_tools``'s own ``tool_table_first_cells``/``extract_tool_name``: the
    one implementation of "what counts as a Tools table row" and "what name does this cell
    claim" the fabrication check itself uses, so a documented call signature like
    ``visualize(source="annotations", path=<image>)`` is caught the same way a bare
    `` `name` `` cell is. A same-shaped row in an unrelated table (ARCHITECTURE.md's own
    re-exports or module inventories) has a different header and is not flagged; ordinary prose
    quoting the name never takes a table shape at all."""
    from tools.verify_skill_tools import extract_tool_name, tool_table_first_cells

    hits = []
    for rel in files:
        text = _read(rel)
        if text is None:
            continue
        names = {extract_tool_name(cell) for cell in tool_table_first_cells(text)}
        if name in names:
            hits.append(rel)
    return hits


@pytest.mark.parametrize("name", DEMOTED)
def test_demoted_name_has_no_tool_table_row(name):
    files = [f for f in _tracked_files() if _in_scope(f)]
    sites = _demoted_tool_table_sites(name, files)
    assert not sites, f"{name!r} still has a tool-table row in: {sites}"


def test_demoted_tool_table_sites_catches_a_call_signature_row(tmp_path):
    """A demoted name surviving in a Tools table as a documented call signature, not a bare
    backtick-quoted token, is still a stale row: the old bare-name-only regex missed this shape."""
    fixture = tmp_path / "stale.md"
    fixture.write_text(
        "| Tool | Role |\n"
        "|------|------|\n"
        '| `visualize(source="annotations", path=<image>)` | Render annotations |\n',
        encoding="utf-8",
    )

    sites = _demoted_tool_table_sites("visualize", [str(fixture)])

    assert sites == [str(fixture)]
