"""The architecture-doc checker reads both directions: a named path must exist, and a source
file under a covered root must be named."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "check_architecture_doc.py"
INVENTORY_SCRIPT = REPO_ROOT / "tools" / "build_module_inventory.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_architecture_doc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_architecture_doc"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_inventory_builder():
    spec = importlib.util.spec_from_file_location("build_module_inventory", INVENTORY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_module_inventory"] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(*paths: str) -> str:
    rows = "\n".join(f"| {p} | (none found) | 0 | 0 |" for p in paths)
    return "| Module path | Ownership (one line) | In-repo imports | Imported by |\n|---|---|---|---|\n" + rows + "\n"


def test_a_source_file_no_table_names_is_reported(tmp_path):
    checker = _load()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "named.py").write_text("", encoding="utf-8")
    (tmp_path / "tools" / "orphan.py").write_text("", encoding="utf-8")
    rows = checker.parse_module_rows(_table("tools/named.py"))

    findings = checker.check_coverage(rows, tmp_path)

    assert [f["path"] for f in findings] == ["tools/orphan.py"]


def test_a_tree_every_file_of_which_is_named_passes(tmp_path):
    checker = _load()
    frontend = tmp_path / "packages" / "tcip-web" / "frontend" / "src" / "lib"
    frontend.mkdir(parents=True)
    (frontend / "a.ts").write_text("", encoding="utf-8")
    (frontend / "a.test.tsx").write_text("", encoding="utf-8")
    (frontend / "notes.md").write_text("", encoding="utf-8")
    rows = checker.parse_module_rows(_table(
        "packages/tcip-web/frontend/src/lib/a.ts",
        "packages/tcip-web/frontend/src/lib/a.test.tsx",
    ))

    assert checker.check_coverage(rows, tmp_path) == []


def test_build_artifacts_and_caches_are_outside_the_covered_set(tmp_path):
    checker = _load()
    cache = tmp_path / "tools" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "stale.py").write_text("", encoding="utf-8")
    modules = tmp_path / "packages" / "tcip-web" / "frontend" / "src" / "node_modules" / "x"
    modules.mkdir(parents=True)
    (modules / "index.ts").write_text("", encoding="utf-8")

    assert checker.check_coverage([], tmp_path) == []


def test_architecture_md_counts_match_a_fresh_inventory():
    """The gate's own self-check: ARCHITECTURE.md's table counts must match what a freshly
    generated module inventory finds, over the real tree, not a synthetic one. This is the
    check CI runs; it fails whenever a count drifts, which is the point of running it here too."""
    checker = _load()
    builder = _load_inventory_builder()

    inventory = builder.build_inventory()
    md_text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    rows = checker.parse_module_rows(md_text)
    parsed = [r for r in rows if not r.get("unparsed")]

    findings = checker.check_counts(parsed, inventory)

    assert findings == []


def _summary(mcp_modules: int, mcp_lines: int, *, sentence_modules: int, sentence_lines: int) -> str:
    return (
        f"HEAD 1234abcd has {sentence_modules} modules across the six scanned roots "
        f"({sentence_lines} total lines):\n\n"
        "| Package (root) | Modules | Lines |\n"
        "|---|---|---|\n"
        f"| tcip-mcp | {mcp_modules} | {mcp_lines} |\n"
        "| tcip-annotation | 0 | 0 |\n"
        "| tcip-web | 0 | 0 |\n"
        "| tcip-store | 0 | 0 |\n"
        "| tcip-web-frontend | 0 | 0 |\n"
        "| tools | 0 | 0 |\n"
    )


_ONE_MCP_MODULE_INVENTORY = {
    "python_modules": [{"root": "tcip-mcp", "lines": 5}],
    "typescript_modules": [],
    "counts": {
        "python_by_root": {
            "tcip-mcp": 1, "tcip-annotation": 0, "tcip-web": 0, "tcip-store": 0, "tools": 0,
        },
        "typescript_total": 0,
    },
}


def test_a_summary_table_row_that_drifts_from_the_inventory_is_reported():
    """The tcip-mcp row claims 10 lines for its one module; the inventory says that module is
    5 lines. The sentence states the real total (1, 5) so only the row itself drifts."""
    checker = _load()
    md_text = _summary(1, 10, sentence_modules=1, sentence_lines=5)

    sentence, rows = checker.parse_module_count_summary(md_text)
    findings = checker.check_module_count_summary(sentence, rows, _ONE_MCP_MODULE_INVENTORY)

    assert [f["kind"] for f in findings] == ["module_count_row_drift"]
    assert findings[0]["package"] == "tcip-mcp"
    assert findings[0]["doc"] == (1, 10)
    assert findings[0]["real"] == (1, 5)


def test_a_summary_sentence_whose_totals_drift_is_reported():
    """Every row matches the inventory, but the introductory sentence still claims 2 modules
    and 12 total lines against the rows' own 1 module and 5 lines: the sentence is checked
    against the rows' real sum, not merely echoed back."""
    checker = _load()
    md_text = _summary(1, 5, sentence_modules=2, sentence_lines=12)

    sentence, rows = checker.parse_module_count_summary(md_text)
    findings = checker.check_module_count_summary(sentence, rows, _ONE_MCP_MODULE_INVENTORY)

    assert [f["kind"] for f in findings] == ["module_count_sentence_drift"]
    assert findings[0]["doc"] == (2, 12)
    assert findings[0]["real"] == (1, 5)


def test_a_summary_matching_the_inventory_passes():
    checker = _load()
    md_text = _summary(1, 5, sentence_modules=1, sentence_lines=5)

    sentence, rows = checker.parse_module_count_summary(md_text)
    findings = checker.check_module_count_summary(sentence, rows, _ONE_MCP_MODULE_INVENTORY)

    assert findings == []


def test_architecture_md_module_count_summary_matches_a_fresh_inventory():
    """The gate's own self-check for the per-root Modules/Lines summary table and its
    sentence, over the real tree: both must match what a freshly generated module inventory
    finds, the same way test_architecture_md_counts_match_a_fresh_inventory holds the
    per-module tables to it."""
    checker = _load()
    builder = _load_inventory_builder()

    inventory = builder.build_inventory()
    md_text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

    sentence, rows = checker.parse_module_count_summary(md_text)
    findings = checker.check_module_count_summary(sentence, rows, inventory)

    assert findings == []


def _source_line(head: str) -> str:
    return f"Source: the module inventory `tools/build_module_inventory.py` produces, run at HEAD {head}.\n"


def test_a_summary_table_row_naming_an_unknown_package_is_reported():
    """A doc row whose package name is not among the inventory's own roots (a stray or
    renamed row) is its own finding, not silently skipped because no real root answers it."""
    checker = _load()
    md_text = _summary(1, 5, sentence_modules=1, sentence_lines=5).replace(
        "| tools | 0 | 0 |\n", "| tools | 0 | 0 |\n| tcip-nonexistent | 1 | 0 |\n"
    )

    sentence, rows = checker.parse_module_count_summary(md_text)
    findings = checker.check_module_count_summary(sentence, rows, _ONE_MCP_MODULE_INVENTORY)

    assert [f["kind"] for f in findings] == ["module_count_row_unknown_package"]
    assert findings[0]["package"] == "tcip-nonexistent"


def test_a_summary_table_with_no_introductory_sentence_is_reported():
    """A present table with its introductory sentence removed must fail, the same as a present
    sentence over a missing table already does (via module_count_row_missing); a document with
    neither the sentence nor the table has nothing here to check and stays clean."""
    checker = _load()
    with_table = _summary(1, 5, sentence_modules=1, sentence_lines=5)
    headless = "\n".join(
        line for line in with_table.splitlines() if not line.startswith("HEAD ")
    ) + "\n"

    sentence, rows = checker.parse_module_count_summary(headless)
    findings = checker.check_module_count_summary(sentence, rows, _ONE_MCP_MODULE_INVENTORY)

    assert sentence is None
    assert rows
    assert [f["kind"] for f in findings] == ["module_count_sentence_missing"]

    neither_sentence, neither_rows = checker.parse_module_count_summary("nothing here at all\n")
    assert checker.check_module_count_summary(neither_sentence, neither_rows, _ONE_MCP_MODULE_INVENTORY) == []


def test_parse_source_sentence_finds_the_line():
    checker = _load()
    md_text = _source_line("32bc6c58") + "\nsome other text\n"

    sentence = checker.parse_source_sentence(md_text)

    assert sentence == {"line_no": 1, "head": "32bc6c58"}
    assert checker.parse_source_sentence("no such sentence here\n") is None


def test_matching_real_head_sentences_pass_on_this_checkout():
    """Both sentences naming the same commit, one this checkout's own history really carries,
    must pass with no findings and no skip (this repository is a real git checkout)."""
    checker = _load()
    source = {"line_no": 1, "head": "32bc6c58"}
    summary = {"line_no": 5, "head": "32bc6c58", "modules": 1, "lines": 1}

    findings, skips = checker.check_head_sentences(source, summary, REPO_ROOT)

    assert findings == []
    assert skips == []


def test_mismatched_head_sentences_is_a_finding():
    """The two sentences naming different commits is a finding regardless of whether either
    hash is real: pointed at a non-git directory so the realness check contributes nothing but
    a stated skip, isolating the mismatch finding on its own."""
    checker = _load()
    source = {"line_no": 1, "head": "aaaa1111"}
    summary = {"line_no": 5, "head": "bbbb2222", "modules": 1, "lines": 1}

    findings, skips = checker.check_head_sentences(source, summary, pathlib.Path("/does/not/exist"))

    assert [f["kind"] for f in findings] == ["head_mismatch"]
    assert findings[0]["source_head"] == "aaaa1111"
    assert findings[0]["summary_head"] == "bbbb2222"
    assert len(skips) == 2


def test_a_hash_that_is_not_a_real_commit_is_a_finding():
    checker = _load()
    sentence = {"line_no": 1, "head": "notarealcommithashatall"}

    findings, skips = checker.check_head_sentences(sentence, dict(sentence, modules=1, lines=1), REPO_ROOT)

    assert [f["kind"] for f in findings] == ["head_not_a_commit"]
    assert skips == []


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run one git command in ``repo`` and return its stdout, skipping the calling test when git
    refuses: the throwaway repository the test builds is the fixture, so git being unable to
    build it is an absent fixture, never a finding against the checker."""
    result = subprocess.run(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", *args],
        cwd=str(repo), capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"git could not build the two-branch fixture: {result.stderr.strip()}")
    return result.stdout.strip()


def test_a_real_commit_not_on_the_checkouts_own_history_is_a_finding(tmp_path):
    """A commit that resolves in the object database but is not an ancestor of HEAD is the
    shape a worktree's own hash takes once git am renumbers it onto main. The case is built in a
    throwaway repository (a root commit, a diverging commit on a side branch, HEAD back at the
    root) rather than named from this checkout's history, since a commit that exists only in one
    machine's object database is a different finding, head_not_a_commit, on every other clone."""
    checker = _load()
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "root")
    root = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "-b", "aside")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "aside")
    aside = _git(tmp_path, "rev-parse", "--short", "HEAD")
    _git(tmp_path, "checkout", "-q", root)
    sentence = {"line_no": 1, "head": aside}

    findings, skips = checker.check_head_sentences(sentence, dict(sentence, modules=1, lines=1), tmp_path)

    assert [f["kind"] for f in findings] == ["head_not_an_ancestor"]
    assert findings[0]["head"] == aside
    assert skips == []


def test_head_check_is_skipped_not_a_finding_outside_a_git_checkout(tmp_path):
    checker = _load()
    sentence = {"line_no": 1, "head": "32bc6c58"}

    findings, skips = checker.check_head_sentences(sentence, dict(sentence, modules=1, lines=1), tmp_path)

    assert findings == []
    assert len(skips) == 1
    assert "not a git checkout" in skips[0]


def test_a_lone_source_sentence_with_no_summary_partner_is_a_finding():
    checker = _load()
    source = {"line_no": 1, "head": "32bc6c58"}

    findings, skips = checker.check_head_sentences(source, None, REPO_ROOT)

    assert [f["kind"] for f in findings] == ["module_count_sentence_missing"]


def test_a_lone_summary_sentence_with_no_source_partner_is_a_finding():
    checker = _load()
    summary = {"line_no": 5, "head": "32bc6c58", "modules": 1, "lines": 1}

    findings, skips = checker.check_head_sentences(None, summary, REPO_ROOT)

    assert [f["kind"] for f in findings] == ["source_sentence_missing"]


def test_neither_head_sentence_present_is_not_a_finding():
    checker = _load()

    findings, skips = checker.check_head_sentences(None, None, REPO_ROOT)

    assert findings == []
    assert skips == []


# ── --fix ────────────────────────────────────────────────────────────────────

_FIX_INVENTORY = {
    "python_modules": [
        {
            "path": "packages/tcip-mcp/src/tcip_mcp/kept.py", "root": "tcip-mcp", "lines": 10,
            "owns": "Kept module, refreshed rather than dropped.",
            "imports": ["a", "b"], "imported_by_count": 1,
        },
        {
            "path": "packages/tcip-mcp/src/tcip_mcp/new_module.py", "root": "tcip-mcp", "lines": 4,
            "owns": "A module the document never named.",
            "imports": [], "imported_by_count": 0,
        },
    ],
    "typescript_modules": [],
    "counts": {
        "python_by_root": {
            "tcip-mcp": 2, "tcip-annotation": 0, "tcip-web": 0, "tcip-store": 0, "tools": 0,
        },
        "typescript_total": 0,
    },
}


def _fix_fixture_doc() -> str:
    return (
        "## tcip-mcp\n\n"
        "| Module path | Ownership (one line) | In-repo imports | Imported by |\n"
        "|---|---|---|---|\n"
        "| packages/tcip-mcp/src/tcip_mcp/kept.py | (none found) | 99 | 99 |\n"
        "| packages/tcip-mcp/src/tcip_mcp/gone.py | (none found) | 0 | 0 |\n"
        "\n"
        "## Modules with zero importers (0)\n\n"
        "| Root | Module path |\n"
        "|---|---|\n"
        "\n"
        "## Module count summary\n\n"
        "Source: the module inventory `tools/build_module_inventory.py` produces, run at HEAD "
        "aaaaaaaa.\n\n"
        "HEAD aaaaaaaa has 1 modules across the six scanned roots (1 total lines):\n\n"
        "| Package (root) | Modules | Lines |\n"
        "|---|---|---|\n"
        "| tcip-mcp | 1 | 1 |\n"
        "| tcip-annotation | 0 | 0 |\n"
        "| tcip-web | 0 | 0 |\n"
        "| tcip-store | 0 | 0 |\n"
        "| tcip-web-frontend | 0 | 0 |\n"
        "| tools | 0 | 0 |\n"
    )


def test_fix_produces_a_document_the_checker_then_passes():
    checker = _load()
    fixed = checker.fix_architecture_doc(_fix_fixture_doc(), _FIX_INVENTORY, head="deadbeef")

    rows = [r for r in checker.parse_module_rows(fixed) if not r.get("unparsed")]
    assert checker.check_counts(rows, _FIX_INVENTORY) == []

    header_count, zero_rows = checker.parse_zero_importer_section(fixed)
    assert checker.check_zero_importers(header_count, zero_rows, _FIX_INVENTORY) == []

    summary_sentence, summary_rows = checker.parse_module_count_summary(fixed)
    assert checker.check_module_count_summary(summary_sentence, summary_rows, _FIX_INVENTORY) == []

    source_sentence = checker.parse_source_sentence(fixed)
    assert source_sentence == {"line_no": source_sentence["line_no"], "head": "deadbeef"}
    assert summary_sentence["head"] == "deadbeef"


def test_fix_drops_a_rows_whose_path_the_inventory_no_longer_carries():
    checker = _load()
    fixed = checker.fix_architecture_doc(_fix_fixture_doc(), _FIX_INVENTORY, head="deadbeef")

    assert "gone.py" not in fixed


def test_fix_adds_a_row_for_a_module_the_document_never_named():
    checker = _load()
    fixed = checker.fix_architecture_doc(_fix_fixture_doc(), _FIX_INVENTORY, head="deadbeef")

    assert "packages/tcip-mcp/src/tcip_mcp/new_module.py" in fixed
    assert "A module the document never named." in fixed


def test_fix_inserts_a_new_row_in_the_roots_own_sorted_position():
    """A new module whose path sorts before the retained row must land before it, never merely
    appended after the last retained row (the fixture's own new_module.py happens to sort after
    kept.py, so this needs its own path that sorts first to actually exercise the ordering)."""
    checker = _load()
    inventory = {
        "python_modules": [
            {
                "path": "packages/tcip-mcp/src/tcip_mcp/aaa_new.py", "root": "tcip-mcp",
                "lines": 4, "owns": "Sorts before kept.py.",
                "imports": [], "imported_by_count": 0,
            },
            {
                "path": "packages/tcip-mcp/src/tcip_mcp/kept.py", "root": "tcip-mcp", "lines": 10,
                "owns": "Kept module, refreshed rather than dropped.",
                "imports": ["a", "b"], "imported_by_count": 1,
            },
        ],
        "typescript_modules": [],
        "counts": {
            "python_by_root": {
                "tcip-mcp": 2, "tcip-annotation": 0, "tcip-web": 0, "tcip-store": 0, "tools": 0,
            },
            "typescript_total": 0,
        },
    }
    doc = (
        "## tcip-mcp\n\n"
        "| Module path | Ownership (one line) | In-repo imports | Imported by |\n"
        "|---|---|---|---|\n"
        "| packages/tcip-mcp/src/tcip_mcp/kept.py | (none found) | 99 | 99 |\n"
        "\n"
        "## Modules with zero importers (0)\n\n"
        "| Root | Module path |\n"
        "|---|---|\n"
    )

    fixed = checker.fix_architecture_doc(doc, inventory, head="deadbeef")

    new_index = fixed.index("aaa_new.py")
    kept_index = fixed.index("kept.py")
    assert new_index < kept_index


def test_fix_regenerates_the_zero_importer_header_count():
    checker = _load()
    fixed = checker.fix_architecture_doc(_fix_fixture_doc(), _FIX_INVENTORY, head="deadbeef")

    header_count, rows = checker.parse_zero_importer_section(fixed)
    assert header_count == 1
    assert [r["path"] for r in rows] == ["packages/tcip-mcp/src/tcip_mcp/new_module.py"]


def test_fix_preserves_a_rows_queued_marker_while_refreshing_its_counts():
    """Round-tripping "<!-- queued: TEXT -->" through the shared ROW_RE always yields the
    comment group with one trailing space (whatever precedes the closing "-->", by the
    format's own convention); fixing a row must reproduce that same, stable shape rather than
    accumulating whitespace on repeated fixes, never strip the marker outright."""
    checker = _load()
    doc = _fix_fixture_doc().replace(
        "| packages/tcip-mcp/src/tcip_mcp/kept.py | (none found) | 99 | 99 |\n",
        "| packages/tcip-mcp/src/tcip_mcp/kept.py | (none found) | 99 | 99 | "
        "<!-- queued: a pending decision --> \n",
    )
    fixed_once = checker.fix_architecture_doc(doc, _FIX_INVENTORY, head="deadbeef")
    fixed_twice = checker.fix_architecture_doc(fixed_once, _FIX_INVENTORY, head="deadbeef")

    rows = [r for r in checker.parse_module_rows(fixed_once) if not r.get("unparsed")]
    kept = next(r for r in rows if r["path"] == "packages/tcip-mcp/src/tcip_mcp/kept.py")
    assert kept["imports"] == 2
    assert kept["imported_by"] == 1
    assert kept["queued"].strip() == "queued: a pending decision"
    assert fixed_twice == fixed_once, "a second --fix must be a no-op, not accumulate whitespace"


def test_fix_requires_inventory_json():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fix"], cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--inventory-json" in result.stderr


def test_fix_refuses_a_head_that_names_no_commit_and_writes_nothing(tmp_path):
    """guard. A --head the checkout cannot resolve is refused before the document is touched:
    the stamp the checker later verifies would otherwise carry a hash no reader can look up, and
    the rewrite would be reported as done."""
    doc_path = tmp_path / "ARCHITECTURE.md"
    doc_path.write_text(_fix_fixture_doc(), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(_FIX_INVENTORY), encoding="utf-8")
    before = doc_path.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(doc_path), str(REPO_ROOT),
         "--inventory-json", str(inventory_path), "--fix", "--head", "2d33eee1"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )

    assert result.returncode == 1
    assert "REFUSED" in result.stdout and "2d33eee1" in result.stdout
    assert doc_path.read_text(encoding="utf-8") == before


def test_fix_stamps_a_head_the_checkout_does_resolve(tmp_path):
    """coverage that the refusal above admits real work: the same call with this checkout's own
    HEAD rewrites the document and stamps that hash into both sentences."""
    doc_path = tmp_path / "ARCHITECTURE.md"
    doc_path.write_text(_fix_fixture_doc(), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(_FIX_INVENTORY), encoding="utf-8")
    head = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    ).stdout.strip()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(doc_path), str(REPO_ROOT),
         "--inventory-json", str(inventory_path), "--fix", "--head", head],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert head in doc_path.read_text(encoding="utf-8")


def test_without_fix_the_checker_reports_only_and_does_not_write(tmp_path):
    """A legitimate call with drift and no --fix still exits, reporting rather than rewriting."""
    doc_path = tmp_path / "ARCHITECTURE.md"
    doc_path.write_text(_fix_fixture_doc(), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(_FIX_INVENTORY), encoding="utf-8")
    before = doc_path.read_text(encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(doc_path), str(tmp_path),
         "--inventory-json", str(inventory_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )

    assert doc_path.read_text(encoding="utf-8") == before
    assert result.returncode == 1
    assert "COUNT DRIFT" in result.stdout


def test_architecture_md_head_sentences_match_and_are_real_on_this_checkout():
    """The gate's own self-check: ARCHITECTURE.md's two HEAD sentences name one commit, and
    that commit is real on this checkout's own history."""
    checker = _load()
    md_text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

    source_sentence = checker.parse_source_sentence(md_text)
    summary_sentence, _rows = checker.parse_module_count_summary(md_text)
    findings, _skips = checker.check_head_sentences(source_sentence, summary_sentence, REPO_ROOT)

    assert findings == []
