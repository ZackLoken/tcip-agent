"""The architecture-doc checker reads both directions: a named path must exist, and a source
file under a covered root must be named."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_architecture_doc.py"
INVENTORY_SCRIPT = REPO_ROOT / "scripts" / "build_module_inventory.py"


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
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "named.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts" / "orphan.py").write_text("", encoding="utf-8")
    rows = checker.parse_module_rows(_table("scripts/named.py"))

    findings = checker.check_coverage(rows, tmp_path)

    assert [f["path"] for f in findings] == ["scripts/orphan.py"]


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
    cache = tmp_path / "scripts" / "__pycache__"
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
        "| scripts | 0 | 0 |\n"
    )


_ONE_MCP_MODULE_INVENTORY = {
    "python_modules": [{"root": "tcip-mcp", "lines": 5}],
    "typescript_modules": [],
    "counts": {
        "python_by_root": {
            "tcip-mcp": 1, "tcip-annotation": 0, "tcip-web": 0, "tcip-store": 0, "scripts": 0,
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
    return f"Source: the module inventory `scripts/build_module_inventory.py` produces, run at HEAD {head}.\n"


def test_a_summary_table_row_naming_an_unknown_package_is_reported():
    """A doc row whose package name is not among the inventory's own roots (a stray or
    renamed row) is its own finding, not silently skipped because no real root answers it."""
    checker = _load()
    md_text = _summary(1, 5, sentence_modules=1, sentence_lines=5).replace(
        "| scripts | 0 | 0 |\n", "| scripts | 0 | 0 |\n| tcip-nonexistent | 1 | 0 |\n"
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


def test_a_real_commit_not_on_this_checkouts_own_history_is_a_finding():
    """d6e28aee is the worktree hash this fix-up's own bug report names: a real commit object,
    reachable in the object database, that is not an ancestor of this checkout's own HEAD."""
    checker = _load()
    sentence = {"line_no": 1, "head": "d6e28aee"}

    findings, skips = checker.check_head_sentences(sentence, dict(sentence, modules=1, lines=1), REPO_ROOT)

    assert [f["kind"] for f in findings] == ["head_not_an_ancestor"]
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


def test_architecture_md_head_sentences_match_and_are_real_on_this_checkout():
    """The gate's own self-check: ARCHITECTURE.md's two HEAD sentences name one commit, and
    that commit is real on this checkout's own history."""
    checker = _load()
    md_text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

    source_sentence = checker.parse_source_sentence(md_text)
    summary_sentence, _rows = checker.parse_module_count_summary(md_text)
    findings, _skips = checker.check_head_sentences(source_sentence, summary_sentence, REPO_ROOT)

    assert findings == []
