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
