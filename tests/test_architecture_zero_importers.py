"""check_architecture_doc's "Modules with zero importers" list is checked against a regenerated
inventory the same way its module-ownership tables are: header count and membership both."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "check_architecture_doc.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_architecture_doc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_architecture_doc"] = mod
    spec.loader.exec_module(mod)
    return mod


def _section(header_count: int, *rows: tuple[str, str]) -> str:
    body = "\n".join(f"| {root} | {path} |" for root, path in rows)
    return (
        f"## Modules with zero importers ({header_count})\n\n"
        "A module counts as zero-importer when no other module imports it.\n\n"
        "| Root | Module path |\n|---|---|\n" + body + "\n\n"
        "## Public surface\n"
    )


def _inventory(*, zero: list[str], nonzero: list[str]) -> dict:
    return {
        "python_modules": (
            [{"path": p, "imported_by_count": 0} for p in zero]
            + [{"path": p, "imported_by_count": 1} for p in nonzero]
        ),
        "typescript_modules": [],
    }


def test_a_zero_importer_missing_from_the_doc_is_reported():
    checker = _load()
    md = _section(1, ("scripts", "scripts/named.py"))
    header_count, rows = checker.parse_zero_importer_section(md)
    inventory = _inventory(zero=["scripts/named.py", "scripts/orphan.py"], nonzero=[])

    findings = checker.check_zero_importers(header_count, rows, inventory)

    assert {"kind": "zero_importer_missing", "path": "scripts/orphan.py"} in findings


def test_a_doc_entry_that_is_not_really_zero_importer_is_reported():
    checker = _load()
    md = _section(1, ("scripts", "scripts/named.py"))
    header_count, rows = checker.parse_zero_importer_section(md)
    inventory = _inventory(zero=[], nonzero=["scripts/named.py"])

    findings = checker.check_zero_importers(header_count, rows, inventory)

    assert any(
        f["kind"] == "zero_importer_extra" and f["path"] == "scripts/named.py" for f in findings
    )


def test_a_header_count_mismatch_is_reported():
    checker = _load()
    md = _section(5, ("scripts", "scripts/named.py"))
    header_count, rows = checker.parse_zero_importer_section(md)
    inventory = _inventory(zero=["scripts/named.py"], nonzero=[])

    findings = checker.check_zero_importers(header_count, rows, inventory)

    assert {"kind": "zero_importer_header_mismatch", "header": 5, "real": 1} in findings


def test_a_duplicated_row_is_reported_even_when_header_and_membership_both_check_out():
    """Set-wise membership alone would admit this silently: the same path listed twice still
    resolves to one member of ``doc_paths``, so a header count that already matches the true
    zero-importer total hides a physically duplicated row unless it is checked against the row
    list itself."""
    checker = _load()
    md = _section(1, ("scripts", "scripts/named.py"), ("scripts", "scripts/named.py"))
    header_count, rows = checker.parse_zero_importer_section(md)
    inventory = _inventory(zero=["scripts/named.py"], nonzero=[])

    findings = checker.check_zero_importers(header_count, rows, inventory)

    assert any(
        f["kind"] == "zero_importer_duplicate" and f["path"] == "scripts/named.py"
        for f in findings
    )
    assert not any(f["kind"] == "zero_importer_header_mismatch" for f in findings)
    assert not any(f["kind"] in ("zero_importer_missing", "zero_importer_extra") for f in findings)


def test_a_reconciled_list_admits_valid_work():
    checker = _load()
    md = _section(2, ("scripts", "scripts/named.py"), ("scripts", "scripts/other.py"))
    header_count, rows = checker.parse_zero_importer_section(md)
    inventory = _inventory(zero=["scripts/named.py", "scripts/other.py"], nonzero=[])

    assert checker.check_zero_importers(header_count, rows, inventory) == []


def test_the_real_architecture_md_zero_importer_list_matches_a_fresh_inventory():
    """The gate's own self-check, over the real tree rather than a synthetic one."""
    checker = _load()
    inventory_script = REPO_ROOT / "tools" / "build_module_inventory.py"
    git_marker = inventory_script.parent.parent / ".git"
    if not git_marker.exists():
        git_marker.touch()
    spec = importlib.util.spec_from_file_location("build_module_inventory", inventory_script)
    builder = importlib.util.module_from_spec(spec)
    sys.modules["build_module_inventory"] = builder
    spec.loader.exec_module(builder)

    inventory = builder.build_inventory()
    md_text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    header_count, rows = checker.parse_zero_importer_section(md_text)

    assert checker.check_zero_importers(header_count, rows, inventory) == []
