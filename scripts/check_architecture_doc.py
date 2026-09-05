"""Verify ARCHITECTURE.md's module-ownership tables against the tree, for CI.

ARCHITECTURE.md is the adopter-facing map. Its module-ownership tables name a real file per row
with an in-repo-import count and an imported-by count. This check keeps the map from drifting
away from the code: it parses every such table and asserts each named path exists. With a
regenerated module-inventory JSON (the shape scripts under docs/audit produce), it also
cross-checks the two counts per row and reports numeric drift, checks the "Modules with zero
importers" list the same way (every module the inventory records at imported_by_count 0 must
appear in the list, every listed module must really be zero-importer, and the section's own
header count must equal that real count), and checks the per-root Modules/Lines summary table
and its introductory sentence the same way again: each row's module and line count against
`counts.python_by_root` / `counts.typescript_total` and the real per-root line sum, and the
sentence's own totals against the sum of those rows.

Row-level `queued:` HTML-comment markers are checked the same as any other row. Per
ARCHITECTURE.md's own marker convention, a queued-marked sentence states a fact that is true
today paired with a pending decision, not a statement that is false now.

Exit 0 when the tables match the tree, 1 when any named path is missing, any table row fails to
parse, or (when an inventory JSON is supplied) any count or zero-importer drifts. Defaults resolve
ARCHITECTURE.md and the repo root relative to this script, so CI invokes it as
`python scripts/check_architecture_doc.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

ROW_RE = re.compile(
    r"^\|\s*(?P<path>[^|]+?)\s*\|\s*(?P<desc>[^|]*?)\s*\|\s*(?P<imports>-?\d+)\s*\|\s*(?P<imported_by>-?\d+)\s*\|"
    r"(?:\s*<!--\s*(?P<comment>queued:[^>]*)-->)?\s*$"
)
TABLE_HEADER = "| Module path | Ownership (one line) | In-repo imports | Imported by |"

ZERO_IMPORTERS_HEADER_RE = re.compile(r"^## Modules with zero importers \((?P<count>\d+)\)\s*$")
ZERO_IMPORTERS_TABLE_HEADER = "| Root | Module path |"
ZERO_IMPORTER_ROW_RE = re.compile(r"^\|\s*(?P<root>[^|]+?)\s*\|\s*(?P<path>[^|]+?)\s*\|\s*$")

MODULE_COUNT_SENTENCE_RE = re.compile(
    r"^HEAD (?P<head>[0-9a-f]+) has (?P<modules>\d+) modules across the six scanned roots "
    r"\((?P<lines>\d+) total lines\):\s*$"
)
MODULE_COUNT_TABLE_HEADER = "| Package (root) | Modules | Lines |"
MODULE_COUNT_ROW_RE = re.compile(r"^\|\s*(?P<package>[^|]+?)\s*\|\s*(?P<modules>\d+)\s*\|\s*(?P<lines>\d+)\s*\|\s*$")

PYTHON_ROOT_PACKAGES = ("tcip-mcp", "tcip-annotation", "tcip-web", "tcip-store", "scripts")
"""The `counts.python_by_root` keys, in the order the summary table lists them."""
TYPESCRIPT_PACKAGE = "tcip-web-frontend"
"""The summary table's row for `counts.typescript_total`; not a `python_by_root` key."""


def parse_module_rows(md_text: str) -> list[dict]:
    """Extract every data row from every module-ownership table in the document."""
    lines = md_text.splitlines()
    rows: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == TABLE_HEADER:
            i += 2  # header + markdown separator row
            while i < len(lines) and lines[i].startswith("|"):
                m = ROW_RE.match(lines[i])
                if m:
                    rows.append(
                        {
                            "line_no": i + 1,
                            "path": m.group("path").strip(),
                            "imports": int(m.group("imports")),
                            "imported_by": int(m.group("imported_by")),
                            "queued": m.group("comment"),
                        }
                    )
                else:
                    rows.append({"line_no": i + 1, "unparsed": True, "raw": lines[i]})
                i += 1
        else:
            i += 1
    return rows


def check_existence(rows: list[dict], repo_root: Path) -> list[dict]:
    findings = []
    for r in rows:
        if r.get("unparsed"):
            findings.append({"line_no": r["line_no"], "kind": "unparsed_row", "raw": r["raw"]})
            continue
        if not (repo_root / r["path"]).exists():
            findings.append({"line_no": r["line_no"], "kind": "missing_path", "path": r["path"]})
    return findings


COVERED_ROOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("packages/tcip-mcp/src", (".py",)),
    ("packages/tcip-annotation/src", (".py",)),
    ("packages/tcip-web/src", (".py",)),
    ("packages/tcip-store/src", (".py",)),
    ("scripts", (".py",)),
    ("packages/tcip-web/frontend/src", (".ts", ".tsx")),
)
"""The trees the module tables claim to cover, with the extensions a table row names."""

_SKIPPED_PARTS = frozenset({"__pycache__", "node_modules"})


def check_coverage(rows: list[dict], repo_root: Path) -> list[dict]:
    """Every source file under a covered root is named by some table row.

    The existence check reads one direction only (a named path exists); this one reads the
    other, so a module or script that lands with no row fails the gate instead of staying
    undocumented while both directions of the old check were green.
    """
    named = {r["path"].replace("\\", "/") for r in rows if not r.get("unparsed")}
    findings = []
    for root, extensions in COVERED_ROOTS:
        base = repo_root / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in extensions:
                continue
            if _SKIPPED_PARTS & set(path.parts):
                continue
            rel = path.relative_to(repo_root).as_posix()
            if rel not in named:
                findings.append({"kind": "unnamed_path", "path": rel})
    return findings


def parse_zero_importer_section(md_text: str) -> tuple[int | None, list[dict]]:
    """Extract the "Modules with zero importers" section's header count and its data rows."""
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = ZERO_IMPORTERS_HEADER_RE.match(line.strip())
        if not m:
            continue
        header_count = int(m.group("count"))
        j = i + 1
        while j < len(lines) and lines[j].strip() != ZERO_IMPORTERS_TABLE_HEADER:
            j += 1
        j += 2  # table header + markdown separator row
        rows: list[dict] = []
        while j < len(lines) and lines[j].startswith("|"):
            rm = ZERO_IMPORTER_ROW_RE.match(lines[j])
            if rm:
                rows.append(
                    {"line_no": j + 1, "root": rm.group("root").strip(), "path": rm.group("path").strip()}
                )
            j += 1
        return header_count, rows
    return None, []


def check_zero_importers(header_count: int | None, rows: list[dict], inventory: dict) -> list[dict]:
    """The zero-importers list against the regenerated inventory: header count, membership, and
    each row's own uniqueness.

    A module counts as zero-importer exactly when the inventory records imported_by_count 0;
    the header states how many such modules the table below lists. Membership is checked
    set-wise (missing/extra), which by itself would admit a duplicated row silently: two
    identical rows still resolve to one member of the set either side compares, so a repeated
    path is checked separately, against the row list itself rather than its deduplicated set.
    """
    findings: list[dict] = []
    all_records = inventory.get("python_modules", []) + inventory.get("typescript_modules", [])
    real_zero = {m["path"] for m in all_records if m["imported_by_count"] == 0}
    doc_paths = {r["path"] for r in rows}
    for path in sorted(real_zero - doc_paths):
        findings.append({"kind": "zero_importer_missing", "path": path})
    for r in rows:
        if r["path"] not in real_zero:
            findings.append({"kind": "zero_importer_extra", "line_no": r["line_no"], "path": r["path"]})
    first_seen: dict[str, int] = {}
    for r in rows:
        prior = first_seen.get(r["path"])
        if prior is not None:
            findings.append({"kind": "zero_importer_duplicate", "path": r["path"],
                             "line_no": r["line_no"], "first_line_no": prior})
        else:
            first_seen[r["path"]] = r["line_no"]
    if header_count != len(real_zero):
        findings.append(
            {"kind": "zero_importer_header_mismatch", "header": header_count, "real": len(real_zero)}
        )
    return findings


def check_counts(rows: list[dict], inventory: dict) -> list[dict]:
    by_path = {
        m["path"]: m
        for m in inventory.get("python_modules", []) + inventory.get("typescript_modules", [])
    }
    findings = []
    for r in rows:
        if r.get("unparsed"):
            continue
        entry = by_path.get(r["path"])
        if entry is None:
            continue
        real_imports = len(entry.get("imports", []))
        real_imported_by = entry.get("imported_by_count", len(entry.get("imported_by", [])))
        if real_imports != r["imports"] or real_imported_by != r["imported_by"]:
            findings.append(
                {
                    "line_no": r["line_no"],
                    "kind": "count_drift",
                    "path": r["path"],
                    "doc": (r["imports"], r["imported_by"]),
                    "real": (real_imports, real_imported_by),
                }
            )
    return findings


def parse_module_count_summary(md_text: str) -> tuple[dict | None, list[dict]]:
    """The "HEAD <hash> has N modules ... (M total lines):" sentence and the per-root
    Modules/Lines table beneath it. Returns ``(None, [])`` when the sentence is absent, since
    a document without it has nothing here to check."""
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = MODULE_COUNT_SENTENCE_RE.match(line.strip())
        if not m:
            continue
        sentence = {
            "line_no": i + 1,
            "head": m.group("head"),
            "modules": int(m.group("modules")),
            "lines": int(m.group("lines")),
        }
        j = i + 1
        while j < len(lines) and lines[j].strip() != MODULE_COUNT_TABLE_HEADER:
            j += 1
        j += 2  # table header + markdown separator row
        rows: list[dict] = []
        while j < len(lines) and lines[j].startswith("|"):
            rm = MODULE_COUNT_ROW_RE.match(lines[j])
            if rm:
                rows.append(
                    {
                        "line_no": j + 1,
                        "package": rm.group("package").strip(),
                        "modules": int(rm.group("modules")),
                        "lines": int(rm.group("lines")),
                    }
                )
            j += 1
        return sentence, rows
    return None, []


def check_module_count_summary(
    sentence: dict | None, rows: list[dict], inventory: dict
) -> list[dict]:
    """The per-root Modules/Lines summary table and its sentence against the regenerated
    inventory's own totals: each row's module count against `counts.python_by_root` (or
    `counts.typescript_total` for the TypeScript row) and its line count against that root's
    real line sum, then the sentence's totals against the sum of those same rows, so the
    section carries one definition of a module and a line count, not a second one nothing
    checks."""
    if sentence is None:
        return []
    findings: list[dict] = []

    real_lines_by_root: dict[str, int] = {}
    for m in inventory.get("python_modules", []):
        real_lines_by_root[m["root"]] = real_lines_by_root.get(m["root"], 0) + m["lines"]
    real_lines_by_root[TYPESCRIPT_PACKAGE] = sum(
        m["lines"] for m in inventory.get("typescript_modules", [])
    )

    real_modules_by_root: dict[str, int] = dict(inventory.get("counts", {}).get("python_by_root", {}))
    real_modules_by_root[TYPESCRIPT_PACKAGE] = inventory.get("counts", {}).get("typescript_total", 0)

    packages = (*PYTHON_ROOT_PACKAGES, TYPESCRIPT_PACKAGE)
    by_package = {r["package"]: r for r in rows}
    for package in packages:
        row = by_package.get(package)
        real_modules = real_modules_by_root.get(package, 0)
        real_lines = real_lines_by_root.get(package, 0)
        if row is None:
            findings.append({"kind": "module_count_row_missing", "package": package})
            continue
        if row["modules"] != real_modules or row["lines"] != real_lines:
            findings.append(
                {
                    "line_no": row["line_no"],
                    "kind": "module_count_row_drift",
                    "package": package,
                    "doc": (row["modules"], row["lines"]),
                    "real": (real_modules, real_lines),
                }
            )

    real_total_modules = sum(real_modules_by_root.get(p, 0) for p in packages)
    real_total_lines = sum(real_lines_by_root.get(p, 0) for p in packages)
    if sentence["modules"] != real_total_modules or sentence["lines"] != real_total_lines:
        findings.append(
            {
                "line_no": sentence["line_no"],
                "kind": "module_count_sentence_drift",
                "doc": (sentence["modules"], sentence["lines"]),
                "real": (real_total_modules, real_total_lines),
            }
        )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("architecture_md", nargs="?", default=str(_REPO_ROOT / "ARCHITECTURE.md"))
    ap.add_argument("repo_root", nargs="?", default=str(_REPO_ROOT))
    ap.add_argument("--inventory-json", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    md_text = Path(args.architecture_md).read_text(encoding="utf-8")
    rows = parse_module_rows(md_text)
    parsed = [r for r in rows if not r.get("unparsed")]
    queued = [r for r in parsed if r.get("queued")]
    print(f"parsed {len(parsed)} module rows ({len([r for r in rows if r.get('unparsed')])} unparsed); "
          f"{len(queued)} carry a queued marker (checked the same)")

    existence = check_existence(rows, repo_root)
    missing = [f for f in existence if f["kind"] == "missing_path"]
    unparsed = [f for f in existence if f["kind"] == "unparsed_row"]
    if missing:
        print(f"MISSING at HEAD: {len(missing)}")
        for f in missing:
            print(f"  ARCHITECTURE.md:{f['line_no']}  {f['path']}")
    else:
        print("existence: every table-named path exists")
    for f in unparsed:
        print(f"UNPARSED ARCHITECTURE.md:{f['line_no']}: {f['raw']!r}")

    unnamed = check_coverage(parsed, repo_root)
    if unnamed:
        print(f"UNNAMED in the tables: {len(unnamed)}")
        for f in unnamed:
            print(f"  {f['path']}")
    else:
        print("coverage: every source file under a covered root is named by a table row")

    count_findings: list[dict] = []
    zero_findings: list[dict] = []
    module_count_findings: list[dict] = []
    if args.inventory_json and Path(args.inventory_json).exists():
        inventory = json.loads(Path(args.inventory_json).read_text(encoding="utf-8"))
        count_findings = check_counts(parsed, inventory)
        for f in count_findings:
            print(f"COUNT DRIFT ARCHITECTURE.md:{f['line_no']}  {f['path']}  doc={f['doc']} real={f['real']}")
        if not count_findings:
            print("counts: every checkable row matches the regenerated inventory")

        header_count, zero_rows = parse_zero_importer_section(md_text)
        zero_findings = check_zero_importers(header_count, zero_rows, inventory)
        for f in zero_findings:
            if f["kind"] == "zero_importer_missing":
                print(f"ZERO-IMPORTER MISSING FROM DOC: {f['path']}")
            elif f["kind"] == "zero_importer_extra":
                print(f"ZERO-IMPORTER NOT REALLY ZERO ARCHITECTURE.md:{f['line_no']}  {f['path']}")
            elif f["kind"] == "zero_importer_duplicate":
                print(f"ZERO-IMPORTER DUPLICATE ROW ARCHITECTURE.md:{f['line_no']}  {f['path']}  "
                      f"(first listed at line {f['first_line_no']})")
            else:
                print(f"ZERO-IMPORTER HEADER MISMATCH: header={f['header']} real={f['real']}")
        if not zero_findings:
            print("zero-importers: the list's header and membership match the regenerated inventory")

        summary_sentence, summary_rows = parse_module_count_summary(md_text)
        module_count_findings = check_module_count_summary(summary_sentence, summary_rows, inventory)
        for f in module_count_findings:
            if f["kind"] == "module_count_row_missing":
                print(f"MODULE COUNT ROW MISSING FROM SUMMARY TABLE: {f['package']}")
            elif f["kind"] == "module_count_row_drift":
                print(f"MODULE COUNT DRIFT ARCHITECTURE.md:{f['line_no']}  {f['package']}  "
                      f"doc={f['doc']} real={f['real']}")
            else:
                print(f"MODULE COUNT SENTENCE DRIFT ARCHITECTURE.md:{f['line_no']}  "
                      f"doc={f['doc']} real={f['real']}")
        if not module_count_findings:
            print("module counts: the per-root summary table and its sentence match the "
                  "regenerated inventory")

    total = (
        len(missing) + len(unparsed) + len(unnamed) + len(count_findings) + len(zero_findings)
        + len(module_count_findings)
    )
    print(f"{'FAIL' if total else 'PASS'}: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
