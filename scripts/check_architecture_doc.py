"""Verify ARCHITECTURE.md's module-ownership tables against the tree, for CI.

ARCHITECTURE.md is the adopter-facing map. Its module-ownership tables name a real file per row
with an in-repo-import count and an imported-by count. This check keeps the map from drifting
away from the code: it parses every such table and asserts each named path exists. With a
regenerated module-inventory JSON (the shape scripts under docs/audit produce), it also
cross-checks the two counts per row and reports numeric drift.

Row-level `queued:` HTML-comment markers are checked the same as any other row. Per
ARCHITECTURE.md's own marker convention, a queued-marked sentence states a fact that is true
today paired with a pending decision, not a statement that is false now.

Exit 0 when the tables match the tree, 1 when any named path is missing, any table row fails to
parse, or (when an inventory JSON is supplied) any count drifts. Defaults resolve ARCHITECTURE.md
and the repo root relative to this script, so CI invokes it as `python scripts/check_architecture_doc.py`.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("architecture_md", nargs="?", default=str(_REPO_ROOT / "ARCHITECTURE.md"))
    ap.add_argument("repo_root", nargs="?", default=str(_REPO_ROOT))
    ap.add_argument("--inventory-json", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    rows = parse_module_rows(Path(args.architecture_md).read_text(encoding="utf-8"))
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
    if args.inventory_json and Path(args.inventory_json).exists():
        inventory = json.loads(Path(args.inventory_json).read_text(encoding="utf-8"))
        count_findings = check_counts(parsed, inventory)
        for f in count_findings:
            print(f"COUNT DRIFT ARCHITECTURE.md:{f['line_no']}  {f['path']}  doc={f['doc']} real={f['real']}")
        if not count_findings:
            print("counts: every checkable row matches the regenerated inventory")

    total = len(missing) + len(unparsed) + len(unnamed) + len(count_findings)
    print(f"{'FAIL' if total else 'PASS'}: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
