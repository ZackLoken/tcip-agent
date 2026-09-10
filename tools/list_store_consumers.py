"""List every registered store's writers and readers, from the import graph plus a symbol scan.

For every store the registry declares (``tcip_mcp.store_catalogue.bootstrapped_stores``,
descriptors from ``tcip_store.registry``): the store's name, its classification (kind, frozen)
and version ceiling (``schema_version``), the module that declares its descriptor
(``StoreDescriptor.declared_in``), and every other module that references the descriptor's own
symbol, split into writers, readers, and references.

Heuristic, stated once: a store's declaring module names its identity in a module-level constant
following this repository's own convention, ``<NAME>_STORE = "<store_name>"`` (a plain string,
found by a regex over the declaring module's source, never guessed). A module counts as
referencing that symbol when the import graph ``tools/build_module_inventory.py`` builds says it
imports the declaring module, *and* the symbol's own name appears in that module's source text.
Among referencing modules: a **writer** calls one of the writing seam operations (``replace``,
``append``, ``delete``, ``clear_log``) anywhere in its own source; a **reader** calls one of the
read-shaped seam operations (``read``, ``read_versioned``, ``exists``, ``keys``, ``read_log``,
``read_blob_versioned``, ``open_blob``, ``blob_path``) and no writing operation; a module that
references the symbol but calls neither recognizably is printed under **references** rather than
folded into readers. This is textual over the whole module, not tied to the call site that
actually uses this key, so a module calling e.g. ``replace`` for an unrelated reason would
misclassify; it is a starting point for a human sweep, not a proof.

Takes a module-inventory JSON (the shape ``tools/build_module_inventory.py --out`` writes) rather
than building one itself, the same convention ``check_architecture_doc.py`` uses.

    python tools/build_module_inventory.py --out /tmp/inventory.json
    python tools/list_store_consumers.py --inventory-json /tmp/inventory.json
    python tools/list_store_consumers.py --inventory-json /tmp/inventory.json --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WRITE_OPS = ("replace", "append", "delete", "clear_log")
READ_OPS = (
    "read", "read_versioned", "exists", "keys", "read_log", "read_blob_versioned",
    "open_blob", "blob_path",
)


def find_symbol(source_text: str, store_name: str) -> str | None:
    """The module-level constant this repository's own convention assigns the store's name to
    (``<NAME>_STORE = "<store_name>"``), or ``None`` when the declaring module's source carries
    no such line."""
    pattern = re.compile(
        r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<quote>["\'])' + re.escape(store_name)
        + r'(?P=quote)\s*$',
        re.M,
    )
    m = pattern.search(source_text)
    return m.group(1) if m else None


def classify_module(source_text: str) -> str:
    """``writer``, ``reader``, or ``references`` for one module already confirmed to reference
    the symbol: see the module docstring for the exact heuristic and its stated limits."""
    if any(re.search(rf"\b{op}\s*\(", source_text) for op in WRITE_OPS):
        return "writer"
    if any(re.search(rf"\b{op}\s*\(", source_text) for op in READ_OPS):
        return "reader"
    return "references"


def store_consumers(inventory: dict, repo_root: Path, store_name: str, declared_in: str) -> dict:
    """One store's descriptor symbol and every module referencing it, split into writers,
    readers and references. ``declared_in`` is the dotted module name
    ``StoreDescriptor.declared_in`` records; modules come from that module's own
    ``imported_by`` list in ``inventory``.
    """
    by_dotted = {
        m["dotted_name"]: m for m in inventory.get("python_modules", []) if m.get("dotted_name")
    }
    by_path = {m["path"]: m for m in inventory.get("python_modules", [])}
    empty: dict = {"symbol": None, "writers": [], "readers": [], "references": []}

    declaring = by_dotted.get(declared_in)
    if declaring is None:
        return empty
    declaring_text = (repo_root / declaring["path"]).read_text(encoding="utf-8")
    symbol = find_symbol(declaring_text, store_name)
    if symbol is None:
        return empty

    writers: list[str] = []
    readers: list[str] = []
    references: list[str] = []
    for rel_path in declaring.get("imported_by", []):
        candidate = by_path.get(rel_path)
        if candidate is None:
            continue
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        if symbol not in text:
            continue
        target = candidate.get("dotted_name") or rel_path
        kind = classify_module(text)
        (writers if kind == "writer" else readers if kind == "reader" else references).append(target)

    return {
        "symbol": symbol,
        "writers": sorted(writers),
        "readers": sorted(readers),
        "references": sorted(references),
    }


def build_listing(inventory: dict, repo_root: Path) -> list[dict]:
    """One row per registered store, from the live registry (importing it registers every
    store, the same way any other whole-registry sweep in this repository does)."""
    from tcip_mcp.store_catalogue import bootstrapped_stores
    from tcip_store import get_descriptor

    rows = []
    for name in bootstrapped_stores():
        descriptor = get_descriptor(name)
        row = {
            "store": name,
            "kind": descriptor.kind,
            "frozen": descriptor.frozen,
            "schema_version": descriptor.schema_version,
            "declared_in": descriptor.declared_in,
        }
        row.update(store_consumers(inventory, repo_root, name, descriptor.declared_in))
        rows.append(row)
    return rows


def _print_row(row: dict) -> None:
    print(f"{row['store']}  kind={row['kind']} frozen={row['frozen']} "
          f"schema_version={row['schema_version']}")
    print(f"  declared in {row['declared_in']} as {row['symbol'] or '(no <NAME>_STORE constant found)'}")
    for label in ("writers", "readers", "references"):
        names = row[label]
        print(f"  {label}: {', '.join(names) if names else '(none)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory-json", required=True, type=Path,
                    help="a module-inventory JSON, from tools/build_module_inventory.py --out")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--json", dest="json_out", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    inventory = json.loads(args.inventory_json.read_text(encoding="utf-8"))
    rows = build_listing(inventory, args.repo_root)

    if args.json_out:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        _print_row(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
