"""Generate frozen-formats.json, the shipped freeze commitment, from the store registry.

The registry is the authority for every store's classification: freezing is declared on the
descriptor (frozen, schema_version, cannot_carry_field), and every check derives the frozen
set from the registry rather than a hand-kept list. This script renders that registry into a
committed file so CI can hold the code to the last committed edition:
tests/test_frozen_manifest.py regenerates the manifest in process and refuses any drift, so a
store added, dropped, reclassified or version-bumped without regenerating and committing this
file fails the suite. Run from the repo root and commit the result:

    python tools/generate_frozen_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "frozen-formats.json"

COMMITMENT = (
    "Every store this platform registers, with its declared freeze classification and the "
    "schema_version ceiling its readers know. A frozen store's documents carry no version "
    "field until the store's first bump: absence means the frozen version 1, and the first "
    "writer of the field is whichever future change bumps a format. A cannot_carry_field "
    "entry states why that store's documents cannot hold the field at all. This manifest "
    "pins declared classifications and ceilings only; an undeclared shape change inside "
    "version 1 is caught by the stability table's producer-fed round trips and the review "
    "shape, never by this file."
)


def manifest() -> dict[str, Any]:
    """The manifest document, derived from the live registry, deterministically ordered."""
    from tcip_mcp.store_catalogue import bootstrapped_stores
    from tcip_store import get_descriptor

    stores: dict[str, Any] = {}
    for name in sorted(bootstrapped_stores()):
        descriptor = get_descriptor(name)
        stores[name] = {
            "kind": descriptor.kind,
            "frozen": descriptor.frozen,
            "schema_version": descriptor.schema_version,
            "cannot_carry_field": descriptor.cannot_carry_field or None,
        }
    return {"commitment": COMMITMENT, "stores": stores}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare the committed manifest against the live registry instead of "
                         "writing; exit 1 on any drift, naming the stores that differ")
    args = ap.parse_args()

    document = manifest()
    if args.check:
        committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if committed == document:
            print(f"OK: {MANIFEST_PATH.name} matches the registry "
                  f"({len(document['stores'])} stores)")
            return 0
        drifted = sorted(
            set(committed["stores"]) ^ set(document["stores"])
            | {n for n in set(committed["stores"]) & set(document["stores"])
               if committed["stores"][n] != document["stores"][n]}
        )
        print(f"DRIFT: {MANIFEST_PATH.name} no longer matches the registry "
              f"({', '.join(drifted) or 'commitment text'}); regenerate with "
              "python tools/generate_frozen_manifest.py and commit the result with the "
              "change that moved the registry")
        return 1
    MANIFEST_PATH.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    frozen = sum(1 for s in document["stores"].values() if s["frozen"])
    print(f"wrote {MANIFEST_PATH}: {len(document['stores'])} stores, {frozen} frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
