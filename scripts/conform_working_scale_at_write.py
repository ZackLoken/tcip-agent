"""Conform a dataset's stored ``region_completeness`` records to the current
``cells_attested_view`` key: ``working_scale_bar_at_write`` renamed to ``working_scale_at_write``.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md), modelled on ``conform_view_coverage_viewing.py``: the coverage lattice now derives
from the breeder's own set grid zoom rather than an annotation-derived bar, so an attestation's
scale provenance carries the working scale (``WorkingScale {zoom, source}``) in effect at write
time under the new key name, never the old annotation-derived bar. The old bar cannot be mapped
forward to a working scale (they measure different things: a documented default span over saved
annotations, versus a breeder-stated zoom), so this script renames the key and nulls the value on
every entry it touches, reporting each; a later re-attestation is the only way to stamp a real
working scale under the new key. It never runs as part of any runtime path, and the completeness
routes refuse a record still carrying the old key by name, pointing at this script.

    python scripts/conform_working_scale_at_write.py <dataset_root> [<dataset_root> ...]
    python scripts/conform_working_scale_at_write.py --plan <dataset_root>

Exit codes: 0 conformed (or nothing to conform) for every root named; 2 if any root does not
exist or holds no ``.tcip`` directory. Every root named on the command line is still processed
after another root's refusal. The write, when there is one, happens inside the same
``tcip_store.transaction`` lock ``post_completeness`` itself takes on the record's key, so it can
never race a concurrent attestation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_mcp.dataset_layout import region_completeness_key  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

OLD_KEY = "working_scale_bar_at_write"
NEW_KEY = "working_scale_at_write"


def _conform_entry(entry: dict) -> tuple[str, dict | None]:
    """One outcome clause and, when ``entry`` still carries ``OLD_KEY``, its replacement (the
    same entry with ``OLD_KEY`` dropped, ``NEW_KEY`` set to ``None``); ``None`` when the entry is
    already conformed."""
    if OLD_KEY not in entry:
        return "already conformed, unchanged", None
    replacement = {k: v for k, v in entry.items() if k != OLD_KEY}
    replacement[NEW_KEY] = None
    return f"renamed {OLD_KEY} to {NEW_KEY}, nulled (no working scale to carry forward)", \
        replacement


def _conform_record(bucket: str, record: dict, *, plan: bool) -> list[str]:
    """Every outcome line for one bucket's ``cells_attested_view`` entries; mutates ``record`` in
    place with each entry's replacement when ``plan`` is false."""
    outcomes: list[str] = []
    attested_view = record.get("cells_attested_view")
    if not isinstance(attested_view, dict) or not attested_view:
        return outcomes
    for cell, entry in list(attested_view.items()):
        if not isinstance(entry, dict):
            outcomes.append(f"{bucket}/{cell}: not a mapping, left as stored")
            continue
        message, replacement = _conform_entry(entry)
        outcomes.append(f"{bucket}/{cell}: {'would be ' if plan else ''}{message}")
        if replacement is not None and not plan:
            attested_view[cell] = replacement
    return outcomes


def conform_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    """Every outcome line for ``root``, and whether it was refused (no ``.tcip`` directory)."""
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True

    key = region_completeness_key(root)
    if plan:
        store = ts.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        outcomes: list[str] = []
        for bucket, record in store.items():
            if isinstance(record, dict):
                outcomes.extend(_conform_record(bucket, dict(record), plan=True))
        return outcomes, False

    with ts.transaction(key) as txn:
        store = txn.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        outcomes = []
        changed = False
        for bucket, record in store.items():
            if not isinstance(record, dict):
                continue
            before = _conform_record(bucket, record, plan=False)
            if before:
                outcomes.extend(before)
                changed = True
        if changed:
            txn.write(key, store)
    return outcomes, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = conform_root(root, plan=args.plan)
        if refused:
            refused_any = True
        if not outcomes:
            outcomes = ["nothing to conform"]
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
