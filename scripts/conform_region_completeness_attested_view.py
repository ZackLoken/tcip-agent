"""Conform a dataset's stored ``region_completeness`` records to carry the ``cells_attested_view``
key: write-forward ``{}`` onto every bucket record written before the key existed.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md), modelled on ``conform_delivery_events.py``'s acknowledgement write-forward: a record
written before ``cells_attested_view`` existed recorded no cell's scale provenance at all, so an
empty map is the derivable true value, not a guess, since it states the absence of the act rather
than a guess at its content. The coverage routes refuse a record still lacking the key by name,
pointing at this script.

    python scripts/conform_region_completeness_attested_view.py <dataset_root> [<dataset_root> ...]
    python scripts/conform_region_completeness_attested_view.py --plan <dataset_root>

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

_KEY = "cells_attested_view"


def _conform_bucket(bucket: str, record: dict, *, plan: bool) -> tuple[str, bool]:
    """One outcome line for ``bucket``'s stored record, and whether it was (or, in ``plan`` mode,
    would be) changed. Mutates ``record`` in place, setting ``_KEY`` to ``{}``, when it is
    missing and ``plan`` is false."""
    if _KEY in record:
        return f"{bucket}: already carries {_KEY}, unchanged", False
    verb = "would write-forward" if plan else "write-forwarded"
    if not plan:
        record[_KEY] = {}
    return (
        f"{bucket}: {verb} {_KEY} to {{}} (no scale provenance was recorded before the key "
        "existed)"
    ), True


def _corrupt_store_outcome(root: Path, store: object) -> str:
    """The one refusal line for a stored region-completeness document that is present but not
    the dict shape this script recognizes: named rather than silently treated as empty, so a
    corrupt store is never mistaken for one with nothing to conform."""
    return (
        f"refused, the region_completeness store under {root} is a {type(store).__name__}, not "
        "a dict; this script does not know how to conform it"
    )


def conform_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    """Every outcome line for ``root``, and whether it was refused (no ``.tcip`` directory, or
    a stored document present but not the recognized dict shape)."""
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True

    key = region_completeness_key(root)
    if plan:
        store = ts.read(key, default={})
        if not isinstance(store, dict):
            return [_corrupt_store_outcome(root, store)], True
        outcomes = [
            _conform_bucket(bucket, dict(record), plan=True)[0]
            for bucket, record in store.items() if isinstance(record, dict)
        ]
        return outcomes, False

    with ts.transaction(key) as txn:
        store = txn.read(key, default={})
        if not isinstance(store, dict):
            return [_corrupt_store_outcome(root, store)], True
        outcomes = []
        changed = False
        for bucket, record in store.items():
            if not isinstance(record, dict):
                continue
            line, did_change = _conform_bucket(bucket, record, plan=False)
            outcomes.append(line)
            changed = changed or did_change
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
