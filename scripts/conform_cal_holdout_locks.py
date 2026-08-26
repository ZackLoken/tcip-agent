"""Conform every pre-existing ``cal_holdout_split_lock`` record under a root to carry
``split_manifest_dir``, the key every lock this family writes now declares (``None`` for a
whole-directory draw). A record written before this family lacks the key entirely, in both the
record itself and each of its ``redraw_history`` entries' own recorded policy; this is the
one-off operator fix for that, never a runtime migration.

Usage:
    python scripts/conform_cal_holdout_locks.py <root>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def conform_cal_holdout_locks(root: str) -> int:
    """Add ``split_manifest_dir: None`` to every ``cal_holdout_split_lock`` record under
    ``root`` that lacks it, and to the recorded policy of each of its ``redraw_history``
    entries that lacks it there too. Returns the number of records written; a record that
    already carries the key everywhere it belongs is left untouched.
    """
    import tcip_store
    from tcip_mcp.pipelines.data.splits import CAL_HOLDOUT_LOCK_STORE

    root_str = str(Path(root).resolve())
    conformed = 0
    for key in tcip_store.keys(CAL_HOLDOUT_LOCK_STORE, root_str):
        record = tcip_store.read(key)
        if not isinstance(record, dict):
            continue
        history = record.get("redraw_history") or []
        already_conformed = "split_manifest_dir" in record and all(
            isinstance(entry.get("policy"), dict) and "split_manifest_dir" in entry["policy"]
            for entry in history
        )
        if already_conformed:
            continue
        record = dict(record)
        record.setdefault("split_manifest_dir", None)
        new_history = []
        for entry in history:
            entry = dict(entry)
            policy = dict(entry.get("policy") or {})
            policy.setdefault("split_manifest_dir", None)
            entry["policy"] = policy
            new_history.append(entry)
        record["redraw_history"] = new_history
        tcip_store.replace(key, record)
        conformed += 1
    return conformed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root the cal_holdout_split_lock records are stored under.")
    args = parser.parse_args(argv)

    from tcip_store.binding import bind_default
    bind_default()

    count = conform_cal_holdout_locks(args.root)
    print(f"conformed {count} cal_holdout_split_lock record(s) under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
