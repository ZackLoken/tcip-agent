"""Stamp the ``metrics_logged`` marker onto every experiment a root's status record predates.

``log_metrics`` now writes ``status.json["metrics_logged"] = True`` before its append, so
:func:`tcip_mcp.experiments.is_pristine` reads that field instead of scanning the metrics log.
An experiment that logged rows before this change carries no such marker yet, and would read as
pristine (and so be eligible for a full config.json rewrite via
``overwrite_config_if_pristine``) even though it already has real epoch history. This is a
one-off operator conform, not a runtime migration: it is run once against a root whose
experiments predate the change, never invoked from any code path that reads or writes state
during ordinary operation.

    python scripts/conform_metrics_marker.py <root>

Exit codes: 0 on success (regardless of how many experiments needed conforming).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def conform(root: Path) -> list[str]:
    """Stamp the marker on every experiment under ``root`` whose log holds a row and whose
    status record lacks it. Returns the ids touched, in enumeration order."""
    from tcip_store import store

    from tcip_mcp.experiments import (
        experiment_ids_with_status, read_member, read_metrics, status_key,
    )

    touched: list[str] = []
    for experiment_id in experiment_ids_with_status(root):
        if not read_metrics(experiment_id, root=root):
            continue
        status = read_member(status_key(experiment_id, root=root), {})
        if isinstance(status, dict) and status.get("metrics_logged"):
            continue
        key = status_key(experiment_id, root=root)
        with store.transaction(key) as txn:
            record = txn.read(key, default={})
            record["metrics_logged"] = True
            txn.write(key, record)
        touched.append(experiment_id)
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="a project or platform-state root holding .tcip/experiments")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"error: not a directory: {root}")
        return 2

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    touched = conform(root)
    for experiment_id in touched:
        print(f"conformed: {experiment_id}")
    print(f"\nconform_metrics_marker: {len(touched)} experiment(s) stamped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
