"""Check a dataset's on-disk content against its recorded identity: detect changed / moved data.

Recomputes the dataset's fingerprint (the authority) and compares it to (a) the fingerprint cached
in ``<dataset_root>/dataset.json`` (a mismatch means the data changed since it was registered) and
(b) the project's ``.tcip/datasets.json`` by id, so a dataset found at a new path but with the same
fingerprint reads as moved, not changed. Recomputing touches every image on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tcip_store import SchemaVersionRefused

from tcip_mcp.dataset_layout import require_dataset_identity
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, prog=prog)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--project", type=Path, default=None,
                    help="project root holding .tcip/datasets.json (default: dataset_root)")
    args = ap.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    root: Path = args.dataset_root
    try:
        current = dataset_fingerprint(root)
    except SchemaVersionRefused as exc:
        print(f"VERSION-REFUSED: {exc}")
        return 5
    if current is None:
        print(f"no fingerprint for {root} (no images/labels, bespoke or empty)")
        return 0

    try:
        identity = require_dataset_identity(root)
    except SchemaVersionRefused as exc:
        print(f"VERSION-REFUSED: {exc}")
        return 5
    except ValueError as exc:
        print(f"UNREGISTERED: {exc}")
        return 1
    ds_id = identity["id"]
    recorded = identity["fingerprint"]

    if recorded is None:
        print(f"NEVER-RECORDED: {root} carries no recorded fingerprint yet; nothing to compare "
              f"current={current} against. Register with register_dataset to stamp one.")
        status = 4
    elif recorded == current:
        print(f"OK: {root} unchanged (id={ds_id} crop={identity['crop']} fingerprint={current})")
        status = 0
    else:
        print(f"CHANGED: {root} content differs from its recorded identity "
              f"(recorded={recorded} current={current}); a number reproduced from it is no longer valid")
        status = 2

    # Moved: the project registry knows this id at a different path, by identity rather than a
    # stored-versus-passed spelling.
    project = args.project or root
    for r in read_datasets(project):
        if r.get("id") != ds_id:
            continue
        entry_path = dataset_entry_path(project, r)
        try:
            same_as_root = os.path.samefile(entry_path, root)
        except OSError:
            same_as_root = False
        if not same_as_root and r["fingerprint"] == current:
            print(f"  MOVED: id {ds_id} is registered at {entry_path} but the same content is now at {root}")
    return status


if __name__ == "__main__":
    sys.exit(main())
