"""Check a dataset's on-disk content against its recorded identity: detect changed / moved data.

Reproduce-a-number defensibility: a delivered number is only reproducible from the data on disk if that
data still matches what was recorded. This recomputes the dataset's fingerprint (the authority) and
compares it to (a) the fingerprint cached in ``<dataset_root>/dataset.json`` (a mismatch means the
data changed since it was registered) and (b) the project's ``.tcip/datasets.json`` by id, so a
dataset found at a new path but with the same fingerprint reads as moved, not changed.

A dedicated script rather than folded into the ``@audited`` read tools: recomputing touches every
image on disk, which a read tool must not do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcip_mcp.dataset_layout import dataset_identity_path
from tcip_mcp.pipelines.resolution import dataset_fingerprint
from tcip_mcp.tools.project_tools import read_datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--project", type=Path, default=None,
                    help="project root holding .tcip/datasets.json (default: dataset_root)")
    args = ap.parse_args()

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.file_backend import bind_default

    bind_default()

    root: Path = args.dataset_root
    current = dataset_fingerprint(root)
    if current is None:
        print(f"no fingerprint for {root} (no images/labels, bespoke or empty)")
        return 0

    ident_path = dataset_identity_path(root)
    if not ident_path.is_file():
        print(f"UNREGISTERED: {root} has no dataset.json (run register_dataset). current={current}")
        return 1
    stored = json.loads(ident_path.read_text(encoding="utf-8"))
    ds_id = stored.get("id")
    recorded = stored.get("fingerprint")

    if recorded == current:
        print(f"OK: {root} unchanged (id={ds_id} crop={stored.get('crop')} fingerprint={current})")
        status = 0
    else:
        print(f"CHANGED: {root} content differs from its recorded identity "
              f"(recorded={recorded} current={current}); a number reproduced from it is no longer valid")
        status = 2

    # Moved: the project registry knows this id at a different path.
    regs = read_datasets(args.project or root)
    same_id = [r for r in regs if r.get("id") == ds_id]
    for r in same_id:
        if Path(r.get("path", "")) != root and r.get("fingerprint") == current:
            print(f"  MOVED: id {ds_id} is registered at {r.get('path')} but the same content is now at {root}")
    return status


if __name__ == "__main__":
    sys.exit(main())
