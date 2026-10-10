"""Check a dataset's on-disk content against its recorded identity: detect changed / moved data.

Recomputes the dataset's fingerprint (the authority) and compares it to the fingerprint recorded in
``<dataset_root>/dataset.json`` (a mismatch means the data changed since it was registered), then
reads the identity at the path the project's dataset registry holds for the same id and compares
it record to record: the same identity elsewhere is moved, another identity there diverged, none
readable there gone. Recomputing touches every image on disk.
"""

from __future__ import annotations

import argparse
import sys

from tcip_mcp.cli import bound_project
from tcip_mcp.dataset_layout import require_dataset_identity
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_mcp.registry_paths import located
from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, prog=prog)
    ap.add_argument("dataset_root")
    ap.add_argument("--project", required=True,
                    help="the project whose dataset registry names this dataset")
    args = ap.parse_args(argv)
    project = bound_project(args.project)

    root = located(args.dataset_root, project)
    current = dataset_fingerprint(root)
    if current is None:
        print(f"no fingerprint for {root} (no images/labels, bespoke or empty)")
        return 0

    try:
        identity = require_dataset_identity(root)
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
              f"(recorded={recorded} current={current}); "
              "a number reproduced from it is no longer valid")
        status = 2

    for r in read_datasets(project):
        if r["id"] != ds_id:
            continue
        entry_path = dataset_entry_path(project, r)
        try:
            registered = require_dataset_identity(entry_path)
        except (ValueError, OSError) as exc:
            print(f"  GONE: id {ds_id} is registered at {entry_path}, which holds no readable "
                  f"identity ({exc}); it is now at {root}")
            continue
        if registered != identity:
            print(f"  DIVERGED: id {ds_id} is registered at {entry_path}, whose identity "
                  f"{registered} is not {root}'s {identity}")
        elif entry_path.resolve() != root:
            print(f"  MOVED: id {ds_id} is registered at {entry_path} and is now at {root}")
    return status


if __name__ == "__main__":
    sys.exit(main())
