"""Check a dataset's on-disk content against its recorded identity: detect changed / moved data.

Reproduce-a-number defensibility: a delivered number is only reproducible from the data on disk if that
data still matches what was recorded. This recomputes the dataset's fingerprint (the authority) and
compares it to (a) the fingerprint cached in ``<dataset_root>/dataset.json`` (a mismatch means the
data changed since it was registered) and (b) the project's ``.tcip/datasets.json`` by id, so a
dataset found at a new path but with the same fingerprint reads as moved, not changed.

A dedicated script rather than folded into the ``@audited`` read tools: recomputing touches every
image on disk, which a read tool must not do.

``require_dataset_identity`` and ``read_datasets`` refuse a bare pre-prefix fingerprint rather
than serve it as a dataset's current identity; this script's own FORMULA-UNRECORDED and
MOVED-FORMULA-UNRECORDED outcomes exist to report exactly that condition, a diagnostic rather
than a delivered result, so it reads through the raw counterparts
(``read_dataset_identity_document``, ``read_datasets_raw``) instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tcip_store import SchemaVersionRefused

from tcip_mcp.dataset_layout import read_dataset_identity_document
from tcip_mcp.pipelines.data.dataset_fingerprint import (
    FINGERPRINT_FORMULA_VERSION,
    dataset_fingerprint,
    fingerprint_formula_version,
)
from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets_raw


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
        identity = read_dataset_identity_document(root)
    except SchemaVersionRefused as exc:
        print(f"VERSION-REFUSED: {exc}")
        return 5
    except ValueError as exc:
        print(f"UNREGISTERED: {exc}")
        return 1
    ds_id = identity.get("id")
    recorded = identity.get("fingerprint")

    if recorded is None:
        print(f"NEVER-RECORDED: {root} carries no recorded fingerprint yet; nothing to compare "
              f"current={current} against. Register with register_dataset to stamp one.")
        status = 4
    elif recorded == current:
        print(f"OK: {root} unchanged (id={ds_id} crop={identity.get('crop')} fingerprint={current})")
        status = 0
    elif fingerprint_formula_version(recorded) != FINGERPRINT_FORMULA_VERSION:
        print(f"FORMULA-UNRECORDED: {root} carries a bare fingerprint ({recorded!r}) from before "
              f"the formula-version prefix; current={current} cannot be compared to it as same or "
              "changed. Re-register with register_dataset to restamp under the current formula.")
        status = 3
    else:
        print(f"CHANGED: {root} content differs from its recorded identity "
              f"(recorded={recorded} current={current}); a number reproduced from it is no longer valid")
        status = 2

    # Moved: the project registry knows this id at a different path, by identity rather than a
    # stored-versus-passed spelling. Formula-aware: never bare string equality across formulas.
    project = args.project or root
    regs = read_datasets_raw(project)
    same_id = [r for r in regs if r.get("id") == ds_id]
    for r in same_id:
        entry_path = dataset_entry_path(project, r)
        try:
            same_as_root = os.path.samefile(entry_path, root)
        except OSError:
            same_as_root = False
        if same_as_root:
            continue
        r_fp = r.get("fingerprint")
        if r_fp == current:
            print(f"  MOVED: id {ds_id} is registered at {entry_path} but the same content is now at {root}")
        elif r_fp is not None and fingerprint_formula_version(r_fp) != FINGERPRINT_FORMULA_VERSION:
            print(f"  MOVED-FORMULA-UNRECORDED: id {ds_id} is registered at {entry_path} with a "
                  f"fingerprint from before the formula-version prefix ({r_fp!r}); whether it is "
                  f"the same content now at {root} (current={current}) cannot be told without "
                  "re-registering that entry with register_dataset.")
    return status


if __name__ == "__main__":
    sys.exit(main())
