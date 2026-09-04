"""Restamp a bare legacy dataset fingerprint onto the formula-version-prefixed form.

``dataset_fingerprint`` now returns ``v<n>:<hex>`` rather than a bare hex, so a fingerprint a
dataset registered before this family recorded still carries the old bare form in
``dataset.json`` and the project's ``.tcip/datasets.json`` (``register_dataset`` writes both from
one recomputed value, so either can be the one still bare). This script compares the bare hex this
code recomputes right now under the current formula to the bare value already recorded: equal, it
re-registers through ``register_dataset``'s own path, which stamps the prefixed form into both
records; unequal, it refuses and reports the mismatch rather than papering over it, since a formula
match cannot be assumed. A dataset already carrying a prefixed fingerprint in both records is left
alone: nothing here to restamp.

    python scripts/restamp_dataset_fingerprint.py <dataset_root> [--project <project_root>]

A mismatch here has two distinct causes, not one: a real content change since registration
(relabeling, a registry edit, confirming/un-confirming a negative, an image re-encode), which this
script will not paper over, or the dataset predating one of ``dataset_fingerprint``'s own pre-prefix
formula shifts (the confirmations term, or the wider image extension set) with no actual content
change. This script cannot tell the two apart; it refuses either way and names both possibilities
so the operator can judge which applies before re-registering by hand.

Never a runtime migration: the writer already produces the prefixed form, so this exists only
to carry an already-registered dataset onto it. Run ``export_store.py`` on the affected project
afterwards, and ``doctor.py`` to confirm.

``require_dataset_identity`` and ``read_datasets`` now refuse a bare pre-prefix fingerprint by
name, pointing back here; this script reads through their raw counterparts
(``read_dataset_identity_document``, ``read_datasets_raw``) instead, since its whole job is
seeing the very value those refuse on and fixing it.

Exit codes: 0 restamped or nothing to do, 1 the dataset is unregistered, 2 refused (a formula
mismatch, a bare-value disagreement between the two records, an uncomputable recompute, a
recompute mismatch, or what actually landed after re-registering does not match what was compared).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.dataset_layout import read_dataset_identity_document  # noqa: E402
from tcip_mcp.pipelines.data.dataset_fingerprint import (  # noqa: E402
    FINGERPRINT_FORMULA_VERSION,
    dataset_fingerprint,
    fingerprint_formula_version,
)
from tcip_mcp.tools.project_tools import (  # noqa: E402
    read_datasets_raw,
    register_dataset,
)


def _bare_hex(value: object) -> str | None:
    """The bare hex ``value`` states, formula-aware: a prefixed value's hex with the prefix
    stripped, an unprefixed legacy string as is, ``None`` for a value never recorded at all."""
    if value is None:
        return None
    formula = fingerprint_formula_version(value)
    if formula is not None:
        return str(value).split(":", 1)[1]
    return str(value)


def restamp(dataset_root: str, project_root: str | None = None) -> int:
    """Restamp one dataset's fingerprint if it is a bare legacy value that still matches the
    current recompute; refuse and report if it does not. Returns the exit code to use."""
    root = Path(dataset_root).absolute()
    try:
        identity = read_dataset_identity_document(root)
    except ValueError as exc:
        print(f"UNREGISTERED: {exc}")
        return 1

    recorded = identity.get("fingerprint")
    ds_id = identity.get("id")
    crop = identity.get("crop", "")
    proj = Path(project_root).absolute() if project_root else root

    matches = [r for r in read_datasets_raw(proj) if r.get("id") == ds_id]
    registry_fp = matches[0].get("fingerprint") if matches else None

    identity_bare = _bare_hex(recorded)
    registry_bare = _bare_hex(registry_fp) if matches else identity_bare
    if identity_bare != registry_bare:
        print(f"DISAGREE: {root} identity ({recorded!r}) and the registry entry under {proj} "
              f"({registry_fp!r}) name different content; refusing to restamp over an "
              "inconsistency between the two records. Investigate by hand before re-running.")
        return 2

    if recorded is None:
        print(f"OK: {root} never recorded a fingerprint; nothing to restamp. The next "
              "register_dataset stamps the formula-versioned form.")
        return 0

    recorded_formula = fingerprint_formula_version(recorded)
    if recorded_formula is not None and recorded_formula != FINGERPRINT_FORMULA_VERSION:
        print(f"FORMULA-MISMATCH: {root} carries a fingerprint stamped under formula "
              f"{recorded_formula}, but this code computes formula {FINGERPRINT_FORMULA_VERSION}; "
              "this script only restamps a bare pre-prefix value onto the current formula and "
              "does not migrate between formula versions. No action taken.")
        return 2
    if recorded_formula == FINGERPRINT_FORMULA_VERSION and (not matches or registry_fp == recorded):
        print(f"OK: {root} already carries a formula-versioned fingerprint ({recorded!r}); "
              "nothing to restamp")
        return 0

    current = dataset_fingerprint(root)
    if current is None:
        print(f"UNCOMPUTABLE: {root} has no images or labels to fingerprint right now, so this "
              "run cannot verify the recorded value before restamping it")
        return 2
    current_bare = current.split(":", 1)[1]
    if identity_bare != current_bare:
        print(f"MISMATCH: {root} recorded bare fingerprint {identity_bare!r} does not match the "
              f"current recompute {current_bare!r}. Two things produce this: a real content "
              "change since registration, which this script will not paper over, or the dataset "
              "predating one of dataset_fingerprint's own pre-prefix formula shifts (the "
              "confirmations term, or the wider image extension set) with no actual content "
              "change. Confirm which applies before acting; nothing was written by this script.")
        return 2

    result = register_dataset(str(root), crop, str(proj))
    if "error" in result:
        print(f"REFUSED: {root} could not be re-registered: {result['error']}")
        return 2

    written_identity = read_dataset_identity_document(root)
    written_fp = written_identity.get("fingerprint")
    written_matches = [r for r in read_datasets_raw(proj) if r.get("id") == ds_id]
    written_registry_fp = written_matches[0].get("fingerprint") if written_matches else None
    if written_fp != current or written_registry_fp != current:
        print(f"WROTE-BUT-MISMATCH: {root} was re-registered but what actually landed "
              f"(identity={written_fp!r}, registry={written_registry_fp!r}) does not match what "
              f"was compared ({current!r}); the data may have changed during this run. Re-run "
              "the script to re-verify.")
        return 2

    print(f"RESTAMPED: {root} fingerprint {recorded!r} -> {written_fp!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_root")
    ap.add_argument("--project", default=None, help="project root to register under "
                     "(default: dataset_root)")
    args = ap.parse_args()

    bind_default()

    return restamp(args.dataset_root, args.project)


if __name__ == "__main__":
    sys.exit(main())
