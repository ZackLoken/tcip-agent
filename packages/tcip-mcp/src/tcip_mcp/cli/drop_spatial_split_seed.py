"""Drop the recorded ``seed`` key from every persisted spatial-strip split manifest under a
project root: the key never governed the strip layout (a spatial-strip split places every side
by declared order and share alone, never by seed) and the Q45 ruling drops it from the record.

A logged operator command: bind, walk, one outcome line per unit, exit 2 on any refusal or, under
``--plan``, on any record that would change. Its units are the split store's own members
(``EXPERIMENT_SPLIT_STORE``, the ``experiment_split`` record every training run's own
``persist_split_manifest`` writes), enumerated through the store itself
(``store.keys(EXPERIMENT_SPLIT_STORE, experiments_scope(root))``, the same shape
``experiment_ids_with_status`` uses over the status store), so a split record whose status
record is gone is still visited.

Per record, in order:

1. Read through ``read_split_manifest_checked``, whose ``(manifest, decode_error)`` is this
   command's own three-way outcome.
2. A record that will not decode is reported and left, never rewritten.
3. A record with no ``spatial`` block, or a ``spatial`` block already carrying no ``seed``, is
   left byte-identical and reported as such.
4. A record whose ``spatial`` block carries ``seed`` is rewritten without that key through
   ``store.replace`` on the same key (the store's ``last_writer_wins`` policy admits an
   unconditional replace) and one audit line is written naming the experiment and the key
   removed. An ``AuditEntryNotWritten`` out of that line propagates and stops the walk, naming
   the rewritten record whose line was lost, never folded into the exit code, since the rewrite
   has already committed.

This command writes the record directly, past ``refuse_if_terminal``: that check lives inside
``persist_split_manifest``'s own transaction, not as a store policy, and every record this
command targets belongs to a completed run whose own writer will never touch it again. This is
the one operator act that suspends "experiments are immutable" for a record no run will write
again.

Runs under whichever backend the process binds; a rewrite under the database backend leaves any
earlier loose export of that root stale, refreshed by ``tcip export-store``.

``--plan`` previews every outcome without writing anything; a record that would change under a
real run counts toward the exit code the same way a refusal does.

    tcip drop-spatial-split-seed <project_root> [<project_root> ...]
    tcip drop-spatial-split-seed --plan <project_root>

Exit codes: 0 when every visited record in every named root already carries no ``spatial.seed``
or was just conformed; 2 if any root is refused, any record will not decode, or (under
``--plan``) any record would change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tcip_store import store
from tcip_store.binding import bind_default

from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
from tcip_mcp.experiments import (
    EXPERIMENT_SPLIT_STORE, experiments_scope, read_split_manifest_checked, split_key,
)

TOOL_NAME = "drop_spatial_split_seed"


def experiment_ids_with_split(root: Path) -> list[str]:
    """Every experiment id the split store holds a record for under ``root``, sorted: the same
    enumeration shape ``experiment_ids_with_status`` uses over the status store, so a split
    record whose status record is gone is still visited."""
    found = store.keys(EXPERIMENT_SPLIT_STORE, experiments_scope(root))
    return sorted(key.parts[0] for key in found if key.parts[1] == "split")


def process_record(experiment_id: str, *, root: Path, plan: bool) -> tuple[str, bool, bool]:
    """Drop ``spatial.seed`` from one experiment's persisted split record. Returns ``(outcome,
    exit_worthy, changed)``; ``exit_worthy`` is true for an undecodable record and for a record
    that would change under ``--plan``, the same convention ``repair_classified_predictions``
    uses for its own preview and refusal outcomes."""
    manifest, decode_error = read_split_manifest_checked(experiment_id, root=root)
    if decode_error is not None:
        return (f"its split record will not decode, left as it is: {decode_error}", True, False)
    spatial = manifest.get("spatial")
    if not isinstance(spatial, dict) or "seed" not in spatial:
        return ("carries no spatial.seed; left as it is", False, False)
    if plan:
        return ("would drop spatial.seed", True, False)

    new_manifest = {**manifest, "spatial": {k: v for k, v in spatial.items() if k != "seed"}}
    store.replace(split_key(experiment_id, root=root), new_manifest)
    outcome = "dropped spatial.seed"
    try:
        record_event_or_raise(
            TOOL_NAME, {"experiment_id": experiment_id}, scope=root, key_removed="spatial.seed",
        )
    except AuditEntryNotWritten as exc:
        raise AuditEntryNotWritten(
            f"{TOOL_NAME}: experiment {experiment_id!r} under {root} had spatial.seed dropped "
            "and rewritten, but the audit line for it could not be written",
            exc.__cause__ or exc,
        ) from exc
    return (outcome, False, True)


def process_project_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    outcomes: list[str] = []
    refused = False
    changed_count = 0
    unchanged_count = 0
    if not (root / ".tcip").is_dir():
        return ([f"{root}: refused, no .tcip directory found; not a project root; no summary "
                 "applies"], True)
    for experiment_id in experiment_ids_with_split(root):
        outcome, exit_worthy, changed = process_record(experiment_id, root=root, plan=plan)
        outcomes.append(f"{root} {experiment_id}: {outcome}")
        if changed:
            changed_count += 1
        else:
            unchanged_count += 1
        if exit_worthy:
            refused = True
    outcomes.append(
        f"{root}: {changed_count} record(s) changed, {unchanged_count} left as they were")
    return outcomes, refused


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("roots", nargs="*", type=Path)
    ap.add_argument("--plan", action="store_true", help="preview only; nothing is written")
    args = ap.parse_args(argv)

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = process_project_root(root, plan=args.plan)
        if refused:
            refused_any = True
        for line in outcomes:
            print(line)

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())
