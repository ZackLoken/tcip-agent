"""Calibrate + held-out validate a detection operating point over a labeled split.

The confidence operating point is the phenotype for a count trait, so it must be derived per
dataset and validated against held-out ground truth, never pinned. This script runs
:func:`tcip_mcp.pipelines.count_calibration.resolve_count_operating_point`, the resolution the MCP
tool ``calibrate_count_operating_point`` also runs, and prints the full provenance and gate
evidence for inspection. It writes nothing: a validated claim is minted only by the audited door,
and a script writing into the experiment record would route that mutation around the audit log.

Usage:
    tcip calibrate-operating-point \
        --checkpoint <ckpt.pt> --trait <trait_name> \
        --labels-dir <labeled_dir> --images-dir <images_dir> \
        --dataset-root <dataset_root> --project-root <project_root> \
        [--experiment-id <id>] [--val-ratio 0.5] [--device cpu] [--subject <subject>] \
        [--attribute <attribute>] [--split-manifest-dir <dir>]

The checkpoint must be named by a registry entry under --project-root (register it with
register_model first); this script refuses one it is not, naming the digest and the root.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the model .pt checkpoint.")
    parser.add_argument("--trait", required=True, help="Trait name (defines the count objective).")
    parser.add_argument("--labels-dir", required=True, help="Labeled dir (per-image JSON).")
    parser.add_argument("--images-dir", required=True, help="Images for the labels.")
    parser.add_argument("--dataset-root", required=True,
                        help="Root the cal/holdout split lock is stored under, so this script and "
                             "run_inference's own calibration read one lock for these labels. The "
                             "labels' dataset root, or the labels dir itself when the dataset "
                             "layout places it under none.")
    parser.add_argument("--project-root", required=True,
                        help="The registry root the checkpoint must be named under: this script "
                             "binds only the backend, and platform_state_root() falls back to "
                             "the working directory, which would search an empty index.")
    parser.add_argument("--experiment-id", default=None,
                        help="Producing experiment id recorded in the printed provenance.")
    parser.add_argument("--val-ratio", type=float, default=0.5,
                        help="Holdout fraction of the labeled split (disjoint by stem). Only takes "
                             "effect on the first calibration call for this labels_dir's GT identity"
                             ": a cal/holdout split locks on its first draw, and a later run with a "
                             "different --val-ratio/--seed over unchanged labels reuses the locked "
                             "split unchanged (a divergence is printed, not silently ignored).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Split seed for the locked cal/holdout split. Same first-call-only "
                             "semantics as --val-ratio.")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    parser.add_argument("--group-by", default=None,
                        help="Grouping policy for the cal/holdout split: 'tile_prefix' (strips a "
                             "trailing _<x>_<y> tile offset) or 'stem' (one group per image). "
                             "Ignored if --group-key-map is given. Omitted, resolves to "
                             "'tile_prefix' when neither this nor --split-manifest-dir was given; "
                             "conflicts with --split-manifest-dir, whose own grouping policy "
                             "governs the locked draw instead.")
    parser.add_argument("--group-key-map", default=None,
                        help="Path to a JSON file mapping stem -> group key, overriding --group-by.")
    parser.add_argument("--subject", default=None,
                        help="The object class to read name-based labels for. Required with "
                             "--split-manifest-dir; a run's own admission needs one too.")
    parser.add_argument("--attribute", default=None,
                        help="Scope the draw to instances already assessed for this attribute "
                             "of --subject.")
    parser.add_argument("--split-manifest-dir", default=None,
                        help="Restrict the calibration universe to one capture date's "
                             "calibration side of a split manifest (draw_splits' output "
                             "directory) instead of every labeled stem, the same restriction "
                             "run_inference applies. Conflicts with --group-by/--group-key-map; "
                             "requires --subject.")
    args = parser.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    group_key_map = None
    if args.group_key_map:
        with open(args.group_key_map, encoding="utf-8") as f:
            group_key_map = json.load(f)

    from tcip_mcp.model_registry import UnregisteredCheckpoint
    from tcip_mcp.pipelines.count_calibration import (
        CalibrationUsageError, resolve_count_operating_point,
    )

    try:
        result = resolve_count_operating_point(
            checkpoint_path=args.checkpoint, trait=args.trait, labels_dir=args.labels_dir,
            images_dir=args.images_dir, dataset_root=args.dataset_root,
            project_root=args.project_root, subject=args.subject, attribute=args.attribute,
            experiment_id=args.experiment_id, group_by=args.group_by, group_key_map=group_key_map,
            split_manifest_dir=args.split_manifest_dir, val_ratio=args.val_ratio, seed=args.seed,
            device=args.device,
        )
    except (UnregisteredCheckpoint, CalibrationUsageError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    locked = result.locked
    if locked.get("policy_divergence"):
        div = locked["policy_divergence"]
        print(f"Note: a cal/holdout split for this labels_dir is already locked with a different "
              f"policy than requested; the locked split is used unchanged.\n"
              f"  requested: {div['requested']}\n  locked:    {div['locked']}\n"
              f"  Use redraw_calibration_holdout to redraw deliberately.", file=sys.stderr)
    if locked.get("unlocked_stems"):
        print(f"Note: {len(locked['unlocked_stems'])} stem(s) are new since this split was locked "
              "and are excluded from this calibration (the lock stays authoritative for what it "
              "already covers).", file=sys.stderr)

    bundle = result.bundle
    provenance = bundle.to_provenance()
    provenance["gate_evidence"] = bundle.get("conf").gate_evidence

    conf = bundle.get("conf")
    print(f"trait={args.trait} dataset_hash={result.dataset_hash}")
    print(f"conf={conf._raw:.4f} validated_against={conf.validated_against} shippable={bundle.is_shippable}")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
