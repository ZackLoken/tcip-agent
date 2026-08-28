"""Calibrate + held-out validate a detection operating point over a labeled split.

The confidence operating point is the phenotype for a count trait, so it must be derived per
dataset and validated against held-out ground truth, never pinned. This script runs one
low-threshold model pass over a disjoint calibration/holdout split of a labeled dir, resolves the
count-unbiased operating point, checks its held-out count bias, and prints the full provenance and
sweep for inspection. It writes nothing: a validated claim is minted only by the audited doors,
and a script writing into the experiment record would route that mutation around the audit log.

It is a script (not a new MCP tool) per CLAUDE.md: the audited count is produced at ``run_inference``
(which now accepts ``trait`` + ``calibration_labels_dir`` to resolve the same operating point inline);
this is the offline inspector for the sweep behind that resolution.

Usage:
    python scripts/calibrate_operating_point.py \
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
                             "binds only the backend, and project_root() falls back to the "
                             "working directory, which would search an empty index.")
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
                             "calibration side of a split manifest (make_splits' output "
                             "directory) instead of every labeled stem, the same restriction "
                             "run_inference applies. Conflicts with --group-by/--group-key-map; "
                             "requires --subject.")
    args = parser.parse_args(argv)
    if args.split_manifest_dir and not args.subject:
        print("--split-manifest-dir requires --subject.", file=sys.stderr)
        return 2
    if args.split_manifest_dir and (args.group_by is not None or args.group_key_map):
        print("--split-manifest-dir conflicts with --group-by/--group-key-map: the manifest's "
              "own grouping policy governs the locked draw.", file=sys.stderr)
        return 2

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    bind_default()

    from torch.utils.data import DataLoader

    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, manifest_date_key, resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, derive_max_dets_from_counts, records_over_loader,
        resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS, dataset_hash
    from tcip_mcp.pipelines.training.generic_trainer import task_collate

    # --labels-dir is this script's measurement reference, so it clears the one admissibility rail.
    require_reference_ground_truth(args.labels_dir)

    try:
        checkpoint = load_registered_checkpoint(args.checkpoint, project_path=args.project_root)
    except UnregisteredCheckpoint as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Match the MCP path's initial predictor construction (DEFAULT_MAX_DETS); superseded below,
    # once this split's density is known, by set_detector_operating_point.
    predictor = build_predictor(checkpoint, device=args.device, max_dets=DEFAULT_MAX_DETS)
    tile_size = getattr(predictor, "train_tile_size", None)

    probe = build_dataset("detection", images_dir=args.images_dir, labels_dir=args.labels_dir,
                          subject=args.subject, attribute=args.attribute)
    stems = sorted(getattr(probe, "stems", []))
    if len(stems) < 2:
        print(f"Need >=2 labeled stems to split cal/holdout; found {len(stems)}.", file=sys.stderr)
        return 2

    group_key_map = None
    if args.group_key_map:
        with open(args.group_key_map, encoding="utf-8") as f:
            group_key_map = json.load(f)
    group_by = args.group_by

    from tcip_mcp.dataset_layout import annotation_date

    cal_date = annotation_date(args.labels_dir)
    split_manifest_sha256 = None
    if args.split_manifest_dir:
        from tcip_mcp.pipelines.data.splits import resolve_manifest_calibration_universe
        from tcip_mcp.pipelines.resolution import manifest_digest
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(args.split_manifest_dir)
        split_manifest_sha256 = manifest_digest(manifest)
        try:
            stems, group_by, group_key_map, _excluded, cal_date = \
                resolve_manifest_calibration_universe(
                    manifest, args.split_manifest_dir, args.labels_dir, args.images_dir,
                    args.subject, args.attribute, stems)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    dh = dataset_hash(args.labels_dir, stems=(stems if args.split_manifest_dir else None))
    annotation_counts = {
        s: count_label_lines(args.labels_dir, s, subject=args.subject, attribute=args.attribute)
        for s in stems
    }
    # Detector-cap censoring: the flat DEFAULT_MAX_DETS below can still truncate a dense
    # calibration image's raw detections the same way a too-high conf floor censors them, so
    # derive the collection-pass cap from this labeled split's own density (same ~1.5x p99 formula
    # resolve_operating_point uses for the shipped max_dets) so the sweep isn't measured against an
    # arbitrary constant that may sit below what a dense scene actually needs.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    # Locked split: the first call for this labels_dir's GT identity draws and locks the
    # cal/holdout split; a later run of this script over unchanged labels returns the same split
    # rather than a fresh cut that could happen to draw a weaker holdout.
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=args.dataset_root,
        annotation_counts=annotation_counts,
        group_by=(group_by or "tile_prefix"), group_key_map=group_key_map,
        holdout_ratio=args.val_ratio, seed=args.seed,
        split_manifest_dir=args.split_manifest_dir,
    )
    if locked.get("policy_divergence"):
        div = locked["policy_divergence"]
        print(f"Note: a cal/holdout split for this labels_dir is already locked with a different "
              f"policy than requested; the locked split is used unchanged.\n"
              f"  requested: {div['requested']}\n  locked:    {div['locked']}\n"
              f"  Use force_redraw_cal_holdout_split to redraw deliberately.", file=sys.stderr)
    if locked.get("unlocked_stems"):
        print(f"Note: {len(locked['unlocked_stems'])} stem(s) are new since this split was locked "
              "and are excluded from this calibration (the lock stays authoritative for what it "
              "already covers).", file=sys.stderr)
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    # Floor the in-model conf so hesitant detections survive to be swept, and raise the cap to this
    # split's own density (derived above); this call executes after build_predictor's construction-
    # time DEFAULT_MAX_DETS (matching the MCP path's own initial predictor exactly) and wins, so
    # density_cap is the value that actually governs the collection pass. The applied score_thresh
    # (not a re-typed 0.01 literal) is threaded into resolve_operating_point as staged_conf_floor.
    applied, _applied_attribute_path = set_detector_operating_point(
        predictor.model, score_thresh=0.01, detections_per_img=density_cap)

    def _records(sub):
        ds = build_dataset("detection", images_dir=args.images_dir,
                           labels_dir=args.labels_dir, stems=sub,
                           subject=args.subject, attribute=args.attribute)
        loader = DataLoader(ds, batch_size=4, collate_fn=task_collate("detection"))
        return records_over_loader(predictor.model, loader, predictor.device, "detection")

    bundle = resolve_operating_point(
        args.trait, dataset_hash=dh, calibration_records=_records(cal_stems),
        holdout_records=_records(hold_stems), tile_size=tile_size,
        # tile_size read above is the checkpoint's persisted training geometry when present: say
        # so, or resolve_operating_point now (correctly) stamps any unclaimed value "default"
        # rather than assuming truthiness means derived.
        tile_size_source=("derived" if tile_size is not None else "default"),
        # This script's own pass (_records, above) is always untiled: a plain DataLoader over the
        # whole image, never predict_tiled/predict_batch(tile=...), so tiled=False is stated
        # explicitly rather than left to resolve_operating_point's tiled=True default.
        # Omitting this would have resolve_tile_size_param wrongly gate (or falsely validate) a
        # tile_size dimension that was never actually operative for this untiled pass.
        tiled=False,
        experiment_id=args.experiment_id, staged_conf_floor=applied.get("score_thresh"),
        split_manifest_dir=args.split_manifest_dir, calibration_date=manifest_date_key(cal_date),
        calibration_labels_dir=args.labels_dir, split_manifest_sha256=split_manifest_sha256,
    )
    attach_split_policy_provenance(bundle, locked)

    provenance = bundle.to_provenance()
    provenance["sweep"] = bundle.get("conf").sweep

    conf = bundle.get("conf")
    print(f"trait={args.trait} dataset_hash={dh}")
    print(f"conf={conf._raw:.4f} validated_against={conf.validated_against} shippable={bundle.is_shippable}")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
