"""Calibrate + held-out validate a detection operating point over a labeled split.

The confidence operating point is the phenotype for a count trait, so it must be derived per
dataset and validated against held-out ground truth, never pinned. This script runs one
low-threshold model pass over a disjoint calibration/holdout split of a labeled dir, resolves the
count-unbiased operating point, checks its held-out count bias, and persists the full provenance +
sweep to ``.tcip/experiments/<id>/operating_point.json`` for inspection and lineage.

It is a script (not a new MCP tool) per CLAUDE.md: the audited count is produced at ``run_inference``
(which now accepts ``trait`` + ``calibration_labels_dir`` to resolve the same operating point inline);
this is the offline producer for inspecting the sweep and recording the bundle.

Usage:
    python scripts/calibrate_operating_point.py \
        --checkpoint <ckpt.pt> --trait <trait_name> \
        --labels-dir <labeled_dir> --images-dir <images_dir> \
        [--experiment-id <id>] [--val-ratio 0.5] [--device cpu]
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
    parser.add_argument("--experiment-id", default=None,
                        help="Experiment id to persist under (.tcip/experiments/<id>/). "
                             "Defaults to a hash-tagged id.")
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
    parser.add_argument("--group-by", default="tile_prefix",
                        help="Grouping policy for the cal/holdout split: 'tile_prefix' (default, "
                             "strips a trailing _<x>_<y> tile offset) or 'stem' (one group per "
                             "image). Ignored if --group-key-map is given.")
    parser.add_argument("--group-key-map", default=None,
                        help="Path to a JSON file mapping stem -> group key, overriding --group-by.")
    args = parser.parse_args(argv)

    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.splits import count_label_lines, resolve_locked_cal_holdout_split
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, derive_max_dets_from_counts, records_over_loader,
        resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS, dataset_hash
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from tcip_mcp.project_paths import project_root

    # Match the MCP path's own initial predictor construction exactly (DEFAULT_MAX_DETS) rather
    # than leaving the framework default (torchvision 100/300) in place: this value is superseded
    # below, once this split's density is known, by the set_detector_operating_point call that
    # actually governs the collection pass ("detector-cap censoring").
    predictor = build_predictor(checkpoint_path=args.checkpoint, device=args.device,
                                max_dets=DEFAULT_MAX_DETS)
    tile_size = getattr(predictor, "train_tile_size", None)

    probe = build_dataset("detection", images_dir=args.images_dir, labels_dir=args.labels_dir)
    stems = sorted(getattr(probe, "stems", []))
    if len(stems) < 2:
        print(f"Need >=2 labeled stems to split cal/holdout; found {len(stems)}.", file=sys.stderr)
        return 2

    group_key_map = None
    if args.group_key_map:
        with open(args.group_key_map, encoding="utf-8") as f:
            group_key_map = json.load(f)

    dh = dataset_hash(args.labels_dir)
    annotation_counts = {s: count_label_lines(args.labels_dir, s) for s in stems}
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
        stems, identity_hash=dh, annotation_counts=annotation_counts,
        group_by=args.group_by, group_key_map=group_key_map, holdout_ratio=args.val_ratio,
        seed=args.seed,
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
    applied = set_detector_operating_point(predictor.model, score_thresh=0.01,
                                           detections_per_img=density_cap)

    def _records(sub):
        ds = build_dataset("detection", images_dir=args.images_dir,
                           labels_dir=args.labels_dir, stems=sub)
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
    )
    attach_split_policy_provenance(bundle, locked)

    exp_id = args.experiment_id or f"opcal_{args.trait}_{dh}"
    out_dir = project_root() / ".tcip" / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = bundle.to_provenance()
    provenance["sweep"] = bundle.get("conf").sweep
    out_path = out_dir / "operating_point.json"
    out_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    conf = bundle.get("conf")
    print(f"trait={args.trait} dataset_hash={dh}")
    print(f"conf={conf._raw:.4f} validated_against={conf.validated_against} shippable={bundle.is_shippable}")
    print(f"persisted -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
