"""Calibrate + held-out validate a detection operating point over a labeled split (CV0).

The confidence operating point IS the phenotype for a count trait, so it must be DERIVED per
dataset and validated against held-out ground truth — never pinned. This script runs one
low-threshold model pass over a disjoint calibration/holdout split of a labeled dir, resolves the
count-unbiased operating point, checks its held-out count bias, and persists the full provenance +
sweep to ``.tcip/experiments/<id>/operating_point.json`` for inspection and lineage.

It is a script (not a new MCP tool) per CLAUDE.md: the audited count is produced at ``run_inference``
(which now accepts ``trait`` + ``calibration_labels_dir`` to resolve the same operating point inline);
this is the offline producer for inspecting the sweep and recording the bundle.

Usage:
    python scripts/calibrate_operating_point.py \
        --checkpoint <ckpt.pt> --trait catkin \
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
                        help="Holdout fraction of the labeled split (disjoint by stem).")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted).")
    args = parser.parse_args(argv)

    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import (
        records_over_loader, resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from tcip_mcp.project_paths import project_root

    predictor = build_predictor(checkpoint_path=args.checkpoint, device=args.device)
    tile_size = getattr(predictor, "train_tile_size", None)

    probe = build_dataset("detection", images_dir=args.images_dir, labels_dir=args.labels_dir)
    stems = sorted(getattr(probe, "stems", []))
    if len(stems) < 2:
        print(f"Need >=2 labeled stems to split cal/holdout; found {len(stems)}.", file=sys.stderr)
        return 2
    n_hold = max(1, int(round(len(stems) * args.val_ratio)))
    hold_stems, cal_stems = stems[:n_hold], stems[n_hold:]

    set_detector_operating_point(predictor.model, score_thresh=0.01)

    def _records(sub):
        ds = build_dataset("detection", images_dir=args.images_dir,
                           labels_dir=args.labels_dir, stems=sub)
        loader = DataLoader(ds, batch_size=4, collate_fn=task_collate("detection"))
        return records_over_loader(predictor.model, loader, predictor.device, "detection")

    dh = dataset_hash(args.labels_dir)
    bundle = resolve_operating_point(
        args.trait, dataset_hash=dh, calibration_records=_records(cal_stems),
        holdout_records=_records(hold_stems), tile_size=tile_size,
    )

    exp_id = args.experiment_id or f"opcal_{args.trait}_{dh}"
    out_dir = project_root() / ".tcip" / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = bundle.to_provenance()
    provenance["sweep"] = bundle.get("conf").sweep
    out_path = out_dir / "operating_point.json"
    out_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    conf = bundle.get("conf")
    print(f"trait={args.trait} dataset_hash={dh}")
    print(f"conf={conf._raw:.4f} validated_vs_gt={conf.validated_vs_gt} shippable={bundle.is_shippable}")
    print(f"persisted -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
