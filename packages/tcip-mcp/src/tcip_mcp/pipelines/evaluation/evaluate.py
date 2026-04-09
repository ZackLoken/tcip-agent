"""Standalone evaluation tool — evaluate a trained model on a dataset.

Can be used as:
  - MCP tool: evaluate_model(checkpoint_path, images_dir, labels_dir, ...)
  - CLI: python -m tcip_mcp.pipelines.evaluation.evaluate --checkpoint ... --images ... --labels ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from tcip_mcp.pipelines.data.dataset import DetectionDataset
from tcip_mcp.pipelines.data.augmentation import build_val_transforms
from tcip_mcp.pipelines.evaluation.metrics import compute_map
from tcip_mcp.pipelines.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def evaluate_model(
    checkpoint_path: str,
    images_dir: str,
    labels_dir: str,
    iou_thresholds: list[float] | None = None,
    score_threshold: float = 0.5,
    output_path: str | None = None,
    device: str | None = None,
) -> dict:
    """Run full evaluation of a checkpoint on a labeled dataset.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        images_dir: Directory containing test images.
        labels_dir: Directory containing YOLO-format label files.
        iou_thresholds: IoU thresholds for mAP (default: [0.5, 0.75]).
        score_threshold: Minimum prediction confidence.
        output_path: Optional path to save results JSON.
        device: 'cuda' or 'cpu' (auto-detect if None).

    Returns:
        Dict with mAP, per-class AP, counts, and per-image results.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    predictor = Predictor(checkpoint_path, device=device, score_threshold=score_threshold)
    config = predictor.config

    # Build dataset
    val_transforms = build_val_transforms(config.get("augmentation", {}))
    dataset = DetectionDataset(images_dir, labels_dir, transforms=val_transforms)

    logger.info("Evaluating on %d images, device=%s", len(dataset), device)

    all_predictions = []
    all_targets = []
    per_image_results = []

    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        image_tensor = image_tensor.to(device)

        # Run inference
        with torch.no_grad():
            outputs = predictor.model([image_tensor])[0]

        # Filter by score
        keep = outputs["scores"] >= score_threshold
        pred = {
            "boxes": outputs["boxes"][keep].cpu(),
            "scores": outputs["scores"][keep].cpu(),
            "labels": outputs["labels"][keep].cpu(),
        }
        gt = {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }

        all_predictions.append(pred)
        all_targets.append(gt)

        per_image_results.append({
            "stem": dataset.stems[idx],
            "predictions": int(keep.sum()),
            "ground_truth": len(target["labels"]),
        })

    # Compute metrics
    map_results = compute_map(all_predictions, all_targets, iou_thresholds=iou_thresholds)

    # Aggregate per-image stats
    total_preds = sum(r["predictions"] for r in per_image_results)
    total_gt = sum(r["ground_truth"] for r in per_image_results)

    results = {
        "checkpoint": checkpoint_path,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "num_images": len(dataset),
        "total_predictions": total_preds,
        "total_ground_truth": total_gt,
        "score_threshold": score_threshold,
        "iou_thresholds": iou_thresholds,
        **map_results,
        "per_image": per_image_results,
    }

    # Add class name mapping if available
    if predictor.class_map is not None:
        results["class_names"] = predictor.class_map.names

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to %s", output_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained detection model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to YOLO label directory")
    parser.add_argument("--iou", nargs="+", type=float, default=[0.5, 0.75], help="IoU thresholds")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--device", help="Device (cuda/cpu)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        images_dir=args.images,
        labels_dir=args.labels,
        iou_thresholds=args.iou,
        score_threshold=args.score_threshold,
        output_path=args.output,
        device=args.device,
    )

    print(f"\nmAP@0.5: {results['mAP']:.4f}")
    print(f"Images: {results['num_images']}")
    print(f"Predictions: {results['total_predictions']}")
    print(f"Ground truth: {results['total_ground_truth']}")

    for thresh, data in results.get("per_threshold", {}).items():
        print(f"\nmAP@{thresh}: {data['mAP']:.4f}")
        for cls_id, ap in data.get("AP_per_class", {}).items():
            print(f"  Class {cls_id}: AP={ap:.4f}")
