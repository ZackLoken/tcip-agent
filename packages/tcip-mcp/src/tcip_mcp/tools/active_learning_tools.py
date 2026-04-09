"""Active learning MCP tools — score unlabeled images, manage review queue."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.server import mcp


@mcp.tool()
def score_unlabeled(
    checkpoint_path: str,
    unlabeled_dir: str,
    method: str = "combined",
    task: str = "classification",
    budget: int = 50,
) -> dict:
    """Score unlabeled images and select the most informative ones to annotate.

    Uses active learning to maximize labeling efficiency.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        unlabeled_dir: Directory of unlabeled images.
        method: Scoring method ('uncertainty', 'diversity', 'combined').
        task: Task type for uncertainty scoring.
        budget: Number of images to select.
    """
    import torch
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.active_learning.scorer import (
        UncertaintyScorer, DiversityScorer, CombinedScorer,
    )
    from tcip_mcp.pipelines.active_learning.selector import select_batch

    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    predictor = GenericPredictor(checkpoint_path)
    device = predictor.device
    model = predictor.model

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = sorted(str(f) for f in Path(unlabeled_dir).iterdir() if f.suffix.lower() in exts)

    if not paths:
        return {"error": "No images found in unlabeled_dir"}

    if method == "uncertainty":
        scorer = UncertaintyScorer(task=task)
    elif method == "diversity":
        scorer = DiversityScorer()
    else:
        scorer = CombinedScorer(task=task)

    selected = select_batch(scorer, paths, model, device, budget=budget)

    return {
        "method": method,
        "total_unlabeled": len(paths),
        "selected_count": len(selected),
        "selected": selected,
    }


@mcp.tool()
def get_review_queue(
    checkpoint_path: str,
    images_dir: str,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float = 0.8,
) -> dict:
    """Partition predictions into auto-accept, review, and reject queues.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        images_dir: Directory of images to process.
        low: Lower confidence bound for review queue.
        high: Upper confidence bound for review queue.
        auto_threshold: Confidence threshold for auto-accepting labels.
    """
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue

    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    predictor = GenericPredictor(checkpoint_path)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = sorted(str(f) for f in Path(images_dir).iterdir() if f.suffix.lower() in exts)

    predictions = predictor.predict_batch(paths)

    accepted = auto_accept(predictions, threshold=auto_threshold)
    needs_review = review_queue(predictions, low=low, high=high)

    return {
        "total_images": len(predictions),
        "auto_accepted": len(accepted),
        "needs_review": len(needs_review),
        "review_images": [r.get("image", "") for r in needs_review],
        "auto_accepted_images": [a.get("image", "") for a in accepted],
    }
