"""Review -> retrain feedback MCP tools (W5).

``materialize_review_dataset`` turns human review verdicts into a curated YOLO
detection training set (with experiment lineage); ``prioritize_review_queue`` ranks
un-reviewed images by active-learning informativeness for the next review batch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.feedback.materialize import (
    materialize_dataset, reviewed_image_names, select_unreviewed,
)

REVIEW_STATE_FILENAME = "review_stats.json"  # legacy single-file layout; ReviewEngine migrates it
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _review_state_exists(review_state_dir: str) -> bool:
    """True if there's a legacy ``review_stats.json`` or a non-empty ``review/`` shard dir."""
    d = Path(review_state_dir)
    if (d / REVIEW_STATE_FILENAME).is_file():
        return True
    from tcip_annotation.review_engine import REVIEW_SHARD_DIRNAME
    shard_dir = d / REVIEW_SHARD_DIRNAME
    return shard_dir.is_dir() and any(shard_dir.glob("*.json"))


@mcp.tool()
@audited
def materialize_review_dataset(
    review_state_dir: str,
    source_images_dir: str,
    output_dir: str,
    experiment_id: str = "",
    include_hard_negatives: bool = True,
    only_completed: bool = False,
    copy_files: bool = True,
) -> dict:
    """Build a curated YOLO detection dataset from human review verdicts.

    Accepted/edited GT boxes become positive labels; rejected-only images become
    empty-label hard negatives. When ``experiment_id`` is given, records the review
    session as experiment lineage. Output (``images/`` + ``labels/detect/``) chains
    straight into ``make_splits`` / ``launch_training``.

    Args:
        review_state_dir: Directory holding the review state (``review/`` shards, or a
            legacy ``review_stats.json``).
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source).
        experiment_id: Optional experiment to record the review-session lineage on.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
    """
    if not _review_state_exists(review_state_dir):
        return {"error": f"no review state (review/ shards or legacy review_stats.json) in {review_state_dir}"}
    if not Path(source_images_dir).is_dir():
        return {"error": f"Source images dir not found: {source_images_dir}"}

    from tcip_annotation.review_engine import ReviewEngine
    engine = ReviewEngine(review_state_dir)  # migrates a legacy review_stats.json, if any
    review_state = engine.raw_state
    state_path = engine.shard_dir

    result = materialize_dataset(
        review_state, source_images_dir, output_dir,
        review_state_path=str(state_path), include_hard_negatives=include_hard_negatives,
        copy_files=copy_files, only_completed=only_completed,
    )
    result["review_state"] = str(state_path)

    if experiment_id:
        from tcip_mcp.experiments import (
            create_experiment, get_experiment, update_lineage, record_artifact,
        )
        if "error" in get_experiment(experiment_id):
            create_experiment(experiment_id, {"source": "review_feedback"}, data_source=review_state_dir)
        # Set data_source in both branches so lineage points at the review session
        # even when the experiment pre-existed.
        update_lineage(experiment_id, data_source=review_state_dir, review_session={
            "review_state_dir": review_state_dir,
            "review_stats": str(state_path),
            "n_positive": result["positive"],
            "n_hard_negative": result["hard_negative"],
            "n_boxes": result["total_boxes"],
            "manifest": result["manifest"],
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        })
        record_artifact(experiment_id, "curated_dataset", result["output_dir"])
        result["experiment_id"] = experiment_id

    return result


@mcp.tool()
@audited
def prioritize_review_queue(
    checkpoint_path: str,
    images_dir: str,
    review_state_dir: str = "",
    strategy: str = "informativeness",
    method: str = "combined",
    task: str = "detection",
    budget: int = 50,
    skip_reviewed: bool = True,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float = 0.8,
) -> dict:
    """Order un-reviewed images for the next review batch.

    Two strategies:
    - ``informativeness`` (default): rank by active-learning score (``method``) — the most
      uncertain/diverse frames first.
    - ``confidence_triage``: partition by prediction confidence into auto-accept
      (>= ``auto_threshold``) vs needs-review (in ``[low, high]``) queues.

    Args:
        checkpoint_path: Trained model checkpoint (drives scoring / predictions).
        images_dir: Directory of candidate images.
        review_state_dir: Optional review-state dir; with ``skip_reviewed`` excludes
            already-completed images.
        strategy: ``informativeness`` | ``confidence_triage``.
        method: Informativeness scorer — ``uncertainty`` | ``diversity`` | ``combined``.
        task: Task type for the uncertainty scorer.
        budget: Number of images to return (``informativeness`` only).
        skip_reviewed: Exclude already-completed images from the queue.
        low: Lower confidence bound for the review band (``confidence_triage`` only).
        high: Upper confidence bound for the review band (``confidence_triage`` only).
        auto_threshold: Confidence at/above which a label auto-accepts (``confidence_triage`` only).
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    if strategy not in ("informativeness", "confidence_triage"):
        return {"error": f"Unknown strategy {strategy!r}; use 'informativeness' or 'confidence_triage'"}
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return {"error": f"Images dir not found: {images_dir}"}
    paths = sorted(str(f) for f in images_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        return {"error": "No images found in images_dir"}

    reviewed_skipped = 0
    if review_state_dir and skip_reviewed:
        if _review_state_exists(review_state_dir):
            from tcip_annotation.review_engine import ReviewEngine
            reviewed = reviewed_image_names(ReviewEngine(review_state_dir).raw_state)
            before = len(paths)
            paths = select_unreviewed(paths, reviewed)
            reviewed_skipped = before - len(paths)

    try:
        from tcip_mcp.pipelines.inference.predictor import build_predictor
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}

    if strategy == "confidence_triage":
        from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue

        if not paths:
            return {"strategy": strategy, "total_images": 0, "reviewed_skipped": reviewed_skipped,
                    "auto_accepted": 0, "needs_review": 0, "review_images": [],
                    "auto_accepted_images": []}
        predictor = build_predictor(checkpoint_path)
        predictions = predictor.predict_batch(paths)
        accepted = auto_accept(predictions, threshold=auto_threshold)
        needs_review = review_queue(predictions, low=low, high=high)
        return {
            "strategy": strategy,
            "total_images": len(predictions),
            "reviewed_skipped": reviewed_skipped,
            "auto_accepted": len(accepted),
            "needs_review": len(needs_review),
            "review_images": [r.get("image", "") for r in needs_review],
            "auto_accepted_images": [a.get("image", "") for a in accepted],
        }

    if not paths:
        return {"strategy": strategy, "method": method, "task": task, "total_candidates": 0,
                "reviewed_skipped": reviewed_skipped, "selected_count": 0, "queue": []}

    try:
        from tcip_mcp.pipelines.active_learning.helpers import build_scorer, require_composed_detector
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}

    predictor = build_predictor(checkpoint_path)
    guard = require_composed_detector(predictor, purpose="review-queue scoring")
    if guard:
        return {"error": guard}
    scorer = build_scorer(method, task)

    scored = scorer.score(paths, predictor.model, predictor.device)[:budget]
    return {
        "strategy": strategy,
        "method": method,
        "task": task,
        "total_candidates": len(paths),
        "reviewed_skipped": reviewed_skipped,
        "selected_count": len(scored),
        "queue": [{"image": p, "score": round(float(s), 6)} for p, s in scored],
    }
