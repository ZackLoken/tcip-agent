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

REVIEW_STATE_FILENAME = "review_stats.json"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


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
    straight into ``split_dataset`` / ``launch_training``.

    Args:
        review_state_dir: Directory containing ``review_stats.json``.
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source).
        experiment_id: Optional experiment to record the review-session lineage on.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
    """
    state_path = Path(review_state_dir) / REVIEW_STATE_FILENAME
    if not state_path.is_file():
        return {"error": f"review_stats.json not found in {review_state_dir}"}
    if not Path(source_images_dir).is_dir():
        return {"error": f"Source images dir not found: {source_images_dir}"}

    from tcip_annotation.review_engine import ReviewEngine
    review_state = ReviewEngine(review_state_dir).raw_state

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
    method: str = "combined",
    task: str = "detection",
    budget: int = 50,
    skip_reviewed: bool = True,
) -> dict:
    """Rank un-reviewed images by active-learning informativeness for the next review batch.

    Args:
        checkpoint_path: Trained model checkpoint (drives uncertainty/diversity scoring).
        images_dir: Directory of candidate images.
        review_state_dir: Optional review-state dir; with ``skip_reviewed`` excludes
            already-completed images.
        method: ``uncertainty`` | ``diversity`` | ``combined``.
        task: Task type for the uncertainty scorer.
        budget: Number of images to return.
        skip_reviewed: Exclude already-completed images from the queue.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return {"error": f"Images dir not found: {images_dir}"}
    paths = sorted(str(f) for f in images_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        return {"error": "No images found in images_dir"}

    reviewed_skipped = 0
    if review_state_dir and skip_reviewed:
        state_path = Path(review_state_dir) / REVIEW_STATE_FILENAME
        if state_path.is_file():
            from tcip_annotation.review_engine import ReviewEngine
            reviewed = reviewed_image_names(ReviewEngine(review_state_dir).raw_state)
            before = len(paths)
            paths = select_unreviewed(paths, reviewed)
            reviewed_skipped = before - len(paths)

    if not paths:
        return {"method": method, "task": task, "total_candidates": 0,
                "reviewed_skipped": reviewed_skipped, "selected_count": 0, "queue": []}

    try:
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        from tcip_mcp.pipelines.active_learning.scorer import (
            UncertaintyScorer, DiversityScorer, CombinedScorer,
        )
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}

    predictor = GenericPredictor(checkpoint_path)
    if method == "uncertainty":
        scorer = UncertaintyScorer(task=task)
    elif method == "diversity":
        scorer = DiversityScorer()
    else:
        scorer = CombinedScorer(task=task)

    scored = scorer.score(paths, predictor.model, predictor.device)[:budget]
    return {
        "method": method,
        "task": task,
        "total_candidates": len(paths),
        "reviewed_skipped": reviewed_skipped,
        "selected_count": len(scored),
        "queue": [{"image": p, "score": round(float(s), 6)} for p, s in scored],
    }
