"""Review -> retrain feedback MCP tools.

``materialize_review_dataset`` turns human review verdicts into a curated detection training set
(with experiment lineage); ``prioritize_review_queue`` ranks un-reviewed images by
active-learning informativeness for the next review batch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.feedback.materialize import (
    materialize_dataset, reviewed_image_names, select_unreviewed,
)
from tcip_mcp.project_paths import resolve_output_path


def _review_state_exists(review_state_dir: str) -> bool:
    """True if the ``review/`` shard dir holds any review state."""
    from tcip_annotation.review_engine import REVIEW_SHARD_DIRNAME

    shard_dir = Path(review_state_dir) / REVIEW_SHARD_DIRNAME
    return shard_dir.is_dir() and any(shard_dir.glob("*.json"))


@mcp.tool()
@audited(scope_arg="output_dir", scope_via=resolve_output_path)
def materialize_review_dataset(
    review_state_dir: str,
    source_images_dir: str,
    output_dir: str,
    experiment_id: str = "",
    include_hard_negatives: bool = True,
    only_completed: bool = False,
    copy_files: bool = True,
    subject: str | None = None,
) -> dict:
    """Build a curated detection dataset from human review verdicts.

    Accepted/edited GT boxes become positive name-based labels; rejected-only images become
    empty-label hard negatives (keyed under ``subject``, derived from the verdicts when omitted).
    When ``experiment_id`` is given, records the review session as experiment lineage. Output
    (``images/`` + ``annotations/``) chains straight into ``make_splits`` / ``launch_training``.

    Args:
        review_state_dir: Directory holding the review state (``review/`` shards, or a
            ``review/`` shards).
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source). A relative
            path resolves against the project root, never the server process's cwd.
        experiment_id: Optional experiment to record the review-session lineage on.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
        subject: The object the review was about; confirmed negatives are keyed under it. When
            omitted it is derived from the verdicts' own class names, but only when the verdicts
            name exactly one subject; a review touching more than one subject with no explicit
            ``subject`` can't attribute its negatives, and they are silently dropped rather than
            carried into the curated set.
    """
    output_dir = str(resolve_output_path(output_dir))
    if not _review_state_exists(review_state_dir):
        return {"error": f"no review state (review/ shards) in {review_state_dir}"}
    if not Path(source_images_dir).is_dir():
        return {"error": f"Source images dir not found: {source_images_dir}"}

    from tcip_annotation.review_engine import ReviewEngine
    engine = ReviewEngine(review_state_dir)
    review_state = engine.raw_state
    state_path = engine.shard_dir

    result = materialize_dataset(
        review_state, source_images_dir, output_dir, subject=subject,
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
            "review_shards": str(state_path),
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
    auto_threshold: float | None = None,
) -> dict:
    """Order un-reviewed images for the next review batch.

    Two strategies:
    - ``informativeness`` (default): rank by active-learning score (``method``), the most
      uncertain/diverse frames first.
    - ``confidence_triage``: partition by prediction confidence into auto-accept
      (>= ``auto_threshold``) vs needs-review (in ``[low, high]``) queues.

    Args:
        checkpoint_path: Trained model checkpoint (drives scoring / predictions).
        images_dir: Directory of candidate images.
        review_state_dir: Optional review-state dir; with ``skip_reviewed`` excludes
            already-completed images.
        strategy: ``informativeness`` | ``confidence_triage``.
        method: Informativeness scorer. ``uncertainty`` | ``diversity`` | ``combined`` are the
            built-in reference implementations, not the allowed set: register your own with
            ``register_scorer``, or pass a dotted ``module:factory`` you wrote. An
            unresolvable name is refused rather than silently scored as ``combined``.
        task: Task type for the uncertainty scorer.
        budget: Number of images to return (``informativeness`` only).
        skip_reviewed: Exclude already-completed images from the queue.
        low: Lower confidence bound for the review band (``confidence_triage`` only).
        high: Upper confidence bound for the review band (``confidence_triage`` only).
        auto_threshold: Confidence at/above which a prediction is auto-accepted as ground truth
            (``confidence_triage`` only). ``None`` (default) refuses to auto-accept: turning
            predictions into GT at a pinned 0.8 fabricates labels the model was never confirmed to
            get right. Derive this threshold from the model's validated confidence
            distribution and confirm with a breeder spot-check that high-conf actually equals truth,
            then pass it explicitly: the result is stamped as requiring that confirmation.
    """
    if not Path(checkpoint_path).is_file():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    if strategy not in ("informativeness", "confidence_triage"):
        return {"error": f"Unknown strategy {strategy!r}; use 'informativeness' or 'confidence_triage'"}
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return {"error": f"Images dir not found: {images_dir}"}
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    logical = list_logical_images(images_path)
    if not logical:
        return {"error": "No images found in images_dir"}
    # Real sources (a plain path or a BandGroupRef), one per logical image: a band-grouped
    # capture's sibling band files are folded into its one entry instead of each enumerating
    # as its own (spurious) candidate.
    sources = [logical[stem] for stem in sorted(logical)]

    reviewed_skipped = 0
    if review_state_dir and skip_reviewed:
        if _review_state_exists(review_state_dir):
            from tcip_annotation.review_engine import ReviewEngine
            reviewed = reviewed_image_names(ReviewEngine(review_state_dir).raw_state)
            before = len(sources)
            # select_unreviewed compares basenames against review-state img_name, which for a
            # band-grouped capture is the manifest's own filename (the identity every review-state
            # reader/writer in this platform uses), not any one sibling band file's name.
            display = [str(s.manifest_path) if isinstance(s, BandGroupRef) else str(s) for s in sources]
            kept = set(select_unreviewed(display, reviewed))
            sources = [s for s, d in zip(sources, display) if d in kept]
            reviewed_skipped = before - len(sources)

    try:
        from tcip_mcp.pipelines.inference.predictor import build_predictor
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}

    if strategy == "confidence_triage":
        from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue, unscoreable

        if not sources:
            return {"strategy": strategy, "total_images": 0, "reviewed_skipped": reviewed_skipped,
                    "auto_accepted": 0, "needs_review": 0, "review_images": [],
                    "unscoreable_images": [], "auto_accepted_images": []}
        predictor = build_predictor(checkpoint_path)
        predictions = predictor.predict_batch(sources)
        needs_review = review_queue(predictions, low=low, high=high)
        # A prediction with no confidence-bearing signal at all (e.g. a regression head's point
        # estimate) can't be partitioned by auto_accept/review_queue on confidence; route it into
        # review explicitly rather than let it silently vanish from every output, and tag it
        # distinctly from a genuinely medium-confidence review item so a caller can tell why it's
        # here.
        unscoreable_preds = unscoreable(predictions)
        all_review = needs_review + unscoreable_preds
        # Auto-accept turns predictions into GT. Refuse to do so at a pinned threshold: the
        # threshold must be derived from the model's validated conf distribution and breeder
        # spot-checked. With no explicit (confirmed) threshold, accept nothing and say why.
        if auto_threshold is None:
            return {
                "strategy": strategy,
                "total_images": len(predictions),
                "reviewed_skipped": reviewed_skipped,
                "auto_accepted": 0,
                "auto_accept_refused": (
                    "auto_threshold=None: auto-accepting predictions as GT requires a threshold "
                    "derived from the model's validated confidence distribution and confirmed by a "
                    "breeder spot-check; pass auto_threshold explicitly once confirmed."),
                "needs_review": len(all_review),
                "review_images": [r.get("image", "") for r in all_review],
                "unscoreable_images": [p.get("image", "") for p in unscoreable_preds],
                "auto_accepted_images": [],
            }
        accepted = auto_accept(predictions, threshold=auto_threshold)
        return {
            "strategy": strategy,
            "total_images": len(predictions),
            "reviewed_skipped": reviewed_skipped,
            "auto_accepted": len(accepted),
            "auto_accept_requires_breeder_confirmation": (
                "auto-accepted labels are GT only if this threshold was breeder-confirmed on a "
                "high-conf sample"),
            "needs_review": len(all_review),
            "review_images": [r.get("image", "") for r in all_review],
            "unscoreable_images": [p.get("image", "") for p in unscoreable_preds],
            "auto_accepted_images": [a.get("image", "") for a in accepted],
        }

    if not sources:
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
    try:
        scorer = build_scorer(method, task)
    except ValueError as e:  # unknown scorer: refuse rather than silently reordering the queue
        return {"error": str(e)}

    scored = scorer.score(sources, predictor.model, predictor.device)[:budget]
    return {
        "strategy": strategy,
        "method": method,
        "task": task,
        "total_candidates": len(sources),
        "reviewed_skipped": reviewed_skipped,
        "selected_count": len(scored),
        "queue": [
            {"image": str(p.manifest_path) if isinstance(p, BandGroupRef) else str(p),
             "score": round(float(s), 6)}
            for p, s in scored
        ],
    }
