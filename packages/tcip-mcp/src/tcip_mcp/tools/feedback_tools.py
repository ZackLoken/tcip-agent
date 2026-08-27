"""Review -> retrain feedback MCP tools.

``materialize_review_dataset`` turns human review verdicts into a curated detection training set
(with experiment lineage); ``prioritize_review_queue`` ranks un-reviewed images by
active-learning informativeness for the next review batch.

Both are scoped by the dataset root the review was recorded against, which is what derives the
verdict store they read. A review whose verdicts live outside that dataset states its store
instead, and the store it read is reported rather than folded into the derived one.
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
    """True if the verdict store rooted at ``review_state_dir`` holds any review shard.

    Enumerated through the store, the way the immutability guard counts verdicts, rather than by
    globbing the shard directory: shards are placed per prediction bucket, so where a shard sits
    is the store's own layout question and a reader that answers it a second time undercounts.
    """
    import tcip_store

    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    return bool(tcip_store.keys(REVIEW_VERDICTS_STORE, str(review_state_dir)))


def _verdict_store_of(dataset_root: str, review_state_dir: str) -> Path:
    """The verdict store to read: the one the caller stated, else the dataset's own.

    A stated ``review_state_dir`` is used verbatim; with none stated the store is derived from the
    dataset root through :func:`~tcip_mcp.prediction_buckets.review_state_dir_of`, the platform's
    one derivation of where verdicts live. The two are never merged and neither backs the other:
    a stated store holding nothing is a stated store holding nothing.
    """
    from tcip_mcp.prediction_buckets import review_state_dir_of

    if review_state_dir:
        return Path(review_state_dir)
    return review_state_dir_of(dataset_root)


def _load_or_refuse(checkpoint_path: str, project_path: str):
    """The verified checkpoint a review-queue strategy builds its predictor from, or the door's
    own refusal dict when the registry names no entry for it. Returns ``(checkpoint, refusal)``."""
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        return load_registered_checkpoint(checkpoint_path, project_path=project_path or None), None
    except UnregisteredCheckpoint as exc:
        return None, {"error": str(exc)}


def _resolve_review_bucket(engine, bucket: str | None) -> tuple[str | None, str | None]:
    """The prediction bucket to read verdicts from, and the refusal when that is not one answer.

    Verdicts are keyed by the bucket they were recorded against, so a store holding several is
    several reviews and not one; the sole bucket answers when there is exactly one, and several
    are named for the caller to choose among rather than merged into a reference nobody reviewed.
    """
    if bucket is not None:
        return bucket, None
    buckets = engine.reviewed_buckets()
    if len(buckets) == 1:
        return buckets[0], None
    return None, (
        f"review state holds verdicts for {len(buckets)} prediction buckets "
        f"({', '.join(repr(b) for b in buckets)}); pass bucket to name which one to read"
    )


@mcp.tool()
@audited(scope_arg="output_dir", scope_via=resolve_output_path)
def materialize_review_dataset(
    dataset_root: str,
    source_images_dir: str,
    output_dir: str,
    experiment_id: str = "",
    include_hard_negatives: bool = True,
    only_completed: bool = False,
    copy_files: bool = True,
    subject: str | None = None,
    bucket: str | None = None,
    review_state_dir: str = "",
) -> dict:
    """Build a curated detection dataset from human review verdicts.

    Accepted/edited GT boxes become positive name-based labels; rejected-only images become
    empty-label hard negatives (keyed under ``subject``, derived from the verdicts when omitted).
    When ``experiment_id`` is given, records the review session as experiment lineage. Output
    (``images/`` + ``annotations/``) chains straight into ``make_splits`` / ``launch_training``.

    Args:
        dataset_root: Root of the dataset the review was recorded against. It scopes the verdict
            store read when ``review_state_dir`` is not stated (``<dataset_root>/.tcip/state``),
            and it is what the experiment lineage records as the reviewed dataset.
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source). A relative
            path resolves against the project root, never the server process's cwd.
        experiment_id: Optional experiment to record the review-session lineage on.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
        subject: The object the review was about; confirmed negatives are keyed under it. When
            omitted it is derived from every subject the verdicts name, rejections included, and
            only when they name exactly one. A rejected image whose own rejections answer for
            another subject, or for none, is materialized as an unconfirmed empty and reported in
            ``unconfirmed_negatives`` with why, rather than keyed under a subject no verdict on
            that image mentions.
        bucket: Which prediction bucket's verdicts to curate, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several: two buckets are two reviews, and
            merging them would curate one date's verdicts against another's images.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim and the
            response names it as where the shards came from. The two are never merged and neither
            stands in for the other: a stated store holding no shards is refused, never answered
            from the dataset's own store.
    """
    output_dir = str(resolve_output_path(output_dir))
    if not dataset_root:
        return {"error": "dataset_root is required: it names the dataset whose review this curates"}
    store_dir = _verdict_store_of(dataset_root, review_state_dir)
    if not _review_state_exists(str(store_dir)):
        return {"error": f"no review state (review/ shards) in {store_dir}"}
    if not Path(source_images_dir).is_dir():
        return {"error": f"Source images dir not found: {source_images_dir}"}

    if experiment_id:
        # Checked before the curated directory is written: a blob write cannot join the record's
        # own transaction, so this is the one chance to refuse before anything lands on disk.
        from tcip_mcp.experiments import pointer_frozen

        frozen = pointer_frozen(experiment_id, "artifacts", "curated_dataset", output_dir)
        if frozen is not None:
            return {"error": frozen}
        # review_session's own value (the counts below) is unknown this early, so no value here
        # can equal a repeat's; a presence check for a record already populated on this field alone.
        frozen = pointer_frozen(experiment_id, "lineage", "review_session", None)
        if frozen is not None:
            return {"error": frozen}

    from tcip_annotation.review_engine import ReviewEngine
    engine = ReviewEngine(str(store_dir))
    resolved_bucket, refusal = _resolve_review_bucket(engine, bucket)
    if refusal is not None:
        return {"error": refusal}
    review_state = {"image": engine.image_states(resolved_bucket)}
    state_path = engine.shard_dir

    result = materialize_dataset(
        review_state, source_images_dir, output_dir, subject=subject,
        review_state_path=str(state_path), include_hard_negatives=include_hard_negatives,
        copy_files=copy_files, only_completed=only_completed,
    )
    result["review_state"] = str(state_path)
    result["dataset_root"] = dataset_root
    result["review_state_stated"] = bool(review_state_dir)
    result["review_state_origin"] = (
        f"verdict shards read from the stated store {store_dir}, not from this dataset's own "
        f"store at {_verdict_store_of(dataset_root, '')}"
        if review_state_dir
        else f"verdict shards read from this dataset's own store at {store_dir}"
    )

    if experiment_id:
        from tcip_mcp.experiments import (
            create_experiment, get_experiment, update_lineage, record_artifact,
        )
        if "error" in get_experiment(experiment_id):
            create_experiment(experiment_id, {"source": "review_feedback"}, data_source=dataset_root)
        # Set data_source in both branches so lineage names the reviewed dataset even when the
        # experiment pre-existed.
        update_lineage(experiment_id, data_source=dataset_root, review_session={
            "dataset_root": dataset_root,
            "review_state_dir": str(store_dir),
            "review_shards": str(state_path),
            "n_positive": result["positive"],
            "n_hard_negative": result["hard_negative"],
            # Of the rejected-only images, the ones no verdict attributed, so they train as neither.
            "n_unconfirmed_negative": result["unconfirmed_negative"],
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
    dataset_root: str = "",
    strategy: str = "informativeness",
    method: str = "combined",
    task: str = "detection",
    budget: int = 50,
    skip_reviewed: bool = True,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float | None = None,
    bucket: str | None = None,
    review_state_dir: str = "",
    project_path: str = "",
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
        dataset_root: Root of the dataset whose review is in progress. It scopes the verdict store
            (``<dataset_root>/.tcip/state``) that ``skip_reviewed`` reads. With neither this nor
            ``review_state_dir`` stated, no store is read and every candidate image is ranked.
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
        bucket: Which prediction bucket's completed reviews ``skip_reviewed`` skips, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim. The two
            are never merged and neither stands in for the other.
        project_path: Project root the checkpoint's registry entry is looked up under. Empty
            (default) resolves to the process's own root.
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
    if (dataset_root or review_state_dir) and skip_reviewed:
        store_dir = _verdict_store_of(dataset_root, review_state_dir)
        if _review_state_exists(str(store_dir)):
            from tcip_annotation.review_engine import ReviewEngine
            engine = ReviewEngine(str(store_dir))
            resolved_bucket, refusal = _resolve_review_bucket(engine, bucket)
            if refusal is not None:
                return {"error": refusal}
            reviewed = reviewed_image_names({"image": engine.image_states(resolved_bucket)})
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
        checkpoint, refusal = _load_or_refuse(checkpoint_path, project_path)
        if refusal is not None:
            return refusal
        predictor = build_predictor(checkpoint)
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

    checkpoint, refusal = _load_or_refuse(checkpoint_path, project_path)
    if refusal is not None:
        return refusal
    predictor = build_predictor(checkpoint)
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
