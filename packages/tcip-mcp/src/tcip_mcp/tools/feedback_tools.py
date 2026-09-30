"""Review -> retrain feedback MCP tools: ``materialize_review_dataset``,
``prioritize_review_queue`` and ``triage_predictions``, each reading the verdict store of the
dataset root the review was recorded against, or the store the caller states instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tcip_mcp.server import tool
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.feedback.materialize import (
    materialize_dataset, reviewed_image_names, select_unreviewed,
)


def _review_state_exists(review_state_dir: str) -> bool:
    """True if the verdict store at ``review_state_dir`` holds any review shard, enumerated through
    the store.
    """
    import tcip_store

    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    return bool(tcip_store.keys(REVIEW_VERDICTS_STORE, str(review_state_dir)))


def _verdict_store_of(dataset_root: str, review_state_dir: str) -> Path:
    """The verdict store to read: the one the caller stated, else the dataset's own.

    A stated ``review_state_dir`` is used verbatim; with none stated the store is derived from the
    dataset root through :func:`~tcip_mcp.project_paths.project_state_dir`. The two are never
    merged and neither backs the other: a stated store holding nothing is a stated store holding
    nothing.
    """
    from tcip_mcp.project_paths import project_state_dir

    if review_state_dir:
        return Path(review_state_dir)
    return project_state_dir(dataset_root)


def _load_or_refuse(checkpoint_path: str, project: Path):
    """The verified checkpoint either review-queue door builds its predictor from, or the door's
    own refusal dict when ``project``'s registry names no entry for it. Returns ``(checkpoint,
    refusal)``."""
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        return load_registered_checkpoint(checkpoint_path, project=project), None
    except UnregisteredCheckpoint as exc:
        return None, {"error": str(exc)}


def _calibration_stems(checkpoint, images_dir: Path, project: Path) -> set[str] | None:
    """The calibration-side member stems the producing run of ``checkpoint`` in ``project`` froze
    in its partition (``experiments.run_resolution``) whose own image sits in ``images_dir``, or
    ``None``
    for a checkpoint no run of this project produced (``checkpoint.experiment_id``) and for a run
    that bound no selection. A calibration sample from another directory the selection also spans
    is not one of this queue's candidates.
    """
    if checkpoint.experiment_id is None:
        return None
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.split_construction import partition_samples
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.image_utils import stem_of

    partition = run_resolution(checkpoint.experiment_id, project=project)["partition"]
    if partition["selection"] is None:
        return None
    return {stem_of(s.source) for s in partition_samples(partition)
            if s.side == "calibration" and same_directory(Path(s.source).parent, images_dir)}


def _calibration_marks(candidates: list, calibration_stems: set[str]) -> list[bool]:
    """``calibration_member`` for each of ``candidates``, in order: True iff its stem is one of the
    bound selection's calibration-side samples under this queue's own images directory.
    """
    from tcip_mcp.pipelines.image_utils import stem_of

    return [stem_of(source) in calibration_stems for source in candidates]


def _resolve_review_bucket(engine, bucket: str | None) -> tuple[str | None, str | None]:
    """The prediction bucket to read verdicts from, and the refusal when that is not one answer:
    the sole bucket when there is exactly one; several are named for the caller to choose among.
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


@tool()
@audited(scope_arg="output_dir")
def materialize_review_dataset(
    project: Path,
    dataset_root: str,
    source_images_dir: str,
    output_dir: str,
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
    Output is the platform's ``images/`` + ``annotations/`` dataset layout, with its
    ``curated_manifest.json`` naming the review session it was built from and its source.

    The reviewed bucket's scope comes from ``resolution.input_scope``: a stamped bucket's own,
    refusing a ``subject`` stated beside it. Under a classified scope every positive is written
    with the object class in ``subject`` and the confirmed value under the scope's attribute, and
    no rejected-only image is confirmed negative, landing in
    ``unconfirmed_negatives`` instead. The source dataset's own registry is then copied over;
    refuses by name when the source names no dataset root, that root has no ``subjects.json``, or
    the output already holds a registry. A bare directory or a detector scope, and a
    ground-truth-only review with no prediction file, read no classified scope.

    Args:
        dataset_root: Root of the dataset the review was recorded against. It scopes the verdict
            store read when ``review_state_dir`` is not stated (``<dataset_root>/.tcip/state``).
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source); a relative
            path is under the project.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
        subject: The object the review of a bucket with no stamp was about; confirmed negatives
            are keyed under it. A stamped bucket records its own and refuses it. When neither
            states one it is derived from every subject the verdicts name, rejections included,
            and only when they name exactly one. A rejected image whose own rejections answer for
            another subject, or for none, is materialized as an unconfirmed empty and reported in
            ``unconfirmed_negatives`` with why.
        bucket: Which prediction bucket's verdicts to curate, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim and the
            response names it. A stated store holding no shards is refused.
    """
    output_dir = str(Path(project, output_dir))
    if not dataset_root:
        return {"error": "dataset_root is required: it names the dataset whose review this curates"}
    store_dir = _verdict_store_of(dataset_root, review_state_dir)
    if not _review_state_exists(str(store_dir)):
        return {"error": f"no review state (review/ shards) in {store_dir}"}
    if not Path(source_images_dir).is_dir():
        return {"error": f"Source images dir not found: {source_images_dir}"}

    from tcip_annotation.review_engine import NO_BUCKET, ReviewEngine
    engine = ReviewEngine(str(store_dir))
    resolved_bucket, refusal = _resolve_review_bucket(engine, bucket)
    if refusal is not None:
        return {"error": refusal}
    assert resolved_bucket is not None  # _resolve_review_bucket pairs a None refusal with a bucket
    review_state = {"image": engine.image_states(resolved_bucket)}
    state_path = engine.shard_dir

    from tcip_mcp.pipelines.resolution import input_scope
    from tcip_store import StoreError

    scope_dir = None
    if resolved_bucket != NO_BUCKET:
        bucket_path = Path(resolved_bucket)
        if bucket_path.is_absolute():
            scope_dir = bucket_path
        elif review_state_dir:
            return {"error": (
                f"{resolved_bucket!r} is a relative bucket key, meaningful only against the "
                f"dataset root its own store recorded it under; review_state_dir names a "
                f"different store ({store_dir}), so state an absolute bucket path instead."
            )}
        else:
            scope_dir = Path(dataset_root) / resolved_bucket
    try:
        scope, _stamped = input_scope(scope_dir, subject, None)
    except (StoreError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        result = materialize_dataset(
            review_state, source_images_dir, output_dir,
            scope=scope,
            include_hard_negatives=include_hard_negatives,
            copy_files=copy_files, only_completed=only_completed,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    result["review_state"] = str(state_path)
    result["dataset_root"] = dataset_root
    result["review_state_stated"] = bool(review_state_dir)
    result["review_state_origin"] = (
        f"verdict shards read from the stated store {store_dir}, not from this dataset's own "
        f"store at {_verdict_store_of(dataset_root, '')}"
        if review_state_dir
        else f"verdict shards read from this dataset's own store at {store_dir}"
    )
    return result


def _prepare_queue_sources(
    checkpoint_path: str,
    images_dir: str,
    dataset_root: str,
    review_state_dir: str,
    skip_reviewed: bool,
    bucket: str | None,
):
    """The checkpoint-file, images-dir and reviewed-skip plumbing both review-queue doors share, in
    order: checkpoint existence, images directory, logical image enumeration, then which of them
    the dataset's own review state already covers.

    Returns ``(sources, reviewed_skipped, build_predictor, error)``; ``error`` is a ready
    ``{"error": ...}`` dict and the other three are ``None``/``0``/``None`` when it is set. A
    torch-less environment is refused at the ``build_predictor`` import.
    """
    if not Path(checkpoint_path).is_file():
        return None, 0, None, {"error": f"Checkpoint not found: {checkpoint_path}"}
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return None, 0, None, {"error": f"Images dir not found: {images_dir}"}
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    logical = list_logical_images(images_path)
    if not logical:
        return None, 0, None, {"error": "No images found in images_dir"}
    # Real sources, one per logical image: a band-grouped capture's sibling bands fold into one entry.
    sources = [logical[stem] for stem in sorted(logical)]

    reviewed_skipped = 0
    if (dataset_root or review_state_dir) and skip_reviewed:
        store_dir = _verdict_store_of(dataset_root, review_state_dir)
        if _review_state_exists(str(store_dir)):
            from tcip_annotation.review_engine import ReviewEngine
            engine = ReviewEngine(str(store_dir))
            resolved_bucket, refusal = _resolve_review_bucket(engine, bucket)
            if refusal is not None:
                return None, 0, None, {"error": refusal}
            # _resolve_review_bucket pairs a None refusal with a bucket
            assert resolved_bucket is not None
            reviewed = reviewed_image_names({"image": engine.image_states(resolved_bucket)})
            before = len(sources)
            # A band-grouped capture's review-state identity is its manifest filename, not a sibling band's.
            display = [str(s.manifest_path) if isinstance(s, BandGroupRef) else str(s) for s in sources]
            kept = set(select_unreviewed(display, reviewed))
            sources = [s for s, d in zip(sources, display) if d in kept]
            reviewed_skipped = before - len(sources)

    try:
        from tcip_mcp.pipelines.inference.predictor import build_predictor
    except (ImportError, OSError) as e:
        return None, 0, None, {"error": f"torch/torchvision unavailable: {e}"}

    return sources, reviewed_skipped, build_predictor, None


@tool()
def prioritize_review_queue(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    dataset_root: str = "",
    method: str = "combined",
    budget: int = 50,
    skip_reviewed: bool = True,
    bucket: str | None = None,
    review_state_dir: str = "",
) -> dict:
    """Rank un-reviewed images by active-learning informativeness for the next review batch.

    Scores every candidate with ``method`` and returns the most uncertain/diverse frames first.

    When ``checkpoint_path`` names a checkpoint produced by a run bound to a selection, each
    ``queue`` entry carries ``calibration_member: bool``, matched against the calibration samples
    that run's partition froze whose own source sits in ``images_dir``: reviewing that image edits
    a label inside the bound run's own calibration universe. An unbound run carries no mark on any
    entry.

    Args:
        checkpoint_path: Trained model checkpoint (drives scoring).
        images_dir: Directory of candidate images.
        dataset_root: Root of the dataset whose review is in progress. It scopes the verdict store
            (``<dataset_root>/.tcip/state``) that ``skip_reviewed`` reads. With neither this nor
            ``review_state_dir`` stated, no store is read and every candidate image is ranked.
        method: Informativeness scorer. ``uncertainty`` | ``diversity`` | ``combined`` are the
            built-in reference implementations: register your own with ``register_scorer``, or pass
            a dotted ``module:factory`` you wrote, scoring under the checkpoint's own task. An
            unresolvable name is refused.
        budget: Number of images to return.
        skip_reviewed: Exclude already-completed images from the queue.
        bucket: Which prediction bucket's completed reviews ``skip_reviewed`` skips, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim.
    """
    sources, reviewed_skipped, build_predictor, error = _prepare_queue_sources(
        checkpoint_path, images_dir, dataset_root, review_state_dir, skip_reviewed, bucket)
    if error is not None:
        return error

    checkpoint, refusal = _load_or_refuse(checkpoint_path, project)
    if refusal is not None:
        return refusal
    try:
        task = checkpoint.task
    except ValueError as exc:
        return {"error": str(exc)}

    if not sources:
        return {"method": method, "task": task, "total_candidates": 0,
                "reviewed_skipped": reviewed_skipped, "selected_count": 0, "queue": []}

    try:
        from tcip_mcp.pipelines.active_learning.helpers import build_scorer, require_composed_detector
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}

    predictor = build_predictor(checkpoint)
    guard = require_composed_detector(predictor, purpose="review-queue scoring")
    if guard:
        return {"error": guard}
    try:
        scorer = build_scorer(method, task)
    except ValueError as e:  # unknown scorer: refuse rather than silently reordering the queue
        return {"error": str(e)}

    from tcip_mcp.pipelines.image_utils import BandGroupRef

    scored = scorer.score(sources, predictor)[:budget]
    calibration_stems = _calibration_stems(checkpoint, Path(images_dir), project)
    marks: Sequence[bool | None] = [None] * len(scored)
    if calibration_stems is not None:
        marks = _calibration_marks([p for p, _ in scored], calibration_stems)
    queue = []
    for (p, s), mark in zip(scored, marks):
        entry = {"image": str(p.manifest_path) if isinstance(p, BandGroupRef) else str(p),
                 "score": round(float(s), 6)}
        if mark is not None:
            entry["calibration_member"] = mark
        queue.append(entry)
    result = {
        "method": method,
        "task": task,
        "total_candidates": len(sources),
        "reviewed_skipped": reviewed_skipped,
        "selected_count": len(scored),
        "queue": queue,
    }
    return result


def triage_predictions(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    dataset_root: str = "",
    skip_reviewed: bool = True,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float | None = None,
    bucket: str | None = None,
    review_state_dir: str = "",
) -> dict:
    """Sort a checkpoint's own predictions by confidence into auto-accept, needs-review and
    unscoreable queues.

    Returns predictions at or above ``auto_threshold`` as the confident set for a caller to accept
    as ground truth; this door writes nothing itself. Routes predictions between ``low`` and
    ``high`` into the needs-review queue, which can overlap the confident set when
    ``auto_threshold`` sits below ``high``, and separates out predictions with no
    confidence-bearing signal at all (e.g. a regression head's point estimate) into their own
    ``unscoreable_images`` list.

    Args:
        checkpoint_path: Trained model checkpoint (drives predictions).
        images_dir: Directory of candidate images.
        dataset_root: Root of the dataset whose review is in progress. It scopes the verdict store
            (``<dataset_root>/.tcip/state``) that ``skip_reviewed`` reads. With neither this nor
            ``review_state_dir`` stated, no store is read and every candidate image is triaged.
        skip_reviewed: Exclude already-completed images before triaging.
        low: Lower confidence bound for the needs-review band.
        high: Upper confidence bound for the needs-review band.
        auto_threshold: Confidence at/above which a prediction joins the confident set this door
            returns. ``None`` (default) refuses to auto-accept. Derive it from the model's
            validated confidence distribution and confirm with a breeder spot-check; the result is
            stamped as requiring that confirmation.
        bucket: Which prediction bucket's completed reviews ``skip_reviewed`` skips, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim.
    """
    sources, reviewed_skipped, build_predictor, error = _prepare_queue_sources(
        checkpoint_path, images_dir, dataset_root, review_state_dir, skip_reviewed, bucket)
    if error is not None:
        return error

    from tcip_mcp.pipelines.active_learning.selector import auto_accept, review_queue, unscoreable

    if not sources:
        return {"total_images": 0, "reviewed_skipped": reviewed_skipped,
                "auto_accepted": 0, "needs_review": 0, "review_images": [],
                "unscoreable_images": [], "auto_accepted_images": []}
    checkpoint, refusal = _load_or_refuse(checkpoint_path, project)
    if refusal is not None:
        return refusal
    predictor = build_predictor(checkpoint)
    predictions = predictor.predict_batch(sources)
    needs_review = review_queue(predictions, low=low, high=high)
    # A prediction with no confidence signal at all (a regression head's point estimate) is tagged unscoreable, not dropped.
    unscoreable_preds = unscoreable(predictions)
    all_review = needs_review + unscoreable_preds
    # Refuse to auto-accept at a pinned threshold: it must be derived from the validated conf distribution and breeder-confirmed.
    if auto_threshold is None:
        return {
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
