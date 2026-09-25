"""Review -> retrain feedback MCP tools.

``materialize_review_dataset`` turns human review verdicts into a curated detection training set
(with experiment lineage); ``prioritize_review_queue`` ranks un-reviewed images by
active-learning informativeness for the next review batch; ``triage_predictions`` partitions a
checkpoint's own predictions by confidence, optionally auto-accepting the most confident ones as
ground truth.

All three are scoped by the dataset root the review was recorded against, which is what derives
the verdict store they read. A review whose verdicts live outside that dataset states its store
instead, and the store it read is reported rather than folded into the derived one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.feedback.materialize import (
    materialize_dataset, reviewed_image_names, select_unreviewed,
)
from tcip_mcp.project_paths import resolve_output_path


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
    dataset root through :func:`~tcip_mcp.prediction_buckets.review_state_dir_of`, the platform's
    one derivation of where verdicts live. The two are never merged and neither backs the other:
    a stated store holding nothing is a stated store holding nothing.
    """
    from tcip_mcp.prediction_buckets import review_state_dir_of

    if review_state_dir:
        return Path(review_state_dir)
    return review_state_dir_of(dataset_root)


def _load_or_refuse(checkpoint_path: str, project_path: str):
    """The verified checkpoint either review-queue door builds its predictor from, or the door's
    own refusal dict when the registry names no entry for it. Returns ``(checkpoint, refusal)``."""
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        return load_registered_checkpoint(checkpoint_path, project_path=project_path or None), None
    except UnregisteredCheckpoint as exc:
        return None, {"error": str(exc)}


def _resolve_calibration_ids(
    checkpoint, images_dir: Path, *, project_path: str | None = None,
) -> tuple[set[str] | None, str | None]:
    """The bound run's calibration-side member stems whose own image sits in ``images_dir``, or the
    reason none could be resolved. Returns ``(calibration_stems, marks_unresolved)``.

    ``calibration_stems`` is ``None`` with ``marks_unresolved`` also ``None`` for an unbound run:
    no registry-entry ``experiment_id`` at all (``checkpoint.producer``), or an experiment recorded
    but never bound to a selection (its ``split.json`` carries no ``selection_binding``). A bound
    run's own split record that cannot be read, or whose named selection can no longer be read,
    gets ``calibration_stems=None`` with ``marks_unresolved`` naming why.

    A calibration sample whose recorded source sits in ``images_dir`` is one of this queue's
    candidates; a sample from another directory the selection also spans is not.
    """
    experiment_id = checkpoint.producer
    if not experiment_id:
        return None, None
    from tcip_mcp.project_paths import platform_state_root

    root = project_path or str(platform_state_root())
    from tcip_mcp.experiments import read_run_partition_checked

    split, decode_error = read_run_partition_checked(experiment_id, root=root)
    if decode_error is not None:
        return None, (
            f"this run's split record could not be read to mark the queue's calibration-side "
            f"candidates: {decode_error}"
        )
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.split_construction import bound_selection_dir

    selection_dir = bound_selection_dir(split)
    if selection_dir is None:
        return None, None

    try:
        selection = read_selection(selection_dir)
    except ValueError as exc:
        return None, (
            f"this run is bound to the selection at {selection_dir!r}, but it could not be read "
            f"to mark the queue's calibration-side candidates: {exc}"
        )
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.image_utils import stem_of

    return {stem_of(s.source) for s in selection.on("calibration")
            if same_directory(Path(s.source).parent, images_dir)}, None


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
    When ``experiment_id`` is given, records the review session as experiment lineage. Output is
    the platform's ``images/`` + ``annotations/`` dataset layout.

    The reviewed bucket's own recorded scope (``resolution.bucket_scope``) governs when there is
    one: under a classified scope every positive is written with the object class in ``subject``
    and the confirmed value under the scope's attribute, ``subject`` (if stated) must equal the
    scope's own, and no rejected-only image is confirmed negative, landing in
    ``unconfirmed_negatives`` instead. The source dataset's own registry is then copied over;
    refuses by name when the source names no dataset root, that root has no ``subjects.json``, or
    the output already holds a registry. A bare directory or a detector scope, and a
    ground-truth-only review with no prediction file, read no classified scope.

    Args:
        dataset_root: Root of the dataset the review was recorded against. It scopes the verdict
            store read when ``review_state_dir`` is not stated (``<dataset_root>/.tcip/state``),
            and it is what the experiment lineage records as the reviewed dataset.
        source_images_dir: Directory of the reviewed source images.
        output_dir: Destination for the curated dataset (distinct from the source). A relative path
            resolves against the platform state root, never the server process's cwd.
        experiment_id: Optional run record (``tcip_mcp.experiments``) to record the review-session
            lineage on.
        include_hard_negatives: Emit rejected-only images as empty-label backgrounds.
        only_completed: Restrict to fully-reviewed (``img_status=='completed'``) images.
        copy_files: Copy images (True) or symlink (False).
        subject: The object the review was about; confirmed negatives are keyed under it. When
            omitted it is derived from every subject the verdicts name, rejections included, and
            only when they name exactly one. A rejected image whose own rejections answer for
            another subject, or for none, is materialized as an unconfirmed empty and reported in
            ``unconfirmed_negatives`` with why.
        bucket: Which prediction bucket's verdicts to curate, as
            ``prediction_buckets.bucket_key_of`` spells it. Omitted reads the store's sole bucket
            and refuses, naming them, when it holds several.
        review_state_dir: A verdict store to read instead of the dataset's own. Not stated (the
            default) derives the store from ``dataset_root``; stated, it is read verbatim and the
            response names it. A stated store holding no shards is refused.
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

    from tcip_annotation.review_engine import NO_BUCKET, ReviewEngine
    engine = ReviewEngine(str(store_dir))
    resolved_bucket, refusal = _resolve_review_bucket(engine, bucket)
    if refusal is not None:
        return {"error": refusal}
    assert resolved_bucket is not None  # _resolve_review_bucket pairs a None refusal with a bucket
    review_state = {"image": engine.image_states(resolved_bucket)}
    state_path = engine.shard_dir

    from tcip_mcp.pipelines.resolution import bucket_scope
    from tcip_store import StoreError

    scope = None
    vocabulary = None
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
            scope = bucket_scope(scope_dir)
        except StoreError as exc:
            return {"error": str(exc)}
        if scope is not None and scope.classified:
            from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map

            id_map = bucket_id_map(scope_dir)
            if id_map is None:
                return {"error": (
                    f"{scope_dir} records no id_map: a classified scope's confirmed values need "
                    "the bucket's own recorded vocabulary to check them against, and this bucket's "
                    "stamp carries none. Re-infer this run or repair its stamp before "
                    "materializing its review."
                )}
            vocabulary = set(id_map)

    try:
        result = materialize_dataset(
            review_state, source_images_dir, output_dir, subject=subject,
            review_state_path=str(state_path), include_hard_negatives=include_hard_negatives,
            copy_files=copy_files, only_completed=only_completed, scope=scope,
            vocabulary=vocabulary,
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


@mcp.tool()
def prioritize_review_queue(
    checkpoint_path: str,
    images_dir: str,
    dataset_root: str = "",
    method: str = "combined",
    budget: int = 50,
    skip_reviewed: bool = True,
    bucket: str | None = None,
    review_state_dir: str = "",
    project_path: str = "",
) -> dict:
    """Rank un-reviewed images by active-learning informativeness for the next review batch.

    Scores every candidate with ``method`` and returns the most uncertain/diverse frames first.

    When ``checkpoint_path`` names a registry entry produced by a run bound to a selection
    (``selection_binding`` on that run's ``split.json``), each ``queue`` entry carries
    ``calibration_member: bool``, matched against the selection's calibration samples whose own
    source sits in ``images_dir``: reviewing that image edits a label inside the bound run's own
    calibration universe. An unbound run carries no mark on any entry. A bound run whose own split
    record, or whose named selection, can no longer be read carries no mark either, and the
    response states why under ``marks_unresolved``.

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
        project_path: Project root the checkpoint's registry entry is looked up under. Empty
            (default) resolves to the process's own root.
    """
    sources, reviewed_skipped, build_predictor, error = _prepare_queue_sources(
        checkpoint_path, images_dir, dataset_root, review_state_dir, skip_reviewed, bucket)
    if error is not None:
        return error

    checkpoint, refusal = _load_or_refuse(checkpoint_path, project_path)
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
    calibration_stems, marks_unresolved = _resolve_calibration_ids(
        checkpoint, Path(images_dir), project_path=project_path or None)
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
    if marks_unresolved is not None:
        result["marks_unresolved"] = marks_unresolved
    return result


def triage_predictions(
    checkpoint_path: str,
    images_dir: str,
    dataset_root: str = "",
    skip_reviewed: bool = True,
    low: float = 0.3,
    high: float = 0.8,
    auto_threshold: float | None = None,
    bucket: str | None = None,
    review_state_dir: str = "",
    project_path: str = "",
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
        project_path: Project root the checkpoint's registry entry is looked up under. Empty
            (default) resolves to the process's own root.
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
    checkpoint, refusal = _load_or_refuse(checkpoint_path, project_path)
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
