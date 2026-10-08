"""Review-queue MCP tools: ``prioritize_review_queue`` and ``triage_predictions``, each skipping
the images whose label document marks the named subject finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.active_learning import DEFAULT_REVIEW_BUDGET, DEFAULT_SCORER
from tcip_mcp.server import tool


def _load_or_refuse(checkpoint_path: str, project: Path):
    """The verified checkpoint either review-queue door builds its predictor from, or the door's
    own refusal dict when ``project``'s registry names no entry for it. Returns ``(checkpoint,
    refusal)``."""
    from tcip_mcp.model_registry import UnregisteredCheckpointError, load_registered_checkpoint

    try:
        return load_registered_checkpoint(checkpoint_path, project=project), None
    except UnregisteredCheckpointError as exc:
        return None, {"error": str(exc)}


def _reference_stems(checkpoint, images_dir: Path, project: Path) -> set[str] | None:
    """The reference-side member stems (``selection.REFERENCE_SIDES``) the producing run of
    ``checkpoint`` in ``project`` froze in its partition (``experiments.run_resolution``) whose own
    image sits in ``images_dir``, or ``None`` for a checkpoint no run of this project produced
    (``checkpoint.experiment_id``) and for a run that bound no selection. A reference sample from
    another directory the selection also spans is not one of this queue's candidates.
    """
    if checkpoint.experiment_id is None:
        return None
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
    from tcip_mcp.pipelines.data.split_construction import partition_samples
    from tcip_mcp.pipelines.data.splits import same_directory
    from tcip_mcp.pipelines.image_utils import stem_of

    partition = run_resolution(checkpoint.experiment_id, project=project)["partition"]
    if partition["selection"] is None:
        return None
    return {stem_of(s.source) for s in partition_samples(partition)
            if s.side in REFERENCE_SIDES and same_directory(Path(s.source).parent, images_dir)}


def _prepare_queue_sources(checkpoint_path: str, images_dir: str, subject: str | None):
    """The candidate images of ``images_dir``, in order: checkpoint existence, images directory,
    logical image enumeration, then, with a ``subject`` named, dropping each image whose label
    document marks it finished (:meth:`~tcip_annotation.json_io.LabelDocument.finished`).

    Returns ``(sources, reviewed_skipped, error)``; ``error`` is a ready ``{"error": ...}`` dict
    and the other two are ``None``/``0`` when it is set. With a ``subject`` named, an image
    outside a dataset's image tree, which cannot name its label document, and a label document
    that will not read are that error.
    """
    if not Path(checkpoint_path).is_file():
        return None, 0, {"error": f"Checkpoint not found: {checkpoint_path}"}
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return None, 0, {"error": f"Images dir not found: {images_dir}"}
    from tcip_mcp.pipelines.image_utils import list_logical_images, source_path_of

    logical = list_logical_images(images_path)
    if not logical:
        return None, 0, {"error": "No images found in images_dir"}
    # Real sources, one per logical image: a band-grouped capture's sibling bands fold into one
    # entry.
    sources = [logical[stem] for stem in sorted(logical)]
    if subject is None:
        return sources, 0, None

    from tcip_annotation.json_io import UnreadableLabelDocumentError, read_document_versioned

    from tcip_mcp.dataset_layout import label_key_of

    try:
        kept = [s for s in sources if not read_document_versioned(
            label_key_of(source_path_of(s)))[0].finished(subject)]
    except (ValueError, UnreadableLabelDocumentError) as exc:
        return None, 0, {"error": str(exc)}
    return kept, len(sources) - len(kept), None


def _untiled(checkpoint, stated: Any) -> tuple[Any, dict | None]:
    """``checkpoint`` readied untiled under ``stated``
    (:func:`~tcip_mcp.pipelines.execution.prepare`), or the refusal when torch is not installed.
    Returns ``(preparation, refusal)``."""
    try:
        from tcip_mcp.pipelines.execution import prepare

        return prepare(checkpoint, stated.model_copy(update={"tile": False})), None
    except (ImportError, OSError) as e:
        return None, {"error": f"torch/torchvision unavailable: {e}"}


@tool()
def prioritize_review_queue(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    method: str = DEFAULT_SCORER,
    budget: int = DEFAULT_REVIEW_BUDGET,
    subject: str | None = None,
) -> dict:
    """Rank unfinished images by active-learning informativeness for the next review batch.

    Scores every candidate with ``method`` and returns the most uncertain/diverse frames first,
    each ``queue`` entry naming its image by its logical name.

    When ``checkpoint_path`` names a checkpoint produced by a run bound to a selection, each
    ``queue`` entry carries ``reference_member: bool``, matched against the calibration and holdout
    samples that run's partition froze whose own source sits in ``images_dir``: reviewing that
    image edits a label inside the bound run's own assessment reference. An unbound run carries no
    mark on any entry.

    Args:
        checkpoint_path: Trained model checkpoint (drives scoring).
        images_dir: Directory of candidate images.
        method: Informativeness scorer. ``uncertainty`` | ``diversity`` | ``combined`` are the
            built-in reference implementations: register your own with ``register_scorer``, or pass
            a dotted ``module:factory`` you wrote, scoring under the checkpoint's own task. An
            unresolvable name is refused.
        budget: Number of images to return.
        subject: The subject whose finished images (marked complete in their label document)
            are skipped; omitted ranks every candidate image.
    """
    sources, reviewed_skipped, error = _prepare_queue_sources(checkpoint_path, images_dir, subject)
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

    from tcip_mcp.pipelines.execution import Stated

    prep, refusal = _untiled(checkpoint, Stated())
    if refusal is not None:
        return refusal
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    predictor = prep.predictor
    try:
        scorer = resolve_scorer(method, task)
    except ValueError as e:  # unknown scorer: refuse rather than silently reordering the queue
        return {"error": str(e)}

    from tcip_mcp.pipelines.image_utils import logical_image_name, stem_of

    # A scorer answers the candidates it was handed, whatever their type.
    scored: list[tuple[Any, float]] = list(scorer.score(sources, predictor)[:budget])
    reference_stems = _reference_stems(checkpoint, Path(images_dir), project)
    queue = []
    for p, s in scored:
        entry: dict[str, Any] = {"image": logical_image_name(p), "score": round(float(s), 6)}
        if reference_stems is not None:
            entry["reference_member"] = stem_of(p) in reference_stems
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
    low: float = 0.3,
    high: float = 0.8,
    subject: str | None = None,
    max_dets: int | None = None,
) -> dict:
    """Sort a checkpoint's own predictions by confidence into needs-review and unscoreable
    queues; this door writes nothing.

    Routes predictions between ``low`` and ``high`` into the needs-review queue, and separates out
    predictions with no confidence-bearing signal at all (e.g. a regression head's point estimate)
    into their own ``unscoreable_images`` list. A detector predicts at ``low``, so every box the
    band can hold exists, and at the stated ``max_dets``, refusing without one.

    Args:
        checkpoint_path: Trained model checkpoint (drives predictions).
        images_dir: Directory of candidate images.
        low: Lower confidence bound for the needs-review band.
        high: Upper confidence bound for the needs-review band.
        subject: The subject whose finished images (marked complete in their label document)
            are skipped; omitted triages every candidate image.
        max_dets: The most boxes a detector's frame keeps; required of a detector, refused for
            any other head.
    """
    sources, reviewed_skipped, error = _prepare_queue_sources(checkpoint_path, images_dir, subject)
    if error is not None:
        return error

    from tcip_mcp.pipelines.active_learning.selector import review_queue, unscoreable

    if not sources:
        return {"total_images": 0, "reviewed_skipped": reviewed_skipped, "needs_review": 0,
                "review_images": [], "unscoreable_images": []}
    checkpoint, refusal = _load_or_refuse(checkpoint_path, project)
    if refusal is not None:
        return refusal
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Stated
    from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

    detector = checkpoint.task in DETECTION_TASKS
    prep, refusal = _untiled(checkpoint, Stated(conf=low if detector else None,
                                                max_dets=max_dets))
    if refusal is not None:
        return refusal
    try:
        predictions = prep.runnable().predict(sources)
    except ExecutionRefusedError as exc:
        return {"error": str(exc)}
    # A prediction with no confidence signal at all (a regression head's point estimate) is
    # tagged unscoreable, not dropped.
    unscoreable_preds = unscoreable(predictions)
    all_review = review_queue(predictions, low=low, high=high) + unscoreable_preds
    return {
        "total_images": len(predictions),
        "reviewed_skipped": reviewed_skipped,
        "needs_review": len(all_review),
        "review_images": [r.get("image", "") for r in all_review],
        "unscoreable_images": [p.get("image", "") for p in unscoreable_preds],
    }
