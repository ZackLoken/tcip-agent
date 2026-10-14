"""Review-queue MCP tools: ``prioritize_review_queue`` and ``triage_predictions``, each skipping
the images whose label document marks the named subject finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.active_learning import ScorerSpec
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool


def _load_or_refuse(checkpoint_path: Path, project: Path):
    """``(checkpoint, refusal)``: the verified checkpoint at the established location
    ``checkpoint_path`` (``model_registry.load_registered_checkpoint``) and
    ``None``, or ``None`` and the ``{"error": ...}`` of each refusal that load raises: no file at
    the location, a checkpoint ``project``'s registry names no entry for, or a project record
    that will not read.
    """
    from tcip_mcp.model_registry import (
        CheckpointNotFoundError,
        load_registered_checkpoint,
    )

    try:
        return load_registered_checkpoint(
            checkpoint_path, project=project
        ), None
    except (CheckpointNotFoundError, ValueError) as exc:
        return None, {"error": str(exc)}


def _reference_stems(
    checkpoint, images_dir: Path, project: Path
) -> set[str] | None:
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

    partition = run_resolution(
        checkpoint.experiment_id, project=project
    ).partition
    if partition["selection"] is None:
        return None
    return {
        stem_of(s.source)
        for s in partition_samples(partition)
        if s.side in REFERENCE_SIDES
        and same_directory(Path(s.source).parent, images_dir)
    }


def _prepare_queue_sources(images_dir: str, subject: str | None):
    """The candidate images of the absolute ``images_dir``, in order: images directory, logical
    image enumeration, then, with a ``subject`` named, dropping each image whose label document
    marks it finished (:meth:`~tcip_annotation.json_io.LabelDocument.finished`).

    Returns ``(sources, reviewed_skipped, error)``; ``error`` is a ready ``{"error": ...}`` dict
    and the other two are ``None``/``0`` when it is set. With a ``subject`` named, an image
    outside a dataset's image tree, which cannot name its label document, and a label document
    that will not read are that error.
    """
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return None, 0, {"error": f"Images dir not found: {images_dir}"}
    from tcip_mcp.pipelines.image_utils import (
        list_logical_images,
        source_path_of,
    )

    logical = list_logical_images(images_path)
    if not logical:
        return None, 0, {"error": "No images found in images_dir"}
    # Real sources, one per logical image: a band-grouped capture's sibling bands fold into one
    # entry.
    sources = [logical[stem] for stem in sorted(logical)]
    if subject is None:
        return sources, 0, None

    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_document_versioned,
    )
    from tcip_mcp.dataset_layout import label_key_of

    try:
        kept = [
            s
            for s in sources
            if not read_document_versioned(label_key_of(source_path_of(s)))[
                0
            ].finished(subject)
        ]
    except (ValueError, UnreadableLabelDocumentError) as exc:
        return None, 0, {"error": str(exc)}
    return kept, len(sources) - len(kept), None


def _untiled(checkpoint, stated: Any) -> tuple[Any, dict | None]:
    """``checkpoint`` readied untiled under ``stated``
    (:func:`~tcip_mcp.pipelines.execution.prepare`), or the refusal when torch is not installed.
    Returns ``(preparation, refusal)``.
    """
    try:
        from tcip_mcp.pipelines.execution import prepare

        return prepare(
            checkpoint, stated.model_copy(update={"tile": False})
        ), None
    except (ImportError, OSError) as e:
        return None, {"error": f"torch/torchvision unavailable: {e}"}


@tool()
def prioritize_review_queue(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    scorer: ScorerSpec,
    budget: int,
    rationale: str,
    subject: str | None = None,
) -> dict:
    """Rank unfinished images by active-learning informativeness for the next review batch.

    Scores every candidate with ``scorer`` and returns the ``budget`` most uncertain/diverse frames
    first, each ``queue`` entry naming its image by its logical name, answered beside the scorer
    block, the budget and ``rationale``.

    When ``checkpoint_path`` names a checkpoint produced by a run bound to a selection, each
    ``queue`` entry carries ``reference_member: bool``, matched against the calibration and holdout
    samples that run's partition froze whose own source sits in ``images_dir``: reviewing that
    image edits a label inside the bound run's own assessment reference. An unbound run carries no
    mark on any entry.

    Args:
        checkpoint_path: Trained model checkpoint (drives scoring).
        images_dir: Directory of candidate images.
        scorer: The informativeness scorer block, ``{"name": ..., **values}``: ``uncertainty``
            and ``diversity`` take no values, ``combined`` requires ``uncertainty_weight`` and
            ``diversity_weight``; register your own factory with ``register_scorer``, or name a
            dotted ``module:factory`` you wrote, built from the checkpoint's own task and the
            block's values. An unresolvable name, a required value unstated and a value the
            factory does not take are refused.
        budget: Number of images to return.
        rationale: Why the queue is ranked and sized as stated.
        subject: The subject whose finished images (marked complete in their label document)
            are skipped; omitted ranks every candidate image.

    """
    return ranked_review_queue(
        project,
        located(checkpoint_path, project),
        located(images_dir, project),
        scorer=scorer,
        budget=budget,
        rationale=rationale,
        subject=subject,
    )


def ranked_review_queue(
    project: Path,
    checkpoint_path: Path,
    images_dir: Path,
    *,
    scorer: ScorerSpec,
    budget: int,
    rationale: str,
    subject: str | None,
) -> dict:
    """The unfinished images under the established location ``images_dir`` ranked by the scorer
    ``scorer`` builds (:func:`~tcip_mcp.pipelines.active_learning.scorer.resolve_scorer`) under the
    registered checkpoint at the established location ``checkpoint_path``, the top ``budget`` as
    ``queue`` (each marked ``reference_member`` for a selection-bound run) answered beside the
    scorer block, the budget and ``rationale``, or the refusal ``{"error": ...}``; ``subject``'s
    finished images are skipped.
    """
    checkpoint, refusal = _load_or_refuse(checkpoint_path, project)
    if refusal is not None:
        return refusal
    sources, reviewed_skipped, error = _prepare_queue_sources(
        str(images_dir), subject
    )
    if error is not None:
        return error
    try:
        task = checkpoint.task
    except ValueError as exc:
        return {"error": str(exc)}

    stated = {
        "scorer": scorer.model_dump(),
        "budget": budget,
        "rationale": rationale,
    }
    if not sources:
        return {
            **stated,
            "task": task,
            "total_candidates": 0,
            "reviewed_skipped": reviewed_skipped,
            "selected_count": 0,
            "queue": [],
        }

    from tcip_mcp.pipelines.execution import Stated

    prep, refusal = _untiled(checkpoint, Stated())
    if refusal is not None:
        return refusal
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    predictor = prep.predictor
    try:
        ranking = resolve_scorer(scorer, task)
    except (ValueError, TypeError) as e:
        return {"error": str(e)}

    from tcip_mcp.pipelines.image_utils import logical_image_name, stem_of

    # A scorer answers the candidates it was handed, whatever their type.
    scored: list[tuple[Any, float]] = list(
        ranking.score(sources, predictor)[:budget]
    )
    reference_stems = _reference_stems(checkpoint, images_dir, project)
    queue = []
    for p, s in scored:
        entry: dict[str, Any] = {
            "image": logical_image_name(p),
            "score": round(float(s), 6),
        }
        if reference_stems is not None:
            entry["reference_member"] = stem_of(p) in reference_stems
        queue.append(entry)
    return {
        **stated,
        "task": task,
        "total_candidates": len(sources),
        "reviewed_skipped": reviewed_skipped,
        "selected_count": len(scored),
        "queue": queue,
    }


def triage_predictions(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    low: float,
    high: float,
    rationale: str,
    subject: str | None = None,
) -> dict:
    """Sort a checkpoint's own predictions by confidence into needs-review and unscoreable
    queues; this door writes nothing.

    Routes predictions between ``low`` and ``high`` into the needs-review queue, and separates out
    predictions with no confidence-bearing signal at all (e.g. a regression head's point estimate)
    into their own ``unscoreable_images`` list. ``low`` is read twice: as the band's lower edge,
    and as the conf a detector predicts at, so every box the band can hold exists, each frame
    capped at the object density its checkpoint recorded and no mask collected. The answer
    carries ``low``, ``high`` and ``rationale`` beside the queues.

    Args:
        checkpoint_path: Trained model checkpoint (drives predictions).
        images_dir: Directory of candidate images.
        low: Lower confidence bound for the needs-review band, inclusive, and a detector's
            pass conf.
        high: Upper confidence bound for the needs-review band, exclusive.
        rationale: Why the band is stated as it is.
        subject: The subject whose finished images (marked complete in their label document)
            are skipped; omitted triages every candidate image.

    """
    checkpoint, refusal = _load_or_refuse(
        located(checkpoint_path, project), project
    )
    if refusal is not None:
        return refusal
    sources, reviewed_skipped, error = _prepare_queue_sources(
        str(located(images_dir, project)), subject
    )
    if error is not None:
        return error

    from tcip_mcp.pipelines.active_learning.selector import (
        review_queue,
        unscoreable,
    )

    stated = {"low": low, "high": high, "rationale": rationale}
    if not sources:
        return {
            **stated,
            "total_images": 0,
            "reviewed_skipped": reviewed_skipped,
            "needs_review": 0,
            "review_images": [],
            "unscoreable_images": [],
        }
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Stated
    from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

    prep, refusal = _untiled(
        checkpoint,
        Stated(
            conf=low,
            masks=False if checkpoint.task == "instance_seg" else None,
            rationale=rationale,
        )
        if checkpoint.task in DETECTOR_TASKS
        else Stated(),
    )
    if refusal is not None:
        return refusal
    try:
        predictions = prep.runnable().predict(sources)
    except ExecutionRefusedError as exc:
        return {"error": str(exc)}
    # A prediction with no confidence signal at all (a regression head's point estimate) is
    # tagged unscoreable, not dropped.
    unscoreable_preds = unscoreable(predictions)
    all_review = (
        review_queue(predictions, low=low, high=high) + unscoreable_preds
    )
    return {
        **stated,
        "total_images": len(predictions),
        "reviewed_skipped": reviewed_skipped,
        "needs_review": len(all_review),
        "review_images": [r.get("image", "") for r in all_review],
        "unscoreable_images": [p.get("image", "") for p in unscoreable_preds],
    }
