"""Review-queue doors: ``prioritize_review_queue`` and ``triage_predictions``.

Each skips the images whose label document marks the named subject finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tcip_mcp.pipelines.active_learning import ScorerSpec
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool

if TYPE_CHECKING:
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.execution import Preparation, Stated


class QueueEntry(BaseModel):
    """One image of a review queue: its logical name, its score and its reference membership.

    ``reference_member`` is whether the checkpoint's producing run froze the image on a reference
    side of its partition, ``None`` for a checkpoint no selection-bound run of the project produced.
    """

    image: str
    score: float
    reference_member: bool | None


class ReviewQueue(BaseModel):
    """A ranked review queue: what it was ranked under, its checkpoint's task, counts and entries.

    ``queue`` is the top ``budget`` of ``total_candidates`` images, highest first;
    ``reviewed_skipped`` counts the images skipped as finished for the subject.
    """

    scorer: ScorerSpec
    budget: int
    rationale: str
    task: str
    total_candidates: int
    reviewed_skipped: int
    queue: list[QueueEntry]


def _load_or_refuse(
    checkpoint_path: Path, project: Path
) -> VerifiedCheckpoint | dict[str, str]:
    """Load the registered checkpoint at ``checkpoint_path``, or answer the load's refusal.

    Returns the verified checkpoint at the established location ``checkpoint_path``
    (``model_registry.load_registered_checkpoint``), or the ``{"error": ...}`` of each refusal
    that load raises: no file at the location, a checkpoint ``project``'s registry names no entry
    for, or a project record that will not read.
    """
    from tcip_mcp.model_registry import (
        CheckpointNotFoundError,
        load_registered_checkpoint,
    )

    try:
        return load_registered_checkpoint(checkpoint_path, project=project)
    except (CheckpointNotFoundError, ValueError) as exc:
        return {"error": str(exc)}


def _reference_stems(
    checkpoint: VerifiedCheckpoint, images_dir: Path, project: Path
) -> set[str] | None:
    """Return the reference-side member stems of ``checkpoint``'s run that sit in ``images_dir``.

    The stems are the reference-side members (``selection.REFERENCE_SIDES``) the producing run of
    ``checkpoint`` in ``project`` froze in its partition (``experiments.run_resolution``) whose own
    image sits in ``images_dir``, the two directories compared through
    ``tcip_store.canonical_path``, or ``None`` for a checkpoint no run of this project produced
    (``checkpoint.experiment_id``) and for a run that bound no selection.
    """
    if checkpoint.experiment_id is None:
        return None
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
    from tcip_mcp.pipelines.image_utils import stem_of
    from tcip_store import canonical_path

    partition = run_resolution(
        checkpoint.experiment_id, project=project
    ).partition
    if partition.selection is None:
        return None
    here = canonical_path(images_dir)
    return {
        stem_of(s.source)
        for s in partition.samples
        if s.side in REFERENCE_SIDES
        and canonical_path(Path(s.source).parent) == here
    }


def _prepare_queue_sources(
    images_dir: str, subject: str | None
) -> tuple[list[Path | BandGroupRef], int] | dict[str, str]:
    """List the candidate images of the absolute ``images_dir``.

    The steps, in order: images directory, logical image enumeration, then, with a ``subject``
    named, dropping each image whose label document marks it finished
    (:meth:`~tcip_annotation.json_io.LabelDocument.finished`).

    Returns ``(sources, reviewed_skipped)``, or the ready ``{"error": ...}`` dict of a step that
    refused. With a ``subject`` named, an image outside a dataset's image tree, which cannot name
    its label document, and a label document that will not read are that error.
    """
    images_path = Path(images_dir)
    if not images_path.is_dir():
        return {"error": f"Images dir not found: {images_dir}"}
    from tcip_mcp.pipelines.image_utils import (
        list_logical_images,
        source_path_of,
    )

    logical = list_logical_images(images_path)
    if not logical:
        return {"error": "No images found in images_dir"}
    # Real sources, one per logical image: a band-grouped capture's sibling bands fold into one
    # entry.
    sources = [logical[stem] for stem in sorted(logical)]
    if subject is None:
        return sources, 0

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
        return {"error": str(exc)}
    return kept, len(sources) - len(kept)


def _untiled(
    checkpoint: VerifiedCheckpoint, stated: Stated
) -> Preparation | dict[str, str]:
    """Ready ``checkpoint`` untiled under ``stated``.

    Returns the preparation (:func:`~tcip_mcp.pipelines.execution.prepare`), or the
    ``{"error": ...}`` refusal when torch is not installed.
    """
    try:
        from tcip_mcp.pipelines.execution import prepare

        return prepare(checkpoint, stated.model_copy(update={"tile": False}))
    except (ImportError, OSError) as e:
        return {"error": f"torch/torchvision unavailable: {e}"}


@tool()
def prioritize_review_queue(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    scorer: ScorerSpec,
    budget: int,
    rationale: str,
    subject: str | None = None,
) -> ReviewQueue | dict[str, str]:
    """Rank unfinished images by active-learning informativeness into a review queue.

    Scores every candidate with ``scorer`` and returns the ``budget`` most uncertain/diverse frames
    first as the :class:`ReviewQueue`, each ``queue`` entry a :class:`QueueEntry` naming its image
    by its logical name, beside the scorer block, the budget and ``rationale``.

    When ``checkpoint_path`` names a checkpoint produced by a run bound to a selection, each
    entry's ``reference_member`` is matched against the calibration and holdout samples that run's
    partition froze whose own source sits in ``images_dir``; for an unbound run it is ``None`` on
    every entry.

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
) -> ReviewQueue | dict[str, str]:
    """Rank the unfinished images under ``images_dir`` by the scorer ``scorer`` builds.

    The images under the established location ``images_dir`` are ranked by the scorer
    (:func:`~tcip_mcp.pipelines.active_learning.scorer.resolve_scorer`) under the
    registered checkpoint at the established location ``checkpoint_path``, answered as the
    :class:`ReviewQueue`, or the refusal ``{"error": ...}``; ``subject``'s finished images are
    skipped.
    """
    checkpoint = _load_or_refuse(checkpoint_path, project)
    if isinstance(checkpoint, dict):
        return checkpoint
    queued = _prepare_queue_sources(str(images_dir), subject)
    if isinstance(queued, dict):
        return queued
    sources, reviewed_skipped = queued
    try:
        task = checkpoint.task
    except ValueError as exc:
        return {"error": str(exc)}

    if not sources:
        return ReviewQueue(
            scorer=scorer,
            budget=budget,
            rationale=rationale,
            task=task,
            total_candidates=0,
            reviewed_skipped=reviewed_skipped,
            queue=[],
        )

    from tcip_mcp.pipelines.execution import Stated

    prep = _untiled(checkpoint, Stated())
    if isinstance(prep, dict):
        return prep
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
    return ReviewQueue(
        scorer=scorer,
        budget=budget,
        rationale=rationale,
        task=task,
        total_candidates=len(sources),
        reviewed_skipped=reviewed_skipped,
        queue=[
            QueueEntry(
                image=logical_image_name(p),
                score=round(float(s), 6),
                reference_member=None
                if reference_stems is None
                else stem_of(p) in reference_stems,
            )
            for p, s in scored
        ],
    )


@tool()
def triage_predictions(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    low: float,
    high: float,
    rationale: str,
    subject: str | None = None,
) -> dict:
    """Sort a checkpoint's own predictions by confidence into needs-review and unscoreable queues.

    This door writes nothing.

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
    checkpoint = _load_or_refuse(located(checkpoint_path, project), project)
    if isinstance(checkpoint, dict):
        return checkpoint
    queued = _prepare_queue_sources(str(located(images_dir, project)), subject)
    if isinstance(queued, dict):
        return queued
    sources, reviewed_skipped = queued

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

    prep = _untiled(
        checkpoint,
        Stated(
            conf=low,
            masks=False if checkpoint.task == "instance_seg" else None,
            rationale=rationale,
        )
        if checkpoint.task in DETECTOR_TASKS
        else Stated(),
    )
    if isinstance(prep, dict):
        return prep
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
