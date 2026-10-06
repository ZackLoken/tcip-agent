"""Assessment tools: assess a checkpoint against a held-out reference, against a mosaic's own
reserved regions, and a physical scale against a breeder's reference measurements."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcip_mcp.pipelines.block_calibration import DEFAULT_K_CAL, DEFAULT_K_TEST
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.server import tool

if TYPE_CHECKING:
    from tcip_mcp.assessment import Assessment


def _refusals() -> tuple[type[Exception], ...]:
    """What an assessment door answers ``{"error": ...}`` for rather than raising."""
    from tcip_annotation.json_io import UnreadableLabelDocumentError

    from tcip_mcp.model_registry import UnregisteredCheckpointError
    from tcip_mcp.traits import TraitUnknownError

    return (ValueError, UnreadableLabelDocumentError, UnregisteredCheckpointError,
            TraitUnknownError, FileNotFoundError)


def _answer(assessment: Assessment) -> dict[str, Any]:
    """An assessment as a door answers it: everything but the reference's sample list, which stays
    in ``assessment.json``, counted instead."""
    answer = asdict(assessment)
    samples = answer["reference"].pop("samples")
    return {**answer,
            "execution": assessment.execution.record() if assessment.execution else None,
            "reference": {**answer["reference"], "n_samples": len(samples)}}


@tool()
def assess_checkpoint(
    project: Path,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    selection_dir: str,
    stated: Stated | None = None,
    device: str | None = None,
) -> dict:
    """Assess a checkpoint against the calibration and holdout sides of a selection, for one
    delivered kind of a trait, and record the result as an assessment.

    The operating point is fitted on the calibration side (the trait's count objective picks the
    conf; an unstated merge threshold and detection cap derive from its ground truth) and the
    criterion the kind rests on is measured over the holdout side: the held-out count bias, pooled
    and per class, localization and dispersion for a count; that plus the classifier's agreement
    over matched instances for ``state_crossing_dates``; the scalar skill for an ordinal or
    regression aggregate, a regression scored by the revision's own ``regression_criterion``. The
    reference must share no group or image with the producing run's training or selection side.
    Predictions published under the assessment (``run_inference(assessment_id=...)``) run exactly
    its execution record; a delivery reads its ``passed``.

    Args:
        checkpoint_path: A checkpoint registered in this project.
        trait: The trait whose latest confirmed revision states the criterion's floors.
        delivery_kind: The delivered kind assessed (``per_image_count``,
            ``per_plant_count_aggregate``, ``state_crossing_dates``,
            ``per_plant_ordinal_aggregate``, ``per_plant_regression_aggregate``).
        selection_dir: A selection drawn with calibration and holdout sides (``draw_splits``).
        stated: Execution values to state rather than derive (``execution.Stated``), as
            ``run_inference`` takes them.
        device: cuda / cpu (auto if omitted).
    """
    from tcip_mcp.assessment import assess

    try:
        return _answer(assess(
            project, checkpoint_path=checkpoint_path, trait=trait, delivery_kind=delivery_kind,
            selection_dir=selection_dir, stated=stated or Stated(), device=device))
    except _refusals() as exc:
        return {"error": str(exc)}


@tool()
def assess_reserved_regions(
    project: Path,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    k_cal: int = DEFAULT_K_CAL,
    k_test: int = DEFAULT_K_TEST,
    stated: Stated | None = None,
    device: str | None = None,
) -> dict:
    """Assess a checkpoint trained on one mosaic against that mosaic's own reserved calibration and
    holdout regions, for a count delivery of a trait, and record the result as an assessment.

    The checkpoint's run must have drawn a within-image split with
    ``data.split.calibration_ratio`` set; the regions are cut into ``k_cal`` and
    ``k_test`` buffered bands, predicted through the tiled pass at the split's own tile edge, and
    the count criterion is measured as for :func:`assess_checkpoint`. A raster published under the
    assessment is validated only when it is the same mosaic.

    Args:
        checkpoint_path: A checkpoint registered in this project, produced by a run of it.
        trait: The trait whose latest confirmed revision states the criterion's floors.
        delivery_kind: ``per_image_count`` or ``per_plant_count_aggregate``.
        k_cal / k_test: Bands per reserved region.
        stated: Execution values to state rather than derive (``execution.Stated``); the pass is
            always tiled.
        device: cuda / cpu (auto if omitted).
    """
    from tcip_mcp.assessment import assess_reserved_regions as assess

    try:
        return _answer(assess(
            project, checkpoint_path=checkpoint_path, trait=trait, delivery_kind=delivery_kind,
            k_cal=k_cal, k_test=k_test, stated=stated or Stated(), device=device))
    except _refusals() as exc:
        return {"error": str(exc)}


@tool()
def calibrate_physical_scale(
    project: Path, trait: str, selection_dir: str, reference_csv: str, unit: str,
    reference_subject: str,
) -> dict:
    """Derive a per-pixel physical scale on a selection's calibration side and check it against
    its holdout side, recorded as an assessment a dimensional delivery names.

    Each reference image carries exactly one ``reference_subject`` polygon or mask; the breeder's
    ``reference_csv`` (``image_stem, physical_extent, unit``) states each one's physical extent in
    ``unit`` and is retained with the assessment. The scale passes when the holdout's relative
    dispersion and the scale's deviation from the holdout mean are within the trait revision's
    ``scale_tolerance_frac``.

    Args:
        trait: The trait whose latest confirmed revision states the tolerance.
        selection_dir: A selection of the reference images with calibration and holdout sides.
        reference_csv: The breeder's physical-measurement CSV.
        unit: The linear length unit every reference is in.
        reference_subject: The subject the reference object is annotated as.
    """
    from tcip_mcp.assessment import assess_physical_scale

    try:
        return _answer(assess_physical_scale(
            project, trait=trait, selection_dir=selection_dir, reference_csv=reference_csv,
            unit=unit, reference_subject=reference_subject))
    except _refusals() as exc:
        return {"error": str(exc)}
