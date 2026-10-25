"""Assessment tools for a checkpoint and for a physical scale.

A checkpoint is assessed against a held-out reference or against a mosaic's own reserved regions,
and a physical scale against a breeder's reference measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.server import tool

if TYPE_CHECKING:
    from tcip_mcp.assessment import Assessment

Quantile = Annotated[float, Field(ge=0, le=1)]
"""A share of held-out images in [0, 1]: 0 takes the smallest per-image error, 1 the largest."""


def _refusals() -> tuple[type[Exception], ...]:
    """Return the exception types an assessment refuses with.

    They are ``ValueError``, ``UnreadableLabelDocumentError``, ``UnregisteredCheckpointError``,
    ``TraitUnknownError`` and ``FileNotFoundError``.
    """
    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tcip_mcp.model_registry import UnregisteredCheckpointError
    from tcip_mcp.traits import TraitUnknownError

    return (
        ValueError,
        UnreadableLabelDocumentError,
        UnregisteredCheckpointError,
        TraitUnknownError,
        FileNotFoundError,
    )


def _answer(assessment: Assessment) -> dict[str, Any]:
    """Return an assessment as its record, its reference's sample list replaced by its count."""
    from tcip_mcp.assessment import ASSESSMENT_RECORD

    answer = ASSESSMENT_RECORD.dump_python(assessment, mode="json")
    samples = answer["reference"].pop("samples")
    return {
        **answer,
        "reference": {**answer["reference"], "n_samples": len(samples)},
    }


@tool()
def assess_checkpoint(
    project: Path,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    selection_dir: str,
    equivalence_z: float,
    dispersion_quantile: Quantile,
    rationale: str,
    stated: Stated,
    device: str | None = None,
    *,
    actor: str | None,
) -> dict:
    """Assess a checkpoint on a selection's calibration and holdout sides and record it.

    The assessment is ``actor``'s, for one delivered kind of a trait. The operating point is
    fitted on the calibration side (the trait's count objective picks the conf; its ground truth
    derives the object density each frame's detection cap scales by, and an unstated merge
    threshold) and the criterion the kind rests on is measured over the holdout side: the
    held-out count bias, pooled and per class, localization and dispersion for a count; that plus
    the classifier's agreement over matched instances for ``state_crossing_dates``; the scalar
    skill for an ordinal or regression aggregate, a regression scored by the revision's own
    ``regression_criterion``. The reference must share no group or image with the producing run's
    training or selection side.

    Args:
        checkpoint_path: A checkpoint registered in this project.
        trait: The trait whose latest confirmed revision states the criterion's floors.
        delivery_kind: The delivered kind assessed (``per_image_count``,
            ``per_plant_count_aggregate``, ``state_crossing_dates``,
            ``per_plant_ordinal_aggregate``, ``per_plant_regression_aggregate``).
        selection_dir: A selection drawn with calibration and holdout sides (``draw_splits``).
        equivalence_z: The one-sided standard-error multiplier of a count's bias equivalence
            test (1.645 is a one-sided 95 percent test).
        dispersion_quantile: The share of held-out images whose per-image count error
            ``count_error_tolerance`` bounds (:data:`Quantile`: 0 holds the smallest error to it,
            1 the largest).
        rationale: Why the assessment states these settings, recorded with it.
        stated: Execution values to state rather than derive (``execution.Stated``), as
            ``run_inference`` takes them: a detector's ``conf``, the one its predictions are
            collected at and the floor its fitted conf is censored at, and an instance
            segmentation checkpoint's ``masks`` among them.
        device: cuda / cpu (auto if omitted).

    """
    from tcip_mcp.assessment import assess

    try:
        return _answer(
            assess(
                project,
                checkpoint_path=checkpoint_path,
                trait=trait,
                delivery_kind=delivery_kind,
                selection_dir=selection_dir,
                stated=stated,
                equivalence_z=equivalence_z,
                dispersion_quantile=dispersion_quantile,
                rationale=rationale,
                device=device,
                actor=actor,
            )
        )
    except _refusals() as exc:
        return {"error": str(exc)}


@tool()
def assess_reserved_regions(
    project: Path,
    checkpoint_path: str,
    trait: str,
    delivery_kind: str,
    k_cal: int,
    k_test: int,
    equivalence_z: float,
    dispersion_quantile: Quantile,
    rationale: str,
    stated: Stated,
    device: str | None = None,
    *,
    actor: str | None,
) -> dict:
    """Assess a checkpoint against its own mosaic's reserved regions and record it by ``actor``.

    The checkpoint is one trained on one mosaic, assessed against that mosaic's own reserved
    calibration and holdout regions for a count delivery of a trait. The checkpoint's run must
    have drawn a within-image split with ``data.split.calibration_ratio`` set; the regions are cut
    into ``k_cal`` and ``k_test`` buffered bands, predicted through the tiled pass at the split's
    own tile edge, and the count criterion is measured as for :func:`assess_checkpoint`.

    Args:
        checkpoint_path: A checkpoint registered in this project, produced by a run of it.
        trait: The trait whose latest confirmed revision states the criterion's floors.
        delivery_kind: ``per_image_count`` or ``per_plant_count_aggregate``.
        k_cal / k_test: Bands per reserved region, recorded with the assessment; the count
            criterion needs at least two bands carrying ground truth per side.
        equivalence_z / dispersion_quantile: As :func:`assess_checkpoint` takes them.
        rationale: Why the assessment states these settings, recorded with it.
        stated: Execution values to state rather than derive (``execution.Stated``), its
            ``conf`` as :func:`assess_checkpoint` takes it; the pass is always tiled.
        device: cuda / cpu (auto if omitted).

    """
    from tcip_mcp.assessment import assess_reserved_regions as assess

    try:
        return _answer(
            assess(
                project,
                checkpoint_path=checkpoint_path,
                trait=trait,
                delivery_kind=delivery_kind,
                k_cal=k_cal,
                k_test=k_test,
                stated=stated,
                equivalence_z=equivalence_z,
                dispersion_quantile=dispersion_quantile,
                rationale=rationale,
                device=device,
                actor=actor,
            )
        )
    except _refusals() as exc:
        return {"error": str(exc)}


@tool()
def calibrate_physical_scale(
    project: Path,
    trait: str,
    selection_dir: str,
    reference_csv: str,
    unit: str,
    reference_subject: str,
    *,
    actor: str | None,
) -> dict:
    """Derive a per-pixel physical scale on a selection's calibration side and check it on holdout.

    The result is recorded as an assessment by ``actor``. Each reference image carries exactly one
    ``reference_subject`` polygon or mask; the breeder's ``reference_csv`` (``image_stem,
    physical_extent, unit``) states each one's physical extent in ``unit`` and is retained with
    the assessment. The scale passes when the holdout's relative
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
        return _answer(
            assess_physical_scale(
                project,
                trait=trait,
                selection_dir=selection_dir,
                reference_csv=reference_csv,
                unit=unit,
                reference_subject=reference_subject,
                actor=actor,
            )
        )
    except _refusals() as exc:
        return {"error": str(exc)}
