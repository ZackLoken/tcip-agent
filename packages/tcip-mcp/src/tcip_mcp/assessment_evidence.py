"""The evidence an assessment records and the names of the failures it reads from it.

Each criterion's evidence is its own record, built where the criterion is measured
(:mod:`tcip_mcp.pipelines.operating_point`, :mod:`tcip_mcp.assessment`); an assessment's
``criterion`` is one :class:`AssessmentEvidence` holding the members its producer measured, and
its ``failures`` are names of :data:`FailureName`.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from tcip_mcp.pipelines.data.selection import Side

type FailureName = Literal[
    "conf_censored",
    "insufficient_calibration_gt",
    "insufficient_holdout_gt",
    "insufficient_holdout_images",
    "count_bias_exceeds_tolerance",
    "count_bias_exceeds_tolerance_per_class",
    "insufficient_holdout_images_per_class",
    "holdout_missing_class",
    "localization_quality_floor_failed",
    "holdout_nothing_to_score",
    "count_error_dispersion_too_high",
    "insufficient_calibration_positive_evidence",
    "insufficient_holdout_positive_evidence",
    "insufficient_holdout_items",
    "positive_count_bias_exceeds_tolerance",
    "classifier_agreement_below_floor",
    "criterion_undefined",
    "criterion_not_finite",
    "criterion_below_floor",
    "holdout_shares_calibration_image",
    "reference_shares_training",
    "reference_shares_selection",
    "training_membership_unknown",
    "insufficient_calibration_references",
    "insufficient_holdout_references",
    "holdout_dispersion_exceeds_tolerance",
    "holdout_mean_outside_tolerance",
]
"""Why an assessment did not pass: each failure an assessment can record, by name."""

INSUFFICIENCY_FAILURES: frozenset[FailureName] = frozenset(
    {
        "insufficient_calibration_gt",
        "insufficient_holdout_gt",
        "insufficient_holdout_images",
        "insufficient_holdout_images_per_class",
        "insufficient_calibration_positive_evidence",
        "insufficient_holdout_positive_evidence",
        "insufficient_holdout_items",
        "insufficient_calibration_references",
        "insufficient_holdout_references",
    }
)
"""The failures saying a reference holds too little evidence to judge, not that it judged badly."""

_RECORD = ConfigDict(frozen=True, extra="forbid")


def effective_count_bias_tolerance(
    tolerance_frac: float, typical_count: float, n: int
) -> float:
    """Return the absolute per-image count-bias tolerance one scope is held to.

    A scope is pooled, or one class. The tolerance is ``tolerance_frac`` of the scope's typical
    per-image count, floored at ``1 / n``, one whole miscount spread across the ``n`` samples the
    scope's evidence rests on; ``0.0`` at ``n == 0``.
    """
    return max(tolerance_frac * typical_count, 1.0 / n) if n > 0 else 0.0


def bias_equivalence(
    mean: float,
    std: float,
    n: int,
    *,
    tolerance_frac: float,
    typical_count: float,
    equivalence_z: float,
) -> tuple[bool, float]:
    """Return whether a bias over ``n`` present samples is equivalent to zero, and the tolerance.

    Equivalent means the mean plus ``equivalence_z`` one-sided standard errors lies within the
    scope's tolerance (:func:`effective_count_bias_tolerance`). ``n == 0`` never passes.
    """
    tolerance = effective_count_bias_tolerance(
        tolerance_frac, typical_count, n
    )
    if n == 0:
        return False, tolerance
    return abs(mean) + equivalence_z * std / math.sqrt(
        n
    ) <= tolerance, tolerance


class CountEvidence(BaseModel):
    """The count criterion's evidence: the fitted conf and the holdout judged at it.

    ``conf`` is the conf the count objective picked on the calibration curve, by the picker
    ``conf_derived_from`` names; ``localization`` the criterion both sides matched under;
    ``holdout_at_conf`` and ``calibration_at_conf`` each side's curve row at the conf, each with
    its ``per_class`` rows. ``count_bias_tolerance_frac`` is the trait's fraction,
    ``typical_count`` the holdout's pooled typical per-image count and
    ``per_class_typical_count`` each class's, keyed as the curve rows key it; ``equivalence_z``
    is the one-sided standard errors a bias is held to (:meth:`scope_equivalence`).
    """

    model_config = _RECORD

    conf: float
    conf_derived_from: str
    calibration_curve: dict[str, Any]
    f1_max_conf: float
    localization: dict[str, Any]
    holdout_at_conf: dict[str, Any]
    calibration_at_conf: dict[str, Any]
    count_bias_tolerance_frac: float
    typical_count: float
    per_class_typical_count: dict[str, float]
    holdout_missing_classes: list[str]
    equivalence_z: float
    staged_conf_floor: float
    observed_min_score: float | None
    calibration_cap_saturated_frac: float
    holdout_cap_saturated_frac: float

    def scope_equivalence(
        self, class_id: str | None = None
    ) -> tuple[bool, float]:
        """Return whether one scope's holdout bias is equivalent to zero, and its tolerance.

        The scope is the pooled holdout for ``None``, else the class ``class_id`` keys in
        ``per_class_typical_count``; its bias statistics are its holdout curve row's
        present-scoped ones (:func:`bias_equivalence`).
        """
        row, typical = self.holdout_at_conf, self.typical_count
        if class_id is not None:
            row, typical = (
                row["per_class"][class_id],
                self.per_class_typical_count[class_id],
            )
        return bias_equivalence(
            row["count_bias_mean_present"],
            row["count_bias_std_present"],
            row["n_present"],
            tolerance_frac=self.count_bias_tolerance_frac,
            typical_count=typical,
            equivalence_z=self.equivalence_z,
        )


class ReservedRegionEvidence(BaseModel):
    """How a mosaic's reserved regions were cut into bands for a count criterion.

    ``block_scale_px`` is the buffer between bands and ``block_scale_source`` what it was derived
    from; ``k_cal`` and ``k_test`` the bands asked of each region, ``band_gt_counts`` each side's
    ground-truth count per band, and ``density_uniformity_flags`` the bands whose count is an
    outlier among its siblings.
    """

    model_config = _RECORD

    block_scale_px: int
    block_scale_source: str
    k_cal: int
    k_test: int
    band_gt_counts: dict[Side, dict[str, int]]
    density_uniformity_flags: list[str]


class ClassifierEvidence(BaseModel):
    """The classifier criterion's evidence over matched instances of the positive state.

    ``kappa`` is the holdout's chance-corrected agreement (``None`` when undefined) and
    ``kappa_floor`` the trait's floor; the positive-count bias is its present-scoped mean, standard
    deviation and image count, held to ``positive_count_bias_tolerance`` of the
    ``typical_positive_count``; ``n_calibration`` and ``n_holdout`` count each side's items.
    """

    model_config = _RECORD

    kappa: float | None
    kappa_floor: float
    positive_count_bias: float
    positive_count_bias_std: float
    positive_count_bias_images: int
    typical_positive_count: float
    positive_count_bias_tolerance: float
    n_calibration: int
    n_holdout: int


class ScalarEvidence(BaseModel):
    """The scalar criterion's evidence: ``criterion``'s ``score`` over ``n_holdout`` items.

    ``score`` is ``None`` when it is undefined or not finite, the failure naming which; ``floor``
    is the trait's.
    """

    model_config = _RECORD

    criterion: str
    score: float | None
    floor: float
    n_holdout: int


class ScaleEvidence(BaseModel):
    """A physical scale's evidence: each reference image's implied scale and the checks on them.

    ``implied_scales`` holds each side's implied scale per reference image and
    ``tolerance_frac`` the trait's; the calibration and holdout means, the holdout's relative
    dispersion and the scale's relative deviation from the holdout mean are ``None`` when either
    side held too few references to derive them.
    """

    model_config = _RECORD

    implied_scales: dict[str, dict[str, float]]
    tolerance_frac: float
    calibration_mean: float | None = None
    holdout_mean: float | None = None
    holdout_relative_dispersion: float | None = None
    relative_deviation: float | None = None


_MEASURED_TOGETHER: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"count"}),
        frozenset({"count", "reserved_region"}),
        frozenset({"count", "classifier"}),
        frozenset({"scalar"}),
        frozenset({"scale"}),
    }
)
"""The members one assessment measures together, one set per kind of assessment."""


class AssessmentEvidence(BaseModel):
    """The criteria one assessment measured, each member its criterion's evidence.

    A count delivery measures ``count``, with ``reserved_region`` over a mosaic's reserved
    regions or ``classifier`` for a positive state; a rank or value delivery ``scalar``; a
    physical scale ``scale``. Refuses any other combination of members.
    """

    model_config = _RECORD

    count: CountEvidence | None = None
    reserved_region: ReservedRegionEvidence | None = None
    classifier: ClassifierEvidence | None = None
    scalar: ScalarEvidence | None = None
    scale: ScaleEvidence | None = None

    @model_validator(mode="after")
    def _measured_together(self) -> Self:
        """Refuse a combination of members no assessment measures, naming the members."""
        present = frozenset(
            name
            for name in type(self).model_fields
            if getattr(self, name) is not None
        )
        if present not in _MEASURED_TOGETHER:
            raise ValueError(
                f"an assessment's criterion holds {sorted(present)}, which no assessment "
                f"measures together; one holds {sorted(sorted(s) for s in _MEASURED_TOGETHER)}"
            )
        return self


class RunOverlap(BaseModel):
    """What a reference shares with one side of its producing run: group keys, source digests."""

    model_config = _RECORD

    groups: list[str]
    source_digests: list[str]


class DisjointnessEvidence(BaseModel):
    """Whether a reference is held apart from itself and from the run that produced its model.

    ``holdout_shares_calibration`` lists the source digests its two sides share; ``training`` and
    ``selection`` what it shares with the producing run's training and validation sides, ``None``
    when the run's membership is unknown or the reference is not of a run's model.
    """

    model_config = _RECORD

    holdout_shares_calibration: list[str]
    training: RunOverlap | None = None
    selection: RunOverlap | None = None
