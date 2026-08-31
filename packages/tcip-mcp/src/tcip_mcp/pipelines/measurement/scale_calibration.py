"""Deriving and validating a physical per-pixel scale against real physical measurements.

A physical scale is never agent-guessed and never accepted as a bare candidate: it is derived from
a breeder's own reference measurements (a scale bar, a reference disc, an organ measured with
calipers) and tested against a held-out half of the same references it was not derived from, the
same locked calibration/holdout discipline the scalar-head calibrators use
(``tcip_mcp.pipelines.data.splits.resolve_locked_cal_holdout_split``). A candidate derived from and
then tested against the same references would pass by construction; that is exactly what this
module refuses to do.

The one function here, :func:`resolve_physical_scale`, is this document's registered
``_DOCUMENT_RESOLVERS["resolve_scale"]`` entry (see ``pipelines.resolution``); its caller is
``tools.scale_tools.calibrate_physical_scale``, which reads the references (an annotated reference
object's pixel extent, a breeder-authored CSV's physical extent) and runs this gate over them.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_physical_scale(
    *,
    unit: str,
    references: Mapping[str, Mapping[str, Any]],
    tolerance_frac: float | None,
    dataset_root: str | Path,
    identity_hash: str,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
    capture_id: str | None = None,
) -> dict:
    """Derive a per-pixel scale from a locked calibration half of ``references`` and validate it
    against the holdout half, never against the references it was derived from.

    ``references`` is ``{stem: {"physical_extent": float, "unit": str, "pixel_extent": float}}``,
    one entry per reference image: ``physical_extent``/``unit`` are the breeder's own physical
    measurement and its unit, and ``pixel_extent`` is the reference object's own principal-axis
    extent in that image (``mask_geometry.principal_axis_extent_of_points``, never a bounding box's
    long side). Each reference implies a scale ``physical_extent / pixel_extent``.

    ``tolerance_frac`` is the trait-authored ``TraitSpec.scale_tolerance_frac``. ``None`` (not yet
    authored) refuses: unlike the count gate's own tolerance, there is no platform default
    fallback for how much reference disagreement a physical-scale claim may carry, that is a
    measurement decision only the domain expert can make.

    The candidate is the mean of the calibration half's implied scales. It clears when its relative
    deviation from the holdout mean is within the authored ``tolerance_frac``, and when the holdout's
    own relative dispersion (sample standard deviation of its implied scales over their mean) is
    itself within ``tolerance_frac``: a reference set that disagrees with itself by more than the
    tolerance cannot validate to it. The dispersion check already bounds the holdout's own relative
    standard error at ``tolerance_frac / sqrt(holdout count)``, strictly tighter than
    ``tolerance_frac`` for any holdout of two or more, so a second floor derived from that same
    standard error would never bind once the dispersion check passes; the relative standard error is
    still recorded in ``sweep_data`` as a diagnostic. Either half with fewer than two references
    refuses, naming which.

    Returns ``{validated_against, passed, value, unit, failures, sweep_data}``, the same shape the
    scalar-head resolvers return (never a ``ResolvedParam``, which cannot carry a ``failures``
    list): ``value`` is the derived scale on a pass, ``None`` otherwise; ``sweep_data`` carries every
    implied scale by half, both means, the holdout dispersion, the relative standard error and the
    split identity, so a failed calibration is diagnosable from the returned dict alone. Never raises
    for an evidence-quality failure (too few references, a disagreeing reference set, an unauthored
    tolerance); raises only for a caller-composition error (a non-string ``unit``, a unit that is not
    a linear length unit crops.yml declares, a non-mapping ``references``).
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_PHYSICAL_MEASUREMENT
    from tcip_mcp.traits import crops_length_units

    if not isinstance(unit, str) or not unit:
        raise ValueError("resolve_physical_scale requires a non-empty unit string")
    length_units = crops_length_units()
    if unit not in length_units:
        raise ValueError(
            f"resolve_physical_scale requires unit to be a linear length unit crops.yml declares "
            f"({sorted(length_units)}), got {unit!r}: a per-pixel scale is a length-per-pixel "
            "quantity, and a mass or other non-length unit is a contradiction."
        )
    if not isinstance(references, Mapping):
        raise ValueError(
            "resolve_physical_scale requires references as a stem -> measurement mapping"
        )

    if tolerance_frac is None:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False, "value": None, "unit": unit,
            "failures": ["scale_tolerance_not_authored"],
            "sweep_data": {"note": "TraitSpec.scale_tolerance_frac is not authored for this "
                                   "trait; a physical-scale gate has no platform default "
                                   "fallback, the domain expert must author it."},
        }
    if not references:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False, "value": None, "unit": unit,
            "failures": ["empty_reference_set"], "sweep_data": {},
        }

    malformed = sorted(
        stem for stem, r in references.items()
        if not isinstance(r, Mapping)
        or not isinstance(r.get("pixel_extent"), (int, float)) or r.get("pixel_extent", 0) <= 0
        or not isinstance(r.get("physical_extent"), (int, float))
        or not isinstance(r.get("unit"), str)
    )
    if malformed:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False, "value": None, "unit": unit,
            "failures": ["malformed_reference_set"], "sweep_data": {"malformed_stems": malformed},
        }

    off_unit = sorted(stem for stem, r in references.items() if r["unit"] != unit)
    if off_unit:
        return {
            "validated_against": VALIDATED_FALSE, "passed": False, "value": None, "unit": unit,
            "failures": ["unit_disagreement"], "sweep_data": {"off_unit_stems": off_unit},
        }

    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root,
        resolve_locked_cal_holdout_split,
    )

    stems = sorted(references)
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=identity_hash, scope_root=cal_holdout_scope_root(dataset_root),
        group_by=group_by, group_key_map=group_key_map, seed=seed, holdout_ratio=holdout_ratio,
    )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    def _implied(stem_list: list[str]) -> dict[str, float]:
        return {s: references[s]["physical_extent"] / references[s]["pixel_extent"]
               for s in stem_list}

    cal_implied = _implied(cal_stems)
    hold_implied = _implied(hold_stems)
    sweep_data: dict[str, Any] = {
        "calibration_implied_scales": cal_implied, "holdout_implied_scales": hold_implied,
        "split_identity": identity_hash, "capture_id": capture_id,
    }

    failures: list[str] = []
    if len(cal_implied) < 2:
        failures.append("insufficient_calibration_references")
    if len(hold_implied) < 2:
        failures.append("insufficient_holdout_references")
    if failures:
        return {"validated_against": VALIDATED_FALSE, "passed": False, "value": None, "unit": unit,
                "failures": failures, "sweep_data": sweep_data}

    candidate = statistics.mean(cal_implied.values())
    hold_values = list(hold_implied.values())
    hold_mean = statistics.mean(hold_values)
    hold_std = statistics.stdev(hold_values)
    hold_relative_dispersion = abs(hold_std / hold_mean) if hold_mean else math.inf
    relative_standard_error = hold_relative_dispersion / math.sqrt(len(hold_values))
    relative_deviation = abs(candidate - hold_mean) / abs(hold_mean) if hold_mean else math.inf

    if hold_relative_dispersion > tolerance_frac:
        failures.append("holdout_dispersion_exceeds_tolerance")
    if relative_deviation > tolerance_frac:
        failures.append("holdout_mean_outside_tolerance")

    sweep_data.update({
        "calibration_mean": candidate,
        "holdout_mean": hold_mean,
        "holdout_relative_dispersion": hold_relative_dispersion,
        "tolerance_frac": tolerance_frac,
        "relative_standard_error": relative_standard_error,
        "relative_deviation": relative_deviation,
    })
    passed = not failures
    return {
        "validated_against": VALIDATED_PHYSICAL_MEASUREMENT if passed else VALIDATED_FALSE,
        "passed": passed, "value": candidate if passed else None, "unit": unit,
        "failures": failures, "sweep_data": sweep_data,
    }

