"""Tests of the count criterion an assessment judges a detection reference by.

Covered: the localization-quality and dispersion floors, reference sufficiency and the
equivalence test, pick-then-label from the trait's objective, exact-conf holdout evaluation,
cap-saturation evidence that adds no failure, and the population the bias is measured over.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch"
)  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.operating_point import (
    cap_saturated_frac,
    count_criterion,
)  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._dense_op_fixtures import (  # noqa: E402
    _box,
    dense_records,
    toy_records as _records,  # noqa: E402
)
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
)

ENTRY = fx.with_fields(
    fx.BUD_OPENING, count_bias_tolerance_frac=0.01, count_error_tolerance=100.0
)
"""A 1% relative bias tolerance, and a dispersion tolerance loose enough that no fixture here
reaches it unless a test states a stricter one."""


def _criterion(
    cal,
    hold,
    *,
    entry=ENTRY,
    floor=0.01,
    equivalence_z=EQUIVALENCE_Z,
    dispersion_quantile=DISPERSION_QUANTILE,
):
    """Run ``count_criterion`` staged at ``floor``; return ``(conf, evidence, failures)``."""
    return count_criterion(
        cal,
        hold,
        entry,
        staged_conf_floor=floor,
        staged_conf_floor_attribute_path=None,
        equivalence_z=equivalence_z,
        dispersion_quantile=dispersion_quantile,
    )


# ── Exact-conf holdout evaluation, not a nearest-neighbor snap ─────────────


def _cal_picks_conf_point_nine():
    """Return a 20-image dense calibration reference whose count-unbiased pick is exactly 0.9.

    Each image carries one low-conf spurious detection, scoring 0.05.
    """
    n, obj = 20, 80
    return dense_records(
        n_images=n,
        objects_per_image=obj,
        id_prefix="c",
        miss_pattern=[0] * n,
        fp_pattern=[1] * n,
        score=0.9,
        fp_score=0.05,
    )


def _nearest_neighbor_bias_comparator(holdout_records, tolerance, conf):
    """Return the curve entry nearest ``conf`` on the holdout's own auto-built grid."""
    from tcip_mcp.pipelines.training.evaluation import (
        derive_operating_point_curve,
    )

    sweep = derive_operating_point_curve(
        holdout_records,
        criterion={"kind": "center_match", "tolerance": tolerance},
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    return min(sweep["curve"], key=lambda c: abs(c["conf"] - conf))


def test_exact_conf_eval_catches_a_catastrophic_bias_a_nearest_neighbor_comparator_misses():
    """Exact-conf evaluation catches a catastrophic bias a nearest-neighbor comparator misses.

    Holdout detections all score 0.05 with zero false positives: evaluated exactly at 0.9 every
    detection is filtered out (bias -80/image), while the comparator's nearest grid point, 0.05,
    reads a perfect 0.0.
    """
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(
        n_images=n,
        objects_per_image=obj,
        id_prefix="hA",
        shift=5.0,
        miss_pattern=[0] * n,
        fp_pattern=[0] * n,
        score=0.05,
    )
    assert {d["score"] for r in hold for d in r["dt"]} == {0.05}

    old = _nearest_neighbor_bias_comparator(
        hold, 0.5 * gt_class_avg_size(hold), 0.9
    )
    assert old["conf"] == pytest.approx(0.05)
    assert old["count_bias_mean"] == pytest.approx(0.0)

    conf, evidence, failures = _criterion(cal, hold)
    hb = evidence["holdout_at_conf"]
    assert conf == hb["conf"] == pytest.approx(0.9)
    assert hb["count_bias_mean"] == pytest.approx(-80.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_exact_conf_eval_admits_a_reference_a_nearest_neighbor_comparator_would_fail():
    """Exact-conf evaluation admits a reference a nearest-neighbor comparator reads as biased.

    True matches score 0.99 and false positives 0.89: at exactly 0.9 the false positives drop and
    the bias is zero, while the comparator's nearest point, 0.89, reads +2.0/image.
    """
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(
        n_images=n,
        objects_per_image=obj,
        id_prefix="hB",
        shift=5.0,
        miss_pattern=[0] * n,
        fp_pattern=[2] * n,
        score=0.99,
        fp_score=0.89,
    )

    old = _nearest_neighbor_bias_comparator(
        hold, 0.5 * gt_class_avg_size(hold), 0.9
    )
    assert old["conf"] == pytest.approx(0.89)
    assert old["count_bias_mean"] == pytest.approx(2.0)

    _conf, evidence, failures = _criterion(cal, hold)
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert failures == []


# ── Dispersion and localization-quality floor ──────────────────────────────


def _tp_zero_bias_zero_records(
    id_prefix: str, *, n_images: int = 10, objects_per_image: int = 50
):
    """Return ``n_images`` records of ``objects_per_image`` GT boxes and as many detections.

    The GT boxes are 20 px, laid on a 40 px grid, and each detection, scored 0.9, sits 2000 px
    right of its GT box.
    """
    records = []
    cols = int(objects_per_image**0.5) + 2
    for i in range(n_images):
        gt, dt = [], []
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            cx, cy = 50.0 + col * 40, 50.0 + row * 40
            gt.append({"category_id": 0, "bbox": _box(cx, cy), "iscrowd": 0})
            dt.append(
                {"category_id": 0, "bbox": _box(cx + 2000, cy), "score": 0.9}
            )
        records.append(
            {
                "width": 4000,
                "height": 4000,
                "image_id": f"{id_prefix}_{i}",
                "gt": gt,
                "dt": dt,
                "count": len(dt),
                "cap": len(dt) + 1,
            }
        )
    return records


def test_tp_zero_bias_zero_holdout_fails_the_localization_floor():
    _conf, evidence, failures = _criterion(
        _tp_zero_bias_zero_records("c"),
        _tp_zero_bias_zero_records("h"),
        floor=0.0,
    )
    assert evidence["holdout_at_conf"]["tp"] == 0
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "localization_quality_floor_failed" in failures


def _one_bad_image():
    """Return ``(cal, hold, entry)`` over ten images, one dropping 10 objects and none elsewhere.

    The entry's dispersion tolerance is 0.5 and its bias tolerance loose.
    """
    n_images, objects_per_image = 10, 50
    cal = dense_records(
        n_images=n_images,
        objects_per_image=objects_per_image,
        id_prefix="c",
        miss_pattern=[0] * n_images,
        fp_pattern=[0] * n_images,
        score=0.9,
    )
    hold = dense_records(
        n_images=n_images,
        objects_per_image=objects_per_image,
        id_prefix="h",
        shift=5.0,
        miss_pattern=[0] * (n_images - 1) + [10],
        fp_pattern=[0] * n_images,
        score=0.9,
    )
    strict = fx.with_fields(
        ENTRY, count_error_tolerance=0.5, count_bias_tolerance_frac=1.0
    )
    return cal, hold, strict


def test_an_authored_dispersion_tolerance_fails_one_bad_image_among_many():
    """An authored dispersion tolerance of 0.5 fails one bad image among many.

    The ninetieth-percentile per-image error (1.0) exceeds the tolerance even though the mean bias
    is small.
    """
    cal, hold, strict = _one_bad_image()

    _conf, evidence, failures = _criterion(cal, hold, entry=strict)

    assert evidence["holdout_at_conf"][
        "count_error_at_quantile"
    ] == pytest.approx(1.0)
    assert "count_error_dispersion_too_high" in failures


def test_the_dispersion_check_reads_the_quantile_the_assessment_states():
    """The dispersion check reads the per-image error at the quantile the assessment states.

    The error is numpy's own quantile over the per-image errors, and the curve records that
    quantile: at the median the one bad image does not reach the tolerance.
    """
    import numpy as np

    cal, hold, strict = _one_bad_image()
    errors = [0] * 9 + [10]

    for quantile in (0.5, 0.95):
        _conf, evidence, failures = _criterion(
            cal, hold, entry=strict, dispersion_quantile=quantile
        )
        assert evidence["holdout_at_conf"][
            "count_error_at_quantile"
        ] == pytest.approx(float(np.quantile(errors, quantile)))
        assert evidence["calibration_curve"]["dispersion_quantile"] == quantile
        assert ("count_error_dispersion_too_high" in failures) is (
            quantile == 0.95
        )


# ── Reference sufficiency and the equivalence criterion ─────────────────────


def test_an_all_negative_calibration_or_holdout_fails_as_insufficient():
    real = _records("c")
    all_negative = [
        {
            "width": 400,
            "height": 400,
            "image_id": f"n_{i}",
            "gt": [],
            "dt": [],
            "count": 0,
            "cap": 1,
        }
        for i in range(3)
    ]

    assert (
        "insufficient_holdout_gt"
        in _criterion(real, all_negative, floor=0.0)[2]
    )
    assert (
        "insufficient_calibration_gt"
        in _criterion(all_negative, real, floor=0.0)[2]
    )


def test_single_image_holdout_fails_the_non_degeneracy_floor_alone():
    _conf, evidence, failures = _criterion(
        _records("c"), [_records("h", shift=3.0)[0]], floor=0.3
    )
    assert evidence["holdout_at_conf"]["n_images"] == 1
    assert "insufficient_holdout_images" in failures


def test_n_equals_2_holdout_with_real_variance_fails_equivalence_not_just_degeneracy():
    """Per-image biases [+1, -1] cancel in the mean; the mean+SE equivalence test still refuses."""
    _conf, evidence, failures = _criterion(
        _records("c"), _records("h", shift=3.0), floor=0.3
    )
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "insufficient_holdout_images" not in failures
    assert "count_bias_exceeds_tolerance" in failures


def test_padding_with_empty_records_cannot_dilute_the_equivalence_test():
    """Eight empty records beside a real n=2 disagreement change no equivalence verdict.

    The evidence records ten images with two present, and the padded run fails exactly as the
    unpadded one does, on the bias tolerance.
    """
    loosened = fx.with_fields(ENTRY, count_bias_tolerance_frac=1.0)
    hold_real = _records("h", shift=3.0)
    padding = [
        {
            "width": 400,
            "height": 400,
            "image_id": f"h_pad_{i}",
            "gt": [],
            "dt": [],
            "count": 0,
            "cap": 1,
        }
        for i in range(8)
    ]

    _c, padded, padded_failures = _criterion(
        _records("c"), hold_real + padding, entry=loosened, floor=0.3
    )
    _c, _bare, bare_failures = _criterion(
        _records("c"), hold_real, entry=loosened, floor=0.3
    )

    assert padded["holdout_at_conf"]["n_images"] == 10
    assert padded["holdout_at_conf"]["n_present"] == 2
    assert padded_failures == bare_failures
    assert "count_bias_exceeds_tolerance" in padded_failures


# ── Pick-then-label from the trait's objective ───────────────────────────────


@pytest.mark.parametrize("objective", ["detection_f1", "presence"])
def test_an_f1_objective_picks_f1_max_and_labels_it_accordingly(objective):
    """The detection F1 and presence objectives each pick the F1-max conf under its label."""
    conf, evidence, _failures = _criterion(
        _records(),
        _records("h", shift=3.0),
        entry=fx.with_fields(ENTRY, count_objective=objective),
        floor=0.0,
    )
    assert conf == pytest.approx(
        0.0
    )  # F1-max pick for this fixture (recall-max, low conf)
    assert evidence["conf_derived_from"] == "F1-max count curve"


# ── Cap-saturation evidence (never a failure) ─────────────────────────────────


def test_cap_saturated_frac_is_the_share_of_records_that_hit_the_cap():
    assert cap_saturated_frac(
        [{"count": 4, "cap": 4}, {"count": 3, "cap": 4}]
    ) == (pytest.approx(0.5))
    assert cap_saturated_frac([]) == 0.0


def test_cap_saturation_is_surfaced_but_never_fails_the_criterion():
    cal = [dict(r, cap=r["count"]) for r in _records("c")]
    _conf, evidence, failures = _criterion(
        cal, _records("h", shift=3.0), floor=0.3
    )
    assert evidence["calibration_cap_saturated_frac"] == pytest.approx(1.0)
    assert not any("cap" in f for f in failures)


# ── Genuine per-image dispersion in the admitting direction ─────────────────


def _rotating_noise_pattern(
    n: int, *, low: int = 1, high: int = 5, offset: int = 13
) -> tuple[list[int], list[int]]:
    """Return ``n`` per-image miss counts and the same counts rotated by ``offset`` as FP counts.

    Image ``i`` misses ``low + (3 * i + 1) % (high - low + 1)``.
    """
    span = high - low + 1
    miss = [low + (i * 3 + 1) % span for i in range(n)]
    fp = [miss[(i + offset) % n] for i in range(n)]
    return miss, fp


def _noisy(n: int, obj: int, miss, fp):
    cal = dense_records(
        n_images=n,
        objects_per_image=obj,
        id_prefix="c",
        miss_pattern=miss,
        fp_pattern=fp,
        score=0.9,
        fp_score=0.9,
    )
    hold = dense_records(
        n_images=n,
        objects_per_image=obj,
        id_prefix="h",
        shift=5.0,
        miss_pattern=miss,
        fp_pattern=fp,
        score=0.9,
        fp_score=0.9,
    )
    return _criterion(cal, hold)


def test_a_realistic_noisy_detector_passes_at_n_equals_40():
    miss, fp = _rotating_noise_pattern(40)
    _conf, evidence, failures = _noisy(40, 100, miss, fp)
    hb = evidence["holdout_at_conf"]
    assert hb["count_bias_mean"] == pytest.approx(0.0)
    assert hb["count_bias_std"] == pytest.approx(2.0254787341673333, abs=1e-6)
    assert hb["recall"] == pytest.approx(0.97, abs=1e-6)
    assert hb["precision"] == pytest.approx(0.97, abs=1e-6)
    assert failures == []


def test_the_equivalence_test_runs_at_the_multiplier_the_assessment_states():
    """The equivalence test runs at the standard-error multiplier the assessment states.

    The same noisy detector at n=40 (bias 0, standard error about 0.32 against a tolerance of 1.0)
    passes at 1.645 standard errors and fails at 4, and the evidence records the multiplier
    applied.
    """
    miss, fp = _rotating_noise_pattern(40)
    for equivalence_z, passes in ((1.645, True), (4.0, False)):
        cal, hold = (
            dense_records(
                n_images=40,
                objects_per_image=100,
                id_prefix=prefix,
                shift=shift,
                miss_pattern=miss,
                fp_pattern=fp,
                score=0.9,
                fp_score=0.9,
            )
            for prefix, shift in (("c", 0.0), ("h", 5.0))
        )
        _conf, evidence, failures = _criterion(
            cal, hold, equivalence_z=equivalence_z
        )
        assert evidence["equivalence_z"] == equivalence_z
        assert ("count_bias_exceeds_tolerance" not in failures) is passes


def test_the_same_noisy_detector_on_ten_images_fails_equivalence():
    miss, fp = _rotating_noise_pattern(40)
    _conf, evidence, failures = _noisy(10, 100, miss[:10], fp[:10])
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_the_same_noise_crosses_from_fail_to_pass_as_density_rises():
    """The same noise crosses from fail to pass as object density rises.

    At a fixed n=40 the identical noise fails at 30 objects/image (tolerance 0.30) and passes at
    100 (tolerance 1.00), its bias spread equal at both densities.
    """
    miss, fp = _rotating_noise_pattern(40)
    _c, sparse, sparse_failures = _noisy(40, 30, miss, fp)
    _c, dense, dense_failures = _noisy(40, 100, miss, fp)
    assert sparse["holdout_at_conf"]["count_bias_std"] == pytest.approx(
        dense["holdout_at_conf"]["count_bias_std"]
    )
    assert "count_bias_exceeds_tolerance" in sparse_failures
    assert dense_failures == []


def test_a_dense_realistic_reference_passes_with_every_condition_applied():
    """A dense realistic reference passes the count criterion.

    The reference has perfect recall and a varying handful of low-conf spurious detections per
    image, and the criterion reports no failure.
    """
    n = 24
    fp = [
        2,
        1,
        3,
        2,
        1,
        2,
        3,
        1,
        2,
        2,
        1,
        3,
        2,
        1,
        2,
        3,
        1,
        2,
        2,
        1,
        3,
        2,
        1,
        2,
    ]
    cal = dense_records(
        n_images=n,
        objects_per_image=100,
        id_prefix="c",
        miss_pattern=[0] * n,
        fp_pattern=fp,
        score=0.9,
        fp_score=0.05,
    )
    hold = dense_records(
        n_images=n,
        objects_per_image=100,
        id_prefix="h",
        shift=5.0,
        miss_pattern=[0] * n,
        fp_pattern=fp,
        score=0.9,
        fp_score=0.05,
    )
    assert _criterion(cal, hold)[2] == []


# ── The population: images that carry the thing being counted ────────────────


def _mixed_reference(
    id_prefix: str,
    *,
    n_loaded: int,
    n_negative: int,
    objects_per_image: int,
    fp_per_loaded: int,
    shift: float = 0.0,
) -> list[dict]:
    """Return ``dense_records`` of ``n_loaded`` images, then of ``n_negative`` images of no objects.

    The loaded images pass ``objects_per_image``, ``shift``, no misses and ``fp_per_loaded`` extra
    detections per image, all scored 0.9.
    """
    loaded = dense_records(
        n_images=n_loaded,
        objects_per_image=objects_per_image,
        id_prefix=f"{id_prefix}L",
        shift=shift,
        miss_pattern=[0] * n_loaded,
        fp_pattern=[fp_per_loaded] * n_loaded,
        score=0.9,
        fp_score=0.9,
    )
    negatives = dense_records(
        n_images=n_negative, objects_per_image=0, id_prefix=f"{id_prefix}N"
    )
    return loaded + negatives


def test_a_systematic_overcount_is_not_excused_by_the_negatives_beside_it():
    cal = _mixed_reference(
        "c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=2
    )
    hold = _mixed_reference(
        "h",
        n_loaded=10,
        n_negative=40,
        objects_per_image=100,
        fp_per_loaded=2,
        shift=5.0,
    )

    _conf, evidence, failures = _criterion(cal, hold)

    hb = evidence["holdout_at_conf"]
    assert hb["n_images"] == 50 and hb["n_present"] == 10
    assert hb["count_bias_mean_present"] == pytest.approx(2.0)
    assert evidence["pooled_typical_count"] == pytest.approx(100.0)
    assert evidence["pooled_count_bias_tolerance"] == pytest.approx(1.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_the_same_overcount_without_the_negatives_fails_identically():
    cal = _mixed_reference(
        "c", n_loaded=10, n_negative=0, objects_per_image=100, fp_per_loaded=2
    )
    hold = _mixed_reference(
        "h",
        n_loaded=10,
        n_negative=0,
        objects_per_image=100,
        fp_per_loaded=2,
        shift=5.0,
    )

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"][
        "count_bias_mean_present"
    ] == pytest.approx(2.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_a_clean_detector_still_passes_on_a_reference_full_of_negatives():
    cal = _mixed_reference(
        "c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=0
    )
    hold = _mixed_reference(
        "h",
        n_loaded=10,
        n_negative=40,
        objects_per_image=100,
        fp_per_loaded=0,
        shift=5.0,
    )

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"]["n_present"] == 10
    assert failures == []


def test_a_negative_the_detector_hallucinates_on_is_evidence_not_a_discard():
    def _hallucinating(id_prefix, shift=0.0):
        loaded = dense_records(
            n_images=10,
            objects_per_image=100,
            id_prefix=f"{id_prefix}L",
            shift=shift,
            miss_pattern=[0] * 10,
            fp_pattern=[0] * 10,
            score=0.9,
            fp_score=0.9,
        )
        noisy = dense_records(
            n_images=10,
            objects_per_image=0,
            id_prefix=f"{id_prefix}N",
            miss_pattern=[0] * 10,
            fp_pattern=[3] * 10,
            score=0.9,
            fp_score=0.9,
        )
        return loaded + noisy

    _conf, evidence, failures = _criterion(
        _hallucinating("c"), _hallucinating("h", shift=5.0)
    )

    hb = evidence["holdout_at_conf"]
    assert hb["n_present"] == 20
    assert hb["count_bias_mean_present"] == pytest.approx(
        1.5
    )  # (10 * 0 + 10 * 3) / 20
    assert "count_bias_exceeds_tolerance" in failures


def test_a_holdout_carrying_one_loaded_image_is_not_enough_evidence():
    cal = _mixed_reference(
        "c", n_loaded=4, n_negative=0, objects_per_image=100, fp_per_loaded=0
    )
    hold = _mixed_reference(
        "h",
        n_loaded=1,
        n_negative=99,
        objects_per_image=100,
        fp_per_loaded=0,
        shift=5.0,
    )

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"]["n_images"] == 100
    assert "insufficient_holdout_images" in failures
