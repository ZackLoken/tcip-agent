"""The conditions of the count criterion an assessment judges a detection reference by: the
localization-quality and dispersion floors, reference sufficiency and the equivalence test,
pick-then-label from the trait's objective, exact-conf holdout evaluation (not a nearest-neighbor
snap), cap-saturation evidence (never a failure), and the population the bias is measured over.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests import _trait_fixtures as fx  # noqa: E402
from tests._dense_op_fixtures import _box, dense_records  # noqa: E402
from tests._dense_op_fixtures import toy_records as _records  # noqa: E402
from tcip_mcp.pipelines.operating_point import cap_saturated_frac, count_criterion  # noqa: E402
from tcip_mcp.traits import DETECTION_F1, PRESENCE  # noqa: E402

ENTRY = fx.with_fields(fx.BUD_OPENING, count_bias_tolerance_frac=0.01,
                       count_error_tolerance=100.0)
"""A 1% relative bias tolerance, and a dispersion tolerance loose enough that no fixture here
reaches it unless a test states a stricter one."""


def _criterion(cal, hold, *, entry=ENTRY, floor=0.01):
    """``count_criterion`` staged at ``floor``: ``(conf, evidence, failures)``."""
    return count_criterion(cal, hold, entry, staged_conf_floor=floor,
                           staged_conf_floor_attribute_path=None)


# ── Exact-conf holdout evaluation, not a nearest-neighbor snap ─────────────

def _cal_picks_conf_point_nine():
    """20-image dense calibration reference whose count-unbiased pick is exactly 0.9 (one low-conf
    spurious detection per image, filtered out once conf crosses its 0.05 score). The holdouts
    below carry a sparse score set that excludes 0.9, so the nearest point of the holdout's own
    grid is a genuinely different threshold."""
    n, obj = 20, 80
    return dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                         miss_pattern=[0] * n, fp_pattern=[1] * n, score=0.9, fp_score=0.05)


def _nearest_neighbor_bias_comparator(holdout_records, tolerance, conf):
    """The curve entry nearest ``conf`` on the holdout's own auto-built grid, not an exact
    evaluation at ``conf`` itself."""
    from tcip_mcp.pipelines.training.evaluation import derive_operating_point_curve

    sweep = derive_operating_point_curve(
        holdout_records, criterion={"kind": "center_match", "tolerance": tolerance})
    return min(sweep["curve"], key=lambda c: abs(c["conf"] - conf))


def test_exact_conf_eval_catches_a_catastrophic_bias_a_nearest_neighbor_comparator_misses():
    """Holdout detections all score 0.05 with zero false positives: evaluated exactly at 0.9 every
    detection is filtered out (bias -80/image), while the comparator's nearest grid point, 0.05,
    reads a perfect 0.0."""
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="hA", shift=5.0,
                         miss_pattern=[0] * n, fp_pattern=[0] * n, score=0.05)
    assert {d["score"] for r in hold for d in r["dt"]} == {0.05}

    old = _nearest_neighbor_bias_comparator(hold, 0.5 * gt_class_avg_size(hold), 0.9)
    assert old["conf"] == pytest.approx(0.05)
    assert old["count_bias_mean"] == pytest.approx(0.0)

    conf, evidence, failures = _criterion(cal, hold)
    hb = evidence["holdout_at_conf"]
    assert conf == hb["conf"] == pytest.approx(0.9)
    assert hb["count_bias_mean"] == pytest.approx(-80.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_exact_conf_eval_admits_a_reference_a_nearest_neighbor_comparator_would_fail():
    """True matches score 0.99 and false positives 0.89: at exactly 0.9 the false positives drop
    and the bias is zero, while the comparator's nearest point, 0.89, reads +2.0/image."""
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="hB", shift=5.0,
                         miss_pattern=[0] * n, fp_pattern=[2] * n, score=0.99, fp_score=0.89)

    old = _nearest_neighbor_bias_comparator(hold, 0.5 * gt_class_avg_size(hold), 0.9)
    assert old["conf"] == pytest.approx(0.89)
    assert old["count_bias_mean"] == pytest.approx(2.0)

    _conf, evidence, failures = _criterion(cal, hold)
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert failures == []


# ── Dispersion and localization-quality floor ──────────────────────────────

def _tp_zero_bias_zero_records(id_prefix: str, *, n_images: int = 10, objects_per_image: int = 50):
    """Every image: N GT, N detections, every detection far outside the center-match tolerance, so
    count bias is exactly 0 (fp == fn == N) while not one detection matches (tp=0)."""
    records = []
    cols = int(objects_per_image**0.5) + 2
    for i in range(n_images):
        gt, dt = [], []
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            cx, cy = 50.0 + col * 40, 50.0 + row * 40
            gt.append({"category_id": 0, "bbox": _box(cx, cy), "iscrowd": 0})
            dt.append({"category_id": 0, "bbox": _box(cx + 100, cy), "score": 0.9})
        records.append({"width": 4000, "height": 4000, "image_id": f"{id_prefix}_{i}",
                        "gt": gt, "dt": dt, "cap_hit": False})
    return records


def test_tp_zero_bias_zero_holdout_fails_the_localization_floor():
    _conf, evidence, failures = _criterion(
        _tp_zero_bias_zero_records("c"), _tp_zero_bias_zero_records("h"), floor=0.0)
    assert evidence["holdout_at_conf"]["tp"] == 0
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "localization_quality_floor_failed" in failures


def test_an_authored_dispersion_tolerance_fails_one_bad_image_among_many():
    """One image drops 10 objects, none elsewhere: the p90 per-image error (1.0) exceeds an
    authored 0.5 even though the mean bias is small."""
    n_images, objects_per_image = 10, 50
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=[0] * n_images, fp_pattern=[0] * n_images, score=0.9)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=[0] * (n_images - 1) + [10],
                         fp_pattern=[0] * n_images, score=0.9)
    strict = fx.with_fields(ENTRY, count_error_tolerance=0.5, count_bias_tolerance_frac=1.0)

    _conf, evidence, failures = _criterion(cal, hold, entry=strict)

    assert evidence["holdout_at_conf"]["count_error_p90"] == pytest.approx(1.0)
    assert "count_error_dispersion_too_high" in failures


# ── Reference sufficiency and the equivalence criterion ─────────────────────

def test_an_all_negative_calibration_or_holdout_fails_as_insufficient():
    real = _records("c")
    all_negative = [{"width": 400, "height": 400, "image_id": f"n_{i}", "gt": [], "dt": [],
                     "cap_hit": False} for i in range(3)]

    assert "insufficient_holdout_gt" in _criterion(real, all_negative, floor=0.0)[2]
    assert "insufficient_calibration_gt" in _criterion(all_negative, real, floor=0.0)[2]


def test_single_image_holdout_fails_the_non_degeneracy_floor_alone():
    _conf, evidence, failures = _criterion(_records("c"), [_records("h", shift=3.0)[0]], floor=0.3)
    assert evidence["holdout_at_conf"]["n_images"] == 1
    assert "insufficient_holdout_images" in failures


def test_n_equals_2_holdout_with_real_variance_fails_equivalence_not_just_degeneracy():
    """Per-image biases [+1, -1] cancel in the mean; the mean+SE equivalence test still refuses."""
    _conf, evidence, failures = _criterion(_records("c"), _records("h", shift=3.0), floor=0.3)
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "insufficient_holdout_images" not in failures
    assert "count_bias_exceeds_tolerance" in failures


def test_padding_with_empty_records_cannot_dilute_the_equivalence_test():
    """Eight empty records beside a real n=2 disagreement change no verdict: the test measures over
    the images that carry something, and the statistics are never recomputed on a shrunk sample."""
    loosened = fx.with_fields(ENTRY, count_bias_tolerance_frac=1.0)
    hold_real = _records("h", shift=3.0)
    padding = [{"width": 400, "height": 400, "image_id": f"h_pad_{i}", "gt": [], "dt": [],
                "cap_hit": False} for i in range(8)]

    _c, padded, padded_failures = _criterion(_records("c"), hold_real + padding, entry=loosened,
                                             floor=0.3)
    _c, _bare, bare_failures = _criterion(_records("c"), hold_real, entry=loosened, floor=0.3)

    assert padded["holdout_at_conf"]["n_images"] == 10
    assert padded["holdout_at_conf"]["n_present"] == 2
    assert padded_failures == bare_failures
    assert "count_bias_exceeds_tolerance" in padded_failures


# ── Pick-then-label from the trait's objective ───────────────────────────────

@pytest.mark.parametrize("objective", [DETECTION_F1, PRESENCE])
def test_an_f1_objective_picks_f1_max_and_labels_it_accordingly(objective):
    """Presence deliberately shares the F1-max picker and label with detection F1."""
    conf, evidence, _failures = _criterion(
        _records(), _records("h", shift=3.0),
        entry=fx.with_fields(ENTRY, count_objective=objective), floor=0.0)
    assert conf == pytest.approx(0.0)  # F1-max pick for this fixture (recall-max, low conf)
    assert evidence["conf_derived_from"] == "F1-max count curve"


# ── Cap-saturation evidence (never a failure) ─────────────────────────────────


def test_cap_saturated_frac_is_the_share_of_records_that_hit_the_cap():
    assert cap_saturated_frac([{"cap_hit": True}, {"cap_hit": False}]) == pytest.approx(0.5)
    assert cap_saturated_frac([]) == 0.0


def test_cap_saturation_is_surfaced_but_never_fails_the_criterion():
    cal = [dict(r, cap_hit=True) for r in _records("c")]
    _conf, evidence, failures = _criterion(cal, _records("h", shift=3.0), floor=0.3)
    assert evidence["calibration_cap_saturated_frac"] == pytest.approx(1.0)
    assert not any("cap" in f for f in failures)


# ── Genuine per-image dispersion in the admitting direction ─────────────────

def _rotating_noise_pattern(n: int, *, low: int = 1, high: int = 5, offset: int = 13
                            ) -> tuple[list[int], list[int]]:
    """``n`` per-image miss counts cycling through ``[low, high]``, and a false-positive pattern
    that is a rotation of the same values at the same score, so the mean bias is exactly zero while
    real per-image dispersion remains."""
    span = high - low + 1
    miss = [low + (i * 3 + 1) % span for i in range(n)]
    fp = [miss[(i + offset) % n] for i in range(n)]
    return miss, fp


def _noisy(n: int, obj: int, miss, fp):
    cal = dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="h", shift=5.0,
                         miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
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


def test_the_same_noisy_detector_on_ten_images_fails_equivalence():
    miss, fp = _rotating_noise_pattern(40)
    _conf, evidence, failures = _noisy(10, 100, miss[:10], fp[:10])
    assert evidence["holdout_at_conf"]["count_bias_mean"] == pytest.approx(0.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_the_same_noise_crosses_from_fail_to_pass_as_density_rises():
    """At a fixed n=40 the identical noise fails at 30 objects/image (tolerance 0.30) and passes at
    100 (tolerance 1.00): the disclosed consequence of a relative tolerance."""
    miss, fp = _rotating_noise_pattern(40)
    _c, sparse, sparse_failures = _noisy(40, 30, miss, fp)
    _c, dense, dense_failures = _noisy(40, 100, miss, fp)
    assert sparse["holdout_at_conf"]["count_bias_std"] == pytest.approx(
        dense["holdout_at_conf"]["count_bias_std"])
    assert "count_bias_exceeds_tolerance" in sparse_failures
    assert dense_failures == []


def test_a_dense_realistic_reference_passes_with_every_condition_applied():
    """Perfect recall and a varying handful of low-conf spurious detections per image, all filtered
    at the count-unbiased conf."""
    n = 24
    fp = [2, 1, 3, 2, 1, 2, 3, 1, 2, 2, 1, 3, 2, 1, 2, 3, 1, 2, 2, 1, 3, 2, 1, 2]
    cal = dense_records(n_images=n, objects_per_image=100, id_prefix="c", miss_pattern=[0] * n,
                        fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=n, objects_per_image=100, id_prefix="h", shift=5.0,
                         miss_pattern=[0] * n, fp_pattern=fp, score=0.9, fp_score=0.05)
    assert _criterion(cal, hold)[2] == []


# ── The population: images that carry the thing being counted ────────────────

def _mixed_reference(id_prefix: str, *, n_loaded: int, n_negative: int, objects_per_image: int,
                     fp_per_loaded: int, shift: float = 0.0) -> list[dict]:
    """``n_loaded`` dense images carrying the same systematic over-count plus ``n_negative``
    confirmed negatives (no GT, no detections)."""
    loaded = dense_records(n_images=n_loaded, objects_per_image=objects_per_image,
                           id_prefix=f"{id_prefix}L", shift=shift,
                           miss_pattern=[0] * n_loaded, fp_pattern=[fp_per_loaded] * n_loaded,
                           score=0.9, fp_score=0.9)
    negatives = dense_records(n_images=n_negative, objects_per_image=0, id_prefix=f"{id_prefix}N")
    return loaded + negatives


def test_a_systematic_overcount_is_not_excused_by_the_negatives_beside_it():
    cal = _mixed_reference("c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=2)
    hold = _mixed_reference("h", n_loaded=10, n_negative=40, objects_per_image=100,
                            fp_per_loaded=2, shift=5.0)

    _conf, evidence, failures = _criterion(cal, hold)

    hb = evidence["holdout_at_conf"]
    assert hb["n_images"] == 50 and hb["n_present"] == 10
    assert hb["count_bias_mean_present"] == pytest.approx(2.0)
    assert evidence["pooled_typical_count"] == pytest.approx(100.0)
    assert evidence["pooled_count_bias_tolerance"] == pytest.approx(1.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_the_same_overcount_without_the_negatives_fails_identically():
    cal = _mixed_reference("c", n_loaded=10, n_negative=0, objects_per_image=100, fp_per_loaded=2)
    hold = _mixed_reference("h", n_loaded=10, n_negative=0, objects_per_image=100,
                            fp_per_loaded=2, shift=5.0)

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"]["count_bias_mean_present"] == pytest.approx(2.0)
    assert "count_bias_exceeds_tolerance" in failures


def test_a_clean_detector_still_passes_on_a_reference_full_of_negatives():
    cal = _mixed_reference("c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=0)
    hold = _mixed_reference("h", n_loaded=10, n_negative=40, objects_per_image=100,
                            fp_per_loaded=0, shift=5.0)

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"]["n_present"] == 10
    assert failures == []


def test_a_negative_the_detector_hallucinates_on_is_evidence_not_a_discard():
    def _hallucinating(id_prefix, shift=0.0):
        loaded = dense_records(n_images=10, objects_per_image=100, id_prefix=f"{id_prefix}L",
                               shift=shift, miss_pattern=[0] * 10, fp_pattern=[0] * 10,
                               score=0.9, fp_score=0.9)
        noisy = dense_records(n_images=10, objects_per_image=0, id_prefix=f"{id_prefix}N",
                              miss_pattern=[0] * 10, fp_pattern=[3] * 10, score=0.9, fp_score=0.9)
        return loaded + noisy

    _conf, evidence, failures = _criterion(_hallucinating("c"), _hallucinating("h", shift=5.0))

    hb = evidence["holdout_at_conf"]
    assert hb["n_present"] == 20
    assert hb["count_bias_mean_present"] == pytest.approx(1.5)  # (10 * 0 + 10 * 3) / 20
    assert "count_bias_exceeds_tolerance" in failures


def test_a_holdout_carrying_one_loaded_image_is_not_enough_evidence():
    cal = _mixed_reference("c", n_loaded=4, n_negative=0, objects_per_image=100, fp_per_loaded=0)
    hold = _mixed_reference("h", n_loaded=1, n_negative=99, objects_per_image=100,
                            fp_per_loaded=0, shift=5.0)

    _conf, evidence, failures = _criterion(cal, hold)

    assert evidence["holdout_at_conf"]["n_images"] == 100
    assert "insufficient_holdout_images" in failures
