"""The count criterion is conditioned on class, not pooled across classes.

The criterion cases drive :func:`~tcip_mcp.pipelines.operating_point.count_criterion` over
per-image records; the curve cases call ``derive_operating_point_curve`` and the tolerance cases
``_effective_count_bias_tolerance`` directly.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "torch"
)  # operating_point imports evaluation, which imports torch

from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    _effective_count_bias_tolerance,
    count_criterion,
)
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    derive_operating_point_curve,
    gt_class_avg_size,
    pick_count_unbiased,
    prediction_record,
    scaled_to,
)
from tests import _trait_fixtures as fx  # noqa: E402
from tests._training_values import (  # noqa: E402
    COUNT_TEST,
    DISPERSION_QUANTILE,
)

FRAME = 1024  # a frame holding every box the fixtures below lay out
ENTRY = fx.with_fields(
    fx.COUNT_SPEC, count_bias_tolerance_frac=0.01, count_error_tolerance=10.0
)
"""A count trait held to one percent of each scope's typical count, its dispersion bound loose so
the per-class conditioning is what these cases isolate."""


def _criterion(cal, hold):
    return count_criterion(
        cal,
        hold,
        ENTRY,
        staged_conf_floor=0.05,
        staged_conf_floor_attribute_path="self",
        **COUNT_TEST,
    )


def _gt_records(prefix, n, *, swap_classes, classes=(1, 2), offset=0.0):
    """Return ``n`` per-image :func:`_record` entries of eight 40 px boxes each.

    The first four boxes are given ``classes[0]`` and the rest ``classes[1]``, each with a 0.9
    detection on its box; ``swap_classes`` gives every detection ``classes[1]``, and ``offset``
    shifts the whole layout in x.
    """
    recs = []
    for i in range(n):
        gt, dt = [], []
        for k in range(8):
            box = [100.0 * k + 3.0 * i + offset, 50.0 + i, 40.0, 40.0]
            cat = classes[0] if k < 4 else classes[1]
            gt.append({"bbox": box, "category_id": cat, "iscrowd": 0})
            dt.append(
                {
                    "bbox": box,
                    "category_id": classes[1] if swap_classes else cat,
                    "score": 0.9,
                }
            )
        recs.append(_record(prefix, i, gt, dt))
    return recs


def _record(prefix, i, gt, dt):
    return prediction_record(
        dt,
        gt,
        width=FRAME,
        height=FRAME,
        cap=len(dt) + 1,
        count=len(dt),
        image_id=f"{prefix}{i}",
    )


def test_a_class_compensating_reference_the_pooled_bias_calls_unbiased_fails_per_class():
    _conf, evidence, failures = _criterion(
        _gt_records("cal", 4, swap_classes=True),
        _gt_records("hold", 4, swap_classes=True, offset=5000.0),
    )

    assert evidence["holdout_at_conf"][
        "count_bias_mean_present"
    ] == pytest.approx(0.0)
    assert {
        cid for cid, c in evidence["per_class"].items() if not c["passed"]
    } == {"1", "2"}
    assert failures == ["count_bias_exceeds_tolerance_per_class"]


def test_the_same_geometry_called_correctly_passes():
    _conf, evidence, failures = _criterion(
        _gt_records("cal", 4, swap_classes=False),
        _gt_records("hold", 4, swap_classes=False, offset=5000.0),
    )

    assert failures == []
    assert set(evidence["per_class"]) == {"1", "2"}
    assert evidence["holdout_missing_classes"] == []


def test_a_single_class_reference_is_unaffected_by_the_conditioning():
    """A single-class reference is unaffected by conditioning the count criterion on class.

    With one detection class, conditioning on class is a no-op, the one class's statistics
    the pooled ones, and an independently class-filtered sweep over the same records agreeing.
    """
    cal = _gt_records("cal", 4, swap_classes=False, classes=(1, 1))
    hold = _gt_records(
        "hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0
    )
    conf, evidence, failures = _criterion(cal, hold)
    hb = evidence["holdout_at_conf"]

    assert failures == []
    assert list(hb["per_class"]) == ["1"]
    explicit = derive_operating_point_curve(
        hold,
        criterion=scaled_to(evidence["localization"], hold),
        class_id=1,
        conf_grid=[conf],
        dispersion_quantile=DISPERSION_QUANTILE,
    )["curve"][0]
    for key in (
        "tp",
        "fp",
        "fn",
        "count_bias_mean",
        "count_bias_std",
        "n_images",
        "n_present",
    ):
        assert hb["per_class"]["1"][key] == pytest.approx(explicit[key])
        assert hb["per_class"]["1"][key] == pytest.approx(hb[key])


def test_a_class_the_holdout_never_carries_cannot_pass_by_its_absence():
    """A class the holdout never carries fails the criterion as missing from the holdout.

    The model confuses classes 1 and 2 in calibration and the holdout holds only class 1: each
    per-class entry the holdout shows passes, and class 2 is reported missing from the holdout.
    """
    _conf, evidence, failures = _criterion(
        _gt_records("cal", 4, swap_classes=True),
        _gt_records(
            "hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0
        ),
    )

    assert all(c["passed"] for c in evidence["per_class"].values())
    assert evidence["holdout_missing_classes"] == ["2"]
    assert "holdout_missing_class" in failures
    assert "insufficient_holdout_images_per_class" not in failures


def test_a_class_scarce_in_the_holdout_cannot_be_diluted_to_a_pass():
    """A class scarce in the holdout cannot be diluted to a pass by the images lacking it.

    Class 2 is present on 2 of 20 holdout images and missed outright both times: its bias is -4
    over the two images that carry it, and it fails.
    """

    def calibration(prefix, n, offset):
        recs = []
        for i in range(n):
            gt, dt = [], []
            for cat in (1, 2):
                box = [100.0 * cat + offset, 50.0 + i, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": cat, "iscrowd": 0})
                dt.append({"bbox": box, "category_id": cat, "score": 0.9})
            recs.append(_record(prefix, i, gt, dt))
        return recs

    def holdout(prefix, n, offset):
        recs = []
        for i in range(n):
            gt, dt = [], []
            for k in range(2):
                box = [100.0 * k + offset, 50.0 + i, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": 1, "iscrowd": 0})
                dt.append({"bbox": box, "category_id": 1, "score": 0.9})
            if i < 2:
                for k in range(4):
                    gt.append(
                        {
                            "bbox": [
                                500.0 + 100.0 * k + offset,
                                50.0 + i,
                                40.0,
                                40.0,
                            ],
                            "category_id": 2,
                            "iscrowd": 0,
                        }
                    )
            recs.append(_record(prefix, i, gt, dt))
        return recs

    _conf, evidence, failures = _criterion(
        calibration("cal", 4, 0.0), holdout("hold", 20, 5000.0)
    )

    assert evidence["per_class"]["2"]["n_present"] == 2
    assert evidence["per_class"]["2"]["bias"] == pytest.approx(-4.0)
    assert evidence["per_class"]["2"]["passed"] is False
    assert "count_bias_exceeds_tolerance_per_class" in failures


def test_no_conf_in_the_sweep_escapes_a_wholesale_class_swap():
    """No conf in the sweep escapes a wholesale class swap.

    For a model that calls every class-1 object class 2, each conf on the curve leaves some
    class's absolute mean bias above one object per image, and each class's mean bias matches,
    within ``pytest.approx``, the mean of ``|dt_c| - |gt_c|`` over the images at that conf.
    """
    recs = _gt_records("cal", 4, swap_classes=True)
    for r in recs:  # a real score spread, so confs actually filter
        for k, d in enumerate(r["dt"]):
            d["score"] = 0.15 + 0.1 * k
    sweep = derive_operating_point_curve(
        recs,
        criterion={
            "kind": "center_match",
            "tolerance": 0.5 * gt_class_avg_size(recs),
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    assert len(sweep["curve"]) > 4
    assert [c for c in sweep["curve"] if c["fp"] + c["tp"] < 32], (
        "no conf filters any detection"
    )
    for entry in sweep["curve"]:
        for cid, stats in entry["per_class"].items():
            expected = sum(
                len(
                    [
                        d
                        for d in r["dt"]
                        if d["category_id"] == int(cid)
                        and d["score"] >= entry["conf"]
                    ]
                )
                - len([a for a in r["gt"] if a["category_id"] == int(cid)])
                for r in recs
            ) / len(recs)
            assert stats["count_bias_mean"] == pytest.approx(expected)
        assert (
            max(abs(s["count_bias_mean"]) for s in entry["per_class"].values())
            > 1.0
        )


def _three_class_records(prefix, offset, *, background: int):
    """Return six records of classes 1, 2 and 3, the layout shifted by ``offset`` in x.

    Classes 1 and 2 each hold one object with a 0.95 detection on its box and a second 0.95
    detection at y 400; class 3 holds three objects, the first two with detections on their boxes
    at 0.95 and 0.4. ``background`` adds that many objects per class, each with a 0.95 detection
    on its box.
    """
    recs = []
    for i in range(6):
        gt, dt = [], []
        for cat in (1, 2):
            box = [200.0 * cat + offset, 50.0, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": cat, "iscrowd": 0})
            dt.append({"bbox": box, "category_id": cat, "score": 0.95})
            dt.append(
                {
                    "bbox": [200.0 * cat + offset, 400.0, 40.0, 40.0],
                    "category_id": cat,
                    "score": 0.95,
                }
            )
        for k in range(3):
            gt.append(
                {
                    "bbox": [700.0 + 100.0 * k + offset, 50.0, 40.0, 40.0],
                    "category_id": 3,
                    "iscrowd": 0,
                }
            )
        dt.append(
            {
                "bbox": [700.0 + offset, 50.0, 40.0, 40.0],
                "category_id": 3,
                "score": 0.95,
            }
        )
        dt.append(
            {
                "bbox": [800.0 + offset, 50.0, 40.0, 40.0],
                "category_id": 3,
                "score": 0.4,
            }
        )
        for cat, x0 in ((1, 7000.0), (2, 11000.0), (3, 1500.0)):
            for k in range(background):
                box = [x0 + 60.0 * k + offset, 900.0, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": cat, "iscrowd": 0})
                dt.append({"bbox": box, "category_id": cat, "score": 0.95})
        recs.append(_record(prefix, i, gt, dt))
    return recs


def test_the_pick_serves_the_worst_class_and_a_sparse_reference_still_fails_it():
    """The pick serves the worst class, and a sparse reference fails the criterion at that pick.

    At conf 0.95 the pooled bias is 0 while class 3 sits at -2; at 0.4 every class's bias is at
    most 1, so the objective picks 0.4. At this density one permanent miscount per image is a
    large relative error in every class, and the criterion refuses all three.
    """
    recs = _three_class_records("w", 0.0, background=0)
    sweep = derive_operating_point_curve(
        recs,
        criterion={
            "kind": "center_match",
            "tolerance": 0.5 * gt_class_avg_size(recs),
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    assert at[0.95]["count_bias_mean"] == pytest.approx(0.0)
    assert at[0.95]["per_class"]["3"]["count_bias_mean"] == pytest.approx(-2.0)
    worst = max(
        abs(s["count_bias_mean"]) for s in at[0.4]["per_class"].values()
    )
    assert worst == pytest.approx(1.0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.4)

    conf, evidence, failures = _criterion(
        recs, _three_class_records("h", 5000.0, background=0)
    )

    assert conf == pytest.approx(0.4)
    assert "count_bias_exceeds_tolerance" in failures
    assert {
        cid for cid, c in evidence["per_class"].items() if not c["passed"]
    } == {"1", "2", "3"}


def test_the_same_pick_passes_once_every_class_is_dense_enough():
    """The 0.4 pick passes when each class holds two hundred always-found objects.

    The objects make each class's one permanent miscount per image a small fraction of its
    typical count, and the criterion admits the pick.
    """
    recs = _three_class_records("w", 0.0, background=200)
    conf, _evidence, failures = _criterion(
        recs, _three_class_records("h", 5000.0, background=200)
    )

    assert conf == pytest.approx(0.4)
    assert failures == []


def test_the_effective_tolerance_floor_governs_a_near_zero_fraction():
    tolerance = _effective_count_bias_tolerance(0.01, typical_count=1.0, n=5)
    assert tolerance == pytest.approx(0.2)
    assert tolerance > 0.01 * 1.0


def test_the_effective_tolerance_floor_shrinks_as_evidence_grows():
    assert _effective_count_bias_tolerance(
        0.0, typical_count=0.0, n=5
    ) == pytest.approx(0.2)
    assert _effective_count_bias_tolerance(
        0.0, typical_count=0.0, n=50
    ) == pytest.approx(0.02)


def test_the_effective_tolerance_floor_stays_at_or_below_one_half_at_two_or_more_images():
    for n in range(2, 60):
        assert (
            _effective_count_bias_tolerance(0.0, typical_count=0.0, n=n) <= 0.5
        )


def test_the_fraction_term_governs_a_dense_reference():
    assert _effective_count_bias_tolerance(
        0.01, typical_count=100.0, n=40
    ) == pytest.approx(1.0)


def test_a_sparse_classes_recorded_tolerance_is_the_floor_not_the_fraction_term():
    """A sparse class's recorded tolerance is the floor, not the fraction term.

    Class 2 holds two objects per image and one spurious detection on one of five images: its
    recorded tolerance is the floor (1 / 5), an order of magnitude above the fraction term
    (0.01 * 2).
    """

    def build(prefix, offset):
        recs = []
        for i in range(5):
            gt, dt = [], []
            for cat, y, n in ((1, 50.0, 20), (2, 900.0, 2)):
                for k in range(n):
                    box = [50.0 + 30.0 * k + offset, y, 20.0, 20.0]
                    gt.append({"bbox": box, "category_id": cat, "iscrowd": 0})
                    dt.append({"bbox": box, "category_id": cat, "score": 0.95})
            if i == 0:
                dt.append(
                    {
                        "bbox": [2000.0 + offset, 900.0, 20.0, 20.0],
                        "category_id": 2,
                        "score": 0.95,
                    }
                )
            recs.append(_record(prefix, i, gt, dt))
        return recs

    _conf, evidence, _failures = _criterion(
        build("c", 0.0), build("h", 5000.0)
    )

    assert evidence["per_class"]["2"]["typical_count"] == pytest.approx(2.0)
    assert evidence["per_class"]["2"]["tolerance"] == pytest.approx(0.2)


def test_a_class_present_on_exactly_one_holdout_image_cannot_pass_on_it_alone():
    """A class present on exactly one holdout image is refused as insufficient evidence.

    Class 2 shows up on one image of ten, in a dense frame, and the criterion reports
    ``insufficient_holdout_images_per_class``.
    """

    def build(prefix, offset):
        recs = []
        for i in range(10):
            gt, dt = [], []
            for k in range(20):
                box = [50.0 + 30.0 * k + offset, 50.0, 20.0, 20.0]
                gt.append({"bbox": box, "category_id": 1, "iscrowd": 0})
                dt.append({"bbox": box, "category_id": 1, "score": 0.95})
            if i == 0:
                for k in range(150):
                    box = [50.0 + 30.0 * k + offset, 900.0, 20.0, 20.0]
                    gt.append({"bbox": box, "category_id": 2, "iscrowd": 0})
                    dt.append({"bbox": box, "category_id": 2, "score": 0.95})
                for k in range(3):
                    dt.append(
                        {
                            "bbox": [
                                10000.0 + 30.0 * k + offset,
                                900.0,
                                20.0,
                                20.0,
                            ],
                            "category_id": 2,
                            "score": 0.95,
                        }
                    )
            recs.append(_record(prefix, i, gt, dt))
        return recs

    _conf, evidence, failures = _criterion(build("c", 0.0), build("h", 5000.0))

    assert evidence["per_class"]["2"]["n_present"] == 1
    assert "insufficient_holdout_images_per_class" in failures
