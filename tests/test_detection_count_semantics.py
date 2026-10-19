"""Operating-point counting semantics: which detection is a hit, and which ratio side it lands on.

Two properties, each over an asymmetric fixture:

* a reference whose false positives and misses differ gives a precision over everything reported
  and a recall over everything real, two different numbers;
* the center-match matcher consumes a ground-truth object when a detection claims it, so a cluster
  of detections on one object contributes one true positive and the rest as false positives.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    derive_operating_point_curve,
    detection_metrics,
    governing_counts,
    gt_record,
    prediction_record,  # noqa: E402
)
from tests._training_values import DISPERSION_QUANTILE  # noqa: E402

CENTER_MATCH_TOLERANCE = 10.0


# -- precision and recall carry different denominators -----------------------------------------


def _asymmetric_iou_records() -> list[dict]:
    """Return two frames of different shapes, all boxes of class 1.

    Frame one (640x400): three ground-truth boxes of three aspect ratios, a detection on the very
    box of two of them, and three detections overlapping no ground-truth box. Frame two
    (300x500): two ground-truth boxes and one detection, on the very box of the first.
    """
    wide_gt = gt_record([50, 60, 40, 20], 1, 0)
    tall_gt = gt_record([200, 100, 30, 60], 1, 0)
    flat_gt = gt_record([400, 250, 80, 25], 1, 0)
    one_dt = [
        {"category_id": 1, "bbox": [50, 60, 40, 20], "score": 0.90},
        {"category_id": 1, "bbox": [200, 100, 30, 60], "score": 0.80},
        {"category_id": 1, "bbox": [500, 20, 15, 35], "score": 0.70},
        {"category_id": 1, "bbox": [30, 300, 25, 15], "score": 0.60},
        {"category_id": 1, "bbox": [600, 350, 20, 40], "score": 0.55},
    ]
    two_dt = [{"category_id": 1, "bbox": [40, 40, 60, 15], "score": 0.95}]
    frame_one = prediction_record(
        one_dt,
        [wide_gt, tall_gt, flat_gt],
        width=640,
        height=400,
        cap=None,
        count=len(one_dt),
    )
    frame_two = prediction_record(
        two_dt,
        [
            gt_record([40, 40, 60, 15], 1, 0),
            gt_record([150, 300, 25, 90], 1, 0),
        ],
        width=300,
        height=500,
        cap=None,
        count=len(two_dt),
    )
    return [frame_one, frame_two]


def test_precision_and_recall_are_not_interchangeable() -> None:
    """Precision and recall differ on an asymmetric reference, each over its own denominator.

    On a reference with three false positives and two misses the two ratios differ, and each is
    reported against its own denominator: precision over everything reported, recall over
    everything real.
    """
    m = detection_metrics(
        _asymmetric_iou_records(),
        trait=None,
        conf_threshold=0.25,
        iou_threshold=0.5,
        by_mask=False,
    )
    assert (m["tp"], m["fp"], m["fn"]) == (3, 3, 2)
    assert m["precision"] == pytest.approx(
        0.5
    )  # 3 of 6 reported detections are real
    assert m["recall"] == pytest.approx(
        0.6
    )  # 3 of 5 real objects were reported
    assert m["f1"] == pytest.approx(2 * 0.5 * 0.6 / 1.1)


def test_extra_false_positives_move_precision_and_leave_recall_alone() -> None:
    """Extra false positives move precision and leave recall alone.

    Two more spurious detections add two false positives, lower precision, and leave the true
    positives, the misses and recall as they were.
    """
    base = _asymmetric_iou_records()
    noisier = _asymmetric_iou_records()
    noisier[1]["dt"] = [
        *noisier[1]["dt"],
        {"category_id": 1, "bbox": [10, 400, 30, 40], "score": 0.65},
        {"category_id": 1, "bbox": [220, 60, 40, 20], "score": 0.45},
    ]

    before, after = (
        detection_metrics(
            records,
            trait=None,
            conf_threshold=0.25,
            iou_threshold=0.5,
            by_mask=False,
        )
        for records in (base, noisier)
    )
    assert (after["tp"], after["fn"]) == (before["tp"], before["fn"])
    assert after["fp"] == before["fp"] + 2
    assert after["recall"] == pytest.approx(before["recall"])
    assert after["precision"] < before["precision"]


# -- center matching consumes the object it matched --------------------------------------------


def _clustered_center_match_records() -> list[dict]:
    """Return two frames where one object has two detections centered near its center.

    Frame one (500x300) holds two objects: the first has detections centered 0 and about 5.8 px
    from its center, the second one detection centered on its own. Frame two (800x600) holds
    three objects and one detection, centered on the first. Every box carries class id 7.
    """
    one_dt = [
        {"category_id": 7, "bbox": [90, 94, 20, 12], "score": 0.90},
        {"category_id": 7, "bbox": [97, 97, 16, 12], "score": 0.85},
        {"category_id": 7, "bbox": [290, 190, 20, 20], "score": 0.50},
    ]
    two_dt = [{"category_id": 7, "bbox": [396, 296, 8, 8], "score": 0.70}]
    frame_one = prediction_record(
        one_dt,
        [
            {"category_id": 7, "bbox": [88, 92, 24, 16], "iscrowd": 0},
            {"category_id": 7, "bbox": [285, 185, 30, 30], "iscrowd": 0},
        ],
        width=500,
        height=300,
        cap=None,
        count=len(one_dt),
    )
    frame_two = prediction_record(
        two_dt,
        [
            {"category_id": 7, "bbox": [370, 290, 60, 20], "iscrowd": 0},
            {"category_id": 7, "bbox": [590, 80, 20, 40], "iscrowd": 0},
            {"category_id": 7, "bbox": [85, 490, 30, 20], "iscrowd": 0},
        ],
        width=800,
        height=600,
        cap=None,
        count=len(two_dt),
    )
    return [frame_one, frame_two]


def test_second_detection_of_one_object_is_a_false_positive() -> None:
    """A second detection of one ground-truth object is a false positive.

    A ground-truth object is claimed by one detection. The runner-up detection on a matched
    object is a false positive, so five objects and four detections leave tp=3, fp=1, fn=2.
    """
    records = _clustered_center_match_records()
    criterion = {"kind": "center_match", "tolerance": CENTER_MATCH_TOLERANCE}
    counts = governing_counts(records, criterion, conf_threshold=0.25)

    n_gt = sum(len(r["gt"]) for r in records)
    n_dt = sum(len(r["dt"]) for r in records)
    assert (n_gt, n_dt) == (5, 4)
    assert (counts["tp"], counts["fp"], counts["fn"]) == (3, 1, 2)
    assert counts["tp"] + counts["fn"] == n_gt
    assert counts["tp"] + counts["fp"] == n_dt
    assert counts["precision"] == pytest.approx(0.75)
    assert counts["recall"] == pytest.approx(0.6)


def test_sweep_records_a_doubly_detected_object_as_one_hit_and_one_false_alarm() -> (
    None
):
    """The conf sweep records a doubly detected object as one hit and one false alarm.

    A cluster on one object reads as one hit plus a false alarm at the confs that keep the
    cluster, 0.5 and 0.85 among them.
    """
    sweep = derive_operating_point_curve(
        _clustered_center_match_records(),
        criterion={
            "kind": "center_match",
            "tolerance": CENTER_MATCH_TOLERANCE,
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    assert set(at) == {0.0, 0.5, 0.7, 0.85, 0.9}

    assert (at[0.5]["tp"], at[0.5]["fp"], at[0.5]["fn"]) == (3, 1, 2)
    assert at[0.5]["precision"] == pytest.approx(0.75)
    # conf 0.85 keeps only the cluster: one hit, one false alarm, four objects never reported.
    assert (at[0.85]["tp"], at[0.85]["fp"], at[0.85]["fn"]) == (1, 1, 4)
    assert at[0.85]["recall"] == pytest.approx(0.2)


def test_uncontested_detections_all_match() -> None:
    """Uncontested detections within tolerance all match.

    One detection per object, each within tolerance, counts as all true positives with nothing
    left over, an F1 of one.
    """
    dt = [
        {"category_id": 7, "bbox": [90, 94, 20, 12], "score": 0.90},
        {"category_id": 7, "bbox": [292, 192, 16, 16], "score": 0.50},
    ]
    records = [
        prediction_record(
            dt,
            [
                {"category_id": 7, "bbox": [88, 92, 24, 16], "iscrowd": 0},
                {"category_id": 7, "bbox": [285, 185, 30, 30], "iscrowd": 0},
            ],
            width=500,
            height=300,
            cap=None,
            count=len(dt),
        )
    ]
    counts = governing_counts(
        records,
        {"kind": "center_match", "tolerance": CENTER_MATCH_TOLERANCE},
        conf_threshold=0.25,
    )
    assert (counts["tp"], counts["fp"], counts["fn"]) == (2, 0, 0)
    assert counts["f1"] == pytest.approx(1.0)
