"""Operating-point counting semantics: which detection is a hit, and which side of the ratio it lands on.

Two properties the symmetric fixtures elsewhere in the suite cannot separate:

* precision and recall answer different questions (how much of what the model reported is real,
  versus how much of what is real the model reported), so a reference whose false-positive burden
  differs from its false-negative burden must produce two different numbers, each attached to the
  right denominator;
* the center-match matcher consumes a ground-truth object when a detection claims it, so a cluster
  of detections on one object contributes one true positive and the rest as false positives. The
  per-image count bias cannot see that failure at all (``fp - fn`` is the raw detection-minus-truth
  difference whatever the matcher does with the pairing), so the tp/fp/fn split is the only place a
  double-counted object is recorded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # evaluation.py imports torch at module load
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    build_coco_image_record,
    coco_detection_metrics,
    governing_counts,
    sweep_operating_point,
)

CENTER_MATCH_TOLERANCE = 10.0


# -- precision and recall carry different denominators -----------------------------------------

def _asymmetric_iou_records() -> list[dict]:
    """Two frames of different shapes whose hit rate and hallucination rate genuinely differ.

    Frame one (640x400): three ground-truth boxes of three aspect ratios, two matched exactly and
    three detections placed well clear of any of them. Frame two (300x500): two ground-truth boxes,
    one matched, nothing spurious. Pooled that is tp=3, fp=3, fn=2, so precision (3/6) and recall
    (3/5) are distinct numbers and neither equals the F1 they combine into.
    """
    wide_gt = {"category_id": 1, "bbox": [50, 60, 40, 20], "iscrowd": 0}
    tall_gt = {"category_id": 1, "bbox": [200, 100, 30, 60], "iscrowd": 0}
    flat_gt = {"category_id": 1, "bbox": [400, 250, 80, 25], "iscrowd": 0}
    frame_one = build_coco_image_record(
        640, 400,
        [wide_gt, tall_gt, flat_gt],
        [
            {"category_id": 1, "bbox": [50, 60, 40, 20], "score": 0.90},
            {"category_id": 1, "bbox": [200, 100, 30, 60], "score": 0.80},
            {"category_id": 1, "bbox": [500, 20, 15, 35], "score": 0.70},
            {"category_id": 1, "bbox": [30, 300, 25, 15], "score": 0.60},
            {"category_id": 1, "bbox": [600, 350, 20, 40], "score": 0.55},
        ],
    )
    frame_two = build_coco_image_record(
        300, 500,
        [{"category_id": 1, "bbox": [40, 40, 60, 15], "iscrowd": 0},
         {"category_id": 1, "bbox": [150, 300, 25, 90], "iscrowd": 0}],
        [{"category_id": 1, "bbox": [40, 40, 60, 15], "score": 0.95}],
    )
    return [frame_one, frame_two]


def test_precision_and_recall_are_not_interchangeable():
    """On a reference with three false positives and two misses the two ratios differ, and each
    must be reported against its own denominator: precision over everything reported, recall over
    everything real."""
    m = coco_detection_metrics(_asymmetric_iou_records())
    assert (m["tp"], m["fp"], m["fn"]) == (3, 3, 2)
    assert m["precision"] == pytest.approx(0.5)      # 3 of 6 reported detections are real
    assert m["recall"] == pytest.approx(0.6)         # 3 of 5 real objects were reported
    assert m["f1"] == pytest.approx(2 * 0.5 * 0.6 / 1.1)


def test_extra_false_positives_move_precision_and_leave_recall_alone():
    """Hallucinated detections cost precision only. Nothing about how much of the ground truth was
    found changes when a detector reports more objects that are not there."""
    base = _asymmetric_iou_records()
    noisier = _asymmetric_iou_records()
    noisier[1]["dt"] = [*noisier[1]["dt"],
                        {"category_id": 1, "bbox": [10, 400, 30, 40], "score": 0.65},
                        {"category_id": 1, "bbox": [220, 60, 40, 20], "score": 0.45}]

    before = coco_detection_metrics(base)
    after = coco_detection_metrics(noisier)
    assert (after["tp"], after["fn"]) == (before["tp"], before["fn"])
    assert after["fp"] == before["fp"] + 2
    assert after["recall"] == pytest.approx(before["recall"])
    assert after["precision"] < before["precision"]


# -- center matching consumes the object it matched --------------------------------------------

def _clustered_center_match_records() -> list[dict]:
    """Two frames where one object attracts two detections inside the match tolerance.

    Frame one (500x300) holds two objects: the first draws detections at distance 0 and about
    5.8 px, both inside the 10 px tolerance, the second draws one. Frame two (800x600) holds three
    objects, only one of them detected, so the reference also carries plain misses. Class id 7,
    a sparse id, so nothing rests on ids being dense or zero-based.
    """
    frame_one = build_coco_image_record(
        500, 300,
        [{"category_id": 7, "bbox": [88, 92, 24, 16], "iscrowd": 0},
         {"category_id": 7, "bbox": [285, 185, 30, 30], "iscrowd": 0}],
        [{"category_id": 7, "bbox": [90, 94, 20, 12], "score": 0.90},
         {"category_id": 7, "bbox": [97, 97, 16, 12], "score": 0.85},
         {"category_id": 7, "bbox": [290, 190, 20, 20], "score": 0.50}],
    )
    frame_two = build_coco_image_record(
        800, 600,
        [{"category_id": 7, "bbox": [370, 290, 60, 20], "iscrowd": 0},
         {"category_id": 7, "bbox": [590, 80, 20, 40], "iscrowd": 0},
         {"category_id": 7, "bbox": [85, 490, 30, 20], "iscrowd": 0}],
        [{"category_id": 7, "bbox": [396, 296, 8, 8], "score": 0.70}],
    )
    return [frame_one, frame_two]


def test_second_detection_of_one_object_is_a_false_positive():
    """A ground-truth object is claimed once. The runner-up detection on an already-matched object
    is a false positive, so five objects and four detections leave tp=3, fp=1, fn=2 rather than a
    fourth true positive and a negative miss count."""
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


def test_sweep_records_a_doubly_detected_object_as_one_hit_and_one_false_alarm():
    """The conf sweep counts through the same matcher, so a cluster on one object reads as one hit
    plus a false alarm at every conf that keeps the cluster. The signed count bias is blind to this
    (it is detections minus objects however the pairing resolves), which is why the tp/fp/fn split
    is what the entry has to be read on."""
    sweep = sweep_operating_point(_clustered_center_match_records(),
                                  tolerance=CENTER_MATCH_TOLERANCE)
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    assert set(at) == {0.0, 0.5, 0.7, 0.85, 0.9}

    assert (at[0.5]["tp"], at[0.5]["fp"], at[0.5]["fn"]) == (3, 1, 2)
    assert at[0.5]["precision"] == pytest.approx(0.75)
    # conf 0.85 keeps only the cluster: one hit, one false alarm, four objects never reported.
    assert (at[0.85]["tp"], at[0.85]["fp"], at[0.85]["fn"]) == (1, 1, 4)
    assert at[0.85]["recall"] == pytest.approx(0.2)


def test_uncontested_detections_all_match():
    """The tolerance still admits every detection that has its own object: one detection per object,
    each within tolerance, is all true positives with nothing left over."""
    records = [
        build_coco_image_record(
            500, 300,
            [{"category_id": 7, "bbox": [88, 92, 24, 16], "iscrowd": 0},
             {"category_id": 7, "bbox": [285, 185, 30, 30], "iscrowd": 0}],
            [{"category_id": 7, "bbox": [90, 94, 20, 12], "score": 0.90},
             {"category_id": 7, "bbox": [292, 192, 16, 16], "score": 0.50}],
        ),
    ]
    counts = governing_counts(records, {"kind": "center_match", "tolerance": CENTER_MATCH_TOLERANCE},
                              conf_threshold=0.25)
    assert (counts["tp"], counts["fp"], counts["fn"]) == (2, 0, 0)
    assert counts["f1"] == pytest.approx(1.0)
