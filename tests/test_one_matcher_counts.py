"""Detection counts have one producer, the platform's matcher: an evaluation's counts are the
matcher's own at the operating point, inclusive of a detection scoring exactly at it, and the
prediction scorer's per-image matchings sum to the evaluation's counts over the same records."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _records():
    from tcip_mcp.pipelines.training.evaluation import gt_record, prediction_record

    dt = [{"category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.5},
          {"category_id": 1, "bbox": [61.0, 61.0, 20.0, 20.0], "score": 0.9},
          {"category_id": 1, "bbox": [0.0, 80.0, 10.0, 10.0], "score": 0.3}]
    return [prediction_record(dt, [gt_record([10.0, 10.0, 20.0, 20.0], 1, 0),
                                   gt_record([60.0, 60.0, 20.0, 20.0], 1, 0)],
                              width=100, height=100, cap=None, count=len(dt))]


def test_an_evaluations_counts_are_the_matchers_inclusive_at_the_threshold():
    from tcip_mcp.pipelines.training.evaluation import (
        detection_metrics, governing_counts, iou_criterion,
    )

    m = detection_metrics(_records(), trait=None, conf_threshold=0.5, iou_threshold=0.5,
                          by_mask=False)
    counts = governing_counts(_records(), iou_criterion(0.5, by_mask=False), conf_threshold=0.5)

    assert (m["tp"], m["fp"], m["fn"]) == (counts["tp"], counts["fp"], counts["fn"])
    # The detection scoring exactly at the threshold counts: two true positives, no miss.
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)


def test_the_scorers_matchings_and_the_evaluations_counts_agree_over_one_reference():
    from tcip_annotation.matching import pair_proposals
    from tcip_annotation.state import Annotation, BBox, Polygon
    from tcip_mcp.pipelines.training.evaluation import (
        detection_metrics, records_from_annotation,
    )

    gt = [Annotation(subject="fruit", geometry=BBox(10, 10, 30, 30)),
          Annotation(subject="fruit", geometry=Polygon(rings=[[(60, 60), (80, 60), (80, 80),
                                                               (60, 80)]]))]
    preds = [Annotation(subject="fruit", geometry=BBox(11, 11, 30, 30), score=0.8),
             Annotation(subject="fruit", geometry=BBox(0, 80, 10, 90), score=0.6)]
    record = records_from_annotation(gt, preds, width=100, height=100, cap=None)
    metrics = detection_metrics([record], trait=None, conf_threshold=0.5, iou_threshold=0.5,
                                by_mask=True)

    matching = pair_proposals(gt, preds, metrics["governing_criterion"], conf_threshold=0.5)
    assert matching.counts == (metrics["tp"], metrics["fp"], metrics["fn"])
    assert matching.counts == (1, 1, 1)


def test_masks_that_share_a_box_but_not_a_region_do_not_match_on_either_route():
    from tcip_annotation.matching import pair_proposals
    from tcip_annotation.state import Annotation, Polygon
    from tcip_mcp.pipelines.training.evaluation import detection_metrics, records_from_annotation

    lower = Polygon(rings=[[(60, 60), (60, 80), (80, 80)]])
    upper = Polygon(rings=[[(60, 60), (80, 60), (80, 80)]])
    gt = [Annotation(subject="fruit", geometry=lower)]
    preds = [Annotation(subject="fruit", geometry=upper, score=0.9)]
    record = records_from_annotation(gt, preds, width=100, height=100, cap=None)
    metrics = detection_metrics([record], trait=None, conf_threshold=0.5, iou_threshold=0.5,
                                by_mask=True)

    matching = pair_proposals(gt, preds, metrics["governing_criterion"], conf_threshold=0.5)
    assert matching.counts == (metrics["tp"], metrics["fp"], metrics["fn"])
    assert matching.counts == (0, 1, 1)


def test_one_scoring_matches_each_class_once_per_distinct_criterion(monkeypatch):
    """Counts, average precision and the displayed matching share one matching per image, class
    and criterion over the same detections: nothing is matched twice."""
    from collections import Counter

    import tcip_annotation.matching as matching
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import (
        AP_IOU_THRESHOLDS, detection_metrics, records_from_annotation,
    )

    calls: Counter = Counter()
    real = matching.pair_detections

    def counted(gt, dt, criterion, **kwargs):
        calls[(tuple(r["category_id"] for r in (*gt, *dt)), tuple(len(x) for x in (gt, dt)),
               criterion["kind"], criterion.get("iou_threshold"))] += 1
        return real(gt, dt, criterion, **kwargs)

    monkeypatch.setattr(matching, "pair_detections", counted)
    gt = [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30)),
          Annotation(subject="fruit", geometry=BBox(60, 60, 80, 80))]
    preds = [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30), score=0.9),
             Annotation(subject="fruit", geometry=BBox(60, 60, 80, 80), score=0.9)]
    metrics = detection_metrics(
        [records_from_annotation(gt, preds, width=100, height=100, cap=None)],
        trait=None, conf_threshold=0.5, iou_threshold=0.5, by_mask=False)

    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 0, 0)
    assert set(calls.values()) == {1}, calls
    assert len(calls) == 2 * len(AP_IOU_THRESHOLDS)


def test_a_detection_of_another_subject_on_an_object_is_a_miss_and_a_false_find_on_both_routes():
    from tcip_annotation.matching import pair_proposals
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import detection_metrics, records_from_annotation

    gt = [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30)),
          Annotation(subject="fruit", geometry=BBox(60, 60, 80, 80))]
    preds = [Annotation(subject="fruit", geometry=BBox(10, 10, 30, 30), score=0.5),
             Annotation(subject="fruit", geometry=BBox(60, 60, 80, 80), score=0.9)]
    record = records_from_annotation(gt, preds, width=100, height=100, cap=None)
    metrics = detection_metrics([record], trait=None, conf_threshold=0.5, iou_threshold=0.5,
                                by_mask=False)

    matching = pair_proposals(gt, preds, metrics["governing_criterion"], conf_threshold=0.5)
    assert matching.counts == (metrics["tp"], metrics["fp"], metrics["fn"])
    assert matching.counts == (1, 1, 1)
