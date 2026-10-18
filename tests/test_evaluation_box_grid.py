"""A box scored against ground truth sits on the grid the stored labels are written on.

A stored label document carries a 2-decimal coordinate quantum and a raw prediction does not.
``records_from_detector`` puts a prediction on that grid and leaves ground truth already on it
unchanged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

SUB_QUANTUM_BOX = (10.123456, 20.987654, 30.5, 40.5)
QUANTIZED_BOX = [10.12, 20.99, 20.38, 19.51]
ON_GRID_BOX = (12.25, 30.5, 60.75, 80.0)


def _detector_pass(gt_boxes, pred_boxes):
    """Return a target of ``gt_boxes`` and an output of ``pred_boxes`` scored 0.9, all class 1."""
    target = {
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
        "iscrowd": torch.zeros(
            len(gt_boxes), dtype=torch.int64
        ),  # every row one object
        "image_id": 1,
    }
    output = {
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
        "labels": torch.ones(len(pred_boxes), dtype=torch.int64),
        "scores": torch.full((len(pred_boxes),), 0.9),
    }
    return target, output


def test_a_prediction_off_the_stored_grid_is_put_on_it_before_scoring():
    """``records_from_detector`` quantizes a raw prediction to the stored label grid."""
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target, output = _detector_pass([ON_GRID_BOX], [SUB_QUANTUM_BOX])
    record = records_from_detector(
        target,
        output,
        width=200,
        height=200,
        cap=2,
        mask_cut=None,
    )

    assert record["dt"][0]["bbox"] == QUANTIZED_BOX


def test_ground_truth_already_on_the_stored_grid_is_unchanged():
    """Ground truth already on the stored grid leaves ``records_from_detector`` unchanged."""
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target, output = _detector_pass([ON_GRID_BOX], [])
    record = records_from_detector(
        target,
        output,
        width=200,
        height=200,
        cap=1,
        mask_cut=None,
    )

    x1, y1, x2, y2 = ON_GRID_BOX
    assert record["gt"][0]["bbox"] == [x1, y1, x2 - x1, y2 - y1]


def test_an_annotation_prediction_off_the_stored_grid_is_put_on_it_before_scoring():
    """``records_from_annotation`` puts both sides of a match on the stored label grid."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    gt = [Annotation(subject="target", geometry=BBox(*ON_GRID_BOX))]
    preds = [
        Annotation(
            subject="target", geometry=BBox(*SUB_QUANTUM_BOX), score=0.9
        )
    ]

    record = records_from_annotation(
        gt, preds, width=200, height=200, cap=None
    )

    x1, y1, x2, y2 = ON_GRID_BOX
    assert record["gt"][0]["bbox"] == [x1, y1, x2 - x1, y2 - y1]
    assert record["dt"][0]["bbox"] == QUANTIZED_BOX


def test_the_grid_is_the_one_the_label_writer_owns():
    """The scorer's box conversion is the label writer's own."""
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.training import evaluation

    assert evaluation.xywh is json_io.xywh
    assert not hasattr(evaluation, "_xyxy_to_xywh")


def test_scoring_labels_read_back_from_disk_is_unaffected_by_the_shared_grid(
    tmp_path,
):
    """Stored labels scored against predictions reproducing them match exactly."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
    from tcip_mcp.pipelines.training.evaluation import (
        detection_metrics,
        records_from_annotation,
    )

    label = label_key(tmp_path, UNDATED_BUCKET, "image_one")
    json_io.write_label_document(
        label,
        [
            Annotation(subject="target", geometry=BBox(*ON_GRID_BOX)),
            Annotation(
                subject="target", geometry=BBox(100.0, 100.5, 140.25, 150.75)
            ),
        ],
        200,
        200,
    )

    stored = json_io.read_label_document(label).annotations
    preds = [
        Annotation(subject=a.subject, geometry=a.geometry, score=0.9)
        for a in stored
    ]
    record = records_from_annotation(
        stored, preds, width=200, height=200, cap=None
    )

    metrics = detection_metrics(
        [record],
        trait=None,
        iou_threshold=0.5,
        conf_threshold=0.25,
        by_mask=False,
    )

    assert metrics["map50"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
