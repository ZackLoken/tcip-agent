"""Every box scored against ground truth sits on the grid the stored labels are written on.

Ground truth reaches evaluation from a stored label document, which carries a 2-decimal coordinate
quantum. A raw prediction does not, so the two sides of a match would otherwise sit on grids that
differ by a fraction of the quantum, enough to decide a hit sitting exactly on the operating IoU.
One derivation, owned by the writer that defines the grid, puts both sides on it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

SUB_QUANTUM_BOX = (10.123456, 20.987654, 30.5, 40.5)
QUANTIZED_BOX = [10.12, 20.99, 20.38, 19.51]
ON_GRID_BOX = (12.25, 30.5, 60.75, 80.0)


def _detector_pass(gt_boxes, pred_boxes):
    """A torchvision-shaped GT target and detector output over one image."""
    target = {"boxes": torch.tensor(gt_boxes, dtype=torch.float32),
              "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
              "image_id": 1}
    output = {"boxes": torch.tensor(pred_boxes, dtype=torch.float32),
              "labels": torch.ones(len(pred_boxes), dtype=torch.int64),
              "scores": torch.full((len(pred_boxes),), 0.9)}
    return target, output


def test_a_prediction_off_the_stored_grid_is_put_on_it_before_scoring():
    """A raw detector coordinate carries more precision than any stored box it will be matched
    against, so it is quantized to the stored grid rather than compared across two grids."""
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target, output = _detector_pass([ON_GRID_BOX], [SUB_QUANTUM_BOX])
    record = records_from_detector(target, output, width=200, height=200)

    assert record["dt"][0]["bbox"] == QUANTIZED_BOX


def test_ground_truth_already_on_the_stored_grid_is_unchanged():
    """The shared derivation must be an identity for the side that defines the grid, or it would be
    moving ground truth to meet the predictions instead of the other way round."""
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target, output = _detector_pass([ON_GRID_BOX], [])
    record = records_from_detector(target, output, width=200, height=200)

    x1, y1, x2, y2 = ON_GRID_BOX
    assert record["gt"][0]["bbox"] == [x1, y1, x2 - x1, y2 - y1]


def test_an_annotation_prediction_off_the_stored_grid_is_put_on_it_before_scoring():
    """The annotation-shaped door onto the same records shares the grid, not just the tensor one."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    gt = [Annotation(subject="target", geometry=BBox(*ON_GRID_BOX))]
    preds = [Annotation(subject="target", geometry=BBox(*SUB_QUANTUM_BOX), score=0.9)]

    _iou_type, record = records_from_annotation(gt, preds, width=200, height=200)

    x1, y1, x2, y2 = ON_GRID_BOX
    assert record["gt"][0]["bbox"] == [x1, y1, x2 - x1, y2 - y1]
    assert record["dt"][0]["bbox"] == QUANTIZED_BOX


def test_the_grid_is_the_one_the_label_writer_owns():
    """Two implementations of the same quantum would drift; the scorer derives through the writer's
    own, so a change to the stored grid can never leave the scorer behind."""
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.training import evaluation

    assert evaluation.xywh is json_io.xywh
    assert not hasattr(evaluation, "_xyxy_to_xywh")


def test_scoring_labels_read_back_from_disk_is_unaffected_by_the_shared_grid(tmp_path):
    """Ordinary work must still pass: stored labels scored against predictions that reproduce them
    match exactly, with no metric moved by putting both sides through the same derivation."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.training.evaluation import (
        coco_detection_metrics, records_from_annotation,
    )

    label_path = tmp_path / "image_one.json"
    json_io.write_annotations(
        str(label_path),
        [Annotation(subject="target", geometry=BBox(*ON_GRID_BOX)),
         Annotation(subject="target", geometry=BBox(100.0, 100.5, 140.25, 150.75))],
        200, 200)

    stored = json_io.read_annotations(str(label_path))
    preds = [Annotation(subject=a.subject, geometry=a.geometry, score=0.9) for a in stored]
    _iou_type, record = records_from_annotation(stored, preds, width=200, height=200)

    metrics = coco_detection_metrics([record], iou_threshold=0.5, conf_threshold=0.25)

    assert metrics["map50"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
