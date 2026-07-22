"""Overnight-audit Wave A: data-fidelity fixes locked by test.

Covers: confirmed negatives survive every save door (MCP save_annotations, ReviewEngine.save_gt),
inference predictions carry model provenance, SAM staging carries created_at, review label backups
capture the canonical JSON format, and stratified splits count JSON objects (not JSON lines).
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import BBox


def _img(tmp_path, name="IMG_0001.JPG", size=(100, 80)):
    p = tmp_path / "images" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(p)
    return p


def test_mcp_save_annotations_empty_records_negative(tmp_path):
    """boxes=[] must write a confirmed-negative {objects: []} file, never delete GT."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    det = tmp_path / "det.json"
    json_io.write_detect(det, [BBox(1, 1, 9, 9, 0)], 100, 80)  # existing GT
    res = save_annotations(str(img), boxes=[], detect_path=str(det))
    assert "error" not in res
    assert det.is_file()                                        # not deleted
    assert json.loads(det.read_text())["objects"] == []         # confirmed negative


def test_review_engine_save_gt_empty_records_negative(tmp_path):
    from tcip_annotation.review_engine import ReviewContext, ReviewEngine

    eng = ReviewEngine(state_dir=str(tmp_path / "state"), class_names={0: "catkin"})
    ctx = ReviewContext(img_name="a.jpg", img_width=100, img_height=80,
                        gt_boxes=[], gt_polygons=[])
    det = tmp_path / "labels" / "a.json"
    assert eng.save_gt(ctx, detect_path=str(det)) is True
    assert det.is_file()
    assert json.loads(det.read_text())["objects"] == []


def test_write_predictions_json_stamps_model_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    p = tmp_path / "pred.json"
    write_predictions_json(p, {"width": 100, "height": 80,
                               "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1]},
                           created_by="model:best_catkin")
    obj = json.loads(p.read_text())["objects"][0]
    assert obj["created_by"] == "model:best_catkin"
    assert obj["created_at"]
    assert obj["score"] == pytest.approx(0.9)


def test_backup_original_labels_captures_json(tmp_path):
    from tcip_annotation.review_engine import ReviewEngine

    eng = ReviewEngine(state_dir=str(tmp_path / "state"), class_names={})
    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_detect(d / "a.json", [BBox(1, 1, 9, 9, 0)], 100, 80)
    json_io.write_detect(d / "b.json", [BBox(2, 2, 8, 8, 0)], 100, 80)
    captured = eng.backup_original_labels(str(d))
    assert captured == 2                                        # both canonical .json labels
    assert (d / ".original" / "a.json").is_file()


def test_make_splits_counts_json_objects_not_lines(tmp_path):
    """A pretty-printed negative ({objects: []}) is ~5 text lines; the stratifier must see 0."""
    from tcip_mcp.tools.data_tools import make_splits

    for i in range(4):
        _img(tmp_path, name=f"img_{i}.JPG")
    labels = tmp_path / "annotations" / "default" / "detect"
    labels.mkdir(parents=True)
    json_io.write_detect(labels / "img_0.json", [BBox(1, 1, 9, 9, 0)] * 3, 100, 80)
    json_io.write_detect(labels / "img_1.json", [BBox(1, 1, 9, 9, 0)], 100, 80)
    json_io.write_detect(labels / "img_2.json", [], 100, 80, keep_empty=True)  # negative
    json_io.write_detect(labels / "img_3.json", [], 100, 80, keep_empty=True)  # negative

    res = make_splits(str(tmp_path), train_ratio=0.5, val_ratio=0.5, test_ratio=0.0,
                      group_by="stem")
    assert "error" not in res
    # foreground_annotations sums per split: true total is 3+1+0+0. Counting raw JSON text
    # lines (the old bug) would have reported dozens — negatives alone read as ~5 each.
    assert sum(res["foreground_annotations"].values()) == 4


def test_coco_roundtrip_preserves_provenance(tmp_path):
    """GT exported to dataset-COCO and re-imported must keep created_by/accepted_by."""
    from tcip_annotation.format_io import parse_coco_detect, write_coco_detect
    import json as _json

    b = BBox(10, 10, 40, 40, 0, created_by="derived:user:zack",
             created_at="2026-02-11T00:00:00+00:00", accepted_by="user:zack")
    p = tmp_path / "dataset.json"
    write_coco_detect(str(p), {"IMG_0001.JPG": ([b], 100, 80)})
    boxes, _ = parse_coco_detect(_json.loads(p.read_text()), file_name="IMG_0001.JPG")
    assert boxes[0].created_by == "derived:user:zack"
    assert boxes[0].accepted_by == "user:zack"
