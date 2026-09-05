"""Data-fidelity coverage: confirmed negatives survive every save door (MCP save_annotations,
ReviewEngine.save_gt), inference predictions carry model provenance, SAM staging carries
created_at, review label backups capture the canonical JSON format, and stratified splits count
JSON objects, not JSON lines.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox


def _img(tmp_path, name="IMG_0001.JPG", size=(100, 80)):
    p = tmp_path / "images" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(p)
    return p


def test_mcp_save_annotations_empty_refuses_and_preserves_gt(tmp_path):
    """An empty save is refused (each annotation needs a subject) and never deletes existing GT.

    A confirmed negative is now an empty label + a human Complete in image_status.json, not a
    product of an empty save door, so the door that used to write ``{objects: []}`` is gone. What
    stays load-bearing is that a save call cannot destroy annotated ground truth.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    det = tmp_path / "det.json"
    json_io.write_annotations(det, [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
                              100, 80)  # existing GT
    res = save_annotations(str(img), annotations=[], path=str(det))
    assert "error" in res                                       # refused, never silently applied
    assert det.is_file()                                        # existing GT not deleted
    assert len(json_io.read_annotations(det)) == 1              # GT preserved intact


def test_review_engine_save_gt_empty_keeps_record(tmp_path):
    """An emptied GT keeps an ``{"annotations": []}`` record (not a negative until confirmed),
    never deleting the label file."""
    from tcip_annotation.review_engine import ReviewContext, ReviewEngine

    eng = ReviewEngine(state_dir=str(tmp_path / "state"))
    ctx = ReviewContext(img_name="a.jpg", img_width=100, img_height=80, gt=[], preds=[])
    det = tmp_path / "labels" / "a.json"
    assert eng.save_gt(ctx, path=str(det)) is True
    assert det.is_file()
    assert json.loads(det.read_text())["annotations"] == []


def test_write_predictions_json_stamps_model_provenance(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    p = tmp_path / "pred.json"
    write_predictions_json(p, {"width": 100, "height": 80,
                               "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1]},
                           created_by="model:best_bud", subject="bud", attribute=None)
    obj = json.loads(p.read_text())["annotations"][0]
    assert obj["created_by"] == "model:best_bud"
    assert obj["created_at"]
    assert obj["score"] == pytest.approx(0.9)


def test_backup_original_labels_captures_json(tmp_path):
    from tcip_annotation.review_engine import ReviewEngine

    eng = ReviewEngine(state_dir=str(tmp_path / "state"))
    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_annotations(d / "a.json", [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
                              100, 80)
    json_io.write_annotations(d / "b.json", [Annotation(subject="bud", geometry=BBox(2, 2, 8, 8))],
                              100, 80)
    captured = eng.backup_original_labels(str(d))
    assert captured == 2                                        # both canonical .json labels
    assert (d / ".original" / "a.json").is_file()


def test_draw_splits_counts_json_objects_not_lines(tmp_path):
    """A pretty-printed negative ({annotations: []}) is several text lines; the stratifier sees 0."""
    from tcip_mcp.tools.data_tools import draw_splits

    for i in range(4):
        _img(tmp_path, name=f"img_{i}.JPG")
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)

    def _box() -> Annotation:
        return Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))

    json_io.write_annotations(labels / "img_0.json", [_box(), _box(), _box()], 100, 80)
    json_io.write_annotations(labels / "img_1.json", [_box()], 100, 80)
    json_io.write_annotations(labels / "img_2.json", [], 100, 80, keep_empty=True)  # negative
    json_io.write_annotations(labels / "img_3.json", [], 100, 80, keep_empty=True)  # negative

    res = draw_splits(str(tmp_path), train_ratio=0.5, val_ratio=0.5, calibration_ratio=0.0,
                      group_by="stem")
    assert "error" not in res
    # foreground_annotations sums per split: true total is 3+1+0+0. Counting raw JSON text
    # lines (the old bug) would have reported dozens: negatives alone read as several each.
    assert sum(res["foreground_annotations"].values()) == 4


def test_coco_roundtrip_preserves_provenance(tmp_path):
    """GT exported to dataset-COCO and re-imported must keep created_by/accepted_by."""
    from tcip_annotation.format_io import parse_coco_annotations, write_coco
    import json as _json

    b = Annotation(subject="bud", geometry=BBox(10, 10, 40, 40), created_by="derived:user:breeder",
                   created_at="2026-02-11T00:00:00+00:00", accepted_by="user:breeder")
    p = tmp_path / "dataset.json"
    write_coco(str(p), {"IMG_0001.JPG": ([b], 100, 80)})
    anns = parse_coco_annotations(_json.loads(p.read_text()), file_name="IMG_0001.JPG")
    assert anns[0].created_by == "derived:user:breeder"
    assert anns[0].accepted_by == "user:breeder"
