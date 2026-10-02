"""Data-fidelity coverage: an emptied document survives the MCP save door, inference predictions
carry model provenance, and stratified splits count JSON objects, not JSON lines.
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


def test_mcp_save_annotations_empty_writes_the_empty_document_the_route_writes(tmp_path):
    """An empty save writes an empty label document through the one save both label doors call:
    the file stays, holding nothing, never deleted."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    det = tmp_path / "det.json"
    json_io.write_annotations(det, [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
                              100, 80)  # existing GT
    res = save_annotations(tmp_path, tmp_path.parent, str(img), annotations=[], path=str(det))
    assert res == {"written": [str(det)], "count": 0}
    assert det.is_file()
    assert json_io.read_annotations(det) == []


@pytest.mark.parametrize("bad", ["a bur", {"bbox": [1, 1, 5, 5]}, {"subject": ""}],
                         ids=["not_an_object", "no_subject", "empty_subject"])
def test_mcp_save_annotations_refuses_by_index_through_the_decoders_checks(tmp_path, bad):
    """Each annotation is admitted by the decoder's own object and subject checks, the refusal
    naming its index, and nothing is written; the valid one beside it is not written alone."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = _img(tmp_path)
    det = tmp_path / "det.json"
    res = save_annotations(tmp_path, tmp_path.parent, str(img),
                           annotations=[{"subject": "bur", "bbox": [1, 1, 5, 5]}, bad],
                           path=str(det))
    assert res["error"].startswith("annotation 1 ")
    assert not det.exists()


def test_encode_predictions_stamps_model_provenance(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    data, _dropped = encode_predictions(
        {"image": "pred.jpg", "width": 100, "height": 80,
         "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1]},
        created_by="model:best_bud", scope=registry_scope(tmp_path, "bud"))
    obj = json.loads(data)["annotations"][0]
    assert obj["created_by"] == "model:best_bud"
    assert obj["created_at"]
    assert obj["score"] == pytest.approx(0.9)


def test_draw_splits_counts_json_objects_not_lines(tmp_path):
    """A pretty-printed negative ({annotations: []}) is several text lines; the stratifier sees 0."""
    from tcip_mcp.tools.data_tools import draw_splits

    from tests._producer_fixtures import mark_complete

    for i in range(4):
        _img(tmp_path, name=f"img_{i}.JPG")
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)

    def _box() -> Annotation:
        return Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))

    json_io.write_annotations(labels / "img_0.json", [_box(), _box(), _box()], 100, 80)
    json_io.write_annotations(labels / "img_1.json", [_box()], 100, 80)
    for negative in ("img_2", "img_3"):
        json_io.write_annotations(labels / f"{negative}.json", [], 100, 80, keep_empty=True)
        mark_complete(tmp_path / "images" / f"{negative}.JPG", labels / f"{negative}.json", "bud",
                      project=tmp_path)

    res = draw_splits(tmp_path, str(tmp_path), train_ratio=0.5, val_ratio=0.5, calibration_ratio=0.0,
                      group_by="stem", subject="bud")
    assert "error" not in res
    # foreground_annotations sums per split: true total is 3+1+0+0. Counting raw JSON text
    # lines instead reports dozens, since negatives alone read as several each.
    assert sum(res["foreground_annotations"].values()) == 4
