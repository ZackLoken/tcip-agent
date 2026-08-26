"""score_predictions: a present, unreadable label or prediction document is an error naming the
file, never a raise through the MCP tool boundary, on both the single-image and folder paths.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.tools.annotation_tools import score_predictions


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(path)


def test_score_predictions_single_image_reports_an_unreadable_gt(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    preds = tmp_path / "predictions" / "baseline"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    img = images / "IMG_0000.jpg"
    _write_image(img)
    bad = labels / "IMG_0000.json"
    bad.write_bytes(b"{not json")
    write_annotations(preds / "IMG_0000.json",
                      [Annotation(subject="catkin", geometry=BBox(1, 1, 5, 5), score=0.9)], 100, 80)

    res = score_predictions(str(img))

    assert "error" in res
    assert str(bad) in res["error"]


def test_score_predictions_folder_reports_an_unreadable_prediction(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    images = root / "images"
    labels = root / "annotations"
    preds = root / "predictions" / "baseline"
    labels.mkdir(parents=True)
    preds.mkdir(parents=True)
    _write_image(images / "IMG_0000.jpg")
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="catkin", geometry=BBox(1, 1, 5, 5))], 100, 80)
    bad = preds / "IMG_0000.json"
    bad.write_bytes(b"{not json")

    res = score_predictions(str(root))

    assert "error" in res
    assert str(bad) in res["error"]
