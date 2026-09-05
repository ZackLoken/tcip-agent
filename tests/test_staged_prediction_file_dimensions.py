"""The image dimensions a staged prediction file records.

Geometry is staged in pixels, so the file's own width and height are what every later consumer
normalizes against: review keying, a calibration reference, any re-render. On a non-square image
a transposed pair reads back as geometry that has drifted by the aspect ratio, or left the frame
entirely, while the pixel coordinates still look correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_annotation import Annotation, BBox, json_io
from tcip_mcp.prediction_buckets import stage_prediction_shapes

DATE = "2026-03-04"
LANDSCAPE = (640, 480)
PORTRAIT = (480, 800)


def _stage(root: Path, size: tuple[int, int], geometry: BBox) -> dict:
    return stage_prediction_shapes(
        str(root), "detector", DATE, "IMG_0007",
        annotations=[Annotation(subject="bud", geometry=geometry, score=0.72)],
        img_w=size[0], img_h=size[1],
    )


@pytest.mark.parametrize("size", [LANDSCAPE, PORTRAIT])
def test_staged_file_records_the_width_and_height_it_was_staged_for(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """The header of the staged file carries the staging call's own width and height, each in its
    own field, for both a landscape and a portrait frame."""
    staged = _stage(tmp_path / "proj", size, BBox(10.0, 20.0, 60.0, 90.0))

    header = json.loads(Path(staged["path"]).read_text(encoding="utf-8"))
    assert (header["width"], header["height"]) == size


def test_staged_geometry_stays_inside_the_frame_the_file_declares(tmp_path: Path) -> None:
    """A staged box sits inside the image it was measured on, and stays inside it after a consumer
    normalizes the pixel geometry against the dimensions the same file declares."""
    width, height = LANDSCAPE
    staged = _stage(tmp_path / "proj", LANDSCAPE, BBox(500.0, 60.0, 620.0, 300.0))

    path = Path(staged["path"])
    header = json.loads(path.read_text(encoding="utf-8"))
    (written,) = json_io.read_annotations(path)
    box = written.geometry

    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((500.0, 60.0, 620.0, 300.0))
    assert box.x2 / header["width"] == pytest.approx(620.0 / width)
    assert box.y2 / header["height"] == pytest.approx(300.0 / height)
    assert box.x2 / header["width"] <= 1.0
    assert box.y2 / header["height"] <= 1.0


def test_stage_prediction_shapes_refuses_a_reserved_stem(tmp_path: Path) -> None:
    """An image stem reserved for a bucket's own provenance stamp must never reach a staged
    per-image prediction write, since the stamp write into that same bucket would otherwise
    destroy or refuse over it."""
    with pytest.raises(ValueError, match="operating_point"):
        stage_prediction_shapes(
            str(tmp_path / "proj"), "detector", DATE, "operating_point",
            annotations=[Annotation(subject="bud", geometry=BBox(1.0, 1.0, 5.0, 5.0))],
            img_w=100, img_h=100,
        )
    assert not (tmp_path / "proj").exists()


def test_stage_prediction_shapes_still_writes_an_ordinary_stem(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "proj", LANDSCAPE, BBox(10.0, 20.0, 60.0, 90.0))
    assert Path(staged["path"]).is_file()
