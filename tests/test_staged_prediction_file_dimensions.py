"""The image dimensions a staged prediction file records.

Geometry is staged in pixels, so the file's own width and height are what every later consumer
normalizes against: review keying, a re-render. On a non-square image a transposed pair reads back
as geometry that has drifted by the aspect ratio, or left the frame entirely, while the pixel
coordinates still look correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io

DATE = "2026-03-04"
LANDSCAPE = (640, 480)
PORTRAIT = (480, 800)


def _stage(project: Path, size: tuple[int, int], box: tuple[float, float, float, float]) -> dict:
    """``box`` staged through ``stage_proposals`` for an image of ``size`` under ``project``'s
    dataset ``ds``."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = project / "ds" / "images" / DATE / "IMG_0007.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(image)
    w, h = size
    x1, y1, x2, y2 = box
    staged = stage_proposals(project, str(image), model_name="detector", boxes=[{
        "subject": "bud", "conf": 0.72, "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
        "w": (x2 - x1) / w, "h": (y2 - y1) / h}])
    assert "error" not in staged, staged
    return staged


@pytest.mark.parametrize("size", [LANDSCAPE, PORTRAIT])
def test_staged_file_records_the_width_and_height_it_was_staged_for(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """The header of the staged file carries the image's own width and height, each in its own
    field, for both a landscape and a portrait frame."""
    staged = _stage(tmp_path, size, (10.0, 20.0, 60.0, 90.0))

    header = json.loads(Path(staged["path"]).read_text(encoding="utf-8"))
    assert (header["width"], header["height"]) == size


def test_staged_geometry_stays_inside_the_frame_the_file_declares(tmp_path: Path) -> None:
    """A staged box sits inside the image it was measured on, and stays inside it after a consumer
    normalizes the pixel geometry against the dimensions the same file declares."""
    width, height = LANDSCAPE
    staged = _stage(tmp_path, LANDSCAPE, (500.0, 60.0, 620.0, 300.0))

    path = Path(staged["path"])
    header = json.loads(path.read_text(encoding="utf-8"))
    (written,) = json_io.read_annotations(path)
    box = written.geometry

    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((500.0, 60.0, 620.0, 300.0))
    assert box.x2 / header["width"] == pytest.approx(620.0 / width)
    assert box.y2 / header["height"] == pytest.approx(300.0 / height)
    assert box.x2 / header["width"] <= 1.0
    assert box.y2 / header["height"] <= 1.0
