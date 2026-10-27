"""The image dimensions a staged prediction document records.

Geometry is staged in pixels, and the document records the width and height of the image it was
staged for, landscape or portrait.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io

DATE = "2026-03-04"
LANDSCAPE = (640, 480)
PORTRAIT = (480, 800)


def _staged(
    project: Path,
    size: tuple[int, int],
    box: tuple[float, float, float, float],
) -> json_io.LabelDocument:
    """Stage ``box`` for an image of ``size`` and return the staged document.

    The box is staged by ``staged_box`` under ``project``'s dataset ``ds``; the document is
    returned as its bucket records it.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import staged_box, write_image

    image = write_image(image_dir(project / "ds", DATE) / "IMG_0007.jpg", size)
    bucket = read_bucket(project / "ds", staged_box(project, image, list(box)))
    return json_io.read_label_document(bucket.document_key(image.stem))


@pytest.mark.parametrize("size", [LANDSCAPE, PORTRAIT])
def test_staged_file_records_the_width_and_height_it_was_staged_for(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """A staged file records the width and height of the image it was staged for.

    The header carries the image's own width and height, each in its own field, for both a
    landscape and a portrait frame.
    """
    header = _staged(tmp_path, size, (10.0, 20.0, 60.0, 90.0))

    assert (header.width, header.height) == size


def test_staged_geometry_stays_inside_the_frame_the_file_declares(
    tmp_path: Path,
) -> None:
    """Staged geometry stays inside the frame its own file declares.

    A staged box keeps its pixel coordinates, and normalized against the dimensions the same
    file declares it stays within the unit frame.
    """
    width, height = LANDSCAPE
    header = _staged(tmp_path, LANDSCAPE, (500.0, 60.0, 620.0, 300.0))
    (written,) = header.annotations
    box = written.geometry

    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx(
        (500.0, 60.0, 620.0, 300.0)
    )
    assert box.x2 / header.width == pytest.approx(620.0 / width)
    assert box.y2 / header.height == pytest.approx(300.0 / height)
    assert box.x2 / header.width <= 1.0
    assert box.y2 / header.height <= 1.0
