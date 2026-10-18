"""The COCO parser's reading of a dataset-level COCO document.

These tests call the parser directly; no loader here opens a COCO document.
"""

import pytest

torch = pytest.importorskip("torch")


def test_parse_coco_annotations_decodes_names():
    """A dataset-level COCO parses to name-based Annotations, subjects from its categories."""
    from tcip_annotation.format_io import parse_coco_annotations
    from tcip_annotation.state import Polygon
    from tests._training_values import VERTEX_SPACING

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            }
        ],
        "categories": [{"id": 2, "name": "bud"}],
    }
    anns = parse_coco_annotations(
        coco,
        decode_rle=lambda segmentation: pytest.fail("no run-length mask here"),
        vertex_spacing=VERTEX_SPACING,
    )[1][1]
    assert len(anns) == 1
    assert (
        anns[0].subject == "bud"
    )  # id 2 -> name, from the file's own categories
    assert isinstance(anns[0].geometry, Polygon)
