"""COCO parser correctness, for the import and export doors that read one.

Training reads geometry ground truth as per-image label documents, so no loader here opens a
dataset-level COCO at all; what is covered is the parser those doors share.
"""

import pytest

torch = pytest.importorskip("torch")


def test_parse_coco_annotations_decodes_names():
    """A single-file COCO parses to name-based Annotations, the subject decoded from categories."""
    from tcip_annotation.format_io import parse_coco_annotations
    from tcip_annotation.state import Polygon
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                         "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]}],
        "categories": [{"id": 2, "name": "bud"}],
    }
    anns = parse_coco_annotations(coco, file_name="a.jpg")
    assert len(anns) == 1
    assert anns[0].subject == "bud"       # id 2 -> name, from the file's own categories
    assert isinstance(anns[0].geometry, Polygon)


