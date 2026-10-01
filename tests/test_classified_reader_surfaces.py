"""Reader-side surfaces that hold a classified bucket to its own recorded scope: ``count_by_class``
under a coincidental detector map, a bucket record that no longer decodes, the COCO reader's
``attributes`` handling, and the prediction render's legend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing import phenology

SUBJECT = "bud"
ATTRIBUTE = "opening"


def test_count_by_class_a_detector_map_keyed_by_the_positive_value_name_never_counts_positive(
    tmp_path: Path,
) -> None:
    """A bare single-class detector map that happens to be spelled like the trait's own positive
    value name (a vocabulary coincidence) never counts a positive: only a bucket that classified
    along an attribute assessed a state at all."""
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="open", geometry=BBox(1, 1, 3, 3), score=0.9)], 8, 8)
    scope = ClassScope(subject="open", id_map={"open": 0})

    total, positive, unclassified = phenology.count_by_class(p, "open", scope=scope)

    assert (total, positive, unclassified) == (1, 0, 1)


def test_an_undecodable_bucket_record_refuses_by_name_rather_than_reading_as_unclassified(
    tmp_path: Path,
) -> None:
    """A published bucket whose record no longer decodes has no scope to read its documents
    under: reading it refuses naming the file."""
    pytest.importorskip("torch")
    from tcip_annotation.json_io import BUCKET_RECORD

    from tcip_mcp.buckets import read_bucket
    from tests._chain_fixtures import predicted, published

    scope = {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": {"open": 0, "closed": 1}}
    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "classifier" / "2026-05-02",
                       [predicted("s1", ["open"], scope["id_map"])], scope=scope)
    (bucket.path / BUCKET_RECORD).write_bytes(b"{not json")

    with pytest.raises(ValueError, match=BUCKET_RECORD):
        read_bucket(bucket.path)


def test_the_coco_reader_keeps_a_classified_records_value() -> None:
    from tcip_annotation.format_io import parse_coco_annotations

    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 0, "name": SUBJECT}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 4],
                         "score": 0.9, "attributes": {ATTRIBUTE: "open"}}],
    }

    (restored,) = parse_coco_annotations(
        coco, decode_rle=lambda segmentation: pytest.fail("no run-length mask here"))[1][1]
    assert restored.subject == SUBJECT
    assert restored.attributes == {ATTRIBUTE: "open"}


def test_legend_name_shows_the_classified_value_under_a_classified_scope() -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9,
                      attributes={ATTRIBUTE: "open"})
    scope = ClassScope(subject=SUBJECT, attribute=ATTRIBUTE, id_map={"open": 0, "closed": 1})

    assert _legend_name(pred, scope=scope) == "open"


def test_legend_name_falls_back_to_the_subject_under_no_scope() -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9)

    assert _legend_name(pred, scope=None) == SUBJECT
