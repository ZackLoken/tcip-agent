"""Reader-side surfaces that hold a bucket to its own recorded scope: ``count_by_class`` under a
scope declaring no attribute of the positive state, a bucket record that no longer decodes, the
COCO reader's ``attributes`` handling, and the prediction render's legend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import subject_registry as cr
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.traits import PositiveState

SUBJECT = "object"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "high"))


def _scope(tmp_path: Path, *attributes: cr.Attribute) -> ClassScope:
    """The class space the admission reads for :data:`SUBJECT` over a dataset declaring
    ``attributes`` on it."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tests._producer_fixtures import registry_over

    registry_over(tmp_path, cr.SubjectRegistry(subjects=(
        cr.Subject(name=SUBJECT, attributes=attributes),)))
    (tmp_path / "annotations").mkdir(exist_ok=True)
    return registry_scope(tmp_path / "annotations", SUBJECT)


def test_count_by_class_under_a_scope_declaring_no_attribute_of_the_state_counts_none_positive(
    tmp_path: Path,
) -> None:
    """A bucket whose scope declares no attribute the positive state names never counts a
    positive, even where a record happens to carry that attribute's name and value: only a bucket
    whose model classified that attribute assessed the state at all."""
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 3, 3), score=0.9,
                       attributes={"grade": "high"})], 8, 8)

    total, positive, unclassified = phenology.count_by_class(
        p, PositiveState(attribute="grade", value="high"), scope=_scope(tmp_path, COLOR))

    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_reads_the_states_own_attribute_among_several(tmp_path: Path) -> None:
    """Under a scope declaring two attributes, the positive state's own attribute decides, never
    the other one; a record carrying no value under it refuses by name."""
    p = tmp_path / "img.json"
    json_io.write_annotations(p, [
        Annotation(subject=SUBJECT, geometry=BBox(1, 1, 3, 3), score=0.9,
                   attributes={"color": "red", "grade": "high"}),
        Annotation(subject=SUBJECT, geometry=BBox(4, 4, 6, 6), score=0.9,
                   attributes={"color": "blue", "grade": "low"}),
    ], 8, 8)
    scope = _scope(tmp_path, COLOR, GRADE)

    assert phenology.count_by_class(p, PositiveState(attribute="grade", value="high"),
                                    scope=scope) == (2, 1, 0)
    json_io.write_annotations(p, [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 3, 3),
                                             score=0.9, attributes={"color": "red"})], 8, 8)
    with pytest.raises(json_io.UndeclaredValue, match="'grade'"):
        phenology.count_by_class(p, PositiveState(attribute="grade", value="high"), scope=scope)


def test_an_undecodable_bucket_record_refuses_by_name_rather_than_reading_as_unclassified(
    tmp_path: Path,
) -> None:
    """A published bucket whose record no longer decodes has no scope to read its documents
    under: reading it refuses naming the file."""
    pytest.importorskip("torch")
    from tcip_annotation.json_io import BUCKET_RECORD

    from tcip_mcp.buckets import read_bucket
    from tests._chain_fixtures import predicted, published

    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "classifier" / "2026-05-02",
                       [predicted("s1", ["one"])], scope={"subject": SUBJECT})
    (bucket.path / BUCKET_RECORD).write_bytes(b"{not json")

    with pytest.raises(ValueError, match=BUCKET_RECORD):
        read_bucket(bucket.path)


def test_the_coco_reader_keeps_a_records_attribute_values() -> None:
    from tcip_annotation.format_io import parse_coco_annotations

    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 0, "name": SUBJECT}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 4, 4],
                         "score": 0.9, "attributes": {"color": "blue"}}],
    }

    (restored,) = parse_coco_annotations(
        coco, decode_rle=lambda segmentation: pytest.fail("no run-length mask here"))[1][1]
    assert restored.subject == SUBJECT
    assert restored.attributes == {"color": "blue"}


def test_legend_name_shows_the_subject_and_every_attribute_value_in_declared_order(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9,
                      attributes={"grade": "high", "color": "blue"})

    assert _legend_name(pred, scope=_scope(tmp_path, COLOR, GRADE)) == "object blue high"


def test_legend_name_falls_back_to_the_subject_under_no_scope_or_no_attributes(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9)

    assert _legend_name(pred, scope=None) == SUBJECT
    assert _legend_name(pred, scope=_scope(tmp_path)) == SUBJECT
