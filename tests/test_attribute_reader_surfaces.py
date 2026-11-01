"""Reader-side surfaces that hold a bucket to its own recorded scope.

They are ``count_by_class`` under a scope declaring no attribute of the positive state, a bucket
record that does not decode, the COCO reader's ``attributes`` handling, and the prediction
render's legend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import subject_registry as cr
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.traits import PositiveState
from tests._producer_fixtures import (
    published_document,
    registry_of,
    scope_over,
)
from tests._training_values import VERTEX_SPACING

if TYPE_CHECKING:
    from tcip_store import Key

SUBJECT = "object"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "high"))


def _published(
    tmp_path: Path, name: str, annotations: list[Annotation]
) -> Key:
    """Publish ``annotations`` at 8 px as the bucket ``name``'s document of ``img``; return its key.

    The publication is :func:`~tests._producer_fixtures.published_document`'s.
    """
    image = image_dir(tmp_path, "2024-05-01") / "img.png"
    key = published_document(tmp_path, name, image, annotations, 8, 8).document_key(
        "img"
    )
    assert key is not None
    return key


def test_count_by_class_under_a_scope_declaring_no_attribute_of_the_state_counts_none_positive(
    tmp_path: Path,
) -> None:
    """``count_by_class`` counts no positive under a scope declaring no attribute of the state.

    Under a scope declaring only ``color``, a record carrying the state's ``grade`` value counts
    as one object, no positive and one unclassified.
    """
    p = _published(
        tmp_path,
        "m",
        [
            Annotation(
                subject=SUBJECT,
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"grade": "high"},
            )
        ],
    )

    total, positive, unclassified = phenology.count_by_class(
        p,
        PositiveState(attribute="grade", value="high"),
        scope=scope_over(tmp_path, registry_of(SUBJECT, COLOR), SUBJECT),
    )

    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_reads_the_states_own_attribute_among_several(
    tmp_path: Path,
) -> None:
    """``count_by_class`` reads the positive state's own attribute among several.

    Under a scope declaring ``color`` and ``grade``, the state's own ``grade`` decides, not
    ``color``; a record carrying no ``grade`` value refuses by name.
    """
    p = _published(
        tmp_path,
        "m",
        [
            Annotation(
                subject=SUBJECT,
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"color": "red", "grade": "high"},
            ),
            Annotation(
                subject=SUBJECT,
                geometry=BBox(4, 4, 6, 6),
                score=0.9,
                attributes={"color": "blue", "grade": "low"},
            ),
        ],
    )
    scope = scope_over(tmp_path, registry_of(SUBJECT, COLOR, GRADE), SUBJECT)

    assert phenology.count_by_class(
        p, PositiveState(attribute="grade", value="high"), scope=scope
    ) == (2, 1, 0)
    unassessed = _published(
        tmp_path,
        "m2",
        [
            Annotation(
                subject=SUBJECT,
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"color": "red"},
            )
        ],
    )
    with pytest.raises(json_io.UndeclaredValueError, match="'grade'"):
        phenology.count_by_class(
            unassessed,
            PositiveState(attribute="grade", value="high"),
            scope=scope,
        )


def test_an_undecodable_bucket_record_refuses_by_name_rather_than_reading_as_unclassified(
    tmp_path: Path,
) -> None:
    """Reading a bucket whose record does not decode is refused naming the record.

    Such a bucket has no scope to read its documents under.
    """
    pytest.importorskip("torch")
    import sqlite3

    from tcip_mcp.buckets import NotABucketError, read_bucket
    from tcip_store.file_backend import database_file
    from tests._chain_fixtures import predicted, published

    image = image_dir(tmp_path / "ds", "2026-05-02") / "s1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"")
    bucket = published(
        tmp_path,
        "classifier/2026-05-02",
        [predicted(image, ["one"])],
        scope={"subject": SUBJECT},
    )
    with sqlite3.connect(database_file(str(bucket.root))) as db:
        db.execute(
            "update records set value = ? where store = 'prediction_buckets'",
            (b"{not json",),
        )

    with pytest.raises(
        NotABucketError, match=r"prediction_buckets\['classifier/2026-05-02'\]"
    ):
        read_bucket(bucket.root, bucket.name)


def test_the_coco_reader_keeps_a_records_attribute_values() -> None:
    from tcip_annotation.format_io import parse_coco_annotations

    coco = {
        "images": [
            {"id": 1, "file_name": "img.jpg", "width": 10, "height": 10}
        ],
        "categories": [{"id": 0, "name": SUBJECT}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [1, 1, 4, 4],
                "score": 0.9,
                "attributes": {"color": "blue"},
            }
        ],
    }

    (restored,) = parse_coco_annotations(
        coco,
        decode_rle=lambda segmentation: pytest.fail("no run-length mask here"),
        vertex_spacing=VERTEX_SPACING,
    )[1][1]
    assert restored.subject == SUBJECT
    assert restored.attributes == {"color": "blue"}


def test_legend_name_shows_the_subject_and_every_attribute_value_in_declared_order(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(
        subject=SUBJECT,
        geometry=BBox(1, 1, 5, 5),
        score=0.9,
        attributes={"grade": "high", "color": "blue"},
    )

    assert (
        _legend_name(
            pred,
            scope=scope_over(
                tmp_path, registry_of(SUBJECT, COLOR, GRADE), SUBJECT
            ),
        )
        == "object blue high"
    )


def test_legend_name_falls_back_to_the_subject_under_no_scope_or_no_attributes(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.vision_tools import _legend_name

    pred = Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5), score=0.9)

    assert _legend_name(pred, scope=None) == SUBJECT
    assert (
        _legend_name(
            pred, scope=scope_over(tmp_path, registry_of(SUBJECT), SUBJECT)
        )
        == SUBJECT
    )
