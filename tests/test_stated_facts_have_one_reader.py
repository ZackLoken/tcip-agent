"""A record that does not state a fact its reader needs is refused by name where it is read.

The facts are a prediction's score, a ground-truth record's content, the target a loader builds
from annotations and a completion digest's stored grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._producer_fixtures import image_label_key, label_image

if TYPE_CHECKING:
    from tcip_store import Key

SUBJECT = "strig"
IMG = 100
BOX = BBox(10.0, 10.0, 30.0, 30.0)


def _stored(image: Path, anns: list[Annotation]) -> Key:
    """Write ``anns`` as the document of ``image`` and return its key."""
    label_image(image, anns, IMG, IMG)
    return image_label_key(image)


def _read_back(image: Path, anns: list[Annotation]) -> list[Annotation]:
    return json_io.read_label_document(_stored(image, anns)).annotations


def test_a_prediction_document_stating_no_score_is_refused_where_it_is_read(
    tmp_path: Path,
) -> None:
    key = _stored(
        image_dir(tmp_path, UNDATED_BUCKET) / "p.png",
        [
            Annotation(subject=SUBJECT, geometry=BOX, score=0.8),
            Annotation(subject=SUBJECT, geometry=BOX),
        ],
    )

    with pytest.raises(
        json_io.UnreadableLabelDocumentError, match="record 1 .*'score'"
    ):
        json_io.read_predictions(key)


def test_a_scored_prediction_document_reads_whole(tmp_path: Path) -> None:
    key = _stored(
        image_dir(tmp_path, UNDATED_BUCKET) / "p.png",
        [Annotation(subject=SUBJECT, geometry=BOX, score=0.8)],
    )

    assert [a.score for a in json_io.read_predictions(key).annotations] == [
        0.8
    ]


@pytest.mark.parametrize("reader", ["pairing", "evaluation_record"])
def test_no_reader_stands_in_a_score_for_a_prediction_stating_none(
    tmp_path: Path, reader: str
) -> None:
    from tcip_annotation.matching import pair_proposals
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    gt = _read_back(
        image_dir(tmp_path, UNDATED_BUCKET) / "g.png",
        [Annotation(subject=SUBJECT, geometry=BOX)],
    )
    preds = _read_back(
        image_dir(tmp_path, UNDATED_BUCKET) / "p.png",
        [Annotation(subject=SUBJECT, geometry=BOX)],
    )

    with pytest.raises(ValueError, match="'score'"):
        if reader == "pairing":
            pair_proposals(gt, preds, {"kind": "iou", "iou_threshold": 0.5})
        else:
            records_from_annotation(gt, preds, width=IMG, height=IMG, cap=None)


def test_the_instance_loader_builds_its_target_from_the_polygons_it_reads(
    tmp_path: Path,
) -> None:
    """The instance loader builds its target from the polygons it reads.

    Over the geometry the loader declares it reads, a box beside the polygon is no instance and
    no mask.
    """
    from tests._producer_fixtures import dataset_over, write_image

    images = image_dir(tmp_path, UNDATED_BUCKET)
    _stored(
        write_image(images / "p0.png", (IMG, IMG), (40, 40, 40)),
        [
            Annotation(subject=SUBJECT, geometry=BOX),
            Annotation(
                subject=SUBJECT,
                geometry=Polygon([[(50.0, 50.0), (70.0, 50.0), (70.0, 70.0)]]),
            ),
        ],
    )

    _image, target = dataset_over("instance_seg", images, subject=SUBJECT)[0]

    assert target["boxes"].tolist() == [[50.0, 50.0, 70.0, 70.0]]
    assert target["labels"].tolist() == [1]
    assert len(target["masks"]) == 1 and int(target["masks"][0].sum()) > 0


def test_the_completion_digest_reads_a_box_on_the_writers_own_grid(
    tmp_path: Path,
) -> None:
    """The completion digest reads a box on the label writer's own grid.

    A box off the stored grid digests the same before it is written as after it is read back.
    """
    anns = [
        Annotation(subject=SUBJECT, geometry=BBox(0.005, 10.0, 10.004, 20.0))
    ]

    assert json_io.subject_digest(anns, SUBJECT) == json_io.subject_digest(
        _read_back(image_dir(tmp_path, UNDATED_BUCKET) / "c.png", anns),
        SUBJECT,
    )
