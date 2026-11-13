"""Tests for annotation tools and label I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tcip_annotation import Annotation, BBox
from tcip_annotation.json_io import label_document, read_label_document
from tcip_annotation.matching import iou_matrix, pair_proposals

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject
    from tests._producer_fixtures import WrittenDataset


def test_bbox_creation() -> None:
    b = BBox(x1=10, y1=20, x2=50, y2=60)
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 50, 60)


def test_label_document_reads_the_name_based_record() -> None:
    # A label document is the name-based per-image record: bbox is pixel xywh, subject by name.
    anns = label_document(
        {
            "image": "img_001",
            "width": 640,
            "height": 480,
            "annotations": [
                {"subject": "bud", "bbox": [100, 100, 50, 50]},
                {"subject": "bud", "bbox": [200, 200, 40, 40]},
            ],
        }
    ).annotations
    assert len(anns) == 2
    assert all(isinstance(a.geometry, BBox) for a in anns)
    assert {a.subject for a in anns} == {"bud"}


def test_save_and_read_roundtrip(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, label_key
    from tests._producer_fixtures import label_image

    anns = [
        Annotation(
            subject="bud", geometry=BBox(x1=100, y1=100, x2=200, y2=200)
        ),
        Annotation(
            subject="leaf", geometry=BBox(x1=300, y1=300, x2=350, y2=350)
        ),
    ]
    label_image(
        image_dir(tmp_path, UNDATED_BUCKET) / "test.png", anns, 640, 480
    )
    read_back = read_label_document(
        label_key(tmp_path, UNDATED_BUCKET, "test")
    ).annotations
    assert len(read_back) == 2
    assert {a.subject for a in read_back} == {"bud", "leaf"}
    # Check approximate roundtrip (floating point tolerance)
    for orig, read in zip(anns, read_back):
        assert abs(orig.geometry.x1 - read.geometry.x1) < 2
        assert abs(orig.geometry.y1 - read.geometry.y1) < 2


def test_iou_matrix() -> None:
    iou = iou_matrix(
        [[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]]
    )
    assert iou[0, 0] == 1.0
    assert 0.1 < iou[0, 1] < 0.2  # 25/175 ≈ 0.143
    assert iou[0, 2] == 0.0


def test_pair_proposals() -> None:
    gt = [Annotation(subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10))]
    preds = [
        Annotation(
            subject="bud", geometry=BBox(x1=0, y1=0, x2=10, y2=10), score=0.9
        ),
        Annotation(
            subject="bud", geometry=BBox(x1=50, y1=50, x2=60, y2=60), score=0.8
        ),
    ]
    assert pair_proposals(
        gt, preds, {"kind": "iou", "iou_threshold": 0.5}
    ).pairs == [(0, 0)]


def test_score_predictions_single_image(data_dir: WrittenDataset) -> None:
    from tcip_mcp.tools.annotation_tools import score_predictions

    result = score_predictions(
        data_dir.root,
        str(data_dir.images[0]),
        "live/2-11-26",
        iou_threshold=0.5,
        conf_threshold=0.25,
    )
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result
    assert result["tp"] + result["fn"] == 2  # 2 GT boxes


def test_the_annotation_doors_answer_a_photograph_that_will_not_decode_with_the_headers_sentence(
    project: Path, bound: BoundProject
) -> None:
    """A file named like a photograph that holds none answers the header read's own sentence.

    ``read_annotations`` and ``save_annotations`` are called through the server over a one-byte
    ``.jpg``; each answers the error ``image_path_dimensions`` raises for it.
    """
    import pytest

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import image_path_dimensions
    from tests._producer_fixtures import stub_images
    from tests._server_calls import decoded_answer, results_through_handshake

    (stub,) = stub_images(image_dir(project, "2026-04-01"), ["a_1.jpg"])
    with pytest.raises(OSError) as refused:
        image_path_dimensions(stub)

    results = results_through_handshake(
        [
            ("read_annotations", {"image_path": str(stub)}),
            ("save_annotations", {"image_path": str(stub), "annotations": []}),
        ],
        bound=bound,
    )

    assert not any(result.is_error for result in results), results
    assert [decoded_answer(result) for result in results] == [
        {"error": str(refused.value)}
    ] * 2
