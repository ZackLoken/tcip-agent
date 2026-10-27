"""score_predictions answers an unreadable document with an error naming it.

A present, unreadable label or prediction document is an error naming the document, never a raise
through the MCP tool boundary, on both the single-image and folder paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.tools.annotation_tools import score_predictions
from tests._producer_fixtures import image_label_key, label_image, write_image
from tests._training_values import TRIAGE_WEIGHTS

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.subject_registry import SubjectRegistry

pytest.importorskip("torch")

ONE_BUD = [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))]

SCORED_AT = (0.5, 0.25)
"""A sample IoU and confidence each scoring here states."""


def _published(
    project: Path,
    name: str,
    image: Path,
    registry: SubjectRegistry | None = None,
) -> Bucket:
    """Publish one ``bud`` box at 0.9 on ``image`` as the bucket ``name`` and return the bucket.

    The box carries the last value of each attribute ``registry`` declares on it (none for no
    registry).
    """
    from tests._chain_fixtures import published

    attributes = (
        registry.subjects[0].attributes if registry is not None else ()
    )
    result = {
        "image": str(image),
        "width": 100,
        "height": 80,
        "boxes": [[1.0, 1.0, 5.0, 5.0]],
        "scores": [0.9],
        "labels": [1],
        "cap": 2,
    }
    if attributes:
        result["attributes"] = [[len(a.values) - 1 for a in attributes]]
    return published(
        project, name, [result], scope={"subject": "bud"}, registry=registry
    )


def test_score_predictions_single_image_reports_an_unreadable_gt(
    tmp_path: Path,
) -> None:
    from tests._record_damage_fixtures import damage_record

    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "IMG_0000.jpg")
    label_image(img, ONE_BUD, 100, 80)
    damage_record(image_label_key(img), b"{not json")
    _published(tmp_path, "baseline", img)

    res = score_predictions(tmp_path, str(img), "baseline", *SCORED_AT)

    assert "error" in res
    assert "IMG_0000" in res["error"]


def test_score_predictions_folder_reports_an_unreadable_prediction(
    tmp_path: Path,
) -> None:
    from tests._record_damage_fixtures import damage_record

    images = image_dir(tmp_path / "ds", UNDATED_BUCKET)
    image = write_image(images / "IMG_0000.jpg")
    label_image(image, ONE_BUD, 100, 80)
    bucket = _published(tmp_path, "baseline", image)
    damage_record(bucket.document_key("IMG_0000"), b"{not json")

    res = score_predictions(tmp_path, str(images), "baseline", *SCORED_AT)

    assert "error" in res
    assert "IMG_0000" in res["error"]


def test_score_predictions_over_attributed_documents_scores_the_object_class(
    tmp_path: Path,
) -> None:
    """Scoring attributed predictions scores the object class's localization.

    A prediction carrying attribute values carries the object class in subject, and the score
    over it is one true positive with no error.
    """
    from tcip_mcp import subject_registry as cr
    from tests._producer_fixtures import registry_of

    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "IMG_0000.jpg")
    label_image(img, ONE_BUD, 100, 80)
    _published(
        tmp_path,
        "classifier",
        img,
        registry_of(
            "bud", cr.Attribute("opening", "categorical", ("closed", "open"))
        ),
    )

    res = score_predictions(tmp_path, str(img), "classifier", *SCORED_AT)

    assert "error" not in res
    assert res["tp"] == 1


def test_a_predicted_image_with_no_label_document_is_refused_as_a_reference(
    tmp_path: Path,
) -> None:
    """A predicted image with no label document is refused as a reference, naming it.

    An image the bucket predicted and nobody labeled has no reference: scoring and the count
    triage refuse naming its document as having no record.
    """
    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "IMG_0000.jpg")
    bucket = _published(tmp_path, "baseline", img)

    res = score_predictions(tmp_path, str(img), "baseline", *SCORED_AT)

    assert "IMG_0000" in res["error"] and "has no record" in res["error"], res
    with pytest.raises(
        UnreadableLabelDocumentError, match="IMG_0000.*has no record"
    ):
        get_worst_predictions(bucket, 8, **TRIAGE_WEIGHTS)


def test_an_image_the_bucket_names_no_document_for_is_unknown_never_a_miss(
    tmp_path: Path,
) -> None:
    """A labeled image the bucket did not predict is left out and named, not scored a miss.

    A labeled image the bucket's record did not predict is no false negative and no triage
    score: scoring and triage leave it out and name it, while the image it did predict scores.
    """
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    images = image_dir(tmp_path / "ds", UNDATED_BUCKET)
    predicted, unpredicted = (
        write_image(images / f"{stem}.jpg")
        for stem in ("IMG_0000", "IMG_0001")
    )
    for image in (predicted, unpredicted):
        label_image(image, ONE_BUD, 100, 80)
    bucket = _published(tmp_path, "baseline", predicted)

    scored = score_predictions(tmp_path, str(images), "baseline", *SCORED_AT)
    triaged = get_worst_predictions(bucket, 8, **TRIAGE_WEIGHTS)

    assert "error" not in scored, scored
    assert (scored["total_tp"], scored["total_fn"], scored["image_count"]) == (
        1,
        0,
        1,
    )
    assert scored["not_predicted"] == [str(unpredicted)]
    assert [w["stem"] for w in triaged["worst_images"]] == ["IMG_0000"]
    assert triaged["not_predicted"] == ["IMG_0001"]
    assert (
        "not predicted"
        in score_predictions(
            tmp_path, str(unpredicted), "baseline", *SCORED_AT
        )["error"]
    )
