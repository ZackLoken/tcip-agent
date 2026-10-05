"""score_predictions: a present, unreadable label or prediction document is an error naming the
document, never a raise through the MCP tool boundary, on both the single-image and folder paths.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.tools.annotation_tools import score_predictions
from tests._producer_fixtures import image_label_key, label_image, write_image

pytest.importorskip("torch")

ONE_BUD = [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))]


def _published(project: Path, name: str, image: Path, registry=None):
    """One ``bud`` box at 0.9 on ``image``, carrying the last value of each attribute
    ``registry`` declares on it (none for no registry), published as the bucket ``name``; the
    bucket."""
    from tests._chain_fixtures import published

    attributes = registry.subjects[0].attributes if registry is not None else ()
    result = {"image": str(image), "width": 100, "height": 80, "boxes": [[1.0, 1.0, 5.0, 5.0]],
              "scores": [0.9], "labels": [1]}
    if attributes:
        result["attributes"] = [[len(a.values) - 1 for a in attributes]]
    return published(project, name, [result], scope={"subject": "bud"}, registry=registry)


def test_score_predictions_single_image_reports_an_unreadable_gt(tmp_path: Path) -> None:
    from tests._record_damage_fixtures import damage_record

    img = tmp_path / "images" / UNDATED_BUCKET / "IMG_0000.jpg"
    write_image(img)
    label_image(img, ONE_BUD, 100, 80)
    damage_record(image_label_key(img), b"{not json")
    _published(tmp_path, "baseline", img)

    res = score_predictions(str(img), "baseline")

    assert "error" in res
    assert "IMG_0000" in res["error"]


def test_score_predictions_folder_reports_an_unreadable_prediction(tmp_path: Path) -> None:
    from tests._record_damage_fixtures import damage_record

    images = tmp_path / "ds" / "images" / UNDATED_BUCKET
    write_image(images / "IMG_0000.jpg")
    label_image(images / "IMG_0000.jpg", ONE_BUD, 100, 80)
    bucket = _published(tmp_path, "baseline", images / "IMG_0000.jpg")
    damage_record(bucket.document_key("IMG_0000"), b"{not json")

    res = score_predictions(str(images), "baseline")

    assert "error" in res
    assert "IMG_0000" in res["error"]


def test_score_predictions_over_attributed_documents_scores_the_object_class(
    tmp_path: Path,
) -> None:
    """A prediction carrying attribute values carries the object class in subject, so this
    scores its localization, a valid number about finding the object, never an attribute head's
    call."""
    from tcip_mcp import subject_registry as cr

    img = tmp_path / "images" / UNDATED_BUCKET / "IMG_0000.jpg"
    write_image(img)
    label_image(img, ONE_BUD, 100, 80)
    _published(tmp_path, "classifier", img,
               cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(
                   cr.Attribute("opening", "categorical", ("closed", "open")),)),)))

    res = score_predictions(str(img), "classifier")

    assert "error" not in res
    assert res["tp"] == 1


def test_a_predicted_image_with_no_label_document_is_refused_as_a_reference(tmp_path: Path):
    """An image the bucket predicted and nobody labeled has no reference: scoring and the count
    triage refuse naming its document, never measuring it against an empty one."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    img = tmp_path / "images" / UNDATED_BUCKET / "IMG_0000.jpg"
    write_image(img)
    bucket = _published(tmp_path, "baseline", img)

    res = score_predictions(str(img), "baseline")

    assert "IMG_0000" in res["error"] and "has no record" in res["error"], res
    with pytest.raises(UnreadableLabelDocument, match="IMG_0000.*has no record"):
        get_worst_predictions(bucket)


def test_an_image_the_bucket_names_no_document_for_is_unknown_never_a_miss(tmp_path: Path):
    """A labeled image the bucket's record never predicted is no false negative and no triage
    score: scoring and triage leave it out and name it, while the image it did predict scores."""
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    images = tmp_path / "ds" / "images" / UNDATED_BUCKET
    for stem in ("IMG_0000", "IMG_0001"):
        write_image(images / f"{stem}.jpg")
        label_image(images / f"{stem}.jpg", ONE_BUD, 100, 80)
    bucket = _published(tmp_path, "baseline", images / "IMG_0000.jpg")

    scored = score_predictions(str(images), "baseline")
    triaged = get_worst_predictions(bucket)

    assert "error" not in scored, scored
    assert (scored["total_tp"], scored["total_fn"], scored["image_count"]) == (1, 0, 1)
    assert scored["not_predicted"] == [str(images / "IMG_0001.jpg")]
    assert [w["stem"] for w in triaged["worst_images"]] == ["IMG_0000"]
    assert triaged["not_predicted"] == ["IMG_0001"]
    assert "not predicted" in score_predictions(str(images / "IMG_0001.jpg"), "baseline")["error"]
