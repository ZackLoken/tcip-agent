"""score_predictions: a present, unreadable label or prediction document is an error naming the
file, never a raise through the MCP tool boundary, on both the single-image and folder paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.tools.annotation_tools import score_predictions

pytest.importorskip("torch")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(path)


def _published(project: Path, out: Path, image: Path, registry=None) -> Path:
    """One ``bud`` box at 0.9 on ``image``, carrying the last value of each attribute
    ``registry`` declares on it (none for no registry), published as the bucket ``out``; its
    directory."""
    from tests._chain_fixtures import published

    attributes = registry.subjects[0].attributes if registry is not None else ()
    result = {"image": str(image), "width": 100, "height": 80, "boxes": [[1.0, 1.0, 5.0, 5.0]],
              "scores": [0.9], "labels": [1]}
    if attributes:
        result["attributes"] = [[len(a.values) - 1 for a in attributes]]
    return published(project, out, [result], scope={"subject": "bud"}, registry=registry).path


def test_score_predictions_single_image_reports_an_unreadable_gt(tmp_path: Path) -> None:
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    img = tmp_path / "images" / "IMG_0000.jpg"
    _write_image(img)
    bad = labels / "IMG_0000.json"
    bad.write_bytes(b"{not json")
    preds = _published(tmp_path, tmp_path / "predictions" / "baseline", img)

    res = score_predictions(str(img), str(preds))

    assert "error" in res
    assert str(bad) in res["error"]


def test_score_predictions_folder_reports_an_unreadable_prediction(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    images = root / "images"
    labels = root / "annotations"
    labels.mkdir(parents=True)
    _write_image(images / "IMG_0000.jpg")
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    preds = _published(tmp_path, root / "predictions" / "baseline", images / "IMG_0000.jpg")
    bad = preds / "IMG_0000.json"
    bad.write_bytes(b"{not json")

    res = score_predictions(str(images), str(preds))

    assert "error" in res
    assert str(bad) in res["error"]


def test_score_predictions_over_attributed_documents_scores_the_object_class(
    tmp_path: Path,
) -> None:
    """A prediction carrying attribute values carries the object class in subject, so this
    scores its localization, a valid number about finding the object, never an attribute head's
    call."""
    from tcip_mcp import subject_registry as cr

    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    img = tmp_path / "images" / "IMG_0000.jpg"
    _write_image(img)
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    preds = _published(tmp_path, tmp_path / "predictions" / "classifier", img,
                       cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(
                           cr.Attribute("opening", "categorical", ("closed", "open")),)),)))

    res = score_predictions(str(img), str(preds))

    assert "error" not in res
    assert res["tp"] == 1


def test_an_image_the_bucket_names_no_document_for_is_unknown_never_a_miss(tmp_path: Path):
    """A labeled image the bucket's record never predicted is no false negative and no triage
    score: scoring and triage leave it out and name it, while the image it did predict scores."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    root = tmp_path / "ds"
    images, labels = root / "images", root / "annotations"
    labels.mkdir(parents=True)
    for stem in ("IMG_0000", "IMG_0001"):
        _write_image(images / f"{stem}.jpg")
        write_annotations(labels / f"{stem}.json",
                          [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 100, 80)
    preds = _published(tmp_path, root / "predictions" / "baseline", images / "IMG_0000.jpg")

    scored = score_predictions(str(images), str(preds))
    triaged = get_worst_predictions(read_bucket(preds), str(labels))

    assert "error" not in scored, scored
    assert (scored["total_tp"], scored["total_fn"], scored["image_count"]) == (1, 0, 1)
    assert scored["not_predicted"] == [str(images / "IMG_0001.jpg")]
    assert [w["stem"] for w in triaged["worst_images"]] == ["IMG_0000"]
    assert triaged["not_predicted"] == ["IMG_0001"]
    assert "not predicted" in score_predictions(str(images / "IMG_0001.jpg"), str(preds))["error"]
