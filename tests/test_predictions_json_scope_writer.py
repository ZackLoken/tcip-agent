"""``encode_predictions``: the run's own scope decides what a prediction document says. Every
prediction's ``subject`` is the scope's, and each attribute the scope declares lands under
``attributes`` decoded through its own declared values; a scope declaring none writes none, and a
run that cannot be decoded honestly encodes no document at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation.json_io import annotations_from_bytes
from tcip_mcp import subject_registry as cr
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing.export import encode_predictions

SUBJECT = "object"


def _scope(tmp_path: Path, *attributes: cr.Attribute) -> ClassScope:
    """The class space the admission reads for :data:`SUBJECT` over a dataset declaring
    ``attributes`` on it."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tests._producer_fixtures import registry_over

    registry_over(tmp_path, cr.SubjectRegistry(subjects=(
        cr.Subject(name=SUBJECT, attributes=attributes),)))
    (tmp_path / "annotations").mkdir(exist_ok=True)
    return registry_scope(tmp_path / "annotations", SUBJECT)


COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))


def _result(*, boxes, scores, labels, attributes=None, width=100, height=80) -> dict:
    result = {"image": "img1.jpg", "boxes": boxes, "scores": scores, "labels": labels,
              "width": width, "height": height}
    if attributes is not None:
        result["attributes"] = attributes
    return result


def _decoded(result: dict, scope: ClassScope) -> list:
    data, _dropped = encode_predictions(result, "model:fixture", scope=scope)
    return annotations_from_bytes(data, source="img1.json")


def test_an_attributed_run_writes_every_attributes_value_on_every_box(tmp_path) -> None:
    scope = _scope(tmp_path, COLOR, GRADE)
    first, second = _decoded(_result(boxes=[[1, 1, 5, 5], [10, 10, 20, 20]], scores=[0.9, 0.8],
                                     labels=[1, 1], attributes=[[1, 2], [0, 0]]), scope)
    assert (first.subject, first.attributes) == (SUBJECT, {"color": "blue", "grade": "high"})
    assert (second.subject, second.attributes) == (SUBJECT, {"color": "red", "grade": "low"})
    assert first.score == 0.9


def test_an_empty_scope_publishes_no_name_it_cannot_decode() -> None:
    """A run with no admitted subject has no name to publish a detection under; a raw index
    written as a subject would read as a class no vocabulary declares."""
    with pytest.raises(ValueError, match="records no subject"):
        encode_predictions(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]),
                           "model:fixture", scope=ClassScope())


def test_a_detector_run_declaring_no_attributes_writes_the_subject_alone(tmp_path) -> None:
    (written,) = _decoded(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]),
                          _scope(tmp_path))
    assert written.subject == SUBJECT
    assert written.attributes == {}


@pytest.mark.parametrize("dropped", ["image", "scores", "labels", "width"])
def test_a_result_missing_a_field_a_document_holds_refuses_naming_it(tmp_path, dropped):
    """No stored value stands in for one the head did not give: a result without its source
    image, scores, labels or frame size encodes nothing, rather than a document of defaults."""
    scope = _scope(tmp_path, COLOR)
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1], attributes=[[0]])
    del result[dropped]

    with pytest.raises(ValueError, match=f"carries no \\['{dropped}'\\]"):
        encode_predictions(result, "model:fixture", scope=scope)

    assert len(_decoded(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1],
                                attributes=[[0]]), scope)) == 1


def test_a_regression_pass_publishes_its_own_output_and_the_document_decodes(tmp_path):
    """A head that returns no boxes publishes what it returned through the one publication, in
    its own document shape, rather than being refused by a detection document's fields."""
    pytest.importorskip("torch")
    import json

    from PIL import Image

    from tcip_mcp.buckets import pass_documents, publish
    from tcip_mcp.pipelines.execution import Stated, prepare_pass
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(tmp_path, model_source={
        "builder": "tests.bespoke_models:build_bespoke_regressor", "task": "regression"})
    images = tmp_path / "ds" / "images" / "2026-01-01"
    images.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(90, 90, 90)).save(images / "a.png")
    p = prepare_pass(checkpoint, Stated(), images_dir=str(images))

    bucket = publish(tmp_path, tmp_path / "ds" / "predictions" / "r" / "2026-01-01",
                     pass_documents(p, p.predict(p.paths)), producer=checkpoint.producer,
                     scope=p.scope, execution=p.execution, raster_path=None,
                     raster_identity=None, assessment_id=None, actor=None)

    document = bucket.document("a.png")
    assert document is not None
    decoded = json.loads(document.read_bytes())
    assert (decoded["image"], decoded["task"], decoded["width"]) == ("a", "regression", 32)
    assert decoded["outputs"] and all(isinstance(v, list) for v in decoded["outputs"].values())
