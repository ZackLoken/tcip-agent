"""``encode_predictions``: the run's own scope decides where a decoded label lands. With an
attribute, the decoded value lands under ``attributes[attribute]`` and ``subject`` carries the
object class; without one, the output is a detector run's. Every label decodes through the run's
admitted map, and a run that cannot be decoded honestly encodes no document at all.
"""

from __future__ import annotations

import pytest

from tcip_annotation.json_io import annotations_from_bytes
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing.export import encode_predictions

SUBJECT = "bud"
ATTRIBUTE = "bud_opening"
ID_MAP = {"open": 0, "closed": 1}
CLASSIFIED = ClassScope(subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP)


def _result(*, boxes, scores, labels, width=100, height=80) -> dict:
    return {"image": "img1.jpg", "boxes": boxes, "scores": scores, "labels": labels,
            "width": width, "height": height}


def _decoded(result: dict, scope: ClassScope) -> list:
    data, _dropped = encode_predictions(result, scope=scope)
    return annotations_from_bytes(data, source="img1.json")


def test_a_classified_run_decodes_the_value_under_the_attribute() -> None:
    (written,) = _decoded(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]), CLASSIFIED)
    assert written.subject == SUBJECT
    assert written.attributes == {ATTRIBUTE: "open"}
    assert written.score == 0.9


def test_a_classified_result_carrying_a_label_outside_the_map_refuses_naming_id_and_map() -> None:
    # label 3 decodes to 0-indexed id 2, not a key of ID_MAP (which only has ids 0 and 1).
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[3])

    with pytest.raises(ValueError, match=r"detection 0 decoded to id 2") as excinfo:
        encode_predictions(result, scope=CLASSIFIED)

    assert "[0, 1]" in str(excinfo.value)


def test_a_classified_run_refuses_the_whole_result_at_its_first_unmapped_detection() -> None:
    """The refusal fires per-detection, and nothing is encoded for the whole result: the
    decodable detection ahead of the unmapped one is not either."""
    result = _result(boxes=[[1, 1, 5, 5], [10, 10, 20, 20]], scores=[0.9, 0.8], labels=[1, 3])

    with pytest.raises(ValueError):
        encode_predictions(result, scope=CLASSIFIED)


def test_a_detector_run_with_no_id_map_publishes_no_name_it_cannot_decode() -> None:
    """A run with no admitted map has no name to publish a label under; a raw index written as a
    subject would read as a class no vocabulary declares."""
    with pytest.raises(ValueError, match=r"detection 0 decoded to id 0"):
        encode_predictions(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]),
                           scope=ClassScope())


def test_a_detector_run_with_a_recorded_map_decodes_the_name_into_subject() -> None:
    (written,) = _decoded(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]),
                          ClassScope(subject=SUBJECT, id_map={SUBJECT: 0}))
    assert written.subject == SUBJECT
    assert written.attributes == {}


def test_a_detector_result_carrying_a_label_outside_the_map_refuses() -> None:
    """A detector run's labels are decoded through its admitted map as a classified run's are, so
    a label outside it refuses by name rather than publishing its raw index."""
    with pytest.raises(ValueError, match=r"detection 0 decoded to id 8"):
        encode_predictions(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[9]),
                           scope=ClassScope(subject=SUBJECT, id_map={SUBJECT: 0}))


@pytest.mark.parametrize("dropped", ["image", "scores", "labels", "width"])
def test_a_result_missing_a_field_a_document_holds_refuses_naming_it(dropped):
    """No stored value stands in for one the head did not give: a result without its source
    image, scores, labels or frame size encodes nothing, rather than a document of defaults."""
    result = _result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1])
    del result[dropped]

    with pytest.raises(ValueError, match=f"carries no \\['{dropped}'\\]"):
        encode_predictions(result, scope=CLASSIFIED)

    assert len(_decoded(_result(boxes=[[1, 1, 5, 5]], scores=[0.9], labels=[1]),
                        CLASSIFIED)) == 1


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
                     raster_identity=None, assessment_id=None)

    document = bucket.document("a.png")
    assert document is not None
    decoded = json.loads(document.read_bytes())
    assert (decoded["image"], decoded["task"], decoded["width"]) == ("a", "regression", 32)
    assert decoded["outputs"] and all(isinstance(v, list) for v in decoded["outputs"].values())
