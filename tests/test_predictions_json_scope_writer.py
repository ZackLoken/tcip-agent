"""In ``encode_predictions``, the run's own scope decides what a prediction document says.

Every prediction's ``subject`` is the scope's, and each attribute the scope declares lands under
``attributes`` decoded through its own declared values; a scope declaring none writes none, and a
run that cannot be decoded honestly encodes no document at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tcip_annotation.json_io import label_document
from tcip_mcp import subject_registry as cr
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing.export import encode_predictions
from tests._chain_fixtures import detected, detection_pass
from tests._producer_fixtures import registry_of, scope_over

if TYPE_CHECKING:
    from tcip_mcp.pipelines.inference.passes import Detection, DetectionPass

SUBJECT = "object"


COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))


def _result(*detections: Detection) -> DetectionPass:
    return detection_pass(
        "img1.jpg", (100, 80), list(detections), cap=len(detections) + 1
    )


def _decoded(result: DetectionPass, scope: ClassScope) -> list:
    data, _dropped = encode_predictions(result, "model:fixture", scope=scope)
    return label_document(data).annotations


def test_an_attributed_run_writes_every_attributes_value_on_every_box(
    tmp_path: Path,
) -> None:
    scope = scope_over(tmp_path, registry_of(SUBJECT, COLOR, GRADE), SUBJECT)
    first, second = _decoded(
        _result(
            detected((1, 1, 5, 5), 0.9, attributes=[1, 2]),
            detected((10, 10, 20, 20), 0.8, attributes=[0, 0]),
        ),
        scope,
    )
    assert (first.subject, first.attributes) == (
        SUBJECT,
        {"color": "blue", "grade": "high"},
    )
    assert (second.subject, second.attributes) == (
        SUBJECT,
        {"color": "red", "grade": "low"},
    )
    assert first.score == 0.9


def test_an_empty_scope_publishes_no_name_it_cannot_decode() -> None:
    """An empty scope refuses to encode, saying it records no subject to publish under."""
    with pytest.raises(ValueError, match="records no subject"):
        encode_predictions(
            _result(detected((1, 1, 5, 5), 0.9)),
            "model:fixture",
            scope=ClassScope(),
        )


def test_a_detector_run_declaring_no_attributes_writes_the_subject_alone(
    tmp_path: Path,
) -> None:
    (written,) = _decoded(
        _result(detected((1, 1, 5, 5), 0.9)),
        scope_over(tmp_path, registry_of(SUBJECT), SUBJECT),
    )
    assert written.subject == SUBJECT
    assert written.attributes == {}


@pytest.mark.parametrize("dropped", ["width", "height", "detections", "cap"])
def test_a_pass_result_missing_a_field_a_document_holds_refuses_naming_it(
    tmp_path: Path, dropped: str
) -> None:
    """A detector's pass missing a field its document holds refuses at publication, naming it.

    The predictor's pass for one image, without that field, refuses as the pass publishes it;
    the same pass whole publishes one document.
    """
    pytest.importorskip("torch")
    from pydantic import ValidationError
    from pydantic_core import to_jsonable_python

    from tcip_mcp.buckets import pass_documents
    from tests._chain_fixtures import detector_pass
    from tests._producer_fixtures import write_image

    image = write_image(image_dir(tmp_path / "ds", "2026-01-01") / "a.jpg")
    p = detector_pass(tmp_path, scope={"subject": SUBJECT})
    (result,) = p.predict([image])
    damaged = to_jsonable_python(result)
    del damaged[dropped]

    with pytest.raises(ValidationError, match=dropped):
        list(pass_documents(p, [damaged]))
    assert len(list(pass_documents(p, [result]))) == 1


def test_a_detector_pass_carrying_a_key_no_field_holds_refuses_at_publication(
    tmp_path: Path,
) -> None:
    """A detector pass carrying a key its record does not hold refuses as it is published.

    The predictor's own pass for one image, with one key added, refuses naming the key; the pass
    as the predictor returned it publishes one document.
    """
    pytest.importorskip("torch")
    from pydantic import ValidationError
    from pydantic_core import to_jsonable_python

    from tcip_mcp.buckets import pass_documents
    from tests._chain_fixtures import detector_pass
    from tests._producer_fixtures import write_image

    image = write_image(image_dir(tmp_path / "ds", "2026-01-01") / "a.jpg")
    p = detector_pass(tmp_path, scope={"subject": SUBJECT})
    (result,) = p.predict([image])
    carrying: Any = {**to_jsonable_python(result), "unrecorded": 1}

    with pytest.raises(ValidationError, match="unrecorded"):
        list(pass_documents(p, [carrying]))
    assert len(list(pass_documents(p, [result]))) == 1


def test_a_regression_pass_publishes_its_own_output_and_the_document_decodes(
    tmp_path: Path,
) -> None:
    """A regression pass publishes its own output through the one publication, and it decodes.

    A head that returns no boxes publishes a document whose decoded task is regression, carrying
    its frame width and its outputs.
    """
    pytest.importorskip("torch")
    import tcip_store
    from tcip_mcp.buckets import pass_documents, publish
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        BESPOKE_REGRESSOR,
        FRAME_STATISTICS,
    )
    from tests._mapping_fixtures import CROP
    from tests._producer_fixtures import register_dataset_at, write_image
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    checkpoint = verified_checkpoint(
        tmp_path,
        model_source={
            "builder": BESPOKE_REGRESSOR,
            "source_files": [BESPOKE_MODELS],
            "task": "regression",
            "normalization": FRAME_STATISTICS,
        },
    )
    images = image_dir(tmp_path / "ds", "2026-01-01")
    write_image(images / "a.png", (32, 32), (90, 90, 90))
    register_dataset_at(tmp_path, tmp_path / "ds", crop=CROP)
    p = prepare(checkpoint, Stated(), images_dir=str(images)).runnable()

    bucket = publish(
        tmp_path,
        "r/2026-01-01",
        pass_documents(p, p.predict(p.paths)),
        producer=checkpoint.producer,
        scope=p.scope,
        execution=p.execution,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=None,
    )

    document = bucket.document_key("a")
    assert document is not None
    decoded = tcip_store.read(document)
    assert isinstance(decoded, dict)
    assert (decoded["task"], decoded["width"]) == ("regression", 32)
    outputs = decoded["outputs"]
    assert isinstance(outputs, dict)
    assert outputs and all(isinstance(v, list) for v in outputs.values())
