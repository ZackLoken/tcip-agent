"""In ``encode_predictions``, the run's own scope decides what a prediction document says.

Every prediction's ``subject`` is the scope's, and each attribute the scope declares lands under
``attributes`` decoded through its own declared values; a scope declaring none writes none, and a
run that cannot be decoded honestly encodes no document at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation.json_io import label_document
from tcip_mcp import subject_registry as cr
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing.export import (
    DetectionResult,
    encode_predictions,
)
from tests._producer_fixtures import registry_of, scope_over

SUBJECT = "object"


COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))


def _result(
    *,
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    attributes: list[list[int]] | None = None,
) -> DetectionResult:
    return DetectionResult(
        image="img1.jpg",
        boxes=boxes,
        scores=scores,
        attributes=attributes,
        width=100,
        height=80,
        cap=len(boxes) + 1,
    )


def _decoded(result: DetectionResult, scope: ClassScope) -> list:
    data, _dropped = encode_predictions(result, "model:fixture", scope=scope)
    return label_document(data).annotations


def test_an_attributed_run_writes_every_attributes_value_on_every_box(
    tmp_path: Path,
) -> None:
    scope = scope_over(tmp_path, registry_of(SUBJECT, COLOR, GRADE), SUBJECT)
    first, second = _decoded(
        _result(
            boxes=[(1, 1, 5, 5), (10, 10, 20, 20)],
            scores=[0.9, 0.8],
            attributes=[[1, 2], [0, 0]],
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
            _result(boxes=[(1, 1, 5, 5)], scores=[0.9]),
            "model:fixture",
            scope=ClassScope(),
        )


def test_a_detector_run_declaring_no_attributes_writes_the_subject_alone(
    tmp_path: Path,
) -> None:
    (written,) = _decoded(
        _result(boxes=[(1, 1, 5, 5)], scores=[0.9]),
        scope_over(tmp_path, registry_of(SUBJECT), SUBJECT),
    )
    assert written.subject == SUBJECT
    assert written.attributes == {}


@pytest.mark.parametrize("dropped", ["width", "height", "boxes", "scores", "cap"])
def test_a_pass_result_missing_a_field_a_document_holds_refuses_naming_it(
    tmp_path: Path, dropped: str
) -> None:
    """A detector's result missing a field its document holds refuses at the pass, naming it.

    The predictor's result for one image, without that field, refuses as the pass publishes it;
    the same result whole publishes one document.
    """
    pytest.importorskip("torch")
    from pydantic import ValidationError

    from tests._chain_fixtures import predicted, published
    from tests._producer_fixtures import write_image

    image = write_image(image_dir(tmp_path / "ds", "2026-01-01") / "a.jpg")
    damaged = predicted(image, [SUBJECT])
    del damaged[dropped]

    with pytest.raises(ValidationError, match=dropped):
        published(tmp_path, "damaged", [damaged], scope={"subject": SUBJECT})

    whole = published(
        tmp_path, "whole", [predicted(image, [SUBJECT])], scope={"subject": SUBJECT}
    )
    assert len(whole.documents) == 1


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
    from tests._producer_fixtures import write_image
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
    p = prepare(checkpoint, Stated(), images_dir=str(images)).runnable()

    bucket = publish(
        tmp_path,
        tmp_path / "ds",
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
