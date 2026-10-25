"""``run_inference``'s images regime, publishing a bucket end to end.

The checkpoint's own recorded ``scope`` decodes the detections' attribute ids into named values
on the subject, and the bucket's scope, read back through ``read_bucket``, carries the same
attributes. A detector whose scope declares no attribute writes its detection's subject with no
attributes, and its bucket's scope carries none.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_annotation.state import Annotation
    from tcip_mcp.buckets import Bucket

torch = pytest.importorskip("torch")

from tcip_mcp import subject_registry as cr  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._predictor_fixtures import BOX, StubPredictor, install  # noqa: E402
from tests._producer_fixtures import write_image  # noqa: E402
from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS  # noqa: E402

SUBJECT = "bud"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))


def _checkpoint(tmp_path: Path, *attributes: cr.Attribute) -> str:
    """Return the checkpoint a run of ``tmp_path`` registered by completing.

    The run's dataset declares ``attributes`` on :data:`SUBJECT`.
    """
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(
        tmp_path,
        data={"num_channels": 3, "scope": {"subject": SUBJECT}},
        registry=cr.SubjectRegistry(
            subjects=(
                cr.Subject(
                    name=SUBJECT, description="", attributes=attributes
                ),
            )
        ),
    )


def _attributed_predictor() -> StubPredictor:
    """Return a ``StubPredictor`` of two boxes with attribute ids ``[0, 2]`` and ``[1, 0]``."""
    return StubPredictor(
        boxes=(BOX, (40.0, 40.0, 60.0, 60.0)),
        scores=(0.9, 0.8),
        attributes=[[0, 2], [1, 0]],
    )


def _published(
    tmp_path: Path, checkpoint: str, images_dir: Path
) -> tuple[Bucket, list[Annotation]]:
    """Run inference over ``images_dir`` into the bucket ``out/2026-01-01``.

    Return the bucket and the annotations read from its ``img`` document.
    """
    from tcip_annotation.json_io import read_predictions
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    result = run_inference(
        tmp_path,
        checkpoint,
        str(images_dir),
        bucket="out/2026-01-01",
        stated=Stated(tile=False, **SAMPLE_DETECTOR_PASS),
        actor=None,
    )
    assert "error" not in result, result
    bucket = read_bucket(result["dataset_root"], result["bucket"])
    key = bucket.document_key("img")
    assert key is not None
    return bucket, read_predictions(key).annotations


def test_an_attributed_run_writes_every_attribute_value_and_stamps_its_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "images" / "2026-01-01"
    write_image(images_dir / "img.png", (100, 100))
    checkpoint = _checkpoint(tmp_path, COLOR, GRADE)
    install(monkeypatch, _attributed_predictor())

    bucket, anns = _published(tmp_path, checkpoint, images_dir)

    assert [a.attributes for a in anns] == [
        {"color": "red", "grade": "high"},
        {"color": "blue", "grade": "low"},
    ]
    assert all(a.subject == SUBJECT for a in anns)

    assert bucket.scope.attributes == (COLOR, GRADE)


def test_a_detector_run_declaring_no_attribute_writes_the_ordinary_shape_and_stamps_its_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = tmp_path / "images" / "2026-01-01"
    write_image(images_dir / "img.png", (100, 100))
    checkpoint = _checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())

    bucket, anns = _published(tmp_path, checkpoint, images_dir)

    assert len(anns) == 1
    assert anns[0].subject == SUBJECT
    assert not anns[0].attributes

    assert bucket.scope.attributes == ()
