"""A published prediction bucket is readable by ``buckets.read_bucket``.

This drives the real writer (``run_inference``'s publication) and the real reader
(``buckets.read_bucket``) against each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.execution import MaskCut

torch = pytest.importorskip("torch")


def _publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **stated: bool | int | float | str | MaskCut | None,
) -> dict:
    """Run ``run_inference`` over one dated capture of the dataset ``dataset``, stubbed.

    The dataset is registered in ``tmp_path`` at :data:`~tests._mapping_fixtures.CROP` first.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import detected
    from tests._mapping_fixtures import CROP
    from tests._predictor_fixtures import BOX, StubPredictor, install
    from tests._producer_fixtures import register_dataset_at, write_image
    from tests._verified_checkpoint_fixtures import project_checkpoint

    images_dir = image_dir(tmp_path / "dataset", "2026-03-01")
    write_image(images_dir / "capture_a.png", (160, 120), (70, 90, 110))
    register_dataset_at(tmp_path, tmp_path / "dataset", crop=CROP)
    install(
        monkeypatch,
        StubPredictor(
            width=160,
            height=120,
            detections=[detected(BOX, 0.95)],
            task="detection",
            in_chans=3,
        ),
    )
    return run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        images_dir=str(images_dir),
        bucket="baseline/2026-03-01",
        device="cpu",
        stated=Stated(**stated),
        actor=None,
    )


def test_the_record_the_publication_writes_is_the_record_the_reader_takes_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.model_registry import Producer
    from tcip_mcp.pipelines.execution import EXECUTION_RECORD
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CROSS_TILE_NMS,
        SAMPLE_OVERLAP,
    )

    result = _publish(
        tmp_path,
        monkeypatch,
        tile=True,
        tile_size=64,
        overlap=SAMPLE_OVERLAP,
        conf=0.3,
        postprocess="nms",
        cross_tile_nms=SAMPLE_CROSS_TILE_NMS,
        rationale="a sample reason the test states",
    )

    assert "error" not in result, result
    bucket = read_bucket(result["dataset_root"], result["bucket"])
    assert (
        EXECUTION_RECORD.dump_python(bucket.execution, mode="json")
        == result["execution"]
    )
    assert (
        bucket.execution.tile_size,
        bucket.execution.sources["tile_size"],
    ) == (64, "explicit")
    assert bucket.date == "2026-03-01" and bucket.documents == {
        "capture_a": "capture_a.png"
    }
    assert bucket.producer == Producer.model_validate(result["producer"])
