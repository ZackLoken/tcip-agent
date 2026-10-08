"""The one predictor: a checkpoint the platform trained, which states no predictor kind, is built
by ``GenericPredictor`` directly, predicts, and is ranked by an active-learning scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_a_checkpoint_without_kind_predicts_and_ranks(tmp_path: Path) -> None:
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer
    from tcip_mcp.pipelines.execution import Stated, execution_record
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_MAX_DETS, registered_checkpoint, table_images,
    )

    path = registered_checkpoint(tmp_path)
    checkpoint = load_registered_checkpoint(path, project=tmp_path)
    assert "kind" not in checkpoint.payload

    predictor = GenericPredictor(checkpoint, device="cpu")
    images = [str(p) for p in list_logical_images(
        table_images(tmp_path / "unlabeled")["images_dir"]).values()]
    result = predictor.predict(images[0], execution_record(
        checkpoint, Stated(conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS), None, None))
    assert {"boxes", "scores", "labels", "count", "cap_hit"} <= set(result)

    ranked = resolve_scorer("uncertainty", checkpoint.task).score(images, predictor)
    assert sorted(path for path, _score in ranked) == sorted(images)
