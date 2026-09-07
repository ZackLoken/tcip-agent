"""Shared fixtures for ``clear_prediction_bucket`` tests: a canonical bucket published through
``run_inference`` with a fake predictor, its experiment carried to a terminal state, and review
state recorded against it through ``ReviewEngine``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def assert_source_stamps_absent(bucket: Path) -> None:
    """Assert a ``read_versioned`` of each of ``SIDECAR_FILENAMES``' five stamps answers absent at
    ``bucket``: the one helper every source-empty assertion goes through, rather than each test
    checking ``operating_point`` alone and calling the source empty of stamps on that one answer."""
    import tcip_store as ts
    from tcip_annotation.json_io import SIDECAR_FILENAMES

    from tcip_mcp.pipelines.resolution import sidecar_key

    for filename in SIDECAR_FILENAMES:
        document = filename.removesuffix(".json")
        assert ts.read_versioned(sidecar_key(bucket, document), default=None).value is None, document


def stub_predictor(monkeypatch, *, boxes: tuple[tuple[float, float, float, float], ...] = (
    (10.0, 10.0, 30.0, 30.0),
)) -> None:
    """A ``GenericPredictor`` stand-in that predicts a fixed set of boxes per image, the same
    shape ``tests/test_run_inference_bucket_handling.py``'s own fake predictor uses."""

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [list(b) for b in boxes], "scores": [0.9] * len(boxes),
                     "labels": [1] * len(boxes), "count": len(boxes)}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)


def stub_checkpoint_verification(monkeypatch) -> None:
    """``load_registered_checkpoint`` admits whatever path it is given, the same stand-in
    ``test_run_inference_bucket_handling.py``'s autouse fixture installs."""
    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    def _stub(path, *a, **kw):
        p = Path(path)
        sha = model_registry_mod._sha256_of_bytes(p.read_bytes()) if p.is_file() else "stub-sha256"
        return stub_verified_checkpoint(str(path), sha256=sha)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


def write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(path)


def build_published_bucket(
    tmp_path: Path,
    monkeypatch,
    *,
    experiment_id: str,
    dataset_root: Path | None = None,
    model: str = "m",
    date: str | None = "2026-03-02",
    stems: tuple[str, ...] = ("img",),
    state: str = "completed",
) -> dict[str, Any]:
    """Publish a canonical bucket through ``run_inference`` with a fake predictor, carry
    ``experiment_id`` through ``create_experiment``/``update_status`` to ``state`` (``None``
    leaves it ``running``), and return the pieces a ``clear_prediction_bucket`` test needs:
    ``dataset_root``, ``bucket``, ``images_dir``, ``checkpoint``, ``experiment_id``, ``result``
    (``run_inference``'s own response).
    """
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.inference_tools import run_inference

    stub_predictor(monkeypatch)
    stub_checkpoint_verification(monkeypatch)

    root = dataset_root if dataset_root is not None else (tmp_path / "ds")
    images_dir = tmp_path / f"{experiment_id}_images"
    for stem in stems:
        write_image(images_dir / f"{stem}.png")

    ckpt = tmp_path / f"{experiment_id}.pt"
    if not ckpt.exists():
        ckpt.write_bytes(b"stub")

    create_experiment(experiment_id, {"model_source": {"builder": "x:y"}})
    update_status(experiment_id, "running")

    bucket = prediction_dir(root, model, date)
    result = run_inference(str(ckpt), str(images_dir), output_dir=str(bucket), tile=False,
                           experiment_id=experiment_id)
    assert "error" not in result, result

    if state is not None and state != "running":
        update_status(experiment_id, state)

    return {"dataset_root": root, "bucket": bucket, "images_dir": images_dir,
            "checkpoint": ckpt, "experiment_id": experiment_id, "result": result}


def record_review_verdict(bucket: Path, review_state_dir: Path, img_name: str) -> None:
    """Record one accepted detection verdict against ``img_name`` under ``bucket``'s own key,
    through the review engine (never a raw store write)."""
    from tcip_annotation import Annotation, BBox
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine

    from tcip_mcp.prediction_buckets import bucket_key_of

    engine = ReviewEngine(review_state_dir)
    ctx = ReviewContext(
        img_name=img_name, img_width=100, img_height=100,
        preds=[Annotation(subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0), score=0.9)],
    )
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(bucket), det, ctx, action="accepted")


def mark_bulk_accepted(bucket: Path, review_state_dir: Path, img_name: str) -> None:
    """Mark ``img_name`` reviewed under ``bucket``'s own key with zero detection entries (a
    bulk accept / confirmed negative), through the review engine."""
    from tcip_annotation.review_engine import ReviewEngine

    from tcip_mcp.prediction_buckets import bucket_key_of

    ReviewEngine(review_state_dir).mark_image_reviewed(bucket_key_of(bucket), img_name)
