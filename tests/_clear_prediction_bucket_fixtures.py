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
), scores: tuple[float, ...] | None = None) -> None:
    """A ``GenericPredictor`` stand-in that predicts a fixed set of boxes per image, the same
    shape ``tests/test_run_inference_bucket_handling.py``'s own fake predictor uses.

    ``scores`` names each box's own confidence, one per entry of ``boxes``; omitted, every box
    scores 0.9. Lets a caller pin a prediction's score against an earned admission rule's own
    conf, above or below it, rather than trust an unstated default to land on either side.
    """
    box_scores = list(scores) if scores is not None else [0.9] * len(boxes)
    assert len(box_scores) == len(boxes), "scores must name one confidence per box"

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [list(b) for b in boxes], "scores": list(box_scores),
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
    boxes: tuple[tuple[float, float, float, float], ...] = ((10.0, 10.0, 30.0, 30.0),),
    scores: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Publish a canonical bucket through ``run_inference`` with a fake predictor, carry
    ``experiment_id`` through ``create_experiment``/``update_status`` to ``state`` (``None``
    leaves it ``running``), and return the pieces a ``clear_prediction_bucket`` test needs:
    ``dataset_root``, ``bucket``, ``images_dir``, ``checkpoint``, ``experiment_id``, ``result``
    (``run_inference``'s own response). ``boxes``/``scores`` thread through to
    :func:`stub_predictor`, so a caller can pin a prediction's score against an earned rule.
    """
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.inference_tools import run_inference

    stub_predictor(monkeypatch, boxes=boxes, scores=scores)
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


def earn_validated_stamp(bucket: Path, dataset_root: Path, *, trait: str) -> dict:
    """Replace ``bucket``'s published stamp's ``operating_point`` with one earned through the
    same two-phase gate a producer runs (``open_validation``, then ``seal_validation``), keeping
    the run's own ``experiment_id``, ``checkpoint_sha256``, ``subject`` and ``attribute`` so the
    door's other checks stay meaningful; returns the stamp as stored after the merge.
    """
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.pipelines.resolution import (
        open_validation, operating_point_stamp, read_operating_point_sidecar, seal_validation,
        update_sidecar,
    )

    stored = read_operating_point_sidecar(bucket)
    assert stored is not None

    common = dict(n_images=20, objects_per_image=80, miss_pattern=[0] * 20,
                  fp_pattern=[1] * 20, score=0.9, fp_score=0.05)
    cal = dense_records(id_prefix="c", **common)
    hold = dense_records(id_prefix="h", shift=5.0, **common)
    labels_dir = dataset_root / "annotations" / "2026-03-04"
    labels_dir.mkdir(parents=True, exist_ok=True)

    draft = open_validation(
        document="operating_point",
        evidence={"resolver": "resolve_operating_point",
                  "inputs": {"dataset_hash": "h1", "calibration_records": cal,
                             "holdout_records": hold, "staged_conf_floor": 0.01,
                             "tiled": False}},
        trait=trait, checkpoint_sha256=stored.get("checkpoint_sha256"),
        producing_experiment_id=None,
        reference_inputs={"dataset_root": str(dataset_root),
                          "label_dirs": {"calibration": labels_dir},
                          "stated_values": {"split_identity": "clear-bucket-admitting"}},
    )
    earned_body = operating_point_stamp(
        draft.result.to_provenance()["operating_point"], validated=True, validated_by=None,
        tile_size_validated=None, shippable_issues=draft.result.shippable_issues(), id_map=None,
        subject=stored.get("subject"), attribute=stored.get("attribute"), trait=trait,
        dataset_hash="h1", checkpoint=stored.get("checkpoint"),
        checkpoint_sha256=stored.get("checkpoint_sha256"), experiment_id=stored.get("experiment_id"),
        images_dir=stored.get("images_dir"), raster_path=stored.get("raster_path"),
        produced_at=stored.get("produced_at"),
    )
    stamped = seal_validation(
        draft, dataset_root=dataset_root, bucket_dirs=[bucket], stamp_body=earned_body)

    def _merge(current: dict) -> dict:
        return {**current, "validated": True, "validated_by": stamped["validated_by"],
                "operating_point": earned_body["operating_point"], "trait": trait}

    assert update_sidecar(bucket, _merge) is True
    updated = read_operating_point_sidecar(bucket)
    assert updated is not None
    assert updated.get("validated_by") is not None
    return updated


def mark_bulk_accepted(bucket: Path, review_state_dir: Path, img_name: str) -> None:
    """Mark ``img_name`` reviewed under ``bucket``'s own key with zero detection entries (a
    bulk accept / confirmed negative), through the review engine."""
    from tcip_annotation.review_engine import ReviewEngine

    from tcip_mcp.prediction_buckets import bucket_key_of

    ReviewEngine(review_state_dir).mark_image_reviewed(bucket_key_of(bucket), img_name)
