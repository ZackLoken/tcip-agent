"""``/api/review/validate_reference``'s stamp reads: the route's one strict read of a bucket's
``operating_point.json`` before its two early returns, then, before the review's operating point
is resolved, the refusal for a bucket carrying no stamp at all and the writer-side scope rail a
promotion that merges into an existing stamp is held to, through ``update_sidecar``'s own rail
rather than a second check in this route. A bucket a producer already stamped carries its pair
forward through the promotion untouched.

A stamp lacking the pair, or one that will not decode, predates the writer rail: no live
producer mints either shape, and ``/api/review/action`` itself now refuses to record a fresh
verdict against one (``_review_scope``'s own strict read). Its verdict is therefore seeded
straight through the review store, the same posture ``test_repair_classified_predictions.py``
takes for a bucket the rail would otherwise refuse to build.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import tcip_store as ts
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.review_engine import ReviewEngine
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, sidecar_key
from tcip_mcp.prediction_buckets import bucket_key_of, review_state_dir_of, stage_prediction_shapes
from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec
from tcip_web.app import app

DATE = "2026-04-05"
STEM = "IMG_0300"
IMG_W, IMG_H = 200, 150
BOX = (20.0, 20.0, 60.0, 60.0)
SUBJECT = "bud"
TRAIT = TraitSpec(
    name=SUBJECT,
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    localization_tolerance="half_class_avg_size",
    localization_tolerance_frac=0.5,
    holdout_match_quality_floor=0.5,
    positive_class_name="open",
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    majority_milestone="95per",
    majority_provisional=True,
    phenology_prefix="bud",
    majority_label="opening",
    sliver_policy="class_avg_size",
    sliver_frac=0.5,
    delivers=("leaf_out_05per_date", "leaf_out_50per_date"),
    notes="A neutral fixture trait, not any real crop's own.",
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def seed_bud_trait_spec(tmp_path: Path, _pin_platform_root):
    data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(TRAIT).items()}
    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    ts.replace(traits.trait_spec_key(specs_dir, SUBJECT), data, expect=ts.Version.ABSENT)


def _image(dataset_root: Path) -> Path:
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{STEM}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(90, 100, 80)).save(path)
    return path


def _stage(dataset_root: Path) -> dict:
    return stage_prediction_shapes(
        str(dataset_root), "detector", DATE, STEM,
        annotations=[Annotation(subject=SUBJECT, geometry=BBox(*BOX), score=0.83)],
        img_w=IMG_W, img_h=IMG_H,
    )


def _write_stamp(bucket: Path, stamp: dict) -> None:
    bucket.mkdir(parents=True, exist_ok=True)
    ts.replace(sidecar_key(bucket, "operating_point"), stamp, expect=ts.Version.ABSENT)


def _tiled_stamp(**overrides) -> dict:
    """A stamp that resolves the promotion's own tiled precondition (an untiled run), so the
    scope rail under test is what a request actually refuses on, never an unrelated geometry gate.
    """
    stamp = {
        "checkpoint_sha256": "sha-detector", "experiment_id": None, "validated": False,
        "id_map": {SUBJECT: 0}, "subject": SUBJECT, "attribute": None,
        "operating_point": {"tiled": {"value": False}, "conf": {"value": 0.25}},
    }
    stamp.update(overrides)
    return stamp


def _seed_accepted_verdict(dataset_root: Path, bucket: Path) -> None:
    """A completed, accepted verdict against ``bucket``, filed straight through the review store:
    the posture for a bucket ``/api/review/action`` itself now refuses to record a fresh verdict
    against, since its own scope resolution reads the same stamp strictly.
    """
    key = (bucket_key_of(bucket), f"{STEM}.jpg")
    state_dir = review_state_dir_of(dataset_root)
    engine = ReviewEngine(str(state_dir))
    engine.raw_state.update({"verdicts": {key: {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": SUBJECT,
         "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]}}})
    engine.save_review_state()


def _damage_stamp(bucket: Path) -> None:
    """Corrupt the already-written stamp's bytes in place, wherever the bound backend keeps it."""
    import os

    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = sidecar_key(bucket, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute("update records set value = ? where store = ? and parts = ?",
                    (b"{not json", key.store, encode_parts(key.parts)))
    finally:
        conn.close()


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_over_a_neither_key_stamp_refuses_naming_the_repair_command(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    _image(dataset_root)
    _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    neither_key = {k: v for k, v in _tiled_stamp().items() if k not in ("subject", "attribute")}
    _write_stamp(bucket, neither_key)
    _seed_accepted_verdict(dataset_root, bucket)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 400
    assert "repair-classified-predictions" in resp.json()["detail"]
    assert "subject" not in (read_operating_point_sidecar(bucket) or {})


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_over_an_undecodable_stamp_returns_400_with_the_seams_message(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    _image(dataset_root)
    _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    _write_stamp(bucket, _tiled_stamp())
    _seed_accepted_verdict(dataset_root, bucket)
    _damage_stamp(bucket)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"] or "decode" in resp.json()["detail"].lower()


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_over_an_already_stamped_bucket_carries_its_pair_forward(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    _image(dataset_root)
    _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    _write_stamp(bucket, _tiled_stamp())
    _seed_accepted_verdict(dataset_root, bucket)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 200, resp.text
    stamp = read_operating_point_sidecar(bucket)
    assert (stamp["subject"], stamp["attribute"]) == (SUBJECT, None)


def _accept(client: TestClient, dataset_root: Path, img: Path, gt: Path, pred_path: str) -> None:
    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": img.name, "image_path": str(img),
        "gt_path": str(gt), "pred_path": pred_path,
        "det_type": "fp", "class_name": SUBJECT, "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0, "bbox": list(BOX), "action": "accepted",
        "user": "breeder",
    })
    assert resp.status_code == 200, resp.text


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_over_a_bare_bucket_refuses_before_any_stamp_is_written(
    client: TestClient, tmp_path: Path,
) -> None:
    """A bucket no producer ever stamped (the staging door's own shape) is refused before the
    review's operating point is resolved: no checkpoint, experiment or generation conf stands
    behind it, and a staged bucket is reviewed through the accept path, never promoted."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    assert read_operating_point_sidecar(bucket) is None  # genuinely bare: no producer ever stamped it

    gt = dataset_root / "annotations" / DATE / f"{STEM}.json"
    _accept(client, dataset_root, img, gt, staged["path"])

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "carries no operating_point.json stamp" in detail
    assert "never promoted to a validation reference" in detail
    assert read_operating_point_sidecar(bucket) is None  # refused before any stamp was written


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_reports_a_stamp_read_contention_as_retryable(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock timeout at the promotion's own stamp read is retryable infrastructure, not a bad
    request, the same posture the stamping block's own StoreBusy handling already takes. The
    contention is injected because a real lock timeout is not constructible through one client."""
    import tcip_mcp.pipelines.resolution as resolution_mod
    from tcip_store.errors import StoreBusy

    dataset_root = tmp_path / "data"
    _image(dataset_root)
    _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    _write_stamp(bucket, _tiled_stamp())
    _seed_accepted_verdict(dataset_root, bucket)
    busy_key = sidecar_key(bucket, "operating_point")

    def _busy(*args, **kwargs):
        raise StoreBusy((busy_key,), busy_key, 5.0)

    monkeypatch.setattr(resolution_mod, "read_operating_point_sidecar", _busy)
    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 503, resp.text


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_over_a_producer_written_stamp_promotes(
    client: TestClient, tmp_path: Path,
) -> None:
    """The admitting case built through the platform's own producer (operating_point_stamp +
    write_sidecar), not the raw seed the pre-rail fixtures above use for shapes no live producer
    can write."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    dataset_root = tmp_path / "data"
    _image(dataset_root)
    _stage(dataset_root)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE))
    stamp = operating_point_stamp(
        {"tiled": {"value": False}, "conf": {"value": 0.25}},
        validated=False, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map={SUBJECT: 0}, subject=SUBJECT, attribute=None, trait=None, dataset_hash=None,
        checkpoint=None, checkpoint_sha256="sha-detector", experiment_id=None, images_dir=None,
        raster_path=None, produced_at=None,
    )
    write_sidecar(bucket, stamp)
    _seed_accepted_verdict(dataset_root, bucket)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 200, resp.text
    stamp_after = read_operating_point_sidecar(bucket)
    assert (stamp_after["subject"], stamp_after["attribute"]) == (SUBJECT, None)
