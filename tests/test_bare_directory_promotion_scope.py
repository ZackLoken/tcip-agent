"""A bare prediction directory (no ``operating_point.json`` stamp at all) promoted through
``/api/review/validate_reference``: the review it promotes was a detector review, since a bare
directory admits no classified one, so the stamp this route writes for it is
``(subject=req.subject, attribute=None)``, earned only after every prediction record in the
directory positively carries that subject.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import tcip_store as ts
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.state import Annotation, BBox
from tcip_mcp import traits
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
from tcip_mcp.prediction_buckets import stage_prediction_shapes
from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec
from tcip_web.app import app

DATE = "2026-04-01"
IMG_W, IMG_H = 200, 150
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


def _image(dataset_root: Path, stem: str) -> Path:
    d = dataset_root / "images"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(90, 100, 80)).save(path)
    return path


def _stage(dataset_root: Path, stem: str, *, subject: str, box) -> dict:
    return stage_prediction_shapes(
        str(dataset_root), "baseline", DATE, stem,
        annotations=[Annotation(subject=subject, geometry=BBox(*box), score=0.83)],
        img_w=IMG_W, img_h=IMG_H,
    )


def _accept(client: TestClient, dataset_root: Path, img: Path, gt: Path, pred_path: str, box,
            *, class_name: str = SUBJECT) -> None:
    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": img.name, "image_path": str(img),
        "gt_path": str(gt), "pred_path": pred_path,
        "det_type": "fp", "class_name": class_name, "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0, "bbox": list(box), "action": "accepted",
        "user": "breeder",
    })
    assert resp.status_code == 200, resp.text


@pytest.mark.usefixtures("seed_bud_trait_spec")
@pytest.mark.xfail(
    strict=True, reason=(
        "design gap, not yet landed: /api/review/validate_reference resolves 'tiled' for "
        "resolve_operating_point_from_review only from the bucket's own sidecars (routes/"
        "validation.py's review_tiled), which is None for a bucket with no stamp at all; "
        "resolve_operating_point (operating_point.py) then raises ValueError('requires an "
        "explicit tiled=<bool>') before validation.py ever reaches _stamp_body/"
        "_require_bare_bucket_subject, so a genuinely bare (zero-sidecar) bucket 400s on the "
        "tiled precondition and can never reach the (req.subject, None) stamp write the design "
        "states for it."
    ),
)
def test_a_promotion_over_a_bare_directory_writes_the_stated_subject_with_no_attribute(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    stem = "IMG_0001"
    box = (20.0, 20.0, 60.0, 60.0)
    img = _image(dataset_root, stem)
    staged = _stage(dataset_root, stem, subject=SUBJECT, box=box)
    bucket = Path(prediction_dir(dataset_root, "baseline", DATE))
    assert read_operating_point_sidecar(bucket) is None  # genuinely bare: no producer ever stamped it

    gt = dataset_root / "annotations" / DATE / f"{stem}.json"
    _accept(client, dataset_root, img, gt, staged["path"], box)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 200, resp.text
    assert resp.json()["buckets_stamped"] == [str(bucket)]
    stamp = read_operating_point_sidecar(bucket)
    assert stamp is not None
    assert (stamp["subject"], stamp["attribute"]) == (SUBJECT, None)


@pytest.mark.usefixtures("seed_bud_trait_spec")
@pytest.mark.xfail(
    strict=True, reason=(
        "design gap, not yet landed: _require_bare_bucket_subject (routes/validation.py) runs "
        "only after resolve_operating_point_from_review returns without raising, and that call "
        "always raises ValueError('requires an explicit tiled=<bool>') first over a bucket with "
        "no stamp at all (see the sibling test in this module), so the second-subject refusal "
        "this design states is unreachable for a genuinely bare bucket; the request 400s for the "
        "tiled precondition instead, never naming the foreign subject."
    ),
)
def test_a_bare_directory_holding_a_second_subject_refuses_naming_it(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    stem_a, stem_b = "IMG_0001", "IMG_0002"
    box = (20.0, 20.0, 60.0, 60.0)
    img_a = _image(dataset_root, stem_a)
    img_b = _image(dataset_root, stem_b)
    staged_a = _stage(dataset_root, stem_a, subject=SUBJECT, box=box)
    staged_b = _stage(dataset_root, stem_b, subject="shoot", box=box)
    bucket = Path(prediction_dir(dataset_root, "baseline", DATE))

    gt_a = dataset_root / "annotations" / DATE / f"{stem_a}.json"
    gt_b = dataset_root / "annotations" / DATE / f"{stem_b}.json"
    _accept(client, dataset_root, img_a, gt_a, staged_a["path"], box, class_name=SUBJECT)
    _accept(client, dataset_root, img_b, gt_b, staged_b["path"], box, class_name="shoot")

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": SUBJECT, "pred_dir": str(bucket),
        "subject": SUBJECT})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "shoot" in detail and "one subject's" in detail
    assert read_operating_point_sidecar(bucket) is None  # refused before any stamp was written
