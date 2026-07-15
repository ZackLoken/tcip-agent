"""Tests for scripts/migrate_labels_to_json.py — legacy YOLO .txt -> per-image JSON."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation.json_io import read_detect, read_detect_pred, read_segment

REPO = Path(__file__).resolve().parents[1]


def _load_migrator():
    spec = importlib.util.spec_from_file_location(
        "migrate_labels_to_json", REPO / "scripts" / "migrate_labels_to_json.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


migrator = _load_migrator()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Tiny legacy-layout dataset: images + YOLO GT/predictions across the migration cases."""
    for date, stems, size in (
        ("2026-02-11", ("img_a", "img_neg", "img_c"), (640, 480)),
        ("2026-03-02", ("img_b",), (320, 240)),
    ):
        d = tmp_path / "images" / date
        d.mkdir(parents=True)
        for stem in stems:
            Image.new("RGB", size, color=(120, 120, 120)).save(d / f"{stem}.jpg")

    catkin = tmp_path / "annotations" / "catkin" / "2026-02-11" / "detect"
    catkin.mkdir(parents=True)
    # cx=0.5 cy=0.5 w=0.25 h=0.5 on 640x480 -> x1=240 y1=120 x2=400 y2=360
    (catkin / "img_a.txt").write_text("1 0.5 0.5 0.25 0.5\n")
    (catkin / "img_neg.txt").write_text("")  # 0-byte confirmed negative
    (catkin / "img_missing.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # no image on disk
    original = catkin / ".original"
    original.mkdir()
    (original / "img_a.txt").write_text("1 0.5 0.5 0.25 0.5\n")

    bush = tmp_path / "annotations" / "bush" / "2026-03-02" / "segment"
    bush.mkdir(parents=True)
    # points on 320x240 -> (32,24) (160,24) (160,120)
    (bush / "img_b.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")

    leaf = tmp_path / "annotations" / "leaf" / "2026-03-02" / "detect"
    leaf.mkdir(parents=True)
    (leaf / "img_b.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    preds = tmp_path / "predictions" / "baseline" / "2026-02-11" / "detect"
    preds.mkdir(parents=True)
    # cx=0.5 cy=0.5 w=0.1 h=0.1 on 640x480 -> x1=288 y1=216 x2=352 y2=264
    (preds / "img_c.txt").write_text("0 0.9 0.5 0.5 0.1 0.1\n")

    return tmp_path


def test_detect_geometry_and_zack_provenance(dataset: Path):
    summary = migrator.migrate(dataset)
    assert summary["converted"][("annotations", "detect")] == 3  # img_a, img_neg, leaf/img_b
    out = dataset / "annotations" / "catkin" / "2026-02-11" / "detect" / "img_a.json"
    boxes, class_ids = read_detect(out)
    assert class_ids == {1}
    (b,) = boxes
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((240, 120, 400, 360), abs=0.05)
    assert b.created_by == "user:zack"
    datetime.fromisoformat(b.created_at)  # valid ISO 8601
    assert b.accepted_by is None and b.accepted_at is None


def test_segment_geometry_and_emily_provenance(dataset: Path):
    migrator.migrate(dataset)
    out = dataset / "annotations" / "bush" / "2026-03-02" / "segment" / "img_b.json"
    polys, class_ids = read_segment(out)
    assert class_ids == {0}
    (p,) = polys
    assert p.points == pytest.approx([(32, 24), (160, 24), (160, 120)], abs=0.05)
    assert p.created_by == "user:emily"


def test_other_gt_is_unknown(dataset: Path):
    migrator.migrate(dataset)
    boxes, _ = read_detect(dataset / "annotations" / "leaf" / "2026-03-02" / "detect" / "img_b.json")
    assert boxes[0].created_by == "unknown"


def test_negative_becomes_present_empty_objects(dataset: Path):
    summary = migrator.migrate(dataset)
    assert summary["negatives"] == 1
    out = dataset / "annotations" / "catkin" / "2026-02-11" / "detect" / "img_neg.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["objects"] == []
    assert data["width"] == 640 and data["height"] == 480
    assert read_detect(out) == ([], set())


def test_prediction_carries_score_and_model_provenance(dataset: Path):
    summary = migrator.migrate(dataset)
    assert summary["converted"][("predictions", "detect")] == 1
    out = dataset / "predictions" / "baseline" / "2026-02-11" / "detect" / "img_c.json"
    preds, _ = read_detect_pred(out)
    (p,) = preds
    assert p.confidence == pytest.approx(0.9)
    assert (p.x1, p.y1, p.x2, p.y2) == pytest.approx((288, 216, 352, 264), abs=0.05)
    assert p.created_by == "baseline"
    assert json.loads(out.read_text(encoding="utf-8"))["objects"][0]["score"] == pytest.approx(0.9)


def test_missing_image_skipped_and_original_dir_ignored(dataset: Path):
    summary = migrator.migrate(dataset)
    assert summary["skipped_missing_image"] == 1
    catkin = dataset / "annotations" / "catkin" / "2026-02-11" / "detect"
    assert not (catkin / "img_missing.json").exists()
    assert not (catkin / ".original" / "img_a.json").exists()


def test_dry_run_writes_nothing(dataset: Path):
    txts_before = sorted(dataset.rglob("*.txt"))
    summary = migrator.migrate(dataset, dry_run=True)
    assert sum(summary["converted"].values()) == 5
    assert list(dataset.rglob("*.json")) == []
    assert sorted(dataset.rglob("*.txt")) == txts_before


def test_remove_source_deletes_txt(dataset: Path):
    migrator.migrate(dataset, remove_source=True)
    catkin = dataset / "annotations" / "catkin" / "2026-02-11" / "detect"
    assert not (catkin / "img_a.txt").exists()
    assert (catkin / "img_a.json").exists()
    assert not (catkin / "img_neg.txt").exists()
    assert (catkin / "img_neg.json").exists()
    assert (catkin / "img_missing.txt").exists()  # skipped labels keep their source
    assert (catkin / ".original" / "img_a.txt").exists()


def test_second_run_is_idempotent(dataset: Path):
    first = migrator.migrate(dataset)
    out = dataset / "annotations" / "catkin" / "2026-02-11" / "detect" / "img_a.json"
    before = out.read_text(encoding="utf-8")
    second = migrator.migrate(dataset)
    assert second["converted"] == first["converted"]
    assert out.read_text(encoding="utf-8") == before


def test_rerun_after_remove_source_is_noop(dataset: Path):
    migrator.migrate(dataset, remove_source=True)
    jsons = sorted(dataset.rglob("*.json"))
    summary = migrator.migrate(dataset, remove_source=True)
    assert summary["converted"][("annotations", "detect")] == 0
    assert summary["converted"][("annotations", "segment")] == 0
    assert sorted(dataset.rglob("*.json")) == jsons
