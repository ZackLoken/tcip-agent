"""run_inference publishes a bucket once: per-image prediction JSON and one bucket.json, into a
directory the publication creates, never into one that exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._verified_checkpoint_fixtures import project_checkpoint  # noqa: E402

UNTILED = Stated(tile=False)
"""The one execution value every run here states."""


def _stubbed(monkeypatch) -> StubPredictor:
    return install(monkeypatch, StubPredictor(task="detection", in_chans=3))


def _images(directory: Path) -> Path:
    directory.mkdir(parents=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(directory / "img.png")
    return directory


def test_run_inference_writes_each_document_then_the_record_and_one_audit_line(
    tmp_path, monkeypatch,
):
    import hashlib

    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = Path(project_checkpoint(tmp_path))
    images_dir = _images(tmp_path / "images")
    _stubbed(monkeypatch)
    out = tmp_path / "out"

    ran = run_inference(tmp_path, str(ckpt), str(images_dir), output_dir=str(out), stated=UNTILED)

    assert "error" not in ran, ran
    data = json.loads((out / "img.json").read_text())
    assert data["image"] == "img"
    assert (data["width"], data["height"]) == (100, 100)
    (ann,) = data["annotations"]
    assert ann["subject"] == "bud"                 # label 1 -> id 0 -> the checkpoint's own map
    assert ann["score"] == pytest.approx(0.9)
    assert ann["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])  # COCO xywh of xyxy [10,10,30,30]
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert ann["created_by"] == f"model:{ckpt.stem}@{digest[:12]}"
    bucket = read_bucket(out)
    assert bucket.documents == {"img": "img.png"}
    assert bucket.producer["checkpoint_sha256"] == digest and bucket.assessment_id is None
    rows = [r for key in dict.fromkeys((audit_log_key(tmp_path), audit_log_key(out)))
            for r in ts.read_log(key).records if r["tool"] == "prediction_bucket_published"]
    assert [r["arguments"] for r in rows] == [{"predictions_dir": str(out), "dropped_boxes": 0}]


def test_a_second_run_into_a_published_bucket_refuses_before_any_pass_and_leaves_it(
    tmp_path, monkeypatch,
):
    """A bucket is published once: a second run into the same directory refuses before the pass
    runs, leaving the first bucket's documents and record as they were, and a new bucket admits
    the same run."""
    from tcip_mcp.dataset_layout import prediction_root
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(tmp_path / "dataset" / "images" / "2026-01-01")
    out = prediction_root(tmp_path / "dataset") / "baseline" / "2026-01-01"
    predictor = _stubbed(monkeypatch)
    ckpt = project_checkpoint(tmp_path)
    first = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), stated=UNTILED)
    assert "error" not in first, first
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

    again = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), stated=UNTILED)
    preview = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), dry_run=True)
    fresh = run_inference(tmp_path, ckpt, str(images_dir),
                          output_dir=str(out.parent.parent / "baseline-2" / "2026-01-01"),
                          stated=UNTILED)

    assert "already exists" in again["error"]
    assert predictor.calls == 2  # the first run and the fresh one, never the refused one
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    assert preview["bucket_exists"] is True
    assert "error" not in fresh, fresh


def test_a_run_over_an_empty_images_directory_publishes_a_bucket_of_no_documents(
    tmp_path, monkeypatch,
):
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    empty = tmp_path / "empty_images"
    empty.mkdir()
    _stubbed(monkeypatch)
    out = tmp_path / "out"

    ran = run_inference(tmp_path, project_checkpoint(tmp_path), str(empty), output_dir=str(out),
                        stated=UNTILED)

    assert "error" not in ran, ran
    assert ran["image_count"] == 0
    assert read_bucket(out).documents == {}


def test_a_bucket_outside_any_dataset_is_published_where_it_was_asked_for(tmp_path, monkeypatch):
    """A bucket outside any dataset records no capture of one; refusing the write would reject
    legitimate exploratory work."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(tmp_path / "captures")
    _stubbed(monkeypatch)
    out = tmp_path / "scratch_preds"

    ran = run_inference(tmp_path, project_checkpoint(tmp_path), str(images_dir),
                        output_dir=str(out), stated=UNTILED)

    assert "error" not in ran, ran
    assert Path(ran["output_dir"]) == out
    assert read_bucket(out).dataset_id is None
    assert (out / "img.json").is_file()


def test_a_dry_run_previews_the_both_sources_refusal(tmp_path):
    """A preview previews the same refusal a real call would hit."""
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"stub")

    result = run_inference(
        tmp_path, project_checkpoint(tmp_path), images_dir=str(images_dir),
        raster_path=str(raster_path), output_dir=str(tmp_path / "out"), dry_run=True)

    assert "exactly one of images_dir or raster_path" in result["error"]


def test_a_dry_run_names_the_bucket_and_the_execution_and_writes_nothing(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(tmp_path / "images")
    _stubbed(monkeypatch)
    out = tmp_path / "out"

    result = run_inference(tmp_path, project_checkpoint(tmp_path), str(images_dir),
                           output_dir=str(out), stated=UNTILED, dry_run=True)

    assert "error" not in result, result
    assert result["output_dir"] == str(out)
    assert result["bucket_exists"] is False
    assert result["execution"]["tile_size"] is None
    assert not out.exists()
