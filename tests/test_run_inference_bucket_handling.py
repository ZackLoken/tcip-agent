"""run_inference publishes a bucket once: one prediction document per image and the bucket's
record under the images' dataset root, in one commit, never over a bucket that exists."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import tcip_store  # noqa: E402
from tcip_mcp.dataset_layout import bucket_key  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._producer_fixtures import gray_frame  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    SAMPLE_CONF, SAMPLE_DETECTOR_PASS, project_checkpoint,
)

UNTILED = Stated(tile=False, conf=SAMPLE_CONF)
"""The execution values every untiled run here states."""

BUCKET = "out/2026-01-01"


def _stubbed(monkeypatch) -> StubPredictor:
    return install(monkeypatch, StubPredictor(task="detection", in_chans=3))


def _images(directory: Path) -> Path:
    gray_frame(directory, 100)
    return directory


def test_run_inference_publishes_each_document_the_record_and_one_audit_line(
    tmp_path, monkeypatch,
):
    import hashlib

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = Path(project_checkpoint(tmp_path))
    images_dir = _images(tmp_path / "images" / "2026-01-01")
    _stubbed(monkeypatch)

    ran = run_inference(tmp_path, str(ckpt), str(images_dir), bucket=BUCKET, stated=UNTILED)

    assert "error" not in ran, ran
    bucket = read_bucket(tmp_path, BUCKET)
    key = bucket.document_key("img")
    assert key is not None
    data = tcip_store.read(key)
    assert key.parts[-1] == "img" and "image" not in data
    assert (data["width"], data["height"]) == (100, 100)
    (ann,) = data["annotations"]
    assert ann["subject"] == "bud"                 # label 1 -> id 0 -> the checkpoint's own map
    assert ann["score"] == pytest.approx(0.9)
    assert ann["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])  # COCO xywh of xyxy [10,10,30,30]
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert ann["created_by"] == f"model:{ckpt.stem}@{digest[:12]}"
    assert bucket.documents == {"img": "img.png"}
    assert bucket.producer["checkpoint_sha256"] == digest and bucket.assessment_id is None
    rows = [r for r in tcip_store.read_log(audit_log_key(tmp_path)).records
            if r["tool"] == "prediction_bucket_published"]
    assert [r["arguments"] for r in rows] == [
        {"dataset_root": str(tmp_path), "bucket": BUCKET, "dropped_boxes": 0}]


def test_a_second_run_into_a_published_bucket_refuses_and_leaves_it(tmp_path, monkeypatch):
    """A bucket is published once: a second run naming the same bucket refuses, leaving the first
    bucket's documents and record as they were, and a new bucket admits the same run."""
    from tcip_mcp.dataset_layout import PREDICTION_DOCUMENTS
    from tcip_mcp.tools.inference_tools import run_inference

    root = tmp_path / "dataset"
    images_dir = _images(root / "images" / "2026-01-01")
    _stubbed(monkeypatch)
    ckpt = project_checkpoint(tmp_path)
    first = run_inference(tmp_path, ckpt, str(images_dir), bucket=BUCKET, stated=UNTILED)
    assert "error" not in first, first

    def published() -> dict:
        keys = [bucket_key(root, BUCKET),
                *tcip_store.keys(PREDICTION_DOCUMENTS, str(root), (BUCKET,))]
        return {key: tcip_store.read_versioned(key).version for key in keys}

    before = published()

    again = run_inference(tmp_path, ckpt, str(images_dir), bucket=BUCKET, stated=UNTILED)
    preview = run_inference(tmp_path, ckpt, str(images_dir), bucket=BUCKET, dry_run=True,
                            stated=UNTILED)
    fresh = run_inference(tmp_path, ckpt, str(images_dir), bucket="baseline-2/2026-01-01",
                          stated=UNTILED)

    assert "already" in again["error"]
    assert published() == before
    assert preview["bucket_exists"] is True
    assert "error" not in fresh, fresh


def test_a_run_over_an_empty_images_directory_refuses_and_publishes_nothing(
    tmp_path, monkeypatch,
):
    from tcip_mcp.tools.inference_tools import run_inference

    empty = tmp_path / "images" / "2026-01-01"
    empty.mkdir(parents=True)
    _stubbed(monkeypatch)

    ran = run_inference(tmp_path, project_checkpoint(tmp_path), str(empty), bucket=BUCKET,
                        stated=UNTILED)

    assert "no document" in ran["error"]
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))


@pytest.mark.parametrize("where", [("captures",), ("images",), ("images", "2026-01-01", "sub")],
                         ids=["no_image_tree", "the_image_tree", "inside_a_capture"])
def test_an_images_dir_that_is_no_capture_refuses_naming_it(tmp_path, monkeypatch, where):
    """A pass over images publishes its capture, so a directory that is no capture
    (``<root>/images/<capture>``) refuses at the door naming it, before any prediction."""
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(tmp_path.joinpath(*where))
    _stubbed(monkeypatch)

    ran = run_inference(tmp_path, project_checkpoint(tmp_path), str(images_dir), bucket=BUCKET,
                        stated=UNTILED)

    assert f"{str(images_dir)!r} is not a capture" in ran["error"]
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))


def test_a_raster_outside_a_capture_is_admitted(tmp_path, monkeypatch):
    """A raster is the one source published outside a capture: one under the image tree but in no
    capture directory reaches the pass."""
    import numpy as np
    import tifffile

    from tcip_mcp.tools.inference_tools import run_inference

    raster = tmp_path / "images" / "mosaic.tif"
    raster.parent.mkdir(parents=True)
    tifffile.imwrite(str(raster), np.zeros((64, 64, 3), dtype=np.uint8))
    _stubbed(monkeypatch)

    result = run_inference(tmp_path, project_checkpoint(tmp_path), raster_path=str(raster),
                           bucket=BUCKET, dry_run=True,
                           stated=Stated(tile_size=32, overlap=0.0, **SAMPLE_DETECTOR_PASS))

    assert "error" not in result, result
    assert result["dataset_root"] == str(tmp_path)


def test_a_dry_run_previews_the_both_sources_refusal(tmp_path):
    """A preview previews the same refusal a real call would hit."""
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"stub")

    result = run_inference(
        tmp_path, project_checkpoint(tmp_path), images_dir=str(images_dir),
        raster_path=str(raster_path), bucket=BUCKET, dry_run=True)

    assert "exactly one of images_dir or raster_path" in result["error"]


def test_a_dry_run_names_the_bucket_and_the_execution_and_writes_nothing(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(tmp_path / "images" / "2026-01-01")
    _stubbed(monkeypatch)

    result = run_inference(tmp_path, project_checkpoint(tmp_path), str(images_dir),
                           bucket=BUCKET, stated=UNTILED, dry_run=True)

    assert "error" not in result, result
    assert (result["dataset_root"], result["bucket"]) == (str(tmp_path), BUCKET)
    assert result["bucket_exists"] is False
    assert result["execution"]["tile_size"] is None
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))
