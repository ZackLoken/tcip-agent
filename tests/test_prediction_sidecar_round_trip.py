"""A written prediction bucket has to be readable by the readers the delivery doors use.

``export_predictions`` writes ``operating_point.json`` beside the predictions, and every door that
later assembles a phenotype from that bucket reads its validity back out of that file rather than
from a caller's word. These tests drive the real writer and the real readers against each other, so
a stamp the writer produces but no reader can find is a failure here rather than downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

CONF_FROM_THE_DENSE_REFERENCE = 0.9  # every correct detection in the fixture below scores this


class _BucketStub:
    """A predictor returning one detection per image, enough to write a real bucket."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": 160, "height": 120,
                 "boxes": [[10, 10, 30, 30]], "scores": [0.95], "labels": [1], "count": 1}
                for p in paths]


def _held_out_bundle(*, tiled: bool, tile_size: int | None = None,
                     tile_size_source: str = "default"):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=CONF_FROM_THE_DENSE_REFERENCE,
                        fp_score=0.05)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp,
                         score=CONF_FROM_THE_DENSE_REFERENCE, fp_score=0.05)
    return resolve_operating_point(
        "catkin", dataset_hash="H", calibration_records=cal, holdout_records=hold,
        tiled=tiled, tile_size=tile_size, tile_size_source=tile_size_source,
        staged_conf_floor=0.01)


def _export(tmp_path, monkeypatch, *, bundle, tile, tile_size=None):
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.tools.inference_tools as itools

    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (160, 120), color=(70, 90, 110)).save(images_dir / "capture_a.png")

    monkeypatch.setattr(itools, "_calibrate_operating_point", lambda *a, **k: (bundle, "H", 0))
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda **kw: _BucketStub())
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    result = itools.export_predictions(
        str(ckpt), images_dir=str(images_dir), output_dir=str(tmp_path / "bucket"),
        device="cpu", tile=tile, tile_size=tile_size, trait="catkin",
        calibration_labels_dir=str(images_dir))
    assert "error" not in result, result
    return result


def test_the_count_operating_points_validity_survives_the_round_trip_to_disk(tmp_path, monkeypatch):
    """The conf a calibrated run resolved is recoverable from the written bucket by the same
    reconciliation the delivery doors gate on, with its held-out reference intact."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, read_operating_point_sidecar, reconcile_operating_point_validity,
    )

    result = _export(tmp_path, monkeypatch, bundle=_held_out_bundle(tiled=False), tile=False)
    bucket = result["output_dir"]

    assert read_operating_point_sidecar(bucket) is not None
    reconciled = reconcile_operating_point_validity([bucket])
    assert reconciled["missing_sidecars"] == []
    assert reconciled["unvalidated_buckets"] == []
    assert reconciled["on_disk_validated"] is True
    assert reconciled["validated"] == VALIDATED_HELD_OUT
    assert reconciled["conf"] == pytest.approx(CONF_FROM_THE_DENSE_REFERENCE)


def test_the_tile_geometrys_basis_survives_the_round_trip_to_disk(tmp_path, monkeypatch):
    """A tiled bucket's tile scale is readable back as the operative, explicitly-stated basis it
    was written at, the dimension a multi-bucket delivery floors itself against."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_EXPLICIT_GEOMETRY, reconcile_tile_size_validity,
    )

    bundle = _held_out_bundle(tiled=True, tile_size=64, tile_size_source="explicit")
    result = _export(tmp_path, monkeypatch, bundle=bundle, tile=True, tile_size=64)
    bucket = result["output_dir"]

    reconciled = reconcile_tile_size_validity([bucket])
    assert reconciled["operative"] is True
    assert reconciled["unvalidated_buckets"] == []
    assert reconciled["validated"] == VALIDATED_EXPLICIT_GEOMETRY
    assert reconciled["per_bucket"] == {str(Path(bucket)): VALIDATED_EXPLICIT_GEOMETRY}
