"""A per-plant orthomosaic delivery resolves detections only through the raster its bucket was
produced on.

The counts in the delivered CSV are attributed to plants by the raster's own georeferencing, so
the raster is part of the measurement: a pixel-identical copy at a moved tiepoint re-attributes
every count to a neighboring plant, and a far-shifted one reads every plant as an explicit zero.
The bucket record names the raster and its identity; a raster changed on disk since publication
refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests.test_orthomosaic_tools import (  # noqa: E402, F401
    _GRID, _PLANT_PIXELS, SCOPE, TIEPOINT_NATIVE_X, TILE, _bespoke_detection_checkpoint,
    _plant_grid_csv, _plant_registry, _raster, _write_geo_raster,
)

BUCKET = "preds/2026-01-01"
pytestmark = pytest.mark.usefixtures("confirmed_count_aggregate")


def _produced_bucket(tmp_path: Path, raster_path: Path) -> str:
    """A bucket from the real producer: run_inference's whole-raster regime; its name."""
    from tcip_mcp.tools.inference_tools import run_inference
    from tests.test_orthomosaic_tools import RASTER_PASS

    result = run_inference(
        tmp_path, _bespoke_detection_checkpoint(tmp_path), bucket=BUCKET,
        raster_path=str(raster_path), stated=RASTER_PASS.model_copy(update={"overlap": None}))
    assert "error" not in result, result
    return result["bucket"]


def _delivered(tmp_path: Path, bucket: str, raster_path: Path) -> dict:
    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    registry = _plant_registry(tmp_path, _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS))
    return deliver_orthomosaic_plant_counts(
        tmp_path, str(tmp_path / "ds"), bucket, registry, str(tmp_path / "counts.csv"),
        "stem_count", _GRID)


def test_a_raster_whose_tiepoint_moved_since_publication_refuses(tmp_path):
    """Identical pixels at a moved tiepoint resolve every detection onto a different plant, so the
    raster is refused rather than silently believed: the content half alone cannot see this."""
    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path)
    bucket = _produced_bucket(tmp_path, raster_path)
    _write_geo_raster(raster_path, tiepoint_x=TIEPOINT_NATIVE_X + 20.0)

    refused = _delivered(tmp_path, bucket, raster_path)

    assert "georeferencing mismatch" in refused["error"]
    assert "tiepoint_native_x" in refused["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_a_raster_whose_content_changed_since_publication_refuses(tmp_path):
    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path)
    bucket = _produced_bucket(tmp_path, raster_path)
    _write_geo_raster(raster_path, seed=7)

    refused = _delivered(tmp_path, bucket, raster_path)

    assert "content mismatch" in refused["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_a_bucket_of_per_image_predictions_refuses_naming_what_it_is(tmp_path):
    from tests._chain_fixtures import published

    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path)
    bucket = published(tmp_path, "frames/2026-01-01",
                       [{"image": str(raster_path.parent / "img1.jpg"), "width": 64,
                         "height": 64, "boxes": [], "scores": [], "labels": []}],
                       scope=SCOPE).name

    refused = _delivered(tmp_path, bucket, raster_path)

    assert "per-image predictions" in refused["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_the_raster_it_was_produced_on_passes_the_identity_check_and_meets_the_gate(tmp_path):
    """The producer's own recorded identity admits the unchanged raster; this bucket was never
    assessed and this door takes no acknowledgment, so the refusal named is the gate's."""
    from tcip_mcp.buckets import read_bucket

    raster_path = _raster(tmp_path)
    _write_geo_raster(raster_path)
    bucket = _produced_bucket(tmp_path, raster_path)
    recorded = read_bucket(tmp_path / "ds", bucket).raster_identity
    assert recorded is not None and recorded["pixel_checksum"]
    assert recorded["geotransform"]["tiepoint_native_x"] == TIEPOINT_NATIVE_X

    refused = _delivered(tmp_path, bucket, raster_path)

    assert "mismatch" not in refused["error"]
    assert "no assessment answers" in refused["error"]
    assert not (tmp_path / "counts.csv").exists()
