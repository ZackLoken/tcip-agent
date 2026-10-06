"""A within-mosaic calibration rect must be positively attested as reserved, not merely un-trained.

A mosaic reference band has no image identity of its own, so the only proof it was held out is its
geometry: the rect has to sit fully inside a region the run's partition actually recorded as
non-train (``val_region``/``holdout_region``/``calibration_region``) and clear of every recorded
train region. Missing the containment obligation admits a rect that lies in no attested region at
all: a gap between regions, or coordinates the persisted geometry never covered.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.operating_point import spatial_disjointness  # noqa: E402

MOSAIC_W, MOSAIC_H = 4000, 3000
GAPPED = {"train_region": [[0, 0, 400, 1000]], "val_region": [[600, 0, 1000, 1000]],
          "holdout_region": [], "calibration_region": []}
"""A manifest reserving x<400 for training and x>=600 for validation, silent on the strip
between them."""


def test_a_rect_in_an_unattested_gap_between_regions_is_a_leak():
    """A rect drawn from the silent strip touches no train pixel, so an overlap test alone reads it
    clean, yet nothing in the split ever attested it as held out."""
    from tcip_mcp.pipelines.raster_source import rect_contains_rect, rects_overlap

    gap_rect = (440, 100, 560, 300)
    assert not rects_overlap((0, 0, 400, 1000), gap_rect)          # genuinely clear of train
    assert not rect_contains_rect((600, 0, 1000, 1000), gap_rect)  # and attested by nothing

    assert spatial_disjointness(GAPPED, [gap_rect]) == ["[440, 100, 560, 300]"]


def test_a_rect_inside_an_attested_region_is_admitted():
    """The companion obligation: a rect fully inside a recorded non-train region reads clean, or
    a mosaic reference could never be assessed at all."""
    assert spatial_disjointness(GAPPED, [(650, 100, 750, 300)]) == []


def _mosaic_dataset(root: Path) -> Path:
    """One large single-source mosaic with GT spread across its whole extent, enough for the real
    spatial-strip split to derive a four-way layout over it; returns its images directory."""
    from PIL import Image

    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    image = images_dir / "mosaic.png"
    Image.new("RGB", (MOSAIC_W, MOSAIC_H), color=(90, 90, 90)).save(image)
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, MOSAIC_W - 20, 200) for y in range(20, MOSAIC_H - 20, 200)]
    label_image(image, boxes, MOSAIC_W, MOSAIC_H, keep_empty=True)
    return images_dir


def test_persisted_four_way_geometry_admits_its_calibration_region_and_refuses_the_unattested(
        tmp_path):
    """Driven through the real writer rather than a hand-written manifest, so the reader's key
    names are checked against what the run's resolved record actually carries.

    A four-way split reserves its own calibration region; a rect inside it must read clean, and a
    rect outside every persisted region (here beyond the mosaic's own extent, the shape a caller
    passing coordinates from a different raster produces) must read as a leak.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir = _mosaic_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "holdout_ratio": 0.1, "calibration_ratio": 0.15, "seed": 1},
    }
    resolved_run(tmp_path, data_cfg, experiment_id="exp_four_way")
    spatial = run_resolution("exp_four_way", project=tmp_path)["data"]["split"]["spatial_manifest"]
    cal_region = spatial["calibration_region"]
    assert cal_region, "the writer produced no calibration region to read back"

    def _shrunk(rect):
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    assert spatial_disjointness(spatial, [_shrunk(cal_region[0])]) == []

    beyond_extent = (MOSAIC_W + 1000, 100, MOSAIC_W + 2000, 300)
    from tcip_mcp.pipelines.raster_source import rects_overlap
    assert all(not rects_overlap(tuple(tr), beyond_extent) for tr in spatial["train_region"])

    assert spatial_disjointness(spatial, [beyond_extent]) == [str(list(beyond_extent))]


def test_a_spatial_runs_resolved_record_carries_no_drawn_seed(tmp_path):
    """The spatial-strip layout is governed by declared order alone, so the run's resolved
    partition carries no drawn seed and its spatial manifest no ``seed`` key: this route draws
    nothing, whatever ``data.split.seed`` the config states.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir = _mosaic_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "holdout_ratio": 0.1, "calibration_ratio": 0, "seed": 7},
    }
    resolved_run(tmp_path, data_cfg, experiment_id="exp_spatial_no_seed")
    resolved = run_resolution("exp_spatial_no_seed", project=tmp_path)

    assert resolved["partition"]["seed"] is None
    assert "seed" not in resolved["data"]["split"]["spatial_manifest"]
