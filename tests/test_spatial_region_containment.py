"""A within-mosaic calibration rect reads clean only inside a region recorded as non-train.

The rect reads clean when it sits fully inside a region the run's partition recorded as
non-train (``val_region``/``holdout_region``/``calibration_region``), and a rect outside every
such region reads as a leak even when it touches no recorded train region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("torch")

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.schemas import Rect

from tcip_mcp.pipelines.operating_point import spatial_disjointness  # noqa: E402
from tcip_mcp.pipelines.schemas import SpatialManifest  # noqa: E402
from tests._producer_fixtures import big_single_source  # noqa: E402
from tests._training_values import drawn_split, tiling_block  # noqa: E402

MOSAIC_W, MOSAIC_H = 4000, 3000
GAPPED = SpatialManifest.model_construct(
    train_region=[(0, 0, 400, 1000)],
    val_region=[(600, 0, 1000, 1000)],
    holdout_region=[],
    calibration_region=[],
)
"""A manifest whose train region spans x 0 to 400 and val region x 600 to 1000, both over y 0 to
1000, with no holdout or calibration region: the strip from x 400 to 600 lies in no region."""


def test_a_rect_in_an_unattested_gap_between_regions_is_a_leak() -> None:
    """A rect in an unattested gap between regions reads as a leak.

    A rect drawn from the silent strip touches no train pixel and lies inside no recorded
    region, and the check names it.
    """
    from tcip_mcp.pipelines.raster_source import (
        rect_contains_rect,
        rects_overlap,
    )

    gap_rect = (440, 100, 560, 300)
    assert not rects_overlap(
        (0, 0, 400, 1000), gap_rect
    )  # genuinely clear of train
    assert not rect_contains_rect(
        (600, 0, 1000, 1000), gap_rect
    )  # and attested by nothing

    assert spatial_disjointness(GAPPED, [gap_rect]) == ["[440, 100, 560, 300]"]


def test_a_rect_inside_an_attested_region_is_admitted() -> None:
    """A rect fully inside a recorded non-train region reads clean."""
    assert spatial_disjointness(GAPPED, [(650, 100, 750, 300)]) == []


def test_persisted_four_way_geometry_admits_its_calibration_region_and_refuses_the_unattested(
    tmp_path: Path,
) -> None:
    """A persisted four-way split admits its calibration region and refuses an unattested rect.

    The manifest is the run's resolved record, written by the real writer. A four-way split
    reserves its own calibration region; a rect inside it reads clean, and a rect outside every
    persisted region (here beyond the mosaic's own extent) reads as a leak.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir, _stem = big_single_source(tmp_path / "ds", MOSAIC_W, MOSAIC_H)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=128, overlap=0.2),
        "split": drawn_split(
            val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15, seed=1
        ),
    }
    resolved_run(tmp_path, data_cfg, experiment_id="exp_four_way")
    spatial = run_resolution(
        "exp_four_way", project=tmp_path
    ).partition.spatial
    assert spatial is not None
    cal_region = spatial.calibration_region
    assert cal_region, "the writer produced no calibration region to read back"

    def _shrunk(rect: Rect) -> Rect:
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    assert spatial_disjointness(spatial, [_shrunk(cal_region[0])]) == []

    beyond_extent = (MOSAIC_W + 1000, 100, MOSAIC_W + 2000, 300)
    from tcip_mcp.pipelines.raster_source import rects_overlap

    assert all(
        not rects_overlap(tr, beyond_extent) for tr in spatial.train_region
    )

    assert spatial_disjointness(spatial, [beyond_extent]) == [
        str(list(beyond_extent))
    ]


def test_a_spatial_runs_resolved_record_carries_no_drawn_seed(
    tmp_path: Path,
) -> None:
    """A spatial run's resolved record carries no drawn seed.

    The run's resolved partition carries no drawn seed and its spatial manifest no ``seed`` key,
    though the config states ``data.split.seed``.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir, _stem = big_single_source(tmp_path / "ds", MOSAIC_W, MOSAIC_H)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=128, overlap=0.2),
        "split": drawn_split(val_ratio=0.2, holdout_ratio=0.1, seed=7),
    }
    resolved_run(tmp_path, data_cfg, experiment_id="exp_spatial_no_seed")
    resolved = run_resolution("exp_spatial_no_seed", project=tmp_path)

    assert resolved.partition.seed is None
    manifest = resolved.partition.spatial
    assert manifest is not None and "seed" not in manifest.model_dump()
