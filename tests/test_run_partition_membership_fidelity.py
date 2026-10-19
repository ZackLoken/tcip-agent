"""What a run's resolved record claims about a spatial-strip split.

The resolved partition's ``spatial`` split records which units are on the training side, which
on the validation side, and which pixel regions each side occupied. These drive the launcher's
own resolution and launch record and read the result back, directly and through the geometric
disjointness check.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._producer_fixtures import big_single_source

if TYPE_CHECKING:
    from tcip_mcp.pipelines.schemas import Rect, SpatialManifest

pytest.importorskip("torchvision")


def _data_cfg(images_dir: Path, **split: object) -> dict:
    from tests._training_values import drawn_split, tiling_block

    return {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=128, overlap=0.2),
        "split": drawn_split(
            **{"val_ratio": 0.25, "holdout_ratio": 0.1, "seed": 1, **split}
        ),
    }


def _persisted_split(
    project: Path, experiment_id: str, data_cfg: dict
) -> SpatialManifest:
    """Resolve the run ``experiment_id`` over ``data_cfg`` and return its spatial manifest.

    The manifest is the one the run under ``project`` resolved and recorded; the helper asserts
    the training side holds the one source ``mosaic``.
    """
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import (
        partition_side,
        resolved_run,
    )

    resolved_run(project, data_cfg, experiment_id=experiment_id)
    resolved = run_resolution(experiment_id, project=project)
    assert partition_side(resolved.partition, "train") == ["mosaic"]
    assert resolved.partition.spatial is not None
    return resolved.partition.spatial


def test_spatial_split_records_val_membership_from_the_val_side(
    tmp_path: Path,
) -> None:
    """The recorded train and val members are disjoint, and the train region is the wider.

    The recorded train and val members are non-empty and disjoint, and the recorded train region
    is wider along the axis than the val region.
    """
    images_dir, _ = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = _data_cfg(images_dir)

    spatial = _persisted_split(tmp_path, "exp_membership", data_cfg)

    train, val = spatial.train_identities, spatial.val_identities
    assert train and val
    assert set(train).isdisjoint(val)

    # The two sides are not interchangeable: 0.65 of the axis trains against 0.25 validating, so
    # the recorded regions differ in width as well as in membership.
    def _axis_width(region: list[Rect]) -> int:
        return sum(x1 - x0 for x0, _y0, x1, _y1 in region)

    assert (
        _axis_width(spatial.train_region) > _axis_width(spatial.val_region) > 0
    )


def test_persisted_calibration_region_is_reserved_away_from_train(
    tmp_path: Path,
) -> None:
    """A four-way split's calibration band is recorded as its own geometry, disjoint from train.

    The reserved calibration band reaches the resolved record as its own geometry, disjoint from
    the train region, and the geometric disjointness check reads a rect drawn from what was
    actually persisted there as clean while still catching one drawn from train.
    """
    from tcip_mcp.pipelines.operating_point import spatial_disjointness
    from tcip_mcp.pipelines.raster_source import rects_overlap

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = _data_cfg(
        images_dir, val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15
    )

    spatial = _persisted_split(tmp_path, "exp_reserved_cal", data_cfg)
    cal_region, train_region = spatial.calibration_region, spatial.train_region
    assert cal_region and train_region
    for cal_rect in cal_region:
        for train_rect in train_region:
            assert not rects_overlap(cal_rect, train_rect)

    def _shrunk(rect: Rect) -> Rect:
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    assert spatial_disjointness(spatial, [_shrunk(cal_region[0])]) == []
    leaking = _shrunk(train_region[0])
    assert spatial_disjointness(spatial, [leaking]) == [str(list(leaking))]
