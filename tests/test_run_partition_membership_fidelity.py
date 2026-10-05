"""What a run's resolved record claims about a spatial-strip split.

The resolved data section's ``split.spatial_manifest`` is the immutable record a reviewer
reconstructs a metric from: which units trained, which validated, and which pixel regions each
side occupied. These drive the real writer (the launcher's own resolution and launch record) and
read the result back, including through the geometric disjointness check that consumes it.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

pytest.importorskip("torchvision")


def _single_source_mosaic(root: Path, width: int = 4000, height: int = 3000) -> tuple[Path, str]:
    """One large detection source with GT spread across its full extent, the shape that resolves
    to the single-source spatial-strip split; its image directory and stem."""
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    Image.new("RGB", (width, height), color=(70, 90, 60)).save(images_dir / f"{stem}.png")
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    label_image(images_dir / f"{stem}.png", boxes, width, height)
    return images_dir, stem


def _data_cfg(images_dir: Path, **split) -> dict:
    cfg = {"val_ratio": 0.25, "holdout_ratio": 0.1, "calibration_ratio": 0, "seed": 1}
    cfg.update(split)
    return {"images_dir": str(images_dir), "scope": {"subject": "bud"},
            "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
            "split": cfg}


def _persisted_split(project: Path, experiment_id: str, data_cfg: dict) -> dict:
    """The spatial manifest the run ``experiment_id`` under ``project`` over ``data_cfg``
    resolved and recorded."""
    from tcip_mcp.experiments import run_resolution
    from tests._verified_checkpoint_fixtures import partition_side, resolved_run

    resolved_run(project, data_cfg, experiment_id=experiment_id)
    resolved = run_resolution(experiment_id, project=project)
    assert partition_side(resolved["partition"], "train") == ["mosaic"]
    return {"spatial": resolved["data"]["split"]["spatial_manifest"]}


def test_spatial_split_records_val_membership_from_the_val_side(tmp_path: Path) -> None:
    """The recorded val membership is the validation side's own strip identities. Train and val
    held different regions, so the record must show different, non-overlapping members: a metric
    reconstructed from a manifest claiming both sides held the same units is unreproducible."""
    images_dir, _ = _single_source_mosaic(tmp_path / "ds")
    data_cfg = _data_cfg(images_dir)

    split = _persisted_split(tmp_path, "exp_membership", data_cfg)

    train, val = split["spatial"]["train_identities"], split["spatial"]["val_identities"]
    assert train and val
    assert set(train).isdisjoint(val)

    # The two sides are not interchangeable: 0.65 of the axis trains against 0.25 validating, so
    # the recorded regions differ in width as well as in membership.
    def _axis_width(region):
        return sum(x1 - x0 for x0, _y0, x1, _y1 in region)

    assert _axis_width(split["spatial"]["train_region"]) > _axis_width(
        split["spatial"]["val_region"]) > 0


def test_persisted_calibration_region_is_reserved_away_from_train(tmp_path: Path) -> None:
    """A four-way split's reserved calibration band reaches the resolved record as its own geometry,
    disjoint from the train region, and the geometric disjointness check reads a rect drawn from
    what was actually persisted there as clean while still catching one drawn from train."""
    from tcip_mcp.pipelines.raster_source import rects_overlap
    from tcip_mcp.pipelines.operating_point import spatial_disjointness

    images_dir, _stem = _single_source_mosaic(tmp_path / "ds")
    data_cfg = _data_cfg(images_dir, val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15)

    split = _persisted_split(tmp_path, "exp_reserved_cal", data_cfg)
    spatial = split["spatial"]
    cal_region = [tuple(r) for r in spatial["calibration_region"]]
    train_region = [tuple(r) for r in spatial["train_region"]]
    assert cal_region and train_region
    for cal_rect in cal_region:
        for train_rect in train_region:
            assert not rects_overlap(cal_rect, train_rect)

    def _shrunk(rect):
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    assert spatial_disjointness(spatial, [_shrunk(cal_region[0])]) == []
    leaking = _shrunk(train_region[0])
    assert spatial_disjointness(spatial, [leaking]) == [str(list(leaking))]
