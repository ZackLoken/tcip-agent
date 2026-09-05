"""What the persisted split manifest claims about a spatial-strip split.

``split.json`` is the immutable record a reviewer reconstructs a metric from: which units trained,
which validated, and which pixel regions each side occupied. These drive the real writer
(``auto_train_val`` into ``persist_split_manifest``) and read the result back, including through
the geometric disjointness check that consumes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torchvision")


def _single_source_mosaic(root: Path, width: int = 4000, height: int = 3000) -> tuple[Path, Path, str]:
    """One large detection source with GT spread across its full extent, the shape that resolves
    to the single-source spatial-strip split."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    Image.new("RGB", (width, height), color=(70, 90, 60)).save(images_dir / f"{stem}.png")
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, width, height,
                              keep_empty=True)
    return images_dir, labels_dir, stem


def _data_cfg(images_dir: Path, labels_dir: Path, **split) -> dict:
    cfg = {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1}
    cfg.update(split)
    return {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
            "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
            "split": cfg}


def _persisted_split(experiment_id: str, data_cfg: dict) -> dict:
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None, "the fixture must produce a real validation side"
    create_experiment(experiment_id, {})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)
    return read_split_manifest(experiment_id)


def test_spatial_split_records_val_membership_from_the_val_side(tmp_path: Path) -> None:
    """The recorded val membership is the validation side's own strip identities. Train and val
    held different regions, so the record must show different, non-overlapping members: a metric
    reconstructed from a manifest claiming both sides held the same units is unreproducible."""
    images_dir, labels_dir, _ = _single_source_mosaic(tmp_path / "ds")
    data_cfg = _data_cfg(images_dir, labels_dir)

    split = _persisted_split("exp_membership", data_cfg)
    manifest = data_cfg["split"]["spatial_manifest"]

    assert split["group_by"] == "spatial_strip"
    assert split["train"] and split["val"]
    assert set(split["train"]).isdisjoint(split["val"])
    assert split["val"] == manifest["val_identities"]
    assert split["train"] == manifest["train_identities"]

    # The two sides are not interchangeable: 0.65 of the axis trains against 0.25 validating, so
    # the recorded regions differ in width as well as in membership.
    def _axis_width(region):
        return sum(x1 - x0 for x0, _y0, x1, _y1 in region)

    assert _axis_width(split["spatial"]["train_region"]) > _axis_width(
        split["spatial"]["val_region"]) > 0


def test_persisted_calibration_region_is_reserved_away_from_train(tmp_path: Path) -> None:
    """A four-way split's reserved calibration band reaches ``split.json`` as its own geometry,
    disjoint from the train region, and the geometric disjointness check reads a rect drawn from
    what was actually persisted there as clean while still catching one drawn from train."""
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    images_dir, labels_dir, stem = _single_source_mosaic(tmp_path / "ds")
    data_cfg = _data_cfg(images_dir, labels_dir, val_ratio=0.2, test_ratio=0.1,
                         reserve_calibration_fraction=0.15)

    split = _persisted_split("exp_reserved_cal", data_cfg)
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

    clean = _train_disjointness("exp_reserved_cal", {stem}, set(),
                                cal_rects={stem: _shrunk(cal_region[0])})
    assert clean["leaked_groups"] == []
    assert clean["group_check"] == "spatial_strip_geometric"

    leaked = _train_disjointness("exp_reserved_cal", {stem}, set(),
                                 cal_rects={stem: _shrunk(train_region[0])})
    assert leaked["leaked_groups"] == [stem]
