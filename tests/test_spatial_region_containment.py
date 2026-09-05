"""A within-mosaic calibration rect must be positively attested as reserved, not merely un-trained.

A block-calibration reference has no image identity of its own, so the only proof it was held out is
its geometry: the rect has to sit fully inside a region the split manifest actually recorded as
non-train (``val_region``/``test_region``/``calibration_region``) and clear of every recorded train
region. Missing the containment obligation admits a rect that lies in no attested region at all: a
gap between regions, or coordinates the persisted geometry never covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.operating_point import _train_disjointness  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

MOSAIC_W, MOSAIC_H = 4000, 3000


def _write_split(tmp_path: Path, experiment_id: str, spatial: dict) -> None:
    import tcip_store as ts
    from tcip_mcp.experiments import split_key

    ts.replace(split_key(experiment_id, root=tmp_path), {
        "train": ["mosaic::strip_x_0"], "group_by": "spatial_strip", "spatial": spatial,
    })


def test_a_rect_in_an_unattested_gap_between_regions_is_a_leak(tmp_path):
    """The manifest reserves x<400 for training and x>=600 for validation, and says nothing at all
    about the strip between them. A rect drawn from that strip touches no train pixel, so an
    overlap test alone reads it clean, yet nothing in the split ever attested it as held out.
    """
    _write_split(tmp_path, "exp_gap", {
        "train_region": [[0, 0, 400, 1000]],
        "val_region": [[600, 0, 1000, 1000]],
        "test_region": [],
        "calibration_region": [],
    })

    gap_rect = (440, 100, 560, 300)
    from tcip_mcp.pipelines.data.tiling import rect_contains_rect, rects_overlap
    assert not rects_overlap((0, 0, 400, 1000), gap_rect)          # genuinely clear of train
    assert not rect_contains_rect((600, 0, 1000, 1000), gap_rect)  # and attested by nothing

    res = _train_disjointness("exp_gap", {"mosaic"}, set(), cal_rects={"mosaic": gap_rect})
    assert res["group_check"] == "spatial_strip_geometric"
    assert res["leaked_groups"] == ["mosaic"]


def test_a_rect_inside_an_attested_region_is_admitted(tmp_path):
    """The companion obligation: a rect fully inside a recorded non-train region must read clean,
    or block calibration could never resolve at all.
    """
    _write_split(tmp_path, "exp_clean", {
        "train_region": [[0, 0, 400, 1000]],
        "val_region": [[600, 0, 1000, 1000]],
        "test_region": [],
        "calibration_region": [],
    })

    res = _train_disjointness("exp_clean", {"mosaic"}, set(),
                              cal_rects={"mosaic": (650, 100, 750, 300)})
    assert res["leaked_groups"] == []
    assert res["unresolvable"] is False


def _mosaic_dataset(root: Path) -> tuple[Path, Path, str]:
    """One large single-source mosaic with GT spread across its whole extent, enough for the real
    spatial-strip split to derive a four-way layout over it."""
    from PIL import Image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    Image.new("RGB", (MOSAIC_W, MOSAIC_H), color=(90, 90, 90)).save(images_dir / f"{stem}.png")
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
             for x in range(20, MOSAIC_W - 20, 200) for y in range(20, MOSAIC_H - 20, 200)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, MOSAIC_W, MOSAIC_H,
                              keep_empty=True)
    return images_dir, labels_dir, stem


def test_persisted_four_way_geometry_admits_its_calibration_region_and_refuses_the_unattested(
        tmp_path):
    """Driven through the real writer rather than a hand-written manifest, so the reader's key
    names are checked against what the split persister actually records.

    A four-way split reserves its own calibration region; a rect inside it must read clean, and a
    rect outside every persisted region (here beyond the mosaic's own extent, the shape a caller
    passing coordinates from a different raster produces) must read as a leak.
    """
    import tcip_store as ts
    from tcip_mcp.experiments import create_experiment, split_key
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    images_dir, labels_dir, stem = _mosaic_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": 1,
                  "reserve_calibration_fraction": 0.15},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None

    create_experiment("exp_four_way", {})
    persist_split_manifest("exp_four_way", train_ds, val_ds, data_cfg)
    spatial = ts.read(split_key("exp_four_way"))["spatial"]
    cal_region = spatial["calibration_region"]
    assert cal_region, "the writer produced no calibration region to read back"

    def _shrunk(rect):
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    clean = _train_disjointness("exp_four_way", {stem}, set(),
                                cal_rects={stem: _shrunk(cal_region[0])})
    assert clean["group_check"] == "spatial_strip_geometric"
    assert clean["leaked_groups"] == []

    beyond_extent = (MOSAIC_W + 1000, 100, MOSAIC_W + 2000, 300)
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    assert all(not rects_overlap(tuple(tr), beyond_extent) for tr in spatial["train_region"])

    leaked = _train_disjointness("exp_four_way", set(), {stem},
                                 hold_rects={stem: beyond_extent})
    assert leaked["leaked_groups"] == [stem]
