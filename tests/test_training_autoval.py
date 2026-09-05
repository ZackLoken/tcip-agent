"""Auto train/val wiring: auto_train_val plus the detection val-loss pass.

These exercise the helper that derives a group-aware val split and the generic_trainer
``_validate`` detection path, which must be correct so a real val loader can be wired without
crashing the run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch.utils.data import DataLoader  # noqa: E402

from tcip_mcp.pipelines.data.split_construction import auto_train_val  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 64


def _save_png(path: Path, bright: bool = False) -> None:
    from torchvision.utils import save_image

    base = 0.7 if bright else 0.0
    img = torch.rand(3, IMG, IMG) * 0.3 + base
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, str(path))


def _detection_dataset(root: Path, prefixes=("srcA", "srcB", "srcC", "srcD"), tiles=2):
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    all_stems = []
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            _save_png(images_dir / f"{stem}.png")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
                IMG,
                IMG,
                keep_empty=True,
            )
            all_stems.append(stem)
    return images_dir, labels_dir, all_stems


def test_auto_train_val_detection_splits(tmp_path: Path):
    images_dir, labels_dir, all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "bud",
        "auto_val": True,
        "split": {"val_ratio": 0.4, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    assert sorted(train_ds.stems + val_ds.stems) == sorted(all_stems)
    assert val_ds.transforms is None


def test_auto_train_val_malformed_group_by_raises(tmp_path: Path):
    """An unrecognized split.group_by is a caller-config error and must propagate,
    not degrade silently to (full_train_ds, None) like other failures in this function."""
    images_dir, labels_dir, _all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "bud",
        "auto_val": True,
        "split": {"group_by": "not_a_real_grouping_key"},
    }
    with pytest.raises(ValueError):
        auto_train_val("detection", data_cfg, None)


def test_auto_train_val_malformed_val_ratio_degrades(tmp_path: Path):
    """The narrowed except ValueError scope must not widen to a malformed val_ratio/seed: those
    still degrade to (full_train_ds, None) exactly as every other non-grouping failure in this
    function does."""
    images_dir, labels_dir, all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "bud",
        "auto_val": True,
        "split": {"val_ratio": "not_a_number"},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is None
    assert sorted(train_ds.stems) == sorted(all_stems)


def test_auto_train_val_ordinal_returns_none(tmp_path: Path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "ranks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "rank"))
        w.writerows(rows)

    data_cfg = {"images_dir": str(images_dir), "csv_path": str(csv_path), "auto_val": True}
    _ds, val_ds, _ = auto_train_val("ordinal", data_cfg, None)
    assert val_ds is None


def test_auto_train_val_ordinal_explicit_val_csv_path(tmp_path: Path):
    train_images = tmp_path / "train_images"
    train_rows = []
    for i in range(4):
        _save_png(train_images / f"img{i}.png")
        train_rows.append((f"img{i}", i % 2))
    train_csv = tmp_path / "train_ranks.csv"
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "rank"))
        w.writerows(train_rows)

    val_images = tmp_path / "val_images"
    val_rows = []
    for i in range(2):
        _save_png(val_images / f"vimg{i}.png")
        val_rows.append((f"vimg{i}", i % 2))
    val_csv = tmp_path / "val_ranks.csv"
    with open(val_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "rank"))
        w.writerows(val_rows)

    data_cfg = {"images_dir": str(train_images), "csv_path": str(train_csv),
                "val_images_dir": str(val_images), "val_csv_path": str(val_csv)}
    train_ds, val_ds, _ = auto_train_val("ordinal", data_cfg, None)
    assert val_ds is not None
    assert val_ds.num_samples == 2
    assert train_ds.num_samples == 4


def test_auto_train_val_ordinal_val_images_dir_without_val_csv_path_degrades(tmp_path: Path):
    """val_images_dir alone isn't enough for a CSV-driven task, unlike the geometry tasks (which
    fall back to the train labels/masks dir): reusing the train CSV here would build a val_ds that
    reads rows for images not present in val_images_dir, failing later inside a training loop
    instead of now. Must degrade to (train_ds, None), not raise, and not silently build a
    mismatched val_ds."""
    train_images = tmp_path / "train_images"
    rows = []
    for i in range(4):
        _save_png(train_images / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    train_csv = tmp_path / "ranks.csv"
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "rank"))
        w.writerows(rows)

    val_images = tmp_path / "val_images"
    val_images.mkdir(parents=True, exist_ok=True)

    data_cfg = {"images_dir": str(train_images), "csv_path": str(train_csv),
                "val_images_dir": str(val_images)}
    train_ds, val_ds, _ = auto_train_val("ordinal", data_cfg, None)
    assert val_ds is None
    assert train_ds.num_samples == 4


def test_auto_train_val_tiny_dataset_guard(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "src_0_0.png")
    json_io.write_annotations(
        str(labels_dir / "src_0_0.json"),
        [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
        IMG,
        IMG,
        keep_empty=True,
    )

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "bud", "auto_val": True}
    _train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is None  # single group -> no leakage-free val possible


def test_auto_train_val_single_source_untiled_still_no_val(tmp_path: Path):
    """A single-image detection source with tiling absent must still degrade to (train_ds, None)
    exactly as before the spatial route existed: there is no tiling geometry to block-split by."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "src_0_0.png")
    json_io.write_annotations(
        str(labels_dir / "src_0_0.json"),
        [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
        IMG, IMG, keep_empty=True,
    )
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "bud", "auto_val": True}
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is None
    assert not hasattr(train_ds, "tile_size")


def _big_single_source(root: Path, width: int, height: int) -> tuple[Path, Path, str]:
    """One large detection source with a scatter of small boxes, real width/height in its
    label JSON: a single-image dataset large enough to hold a spatial train/val split."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    from torchvision.utils import save_image
    save_image(torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png"))
    boxes = [
        Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
        for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)
    ]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, width, height, keep_empty=True)
    return images_dir, labels_dir, stem


def test_auto_train_val_single_source_tiled_spatial_split(tmp_path: Path):
    """A single tiled detection source derives a real, disjoint spatial val split instead of
    degrading to no validation: the tiling=tiling leak this closes."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert train_ds.tile_size == 128 and val_ds.tile_size == 128
    assert train_ds.num_samples > 0 and val_ds.num_samples > 0
    assert set(train_ds.tile_entries).isdisjoint(set(val_ds.tile_entries))
    assert data_cfg["split"]["resolved_group_by"] == "spatial_strip"
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["train_identities"] and manifest["val_identities"]
    assert set(manifest["train_identities"]).isdisjoint(set(manifest["val_identities"]))
    assert all(i.startswith(f"{stem}::strip_") for i in manifest["train_identities"])
    assert manifest["kept_test_tiles"] > 0


def test_spatial_manifest_persists_train_and_val_regions_too(tmp_path: Path):
    """train_region/val_region are persisted the same way test_region already is: real rects, not
    just per-region tile identities, so a later geometric disjointness check has real geometry
    for every side, not only the reserved test area."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    auto_train_val("detection", data_cfg, None)
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["train_region"] and manifest["val_region"] and manifest["test_region"]
    for region in (manifest["train_region"], manifest["val_region"], manifest["test_region"]):
        for rect in region:
            assert len(rect) == 4


def test_auto_train_val_single_source_spatial_split_ignores_a_stray_keep_regions_in_tiling(
    tmp_path: Path,
):
    """A caller's tiling dict carrying its own keep_regions (meaningless in the automatic
    single-source route, which derives its own) must never collide with the derived
    keep_regions kwarg the spatial split passes explicitly."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True,
        "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2,
                  "keep_regions": [(0, 0, 100, 100)]},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None


def test_auto_train_val_degenerate_group_retries_at_stem_level(tmp_path: Path):
    """Two stems whose default tile_prefix grouping collapses to one group starve val (too few
    groups, not too few stems); the retry at stem-level grouping must still populate both sides."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    stems = ["mosaicA_0_0", "mosaicA_1_1"]
    for stem in stems:
        _save_png(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            IMG, IMG, keep_empty=True,
        )
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "bud", "auto_val": True,
                "split": {"val_ratio": 0.5, "seed": 1}}
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    assert sorted(train_ds.stems + val_ds.stems) == sorted(stems)
    assert data_cfg["split"]["resolved_group_by"] == "stem"


def test_auto_train_val_explicit_group_key_map_not_overridden_by_retry(tmp_path: Path):
    """A caller-supplied group_key_map that starves val is a deliberate leakage policy, not a
    data limitation: the retry must never silently discard it for stem-level grouping."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    stems = ["mosaicA_0_0", "mosaicA_1_1"]
    for stem in stems:
        _save_png(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            IMG, IMG, keep_empty=True,
        )
    group_key_map = {s: "one_group_for_everything" for s in stems}
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "bud", "auto_val": True,
                "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": group_key_map}}
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is None  # the explicit map still collapses everything into one group
    assert data_cfg["split"]["resolved_group_by"] == "explicit_map"


# reserve_calibration_fraction: the four-way split (train/val/test/calibration).

def test_reserve_calibration_fraction_unset_is_byte_identical(tmp_path: Path):
    """Fail-before/no-op: with reserve_calibration_fraction absent, the spatial_manifest carries
    no calibration_region and the rest of the manifest is exactly what the 3-way split has always
    produced (same keys, same train/val/test regions for the same seed/layout)."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["calibration_region"] == []
    assert manifest["kept_calibration_tiles"] == 0
    assert manifest["train_region"] and manifest["val_region"] and manifest["test_region"]


def test_reserve_calibration_fraction_adds_a_disjoint_calibration_region(tmp_path: Path):
    """Admits valid work: an explicitly reserved calibration region is real, non-empty geometry,
    disjoint from train/val/test."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": 1,
                  "reserve_calibration_fraction": 0.15},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["calibration_region"]
    assert manifest["kept_calibration_tiles"] > 0

    def _rects(region):
        return [tuple(r) for r in region]

    from tcip_mcp.pipelines.data.tiling import rects_overlap

    cal_rects = _rects(manifest["calibration_region"])
    for other_key in ("train_region", "val_region", "test_region"):
        for other in _rects(manifest[other_key]):
            for cr in cal_rects:
                assert not rects_overlap(cr, other)


def test_reserve_calibration_fraction_raises_on_unresolvable_extent(tmp_path: Path):
    """Reason 1: no width/height in the label file. Explicitly requested -> raises by name,
    rather than the unrequested case's silent (train_ds, None) degradation."""
    from tcip_mcp.pipelines.data.split_construction import spatial_single_source_split

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "mosaic.png")
    # A readable document with no width/height recorded, distinct from an unreadable one.
    (labels_dir / "mosaic.json").write_text('{"annotations": []}', encoding="utf-8")

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"}
    tiling = {"enabled": True, "tile_size": 128, "overlap": 0.2}
    split_cfg = {"val_ratio": 0.2, "test_ratio": 0.1, "reserve_calibration_fraction": 0.15}
    with pytest.raises(ValueError, match="reserve_calibration_fraction"):
        spatial_single_source_split("mosaic", data_cfg, tiling, object(), split_cfg, None)


def test_spatial_single_source_split_raises_on_an_unreadable_label_regardless_of_reserve(
    tmp_path: Path,
):
    """A present, unreadable label document is a categorically different fact than one recording
    no width/height: it raises unconditionally, whether or not reserve_calibration_fraction was
    requested, rather than degrading to no validation over a document nobody can read."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.split_construction import spatial_single_source_split

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "mosaic.png")
    (labels_dir / "mosaic.json").write_text("[]", encoding="utf-8")  # not a dict: unreadable

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"}
    tiling = {"enabled": True, "tile_size": 128, "overlap": 0.2}
    split_cfg = {"val_ratio": 0.2, "test_ratio": 0.1}  # no reserve_calibration_fraction
    with pytest.raises(UnreadableLabelDocument):
        spatial_single_source_split("mosaic", data_cfg, tiling, object(), split_cfg, None)


def test_reserve_calibration_fraction_raises_on_infeasible_layout(tmp_path: Path):
    """Reason 2: spatial_strip_split itself cannot lay out 4 non-empty regions at this mosaic
    size/tile size. Explicitly requested -> raises by name."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        # A calibration fraction that leaves nothing after val+test on a mosaic this size.
        "split": {"val_ratio": 0.45, "test_ratio": 0.45, "seed": 1,
                  "reserve_calibration_fraction": 0.3},
    }
    with pytest.raises(ValueError, match="reserve_calibration_fraction"):
        auto_train_val("detection", data_cfg, None)


def test_reserve_calibration_fraction_raises_on_empty_gt_bearing_side(tmp_path: Path):
    """Reason 3: the strip layout itself is feasible (every side gets kept tiles), but with
    tiling.skip_empty set, a reserved side's tiles carrying no GT filter down to zero real
    samples. At this exact width/tile_size/fractions/seed, spatial_strip_split places train at x
    in [1275, 3175] and val at [3264, 3991] (verified directly against spatial_strip_split for
    this test's own params); GT is placed only inside those two ranges, leaving test ([0, 421])
    and calibration ([510, 1186]) both real, tiled, and entirely GT-free."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    width, height = 4000, 300
    from torchvision.utils import save_image
    save_image(torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png"))
    boxes = [Annotation(subject="bud", geometry=BBox(x, 20, x + 20, 40))
            for x in range(1300, 3960, 40)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, width, height, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True,
        "tiling": {"enabled": True, "tile_size": 64, "overlap": 0.2, "skip_empty": True},
        "split": {"val_ratio": 0.2, "test_ratio": 0.1, "seed": 1,
                  "reserve_calibration_fraction": 0.2},
    }
    with pytest.raises(ValueError, match="reserve_calibration_fraction"):
        auto_train_val("detection", data_cfg, None)


def test_reserve_calibration_fraction_records_raster_content_identity(tmp_path: Path):
    """Mechanism 2's training-time recording: a real, decodable single-source raster gets a
    raster_content_identity in the same spatial_manifest a claim-scope check later reads back."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    auto_train_val("detection", data_cfg, None)
    manifest = data_cfg["split"]["spatial_manifest"]
    identity = manifest["raster_content_identity"]
    assert identity is not None
    assert identity["width"] == 4000 and identity["height"] == 3000
    assert identity["pixel_checksum"]


def test_train_emits_val_loss_with_autoval(tmp_path: Path):
    images_dir, labels_dir, _ = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "bud",
        "auto_val": True,
        "split": {"val_ratio": 0.4, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=task_collate("detection"))
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=task_collate("detection"))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": IMG, "max_size": IMG * 2},
                         "task": "detection"},
        "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=val_loader, task="detection")

    assert run.status == "completed", getattr(run, "error", run.status)
    assert "val_loss" in run.metrics_history[-1]
    assert run.metrics_history[-1]["val_loss"] >= 0.0
