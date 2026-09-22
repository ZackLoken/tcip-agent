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

from tcip_mcp.pipelines.data.split_construction import (  # noqa: E402
    auto_train_val, recorded_side,
)
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


@pytest.mark.parametrize("labels_dir", [None, "absent"])
def test_auto_train_val_refuses_a_missing_location_in_preflights_own_words(
    tmp_path: Path, labels_dir: str | None,
):
    """A run reads its samples out of the places its config names, so a missing or unset location
    refuses through the one missing-key refusal preflight states, naming the key. An unset value
    never reaches a path as an empty string or a ``None``."""
    images_dir, real_labels, _stems = _detection_dataset(tmp_path / "ds")
    data_cfg: dict = {"images_dir": str(images_dir), "subject": "bud"}
    if labels_dir == "absent":
        data_cfg["labels_dir"] = str(tmp_path / "gone")

    with pytest.raises(ValueError, match="data.labels_dir"):
        auto_train_val("detection", data_cfg, None)

    # Admits valid work: the same config naming a real location trains.
    data_cfg["labels_dir"] = str(real_labels)
    train_ds, _val_ds, _partition = auto_train_val("detection", data_cfg, None)
    assert train_ds.num_samples


def test_auto_train_val_detection_splits(tmp_path: Path):
    images_dir, labels_dir, all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "bud",
        "auto_val": True,
        "split": {"val_ratio": 0.4, "seed": 1},
    }
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    # The loaders index by each sample's own source, so a drawn run and a bound one key alike;
    # the run's recorded partition is what names members as bare stems.
    assert sorted(Path(s).stem for s in train_ds.stems + val_ds.stems) == sorted(all_stems)
    assert sorted(recorded_side(partition, "train") + recorded_side(partition, "val")) == \
        sorted(all_stems)
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
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    assert val_ds is None
    # Still the producer's own samples, indexed by source identity, with every admitted stem
    # recorded as trained: a failed draw drops the validation side, never the membership.
    assert sorted(Path(s).stem for s in train_ds.stems) == sorted(all_stems)
    assert recorded_side(partition, "train") == sorted(all_stems)
    assert recorded_side(partition, "val") == []


def test_auto_train_val_ordinal_draws_over_the_tables_own_rows(tmp_path: Path):
    """An ordinal run's ground truth is a table, so the platform's own producer names one sample
    per admitted row and the draw partitions those samples: the run gets a real validation loader
    and a recorded partition, and each loader reads the row its own sample names."""
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

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path), "auto_val": True,
                "split": {"val_ratio": 0.5, "seed": 1}}
    train_ds, val_ds, partition = auto_train_val("ordinal", data_cfg, None)

    assert val_ds is not None
    assert train_ds.num_samples + val_ds.num_samples == 4
    # Each side reads the rows its own samples name, keyed by their own sources.
    assert set(train_ds._stems).isdisjoint(val_ds._stems)
    assert recorded_side(partition, "train") and recorded_side(partition, "val")
    assert sorted(recorded_side(partition, "train") + recorded_side(partition, "val")) == \
        [f"img{i}" for i in range(4)]
    assert sorted(partition) == [str(csv_path)]
    by_row = dict(rows)
    for key, rank in zip(train_ds._stems, train_ds._ranks):
        assert rank == by_row[train_ds.member_stem_of(key)]


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
    """A single-image detection source with tiling absent degrades to (train_ds, None): there is
    no tiling geometry to block-split by."""
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
    degrading to no validation, train and val tiles sharing no tile and no strip identity."""
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


def test_spatial_manifest_tied_val_test_fractions_place_by_declared_order(tmp_path: Path):
    """``val_ratio == test_ratio`` ties their shares in the center-out tie-break, so which
    strip val lands on comes from declared (``split_names``) order alone: ``spatial_single_
    source_split`` fixes that order itself (``("train", "val", "test")``), so this pins the
    resulting regions against the fixed order's own layout, the way the distinct-fractions
    test above pins its own regions; the placement half is coverage. The manifest carries no
    ``seed`` key; the assertion guards that absence alone."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.2},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["train_region"] == [(1020, 0, 3086, 3000)]
    assert manifest["val_region"] == [(0, 0, 842, 3000)]
    assert manifest["test_region"] == [(3264, 0, 3902, 3000)]
    assert "seed" not in manifest


def test_spatial_manifest_tied_test_calibration_fractions_place_by_declared_order(
    tmp_path: Path,
):
    """``reserve_calibration_fraction == test_ratio`` ties their shares the same way; the
    fixed declared order (``("train", "val", "test", "calibration")``) pins the calibration/test
    regions the same way the val/test tie above pins its own, coverage of the placement, and the
    manifest carries no ``seed`` key; the assertion guards that absence alone."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.15, "reserve_calibration_fraction": 0.15},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["calibration_region"] == [(0, 0, 638, 3000)]
    assert manifest["test_region"] == [(3468, 0, 3902, 3000)]
    assert "seed" not in manifest


def test_spatial_manifest_pins_train_val_and_test_regions_for_distinct_fractions(
    tmp_path: Path,
):
    """Coverage, not a guard: with no tied shares (0.65/0.25/0.1) the declared-order tie-break
    does not run, so the layout comes from the fractions and tile geometry alone. The three
    regions are pinned against this exact width, height, tile_size, overlap and fractions."""
    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    manifest = data_cfg["split"]["spatial_manifest"]
    assert manifest["train_region"] == [(1224, 0, 3494, 3000)]
    assert manifest["val_region"] == [(0, 0, 1046, 3000)]
    assert manifest["test_region"] == [(3672, 0, 3902, 3000)]


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
    assert sorted(Path(s).stem for s in train_ds.stems + val_ds.stems) == sorted(stems)
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
    """With reserve_calibration_fraction absent, the spatial_manifest carries no
    calibration_region and the rest of it is the three-way split's own shape (the same keys and
    the same train/val/test regions for this layout)."""
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


def test_a_single_source_spatial_run_builds_its_loaders_at_the_stated_band_count(tmp_path: Path):
    """The sizes a run's config states reach the one-source spatial route too: a run configured
    for one channel indexes its tile lattice and reads its tiles at one channel, rather than
    probing its own source back to three."""
    images_dir, labels_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "num_channels": 1,
        "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert data_cfg["split"]["resolved_group_by"] == "spatial_strip"
    assert train_ds.expected_channels == val_ds.expected_channels == 1
    image, _target = train_ds[0]
    assert image.shape[0] == 1


def _multiband_source(images_dir: Path, labels_dir: Path, stem: str, bands: int) -> None:
    """One annotated raster of ``bands`` bands, written the way the multi-band readers expect."""
    import numpy as np
    import tifffile

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    array = np.zeros((24, 40, bands), dtype=np.uint8)
    array[12:18, 28:34, :] = 255
    tifffile.imwrite(str(images_dir / f"{stem}.tif"), array)
    json_io.write_annotations(
        str(labels_dir / f"{stem}.json"),
        [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))], 40, 24, keep_empty=True)


def test_a_runs_band_count_is_read_over_every_source_and_a_disagreement_refuses(tmp_path: Path):
    """The band count is one fact per run, read over every source it holds: sources that agree
    size both loaders at the count they carry, whichever side each landed on, and one source of
    another count refuses by name rather than leaving a side to be read at the other's count."""
    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    for stem in ("a", "b"):
        _multiband_source(images_dir, labels_dir, stem, 5)
    split = {"group_by": "stem", "val_ratio": 0.5, "seed": 1}
    base_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"}

    train_ds, val_ds, _ = auto_train_val("detection", {**base_cfg, "split": dict(split)}, None)
    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 5

    _multiband_source(images_dir, labels_dir, "c", 3)
    with pytest.raises(ValueError, match="different band counts"):
        auto_train_val("detection", {**base_cfg, "split": dict(split)}, None)


def _probe_spy(monkeypatch) -> list[str]:
    """Every source the band-count probe is asked about, in call order."""
    from tcip_mcp.pipelines import derivations

    probed: list[str] = []
    real_probe = derivations.probe_channels

    def _record(source):
        probed.append(str(source))
        return real_probe(source)

    monkeypatch.setattr(derivations, "probe_channels", _record)
    return probed


def test_a_stated_band_count_reads_every_source_at_it_and_probes_none(tmp_path: Path,
                                                                      monkeypatch):
    """A config that states how to read its sources is taken at its word: a run over an L source
    and an RGB one, stated at one channel, builds both loaders at one channel rather than
    deriving a disagreement it was told how to resolve, and probes no source to do it."""
    from PIL import Image

    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem, mode in (("grey", "L"), ("colour", "RGB")):
        Image.new(mode, (40, 24)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))], 40, 24, keep_empty=True)
    probed = _probe_spy(monkeypatch)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "bud", "num_channels": 1,
                "split": {"group_by": "stem", "val_ratio": 0.5, "seed": 1}}

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 1
    assert train_ds[0][0].shape[0] == 1
    assert probed == []


def test_a_runs_sources_are_probed_once_each_for_the_whole_run(tmp_path: Path, monkeypatch):
    """The band count is read where the run resolves its sizes and nowhere else: each source is
    probed once for the run, not again inside every loader the run builds."""
    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    for stem in ("a", "b", "c", "d"):
        _multiband_source(images_dir, labels_dir, stem, 5)
    probed = _probe_spy(monkeypatch)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                "split": {"group_by": "stem", "val_ratio": 0.5, "seed": 1}}

    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 5
    assert len(probed) == len(set(probed)) == 4


def test_one_preflight_reads_a_sources_header_once_for_its_sizes(tmp_path: Path, monkeypatch):
    """Preflight resolves this run's sizes once and every leg that needs them takes that answer:
    the reserved-calibration feasibility probe builds the run's own dataset at the sizes already
    resolved rather than resolving a second time over the same source."""
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, labels_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3, "min_size": 64,
                                            "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
                 "split": {"val_ratio": 0.2, "test_ratio": 0.1,
                           "reserve_calibration_fraction": 0.15}},
        "batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}],
    }
    probed = _probe_spy(monkeypatch)

    result = preflight_config(cfg, smoke=True)

    assert result["valid"] is True, result["issues"]
    # One read for the run's sizes; the second is the spatial manifest's own raster identity,
    # which answers for the file on disk rather than the width the run reads at.
    assert len(probed) == 2, probed


def test_a_bound_run_records_the_width_it_read_its_sources_at(tmp_path: Path):
    """The width a run resolved is a fact about the checkpoint it produces, so the run records it
    on its own data config: a later reader takes the recorded width rather than assuming RGB."""
    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    for stem in ("a", "b"):
        _multiband_source(images_dir, labels_dir, stem, 5)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                "split": {"group_by": "stem", "val_ratio": 0.5, "seed": 1}}

    train_ds, _val_ds, _ = auto_train_val("detection", data_cfg, None)

    assert data_cfg["num_channels"] == train_ds.expected_channels == 5
    from tcip_mcp.pipelines.model_build import run_in_chans
    assert run_in_chans({"builder": "m:f"}, data_cfg) == 5


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
    rather than the unrequested case's silent (train_ds, None) degradation. The one source is
    admitted through the producer the run itself admits through, so the split is derived over the
    dataset the run would build."""
    from tcip_mcp.pipelines.data.datasets import resolve_sizes
    from tcip_mcp.pipelines.data.split_construction import spatial_single_source_split
    from tests._producer_fixtures import admit_over

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "mosaic.png")
    # A readable, annotated document recording no width/height, distinct from an unreadable one.
    json_io.write_annotations(
        str(labels_dir / "mosaic.json"),
        [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))], 0, 0, keep_empty=True)

    tiling = {"enabled": True, "tile_size": 128, "overlap": 0.2}
    split_cfg = {"val_ratio": 0.2, "test_ratio": 0.1, "reserve_calibration_fraction": 0.15}
    admitted = admit_over(images_dir, labels_dir, subject="bud")
    with pytest.raises(ValueError, match="reserve_calibration_fraction"):
        spatial_single_source_split(
            admitted.one_sample(), admitted.scope, tiling, split_cfg, None,
            resolve_sizes("detection", {}, admitted.every_sample()))


def test_single_tiled_source_raises_on_an_unreadable_label_regardless_of_reserve(
    tmp_path: Path, caplog,
):
    """A present, unreadable label document is a categorically different fact than one recording
    no width/height: the run aborts, whether or not reserve_calibration_fraction was requested,
    rather than degrading to no validation over a document nobody can read."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "mosaic.png")
    (labels_dir / "mosaic.json").write_text("[]", encoding="utf-8")  # not a dict: unreadable

    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.1},  # no reserve_calibration_fraction
    }
    with pytest.raises(UnreadableLabelDocument):
        auto_train_val("detection", data_cfg, None)
    assert "training without validation" not in caplog.text


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
    tiling.skip_empty set, a side's tiles carrying no GT filter down to zero real samples. At
    this exact width/tile_size/fractions, spatial_strip_split places train at x in [1275,
    3175] (verified directly against spatial_strip_split for this test's own params); GT is
    placed only inside that range plus calibration's own [3264, 3991], leaving val ([510,
    1186]) and test ([0, 421]) both real, tiled, and entirely GT-free."""
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
    run = create_run(cfg, str(tmp_path / "out"), id="auto-run-77")
    run = train(run, train_loader, val_loader=val_loader, task="detection")

    assert run.status == "completed", getattr(run, "error", run.status)
    assert "val_loss" in run.metrics_history[-1]
    assert run.metrics_history[-1]["val_loss"] >= 0.0
