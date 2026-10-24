"""Auto train/val wiring: auto_train_val plus the detection val-loss pass.

These exercise the helper that derives a group-aware val split and the generic_trainer
``validate`` detection path.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import train  # noqa: E402
from tests._chain_fixtures import BLOB_BUILDER, training_config  # noqa: E402
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    big_single_source,
    label_image,
    train_val as _train_val,
)
from tests._training_values import drawn_split, stage, tiling_block  # noqa: E402
from tests._verified_checkpoint_fixtures import partition_side as recorded_side  # noqa: E402

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.split_construction import Partition
    from tcip_mcp.pipelines.raster_source import SourceHeader
    from tcip_mcp.pipelines.schemas import Rect, SpatialManifest
    from tcip_mcp.pipelines.training.envelope import TrainContext

IMG = 64


_save_png = partial(write_noise_image, size=IMG)

_BUD = [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))]

_TILED = tiling_block(tile_size=128, overlap=0.2)
"""The tiling block the spatial-split runs here state."""


def _drawn(**stated: object) -> dict:
    """Return a drawn split at seed 1 grouping each source's tiles, with ``stated`` over it."""
    return drawn_split(**{"seed": 1, "grouping": "tile_prefix", **stated})


def _manifest(partition: Partition) -> SpatialManifest:
    """Return the within-image split the resolved ``partition`` records."""
    assert partition.spatial is not None, (
        "the run resolved no within-image split"
    )
    return partition.spatial


def _detection_dataset(
    root: Path,
    prefixes: tuple[str, ...] = ("srcA", "srcB", "srcC", "srcD"),
    tiles: int = 2,
) -> tuple[Path, list[str]]:
    """Write ``tiles`` labeled images per prefix in ``root``'s undated capture.

    Returns the capture's directory and the stems.
    """
    images_dir = root / "images" / UNDATED_BUCKET
    all_stems = []
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            _save_png(images_dir / f"{stem}.png")
            label_image(images_dir / f"{stem}.png", _BUD, IMG, IMG)
            all_stems.append(stem)
    return images_dir, all_stems


def test_a_run_naming_a_ground_truth_place_that_does_not_exist_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """A run naming a ground-truth place that does not exist is refused, naming it.

    A stated place that does not exist, or is empty, raises naming ``data.labels_dir``; the
    same config naming no place reads the images' own label documents.
    """
    images_dir, _stems = _detection_dataset(tmp_path / "ds")
    (tmp_path / "empty").mkdir()
    data_cfg: dict = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.15),
    }

    for place in (tmp_path / "gone", tmp_path / "empty"):
        with pytest.raises(ValueError, match="data.labels_dir"):
            _train_val(
                tmp_path, "detection", {**data_cfg, "labels_dir": str(place)}
            )

    # Admits valid work: the same config naming no place reads the images' own documents.
    train_ds, _val_ds, _partition, _data = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert train_ds.num_samples


def test_auto_train_val_detection_splits(tmp_path: Path) -> None:
    images_dir, all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.4),
    }
    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    # The loaders index by each sample's own source, so a drawn run and a bound one key alike;
    # the run's recorded partition is what names members as bare stems.
    assert sorted(
        Path(s).stem for s in train_ds.stems + val_ds.stems
    ) == sorted(all_stems)
    assert sorted(
        recorded_side(partition, "train") + recorded_side(partition, "val")
    ) == sorted(all_stems)
    assert val_ds.transforms is None


def test_auto_train_val_malformed_grouping_raises(tmp_path: Path) -> None:
    """An unrecognized split.grouping policy is a caller-config error and propagates."""
    images_dir, _all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(grouping="not_a_real_grouping_key", val_ratio=0.15),
    }
    with pytest.raises(ValueError):
        _train_val(tmp_path, "detection", data_cfg)


def test_auto_train_val_malformed_val_ratio_refuses(tmp_path: Path) -> None:
    """The run refuses a malformed val_ratio, naming the value."""
    images_dir, _all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio="not_a_number"),
    }
    with pytest.raises(ValueError, match="not_a_number"):
        _train_val(tmp_path, "detection", data_cfg)


def test_auto_train_val_ordinal_draws_over_the_tables_own_rows(
    tmp_path: Path,
) -> None:
    """An ordinal run's draw partitions the samples of its table's own rows.

    An ordinal run's ground truth is a table, so the platform's own producer names one sample
    per admitted row and the draw partitions those samples: the run gets a real validation loader
    and a recorded partition, and the training loader reads the row each of its samples names.
    """
    from tests.tiny_trainer_fixtures import write_regression_dataset

    ranks = [i % 2 for i in range(4)]
    images_dir, csv_path = write_regression_dataset(
        tmp_path, [0.2, 0.4, 0.6, 0.8], ranks, height=IMG, width=IMG
    )

    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": _drawn(val_ratio=0.5),
    }
    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "ordinal", data_cfg
    )

    assert val_ds is not None
    assert train_ds.num_samples + val_ds.num_samples == 4
    # Each side reads the rows its own samples name, keyed by their own sources.
    assert set(train_ds.stems).isdisjoint(val_ds.stems)
    assert recorded_side(partition, "train") and recorded_side(
        partition, "val"
    )
    assert sorted(
        recorded_side(partition, "train") + recorded_side(partition, "val")
    ) == [f"img{i}" for i in range(4)]
    assert {s.ground_truth for s in partition.samples} == {str(csv_path)}
    by_row = {f"img{i}": rank for i, rank in enumerate(ranks)}
    for idx, key in enumerate(train_ds.stems):
        assert (
            train_ds[idx][1]["ranks"] == by_row[train_ds.sample_of(key).member]
        )


def test_auto_train_val_single_untiled_source_refuses_validation_and_trains_without_it(
    tmp_path: Path,
) -> None:
    """A single untiled source refuses a requested validation and builds without one if stated.

    A requested val share raises naming every share zero; with a val_ratio of zero, the training
    dataset holds the one sample and no validation dataset is built.
    """
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    _save_png(images_dir / "src_0_0.png")
    label_image(images_dir / "src_0_0.png", _BUD, IMG, IMG)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.15),
    }
    with pytest.raises(ValueError, match="every share zero"):
        _train_val(tmp_path, "detection", data_cfg)

    data_cfg["split"] = _drawn(val_ratio=0.0)
    train_ds, val_ds, _, _data = _train_val(tmp_path, "detection", data_cfg)
    assert val_ds is None and train_ds.num_samples == 1


def test_auto_train_val_single_source_tiled_spatial_split(
    tmp_path: Path,
) -> None:
    """A single tiled detection source derives a real, disjoint spatial val split.

    It does not degrade to no validation; train and val tiles share no tile and no strip
    identity.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    assert train_ds.tile_size == 128 and val_ds.tile_size == 128
    assert train_ds.num_samples > 0 and val_ds.num_samples > 0
    assert set(train_ds.tile_entries).isdisjoint(set(val_ds.tile_entries))
    # The partition holds the one source and its spatial regions.
    assert recorded_side(partition, "train") == [stem]
    manifest = _manifest(partition)
    assert manifest.train_identities and manifest.val_identities
    assert set(manifest.train_identities).isdisjoint(
        set(manifest.val_identities)
    )
    assert all(
        i.startswith(f"{stem}::strip_") for i in manifest.train_identities
    )
    assert manifest.kept_holdout_tiles > 0


def test_a_single_tiled_source_with_no_val_share_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A single tiled source with a reserved share and no val share is refused by name.

    The refusal asks for a val_ratio above zero.
    """
    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.0, holdout_ratio=0.1),
    }
    with pytest.raises(ValueError, match="val_ratio above zero"):
        _train_val(tmp_path, "detection", data_cfg)


def test_spatial_manifest_tied_val_holdout_fractions_place_by_declared_order(
    tmp_path: Path,
) -> None:
    """Tied val and holdout shares are placed by declared order.

    ``val_ratio == holdout_ratio`` ties their shares in the center-out tie-break, so which
    strip val lands on comes from declared (``split_names``) order alone: ``spatial_single_
    source_split`` fixes that order itself (``("train", "val", "holdout")``), so this pins the
    resulting regions against the fixed order's own layout. The manifest carries no ``seed``
    key.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.2, holdout_ratio=0.2),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    manifest = _manifest(partition)
    assert manifest.train_region == [(1030, 0, 3218, 3000)]
    assert manifest.val_region == [(0, 0, 849, 3000)]
    assert manifest.holdout_region == [(3399, 0, 4000, 3000)]
    assert "seed" not in manifest.model_dump()


def test_spatial_manifest_tied_holdout_calibration_fractions_place_by_declared_order(
    tmp_path: Path,
) -> None:
    """Tied holdout and calibration shares are placed by declared order.

    ``calibration_ratio == holdout_ratio`` ties their shares in the center-out tie-break; the
    fixed declared order (``("train", "val", "holdout", "calibration")``) pins the
    calibration/holdout regions, and the manifest carries no ``seed`` key.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(
            val_ratio=0.25, holdout_ratio=0.15, calibration_ratio=0.15
        ),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    manifest = _manifest(partition)
    assert manifest.calibration_region == [(0, 0, 643, 3000)]
    assert manifest.holdout_region == [(3605, 0, 4000, 3000)]
    assert "seed" not in manifest.model_dump()


def test_spatial_manifest_pins_train_val_and_holdout_regions_for_distinct_fractions(
    tmp_path: Path,
) -> None:
    """The spatial manifest places the train, val and holdout regions for distinct shares.

    With shares 0.65/0.25/0.1 over a 4000x3000 source, train is pinned at (1236, 0, 3630,
    3000), val at (0, 0, 1055, 3000) and holdout at (3811, 0, 4000, 3000).
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    manifest = _manifest(partition)
    assert manifest.train_region == [(1236, 0, 3630, 3000)]
    assert manifest.val_region == [(0, 0, 1055, 3000)]
    assert manifest.holdout_region == [(3811, 0, 4000, 3000)]


def test_spatial_manifest_persists_train_and_val_regions_too(
    tmp_path: Path,
) -> None:
    """The spatial manifest persists train and val regions as real rects.

    train_region, val_region and holdout_region each hold real four-number rects, not just
    per-region tile identities.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    manifest = _manifest(_train_val(tmp_path, "detection", data_cfg)[2])
    assert (
        manifest.train_region
        and manifest.val_region
        and manifest.holdout_region
    )
    for region in (
        manifest.train_region,
        manifest.val_region,
        manifest.holdout_region,
    ):
        for rect in region:
            assert len(rect) == 4


def test_auto_train_val_single_source_spatial_split_ignores_a_stray_keep_regions_in_tiling(
    tmp_path: Path,
) -> None:
    """A single-source spatial split builds a validation loader beside a stray keep_regions.

    The tiling block names its own keep_regions, and the run still has a validation side.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": {**_TILED, "keep_regions": [(0, 0, 100, 100)]},
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    train_ds, val_ds, _, _data = _train_val(tmp_path, "detection", data_cfg)
    assert val_ds is not None


def _two_mosaic_tiles(tmp_path: Path) -> tuple[Path, list[str]]:
    """Write the labeled images ``mosaicA_0_0`` and ``mosaicA_1_1``; return their tree and stems."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    stems = ["mosaicA_0_0", "mosaicA_1_1"]
    for stem in stems:
        _save_png(images_dir / f"{stem}.png")
        label_image(images_dir / f"{stem}.png", _BUD, IMG, IMG)
    return images_dir, stems


def test_auto_train_val_degenerate_grouping_refuses_and_a_stem_grouping_draws(
    tmp_path: Path,
) -> None:
    """A degenerate grouping is refused, and the same tree grouped by stem draws.

    Two stems whose default tile_prefix grouping collapses to one group hold one foreground
    group, short of a train and a val side: the run refuses through the draw's own floor, and
    the same tree grouped by stem draws two disjoint sides.
    """
    images_dir, stems = _two_mosaic_tiles(tmp_path)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.5),
    }
    with pytest.raises(
        ValueError, match="fewer than the 2 the requested sides need"
    ):
        _train_val(tmp_path, "detection", data_cfg)
    data_cfg["split"]["grouping"] = "stem"
    train_ds, val_ds, partition, _data = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    assert sorted(
        Path(s).stem for s in train_ds.stems + val_ds.stems
    ) == sorted(stems)
    assert partition.group_by == "stem"


def test_auto_train_val_explicit_grouping_map_starving_val_refuses(
    tmp_path: Path,
) -> None:
    """A caller-supplied grouping map that starves val is refused through the draw's floor."""
    images_dir, stems = _two_mosaic_tiles(tmp_path)
    group_key_map = {
        f"{UNDATED_BUCKET}/{s}": "one_group_for_everything" for s in stems
    }
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.5, grouping=group_key_map),
    }
    with pytest.raises(
        ValueError, match="fewer than the 2 the requested sides need"
    ):
        _train_val(tmp_path, "detection", data_cfg)


def test_a_calibration_ratio_of_zero_reserves_no_calibration_region(
    tmp_path: Path,
) -> None:
    """A calibration_ratio stated at zero reserves no calibration region.

    The spatial_manifest carries no calibration_region and still holds its train, val and
    holdout regions.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    manifest = _manifest(partition)
    assert manifest.calibration_region == []
    assert manifest.kept_calibration_tiles == 0
    assert (
        manifest.train_region
        and manifest.val_region
        and manifest.holdout_region
    )


def test_a_single_source_spatial_run_builds_its_loaders_at_the_stated_band_count(
    tmp_path: Path,
) -> None:
    """A single-source spatial run builds its loaders at the stated band count.

    A run configured for one channel builds both loaders at one channel and reads its tiles at
    one channel.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "num_channels": 1,
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    train_ds, val_ds, partition, resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    assert recorded_side(partition, "train") == [stem]
    assert partition.spatial is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 1
    image, _target = train_ds[0]
    assert image.shape[0] == 1


def _within_image_context(
    tmp_path: Path,
) -> tuple[TrainContext, Partition, str]:
    """Return a within-image split run's ``prepare_run_context`` context, partition and stem.

    The run is opened by ``opened_run`` over a 4000x3000 single source.
    """
    from tcip_mcp.experiments import observe_run
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )
    from tests._verified_checkpoint_fixtures import opened_run, unbuilt_source

    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    observation = observe_run(
        opened_run(
            tmp_path,
            training_config(tmp_path, unbuilt_source("detection"), data_cfg),
        )
    )
    return (
        prepare_run_context(observation),
        observation.record.resolution.partition,
        stem,
    )


def test_a_bodys_loader_on_a_within_image_split_run_holds_only_train_region_tiles(
    tmp_path: Path,
) -> None:
    """A body's loader on a within-image split run holds only train-region tiles.

    A training body's loader over the run's one sample is that sample's train view: every tile
    it serves lies inside the train region the run recorded.
    """
    from tcip_mcp.pipelines.raster_source import rect_contains_rect

    ctx, partition, _stem = _within_image_context(tmp_path)
    spatial = partition.spatial
    assert spatial is not None

    served = ctx.build_dataset(samples=partition.samples)
    assert served.tile_entries
    assert all(
        any(rect_contains_rect(r, box) for r in spatial.train_region)
        for _stem, box in served.tile_entries
    )


def test_a_bodys_loader_on_a_within_image_split_run_refuses_another_sample_naming_its_own(
    tmp_path: Path,
) -> None:
    """A body's loader on a within-image split run refuses another sample, naming its own.

    A loader over a sample list that is not the run's one sample refuses naming that sample.
    """
    from tests._producer_fixtures import samples_over, seed_bud_images

    ctx, _partition, stem = _within_image_context(tmp_path)
    other = seed_bud_images(
        tmp_path / "other" / "images" / UNDATED_BUCKET, n=1
    )
    with pytest.raises(ValueError, match=stem):
        ctx.build_dataset(samples=samples_over(other, subject="bud"))


def _multiband_source(images_dir: Path, stem: str, bands: int) -> None:
    """Write a ``bands``-band 40x24 TIFF ``stem`` in ``images_dir``, with a ``bud`` box label.

    Each band is one ``painted_array`` band painted 255 over (28, 12, 34, 18), and the box is
    that rectangle.
    """
    import numpy as np
    import tifffile

    from tests._producer_fixtures import painted_array

    images_dir.mkdir(parents=True, exist_ok=True)
    band = painted_array(40, 24, [((28, 12, 34, 18), 255)])
    tifffile.imwrite(
        str(images_dir / f"{stem}.tif"), np.stack([band] * bands, axis=-1)
    )
    label_image(
        images_dir / f"{stem}.tif",
        [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))],
        40,
        24,
    )


def test_a_runs_band_count_is_read_over_every_source_and_a_disagreement_refuses(
    tmp_path: Path,
) -> None:
    """Sources agreeing on a band count size both loaders at it, and a disagreement is refused.

    Two five-band sources drawn to the two sides size both loaders at 5; adding a three-band
    source is refused, naming the different band counts.
    """
    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    for stem in ("a", "b"):
        _multiband_source(images_dir, stem, 5)
    split = _drawn(grouping="stem", val_ratio=0.5)
    base_cfg = {"images_dir": str(images_dir), "scope": {"subject": "bud"}}

    train_ds, val_ds, _, _data = _train_val(
        tmp_path, "detection", {**base_cfg, "split": split}
    )
    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 5

    _multiband_source(images_dir, "c", 3)
    with pytest.raises(ValueError, match="different band counts"):
        _train_val(tmp_path, "detection", {**base_cfg, "split": split})


def _probe_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every source whose header's own band count is read; return the record.

    A header's first band-count read appends its source, in call order.
    """
    from functools import cached_property

    from tcip_mcp.pipelines import raster_source

    probed: list[str] = []
    real_channels = raster_source.SourceHeader.channels.func

    def _record(header: SourceHeader) -> int:
        probed.append(str(header.source))
        return real_channels(header)

    recorded = cached_property(_record)
    recorded.__set_name__(raster_source.SourceHeader, "channels")
    monkeypatch.setattr(raster_source.SourceHeader, "channels", recorded)
    return probed


def test_a_stated_band_count_reads_every_source_at_it_and_probes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stated band count sizes both loaders at it and probes no source.

    A run over an L source and an RGB one, stated at one channel, builds both loaders at one
    channel and reads a train tile at one channel.
    """
    from PIL import Image

    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    for stem, mode in (("gray", "L"), ("color", "RGB")):
        Image.new(mode, (40, 24)).save(images_dir / f"{stem}.png")
        label_image(
            images_dir / f"{stem}.png",
            [Annotation(subject="bud", geometry=BBox(28, 12, 34, 18))],
            40,
            24,
        )
    probed = _probe_spy(monkeypatch)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "num_channels": 1,
        "split": _drawn(grouping="stem", val_ratio=0.5),
    }

    train_ds, val_ds, _, _data = _train_val(tmp_path, "detection", data_cfg)

    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 1
    assert train_ds[0][0].shape[0] == 1
    assert probed == []


def test_a_runs_sources_are_probed_once_each_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each of a run's sources is probed once for the whole run.

    Four five-band sources are each probed exactly once while the run builds both loaders.
    """
    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    for stem in ("a", "b", "c", "d"):
        _multiband_source(images_dir, stem, 5)
    probed = _probe_spy(monkeypatch)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(grouping="stem", val_ratio=0.5),
    }

    train_ds, val_ds, _, _data = _train_val(tmp_path, "detection", data_cfg)

    assert val_ds is not None
    assert train_ds.expected_channels == val_ds.expected_channels == 5
    assert len(probed) == len(set(probed)) == 4


def test_one_preflight_reads_a_sources_header_once_for_its_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid smoke preflight with a reserved calibration share reads a band count twice.

    The run is a single tiled source with val, holdout and calibration shares.
    """
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = training_config(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "tiling": _TILED,
            "split": _drawn(
                val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15
            ),
        },
        batch_size=1,
        stages=[stage(0)],
    )
    probed = _probe_spy(monkeypatch)

    result = preflight_config(tmp_path, cfg, smoke=True)

    assert result["valid"] is True, result["issues"]
    # One read for the run's sizes; the second is the spatial manifest's own raster identity,
    # which answers for the file on disk rather than the width the run reads at.
    assert len(probed) == 2, probed


def test_a_bound_run_records_the_width_it_read_its_sources_at(
    tmp_path: Path,
) -> None:
    """A bound run records the band width it read its sources at on its data config.

    The resolved data block records five channels, and ``recorded_model_dims`` reads them back
    as ``in_chans``.
    """
    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    for stem in ("a", "b"):
        _multiband_source(images_dir, stem, 5)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(grouping="stem", val_ratio=0.5),
    }

    train_ds, _val_ds, _, resolved = _train_val(
        tmp_path, "detection", data_cfg
    )

    assert resolved.num_channels == train_ds.expected_channels == 5
    from tcip_mcp.pipelines.model_build import recorded_model_dims
    from tcip_mcp.pipelines.schemas import train_config
    from tests._verified_checkpoint_fixtures import unbuilt_source

    spec = train_config(
        training_config(
            tmp_path, unbuilt_source("detection"), resolved.record()
        )
    )
    assert recorded_model_dims(spec)["in_chans"] == 5


def test_a_calibration_ratio_adds_a_disjoint_calibration_region(
    tmp_path: Path,
) -> None:
    """A reserved calibration region is real, non-empty geometry disjoint from the other sides."""
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(
            val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15
        ),
    }
    train_ds, val_ds, partition, _resolved = _train_val(
        tmp_path, "detection", data_cfg
    )
    assert val_ds is not None
    manifest = _manifest(partition)
    assert manifest.calibration_region
    assert manifest.kept_calibration_tiles > 0

    def _rects(region: list[Rect]) -> list[tuple[int, ...]]:
        return [tuple(r) for r in region]

    from tcip_mcp.pipelines.raster_source import rects_overlap

    cal_rects = _rects(manifest.calibration_region)
    for other_region in (
        manifest.train_region,
        manifest.val_region,
        manifest.holdout_region,
    ):
        for other in _rects(other_region):
            for cr in cal_rects:
                assert not rects_overlap(cr, other)


def test_a_spatial_split_raises_on_unresolvable_extent(tmp_path: Path) -> None:
    """A spatial split over a label document with no width/height is refused.

    The refusal says the document states no positive width and height. The one source is
    admitted through ``admit_over`` and its block resolved through ``run_sizes`` and
    ``_with_lattice``; the same block over a document stating its frame draws a train region.
    """
    from tcip_mcp.pipelines.data.split_construction import (
        _with_lattice,
        run_sizes,
        spatial_single_source_split,
        stated_shares,
    )
    from tcip_mcp.pipelines.schemas import DataSpec
    from tests._producer_fixtures import admit_over

    stated = DataSpec.model_validate(
        {
            "tiling": _TILED,
            "split": _drawn(
                val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.15
            ),
        }
    )

    def _split_over(images_dir: Path) -> SpatialManifest:
        admitted = admit_over(images_dir, subject="bud")
        samples = admitted.every_sample()
        block = _with_lattice(
            "detection",
            run_sizes("detection", stated, admitted.scope, samples),
            samples,
        )
        return spatial_single_source_split(
            "detection", samples[0], block, stated_shares(stated.split)
        )

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    _save_png(images_dir / "mosaic.png")
    # A readable, annotated document recording no width/height, distinct from an unreadable one.
    label_image(images_dir / "mosaic.png", _BUD, 0, 0)
    with pytest.raises(
        ValueError, match="states no positive width and height"
    ):
        _split_over(images_dir)

    framed, _stem = big_single_source(tmp_path / "framed", 4000, 3000)
    assert _split_over(framed).train_region


def test_single_tiled_source_raises_on_an_unreadable_label(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A single tiled source with an unreadable label document raises.

    The document is stored as a list, not an object; the run raises
    UnreadableLabelDocumentError and logs no training without validation.
    """
    import tcip_store
    from tcip_annotation.json_io import UnreadableLabelDocumentError
    from tests._producer_fixtures import image_label_key

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    _save_png(images_dir / "mosaic.png")
    # Not an object: unreadable, stored past the writer.
    tcip_store.replace(image_label_key(images_dir / "mosaic.png"), [])

    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.2),
    }
    with pytest.raises(UnreadableLabelDocumentError):
        _train_val(tmp_path, "detection", data_cfg)
    assert "training without validation" not in caplog.text


def test_a_spatial_split_raises_on_infeasible_layout(tmp_path: Path) -> None:
    """A spatial split whose layout is infeasible at the mosaic size is refused.

    With val, holdout and calibration shares of 0.33 each over a 4000x3000 source, the refusal
    says the layout is infeasible at this mosaic size.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(
            val_ratio=0.33, holdout_ratio=0.33, calibration_ratio=0.33
        ),
    }
    with pytest.raises(ValueError, match="is infeasible at this mosaic size"):
        _train_val(tmp_path, "detection", data_cfg)


def test_a_spatial_split_raises_on_empty_gt_bearing_side(
    tmp_path: Path,
) -> None:
    """A spatial split with a side holding no ground truth under skip_empty is refused.

    With tiling.skip_empty set, ground truth placed between x 1300 and 3960 of a 4000-wide
    mosaic leaves a side whose tiles carry none, and the run refuses naming zero kept tiles.
    """
    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    width, height = 4000, 300
    from torchvision.utils import save_image

    save_image(
        torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png")
    )
    boxes = [
        Annotation(subject="bud", geometry=BBox(x, 20, x + 20, 40))
        for x in range(1300, 3960, 40)
    ]
    label_image(images_dir / f"{stem}.png", boxes, width, height)

    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=64, overlap=0.2, skip_empty=True),
        "split": _drawn(
            val_ratio=0.2, holdout_ratio=0.1, calibration_ratio=0.2
        ),
    }
    with pytest.raises(ValueError, match="zero kept"):
        _train_val(tmp_path, "detection", data_cfg)


def test_a_spatial_split_records_raster_content_identity(
    tmp_path: Path,
) -> None:
    """A spatial split records the single source raster's content identity.

    A real, decodable single-source raster gets a raster_content_identity in its
    spatial_manifest, stating its width, height and pixel checksum.
    """
    images_dir, stem = big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": _TILED,
        "split": _drawn(val_ratio=0.25, holdout_ratio=0.1),
    }
    manifest = _manifest(_train_val(tmp_path, "detection", data_cfg)[2])
    identity = manifest.raster_content_identity
    assert identity is not None
    assert identity.width == 4000 and identity.height == 3000
    assert identity.pixel_checksum


def test_train_emits_val_loss_with_autoval(tmp_path: Path) -> None:
    from tests._producer_fixtures import seed_labeled_images

    images_dir = seed_labeled_images(
        tmp_path / "ds" / "images" / UNDATED_BUCKET,
        _BUD,
        n=8,
        width=IMG,
        height=IMG,
    )
    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "split": _drawn(val_ratio=0.4),
    }
    from tests.tiny_trainer_fixtures import run_context

    ctx = run_context(
        tmp_path, training_config(tmp_path, BLOB_BUILDER, data_cfg)
    )
    assert ctx.val_loader is not None
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    assert "val_loss" in run.metrics_history[-1]
    assert run.metrics_history[-1]["val_loss"] >= 0.0
