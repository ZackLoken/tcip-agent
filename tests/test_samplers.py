"""build_sampler's call shape, the imbalance samplers, and the tile-locality sampler.

The imbalance samplers run over a run's own loaders; the tile-locality sampler is covered for
banded read order, runtime band derivation, per-worker lanes under multi-worker loading, and its
refusals.
"""

from __future__ import annotations

import itertools
import os

import pytest

from tests._training_values import sampler

torch = pytest.importorskip("torch")


class _StubTiledDataset:
    """Duck-typed stand-in exposing the tiled-dataset ordering contract.

    It carries ``tile_entries`` and ``source_frames``, its length is its tile count, and item
    ``i`` is ``(i, pid)`` with the pid of the process serving it.
    """

    def __init__(self, tile_entries, source_frames):
        self.tile_entries = tile_entries
        self.source_frames = source_frames

    def __len__(self):
        return len(self.tile_entries)

    def __getitem__(self, i):
        return i, os.getpid()


def _patch_gdal_cache(monkeypatch, cache_bytes: int) -> None:
    from tcip_mcp.pipelines import raster_source

    monkeypatch.setattr(raster_source, "gdal_cache_bytes", lambda: cache_bytes)


def _frame(width=10000, channels=3, itemsize=1, windowed=True):
    return {
        "width": width,
        "height": 4096,
        "channels": channels,
        "dtype_itemsize": itemsize,
        "windowed": windowed,
    }


def _lattice_entries(
    stems, tile_ys=(0, 512, 1024, 1536), tile_xs=(0, 512, 1024), tile=512
):
    return [
        (stem, (tx, ty, tx + tile, ty + tile))
        for stem in stems
        for ty in tile_ys
        for tx in tile_xs
    ]


def _two_source_dataset():
    return _StubTiledDataset(
        _lattice_entries(["a", "b"]),
        {"a": _frame(), "b": _frame()},
    )


def _band_key(dataset, sampler, idx):
    """Return (stem, band ordinal) for a tile index, from the lattice and the band height."""
    stem, box = dataset.tile_entries[idx]
    ys = sorted({b[1] for s, b in dataset.tile_entries if s == stem})
    return stem, ys.index(box[1]) // sampler.band_tile_rows


def _assert_contiguous_bands(dataset, sampler, indices):
    keys = [_band_key(dataset, sampler, i) for i in indices]
    groups = [key for key, _run in itertools.groupby(keys)]
    assert len(groups) == len(set(groups)), (
        "a (source, band) group was split apart"
    )


def _assert_banded(dataset, sampler, order):
    assert sorted(order) == list(range(len(dataset.tile_entries)))
    _assert_contiguous_bands(dataset, sampler, order)


def _assert_bands_owned_by_one_worker(dataset, sampler, per_worker_streams):
    """Assert every (source, band) group is read by exactly one worker.

    Each worker's stream keeps its bands contiguous, and no (source, band) group appears in two
    workers' streams.
    """
    owner: dict[tuple, int] = {}
    for worker, stream in per_worker_streams.items():
        _assert_contiguous_bands(dataset, sampler, stream)
        for key in {_band_key(dataset, sampler, i) for i in stream}:
            assert key not in owner, (
                f"band {key} split across workers {owner[key]} and {worker}"
            )
            owner[key] = worker


# ── single-lane locality ordering ─────────────────────────────────────


def test_tile_locality_orders_are_banded_and_differ_across_epochs(monkeypatch):
    """Tile-locality orders keep each band contiguous and change across epochs.

    Within an epoch every (source, band) group stays contiguous; across consecutive epochs the
    order changes.
    """
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(
        monkeypatch, 62_000_000
    )  # (31e6 // 30000 rows) // 512 pitch = 2 tile rows
    ds = _two_source_dataset()
    sampler = TileLocalitySampler(ds, num_workers=0)
    assert sampler.band_tile_rows == 2

    torch.manual_seed(0)
    epoch1 = list(iter(sampler))
    epoch2 = list(iter(sampler))
    _assert_banded(ds, sampler, epoch1)
    _assert_banded(ds, sampler, epoch2)
    assert epoch1 != epoch2


def test_tile_locality_order_reproducible_under_global_seed(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    sampler = TileLocalitySampler(_two_source_dataset(), num_workers=1)

    torch.manual_seed(123)
    first = list(iter(sampler))
    torch.manual_seed(123)
    again = list(iter(sampler))
    assert first == again


def test_tile_locality_band_height_derived_from_cache_and_row_cost(
    monkeypatch,
):
    """The tile-locality band height is derived from the cache share and the row cost.

    A cache too small for one tile row bands one row, a larger cache bands two, and a source of
    double width at that cache halves the band to one row.
    """
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    ds = _two_source_dataset()

    _patch_gdal_cache(
        monkeypatch, 8_000_000
    )  # too small for even one full tile row: floor of 1
    assert TileLocalitySampler(ds, num_workers=0).band_tile_rows == 1

    _patch_gdal_cache(monkeypatch, 62_000_000)
    assert TileLocalitySampler(ds, num_workers=0).band_tile_rows == 2

    # A costlier source (double width) halves the rows that fit at the same cache.
    wide = _StubTiledDataset(
        _lattice_entries(["a", "b"]),
        {"a": _frame(), "b": _frame(width=20000)},
    )
    assert TileLocalitySampler(wide, num_workers=0).band_tile_rows == 1


def test_tile_locality_band_height_uses_per_worker_cache_share(monkeypatch):
    """The tile-locality band height shrinks with the worker count.

    At one cache size the band is four tile rows with no workers and two rows with two.
    """
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 124_000_000)
    ds = _two_source_dataset()
    assert TileLocalitySampler(ds, num_workers=0).band_tile_rows == 4
    assert (
        TileLocalitySampler(ds, num_workers=2, batch_size=3).band_tile_rows
        == 2
    )


def test_tile_locality_single_tile_row_sources_band_to_one(monkeypatch):
    """One tile row per source leaves no row pitch to derive from; each source is one band."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    ds = _StubTiledDataset(
        _lattice_entries(["a"], tile_ys=(0,)), {"a": _frame()}
    )
    sampler = TileLocalitySampler(ds, num_workers=0)
    assert sampler.band_tile_rows == 1
    assert sorted(iter(sampler)) == list(range(3))


# ── multi-worker lanes ────────────────────────────────────────────────


def test_tile_locality_multi_worker_order_is_laned_per_dispatched_worker(
    monkeypatch,
):
    """A multi-worker tile-locality order gives each dispatched worker whole bands.

    Each epoch's order covers every index, and chunking it into batches dealt round-robin to
    the workers gives every worker a read stream whose (source, band) groups stay contiguous and
    are its own; consecutive epochs differ.
    """
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(
        monkeypatch, 124_000_000
    )  # per-worker share at 2 workers: band of 2 rows
    ds = _two_source_dataset()
    num_workers, batch_size = 2, 3
    sampler = TileLocalitySampler(
        ds, num_workers=num_workers, batch_size=batch_size
    )
    assert sampler.band_tile_rows == 2

    torch.manual_seed(1)
    epoch1 = list(iter(sampler))
    epoch2 = list(iter(sampler))
    for order in (epoch1, epoch2):
        assert sorted(order) == list(range(len(ds)))
        chunks = [
            order[i : i + batch_size] for i in range(0, len(order), batch_size)
        ]
        per_worker = {
            worker: [
                idx
                for ci, chunk in enumerate(chunks)
                if ci % num_workers == worker
                for idx in chunk
            ]
            for worker in range(num_workers)
        }
        _assert_bands_owned_by_one_worker(ds, sampler, per_worker)
    assert epoch1 != epoch2


def test_tile_locality_lanes_land_on_their_worker_through_a_real_loader(
    monkeypatch,
):
    """Tile-locality lanes land on their worker through a real multiprocessing DataLoader.

    Over four equal bands and two workers, each batch carries one worker's pid, both workers
    serve, the number of indices served equals the dataset's length, and each worker's received
    index stream keeps its (source, band) groups contiguous and its own.
    """
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 124_000_000)
    ds = _two_source_dataset()
    sampler = TileLocalitySampler(ds, num_workers=2, batch_size=3)
    assert sampler.band_tile_rows == 2

    torch.manual_seed(0)
    loader = DataLoader(ds, batch_size=3, sampler=sampler, num_workers=2)
    per_worker: dict[int, list[int]] = {}
    served = 0
    for indices, pids in loader:
        pid_set = set(pids.tolist())
        assert len(pid_set) == 1, (
            "one batch was served by more than one worker"
        )
        per_worker.setdefault(pid_set.pop(), []).extend(indices.tolist())
        served += len(indices)

    assert served == len(ds)
    assert len(per_worker) == 2
    _assert_bands_owned_by_one_worker(ds, sampler, per_worker)


# ── refusals, each naming what is required ────────────────────────────


def test_tile_locality_refuses_dataset_without_tile_contract():
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    with pytest.raises(ValueError, match="tile_entries"):
        TileLocalitySampler(object(), num_workers=0)


def test_tile_locality_refuses_when_no_source_is_windowed(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    ds = _StubTiledDataset(
        _lattice_entries(["a"]), {"a": _frame(windowed=False)}
    )
    with pytest.raises(ValueError, match="windowed"):
        TileLocalitySampler(ds, num_workers=0)


def test_tile_locality_refuses_unknown_worker_regime(monkeypatch):
    """A tile-locality sampler built without the loader's num_workers refuses naming it."""
    from tcip_mcp.pipelines.data.samplers import build_sampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    with pytest.raises(ValueError, match="num_workers"):
        build_sampler(_admitted("tile_locality"), _two_source_dataset())


def test_tile_locality_refuses_multi_worker_without_batch_size(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import build_sampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    with pytest.raises(ValueError, match="batch_size"):
        build_sampler(
            _admitted("tile_locality"), _two_source_dataset(), num_workers=2
        )


def test_tile_locality_refuses_windowed_source_without_dtype_size(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    ds = _StubTiledDataset(
        _lattice_entries(["a"]), {"a": _frame(itemsize=None)}
    )
    with pytest.raises(ValueError, match="dtype_itemsize"):
        TileLocalitySampler(ds, num_workers=0)


# ── build_sampler call shape: the loader context must not disturb the existing names ──


def _admitted(name: str, **values):
    """Return the sampler block naming ``name`` at ``values`` as the schema admits it."""
    from tcip_mcp.pipelines.schemas import SamplerSpec

    return SamplerSpec.model_validate(sampler(name, **values))


def _table_run(tmp_path, values: list[int], block: dict, task: str):
    """Return the context of a ``task`` run whose train loader draws through ``block``.

    The run is over one frame per value of ``values`` (each its label), built as the child's own
    entry builds it (:func:`~tests.tiny_trainer_fixtures.run_context`).
    """
    from tests.tiny_trainer_fixtures import (
        classifier_config,
        distinct_intensities,
        frames_data,
        run_context,
    )

    data = frames_data(
        tmp_path / "ds", distinct_intensities(len(values)), values
    )
    return run_context(
        tmp_path,
        classifier_config(1, data, project=tmp_path, task=task, sampler=block),
    )


def test_build_sampler_builds_tile_locality_through_the_factory(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import (
        TileLocalitySampler,
        build_sampler,
    )

    _patch_gdal_cache(monkeypatch, 62_000_000)
    built = build_sampler(
        _admitted("tile_locality"),
        _two_source_dataset(),
        num_workers=2,
        batch_size=4,
    )
    assert isinstance(built, TileLocalitySampler)


def test_build_sampler_imbalance_samplers_build_with_or_without_the_loader_context(
    tmp_path,
):
    """Each imbalance sampler builds with the loader context absent and with it present.

    The class-balanced, oversampling and weighted blocks each build a sampler drawing at least
    one epoch of indices, with the context absent and with it present.
    """
    from tcip_mcp.pipelines.data import samplers

    ds = _table_run(
        tmp_path, [0, 1, 0, 1], sampler("class_balanced"), "classification"
    ).train_loader.dataset
    for block in (
        _admitted("class_balanced"),
        _admitted("oversample", min_count=3),
        _admitted("weighted_random"),
    ):
        for kwargs in ({}, {"num_workers": 4, "batch_size": 8}):
            built = samplers.build_sampler(block, ds, **kwargs)
            assert len(list(iter(built))) >= len(ds)


@pytest.mark.parametrize(
    "block, named",
    [
        (sampler("oversample"), "min_count"),
        (sampler("class_balanced", min_count=5), "min_count"),
        (sampler("random"), "Unknown sampler 'random'"),
        (sampler("tile_locality", num_workers=4), "train loader supplies"),
    ],
    ids=[
        "oversample-without-its-count",
        "a-count-its-sampler-does-not-take",
        "random",
        "the-loaders-own-worker-count",
    ],
)
def test_a_sampler_block_its_sampler_refuses_is_refused_at_preflight_by_name(
    block, named
):
    """A sampler block its sampler refuses is refused at preflight by name.

    Training's read check refuses, each by name, an oversampler with no ``min_count``, a
    sampler stated with a value it does not take, the sampler name ``random``, and the loader's
    own worker count stated in a block.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import weights_normalization
    from tests.tiny_trainer_fixtures import classifier_config

    config = classifier_config(
        1,
        {},
        project=REPO_ROOT,
        normalization=weights_normalization(),
        sampler=block,
    )
    with pytest.raises(ValueError, match=named):
        train_config(config).trainer_reads()


def test_a_stated_oversampler_count_is_the_count_the_train_loader_draws_at(
    tmp_path,
):
    """A run's train loader draws through the oversampler at the count its config states.

    The train loader built for a run (``generic_trainer.run_loaders``) over four samples of two
    classes, two each, through the oversampler at ``min_count`` 5 reads 4 + 2 * 3 * 2 indices
    an epoch.
    """
    ctx = _table_run(
        tmp_path,
        [0, 0, 1, 1],
        sampler("oversample", min_count=5),
        "classification",
    )

    assert len(list(iter(ctx.train_loader.sampler))) == 4 + 2 * 3 * 2


@pytest.mark.parametrize("task", ["classification", "ordinal"])
def test_each_imbalance_sampler_weighs_the_classes_its_loaders_distribution_counts(
    tmp_path, task
):
    """Each imbalance sampler weighs the classes its loader's distribution counts.

    Over a run's own loader of three frames of class 0 and one of class 1, the oversampler at
    ``min_count`` 5 reads 4 + 3 * 2 + 1 * 5 indices, and the class-balanced and weighted
    samplers weigh each frame by its own class's unnormalized class weight.
    """
    from tcip_mcp.pipelines.components.losses import compute_class_weights
    from tcip_mcp.pipelines.data.samplers import build_sampler

    ctx = _table_run(
        tmp_path, [0, 0, 0, 1], sampler("oversample", min_count=5), task
    )
    ds = ctx.train_loader.dataset
    assert ds.class_distribution == {0: 3, 1: 1}
    weight = compute_class_weights(ds.class_distribution, normalize=False)
    expected = [weight[label].item() for label in (0, 0, 0, 1)]
    balanced = build_sampler(_admitted("class_balanced"), ds)
    weighted = build_sampler(_admitted("weighted_random"), ds)
    assert weighted._sampler.weights.tolist() == pytest.approx(expected)
    assert balanced._weights.tolist() == pytest.approx(expected)
    assert len(list(iter(ctx.train_loader.sampler))) == 4 + 3 * 2 + 1 * 5


def test_a_detection_loaders_example_classes_are_its_distributions_keys(
    tmp_path,
):
    """A detection loader's example classes are its class distribution's keys.

    A detection loader's 1-indexed labels reach a sampler as the 0-indexed classes its class
    distribution counts.
    """
    from tests._producer_fixtures import dataset_over
    from tests._verified_checkpoint_fixtures import detection_images

    images = detection_images(tmp_path, {"subject": "bud"})["images_dir"]
    ds = dataset_over("detection", images, subject="bud")

    assert ds.class_distribution == {0: 2}
    assert ds.sample_classes() == [0, 0]


def test_an_imbalance_sampler_over_a_loader_stating_no_classes_refuses_naming_both(
    tmp_path,
):
    """An imbalance sampler over a loader stating no classes refuses naming sampler and task.

    A regression loader states no class distribution, so a run stating a class-balanced
    sampler refuses when its train loader is built, naming the sampler and the task.
    """
    with pytest.raises(
        ValueError, match=r"ClassBalancedSampler.*'regression'"
    ):
        _table_run(
            tmp_path, [0, 1, 0, 1], sampler("class_balanced"), "regression"
        )


def test_the_oversampler_over_an_instance_segmentation_loader_draws_at_its_distribution(
    tmp_path,
):
    """The oversampler over an instance-segmentation loader draws at its distribution.

    An instance-segmentation loader over two frames states a class distribution of two in class
    0, and the oversampler at ``min_count`` 5 over it reads 2 + 2 * 3 indices.
    """
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tests._producer_fixtures import dataset_over
    from tests._verified_checkpoint_fixtures import detection_images

    images = detection_images(tmp_path, {"subject": "bud"}, polygons=True)
    ds = dataset_over("instance_seg", images["images_dir"], subject="bud")

    assert ds.class_distribution == {0: 2}
    built = build_sampler(_admitted("oversample", min_count=5), ds)
    assert len(list(iter(built))) == 2 + 2 * 3


def test_the_oversampler_reshuffles_per_epoch_under_the_global_seed(tmp_path):
    from tcip_mcp.pipelines.data.samplers import OverSampler

    ds = _table_run(
        tmp_path,
        [0] * 8 + [1],
        sampler("oversample", min_count=4),
        "classification",
    ).train_loader.dataset
    s = OverSampler(ds, min_count=4)
    assert len(s) > len(ds)
    torch.manual_seed(0)
    epoch1, epoch2 = list(s), list(s)
    assert sorted(epoch1) == sorted(epoch2)
    assert epoch1 != epoch2
    torch.manual_seed(0)
    assert list(s) == epoch1
