"""build_sampler call shape and the tile-locality sampler: banded read order, runtime band
derivation, per-worker lanes under multi-worker loading, and the refusals that keep it
honest."""

from __future__ import annotations

import itertools
import os

import pytest

torch = pytest.importorskip("torch")


class _StubTiledDataset:
    """Duck-typed stand-in exposing the tiled-dataset ordering contract (``tile_entries`` and
    ``source_frames``), loadable through a real DataLoader: samples carry the serving process
    pid so a multi-worker run reveals which worker fetched which index."""

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
    return {"width": width, "height": 4096, "channels": channels,
            "dtype_itemsize": itemsize, "windowed": windowed}


def _lattice_entries(stems, tile_ys=(0, 512, 1024, 1536), tile_xs=(0, 512, 1024), tile=512):
    return [(stem, (tx, ty, tx + tile, ty + tile))
            for stem in stems for ty in tile_ys for tx in tile_xs]


def _two_source_dataset():
    return _StubTiledDataset(
        _lattice_entries(["a", "b"]),
        {"a": _frame(), "b": _frame()},
    )


def _band_key(dataset, sampler, idx):
    """(stem, band ordinal) for a tile index, from the lattice and the derived band height."""
    stem, box = dataset.tile_entries[idx]
    ys = sorted({b[1] for s, b in dataset.tile_entries if s == stem})
    return stem, ys.index(box[1]) // sampler.band_tile_rows


def _assert_contiguous_bands(dataset, sampler, indices):
    keys = [_band_key(dataset, sampler, i) for i in indices]
    groups = [key for key, _run in itertools.groupby(keys)]
    assert len(groups) == len(set(groups)), "a (source, band) group was split apart"


def _assert_banded(dataset, sampler, order):
    assert sorted(order) == list(range(len(dataset.tile_entries)))
    _assert_contiguous_bands(dataset, sampler, order)


def _assert_bands_owned_by_one_worker(dataset, sampler, per_worker_streams):
    """Every (source, band) group is read by exactly one worker, not merely contiguous within
    whichever worker happens to receive a fragment of it: a band split across workers (each
    decoding its own share from scratch) would still pass a purely-per-worker contiguity check,
    since each worker's own fragment stays internally ordered even though the band itself was
    split, which is exactly the amplification the lanes exist to prevent."""
    owner: dict[tuple, int] = {}
    for worker, stream in per_worker_streams.items():
        _assert_contiguous_bands(dataset, sampler, stream)
        for key in {_band_key(dataset, sampler, i) for i in stream}:
            assert key not in owner, f"band {key} split across workers {owner[key]} and {worker}"
            owner[key] = worker


# ── single-lane locality ordering ─────────────────────────────────────


def test_tile_locality_orders_are_banded_and_differ_across_epochs(monkeypatch):
    """Within an epoch every (source, band) group stays contiguous; across consecutive
    epochs the order changes (a sampler seeded once at construction would freeze it)."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)  # (31e6 // 30000 rows) // 512 pitch = 2 tile rows
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


def test_tile_locality_band_height_derived_from_cache_and_row_cost(monkeypatch):
    """The band height is a runtime derivation, half the per-reader cache share over the
    costliest windowed source's row bytes over the lattice row pitch, never a pinned
    constant."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    ds = _two_source_dataset()

    _patch_gdal_cache(monkeypatch, 8_000_000)  # too small for even one full tile row: floor of 1
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
    """Each DataLoader worker process holds its own GDAL cache, so the band a worker can
    keep resident shrinks with the worker count."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 124_000_000)
    ds = _two_source_dataset()
    assert TileLocalitySampler(ds, num_workers=0).band_tile_rows == 4
    assert TileLocalitySampler(ds, num_workers=2, batch_size=3).band_tile_rows == 2


def test_tile_locality_single_tile_row_sources_band_to_one(monkeypatch):
    """One tile row per source leaves no row pitch to derive from; each source is one band."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    ds = _StubTiledDataset(_lattice_entries(["a"], tile_ys=(0,)), {"a": _frame()})
    sampler = TileLocalitySampler(ds, num_workers=0)
    assert sampler.band_tile_rows == 1
    assert sorted(iter(sampler)) == list(range(3))


# ── multi-worker lanes ────────────────────────────────────────────────


def test_tile_locality_multi_worker_order_is_laned_per_dispatched_worker(monkeypatch):
    """The DataLoader dispatches batch i to worker i mod num_workers, so chunking the
    emitted order into batches and dealing chunks round-robin must reconstruct, for every
    worker, a read stream whose (source, band) groups stay contiguous; consecutive epochs
    still differ."""
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 124_000_000)  # per-worker share at 2 workers: band of 2 rows
    ds = _two_source_dataset()
    num_workers, batch_size = 2, 3
    sampler = TileLocalitySampler(ds, num_workers=num_workers, batch_size=batch_size)
    assert sampler.band_tile_rows == 2

    torch.manual_seed(1)
    epoch1 = list(iter(sampler))
    epoch2 = list(iter(sampler))
    for order in (epoch1, epoch2):
        assert sorted(order) == list(range(len(ds)))
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        per_worker = {
            worker: [idx for ci, chunk in enumerate(chunks)
                     if ci % num_workers == worker for idx in chunk]
            for worker in range(num_workers)
        }
        _assert_bands_owned_by_one_worker(ds, sampler, per_worker)
    assert epoch1 != epoch2


def test_tile_locality_lanes_land_on_their_worker_through_a_real_loader(monkeypatch):
    """End to end through torch's multiprocessing DataLoader: each real worker's received
    index stream stays banded, which holds only if the round-robin batch dispatch delivers
    each lane's batches to one worker. Sample values carry the worker pid, so the mapping is
    observed, not simulated. The lattice is sized so lanes are equal (4 equal bands over 2
    workers, band a whole number of batches) and no tail misalignment can excuse a split."""
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
        assert len(pid_set) == 1, "one batch was served by more than one worker"
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
    ds = _StubTiledDataset(_lattice_entries(["a"]), {"a": _frame(windowed=False)})
    with pytest.raises(ValueError, match="windowed"):
        TileLocalitySampler(ds, num_workers=0)


def test_tile_locality_refuses_unknown_worker_regime(monkeypatch):
    """Built without the loader's num_workers (e.g. a call site that never threads it), the
    sampler refuses and names what to pass rather than assuming a single-worker loader."""
    from tcip_mcp.pipelines.data.samplers import build_sampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    with pytest.raises(ValueError, match="num_workers"):
        build_sampler({"name": "tile_locality"}, _two_source_dataset())


def test_tile_locality_refuses_multi_worker_without_batch_size(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import build_sampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    with pytest.raises(ValueError, match="batch_size"):
        build_sampler({"name": "tile_locality"}, _two_source_dataset(), num_workers=2)


def test_tile_locality_refuses_windowed_source_without_dtype_size(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    ds = _StubTiledDataset(_lattice_entries(["a"]), {"a": _frame(itemsize=None)})
    with pytest.raises(ValueError, match="dtype_itemsize"):
        TileLocalitySampler(ds, num_workers=0)


# ── build_sampler call shape: the loader context must not disturb the existing names ──


class _TinyClassDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return "x", {"label": i % 2}

    @property
    def class_distribution(self):
        return {0: 2, 1: 2}


def test_build_sampler_builds_tile_locality_through_the_factory(monkeypatch):
    from tcip_mcp.pipelines.data.samplers import TileLocalitySampler, build_sampler

    _patch_gdal_cache(monkeypatch, 62_000_000)
    sampler = build_sampler({"name": "tile_locality"}, _two_source_dataset(),
                            num_workers=2, batch_size=4)
    assert isinstance(sampler, TileLocalitySampler)


def test_build_sampler_imbalance_samplers_build_with_or_without_the_loader_context():
    """The imbalance samplers neither require nor receive the loader context: each block builds
    with the context absent and with it present but ignored."""
    from tcip_mcp.pipelines.data import samplers

    ds = _TinyClassDataset()
    for block in ({"name": "class_balanced"}, {"name": "oversample", "min_count": 3},
                  {"name": "weighted_random"}):
        for kwargs in ({}, {"num_workers": 4, "batch_size": 8}):
            sampler = samplers.build_sampler(block, ds, **kwargs)
            assert len(list(iter(sampler))) >= len(ds)


@pytest.mark.parametrize("block, named", [
    ({"name": "oversample"}, "min_count"),
    ({"name": "class_balanced", "min_count": 5}, "min_count"),
    ({"name": "random"}, "Unknown sampler 'random'"),
    ({"name": "tile_locality", "num_workers": 4}, "train loader supplies"),
], ids=["oversample-without-its-count", "a-count-its-sampler-does-not-take", "random",
        "the-loaders-own-worker-count"])
def test_a_sampler_block_its_sampler_refuses_is_refused_at_preflight_by_name(block, named):
    """guard. The oversampler's count has no default: a config naming the oversampler with no
    ``min_count``, or a sampler stated with a value it does not take, refuses naming it where
    training's reads are checked; ``random`` is no sampler name, the absent block being the
    shuffled loader; and the loader's own worker count and batch size are the loader's to
    supply, refused in a block by name."""
    from tcip_mcp.pipelines.schemas import train_config
    from tests.tiny_trainer_fixtures import classifier_config

    with pytest.raises(ValueError, match=named):
        train_config(classifier_config(1, sampler=block)).trainer_reads()


def test_a_stated_oversampler_count_is_the_count_the_train_loader_draws_at(tmp_path):
    """guard. The train loader a run builds (``generic_trainer.run_loaders``) draws through the
    oversampler at the count its config states: four samples of two classes, two each,
    duplicated up to five per class, read 4 + 2 * 3 * 2 indices an epoch."""
    from tcip_mcp.pipelines.training.generic_trainer import run_loaders
    from tests.tiny_trainer_fixtures import classifier_config, trainer_run

    run = trainer_run(classifier_config(1, sampler={"name": "oversample", "min_count": 5}),
                      tmp_path / "out", project=tmp_path, has_val_loader=False)
    train_loader, _ = run_loaders(run, _TinyClassDataset(), None)

    assert len(list(iter(train_loader.sampler))) == 4 + 2 * 3 * 2
