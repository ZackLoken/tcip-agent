"""What a run's train dataset serves (its object density, its native frame) lands in the
persisted config, and a run's tiling block is the one its resolution wrote, so a
requested-but-unrealized tiling record never survives an untiled run."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest
from tests._chain_fixtures import BESPOKE_DETECTION, training_config  # noqa: E402
from tests._producer_fixtures import dataset_over, registry_over  # noqa: E402
from tests._training_values import evaluation_block  # noqa: E402

torch = pytest.importorskip("torch")


class _TiledStub:
    """A tiler serving 96 px tiles, one of them holding two objects."""

    tile_size = 96

    @property
    def regions(self):
        from tests._verified_checkpoint_fixtures import objects_over

        return [objects_over([[0, 0, 4, 4], [10, 10, 14, 14]], 96 * 96)]


class _OpaqueStub:
    """No regions and no source list: nothing can be counted or probed."""


def _geometry(data_cfg: dict, train_ds, task: str = "detection"):
    """``data_cfg`` validated as a ``task`` run's data block with what ``train_ds`` serves
    recorded on it (``generic_trainer.effective_data_geometry``)."""
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.pipelines.training.generic_trainer import effective_data_geometry

    return effective_data_geometry(task, DataSpec.model_validate(data_cfg), train_ds)


def test_stamp_tiled_run_records_its_tiles_density_and_leaves_the_lattice_it_was_given():
    """The lattice a tiled run serves is the one its resolution wrote into ``data.tiling``; the
    stamp adds the density over the tiles and nothing else."""
    tiling = {"enabled": True, "tile_size": 96, "overlap": 0.25}
    stamped = _geometry({"tiling": tiling}, _TiledStub())

    assert stamped.record()["tiling"] == tiling
    assert stamped.train_object_density == pytest.approx(2 / (96 * 96))
    assert stamped.train_native_size is None


def test_a_detector_dataset_records_its_density_whoever_built_it():
    """A bespoke builder's detector dataset exposing regions records their density as a
    platform-built one does; a detector dataset exposing none refuses naming the interface, and
    another head's records none."""
    bespoke = {"dataset_source": {"builder": "my_module:build_ds"}}

    assert _geometry(bespoke, _TiledStub()).train_object_density == pytest.approx(2 / (96 * 96))
    with pytest.raises(ValueError, match="regions"):
        _geometry(bespoke, _OpaqueStub())
    assert _geometry(bespoke, _OpaqueStub(), "classification").train_object_density is None


def _detection_dataset(tmp_path, sizes):
    """A real DetectionDataset over tiny generated images, one per (width, height) in sizes."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject

    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    registry_over(tmp_path, SubjectRegistry((Subject("bud"),)))
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h)).save(images_dir / f"img{i}.png")
        label_image(images_dir / f"img{i}.png",
                    [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], w, h)
    return dataset_over('detection', str(images_dir), subject="bud")


def test_stamp_untiled_uniform_frames_record_train_native_size(tmp_path):
    """An untiled run records the shared native frame as [width, height], and the object
    density counted over those frames."""
    stamped = _geometry({"tiling": {"enabled": False}},
                        _detection_dataset(tmp_path, [(64, 48), (64, 48)]))

    assert stamped.train_native_size == [64, 48]
    assert stamped.train_object_density == pytest.approx(1 / (64 * 48))


def test_stamp_untiled_mixed_frames_record_nothing(tmp_path):
    """Mixed source sizes have no single native frame; stamping any one of them would be a
    guess, so no train_native_size is written."""
    stamped = _geometry({"tiling": {"enabled": False}},
                        _detection_dataset(tmp_path, [(64, 48), (32, 32)]))

    assert stamped.train_native_size is None


# ── the run's resolved record, for a launched run and an HPO trial alike ──


def _base_config(tiling, project: Path, *, task: str = "detection"):
    """A ``task`` run's config over frames of its own under ``project`` (a detection run's
    labeled by documents, a classification run's by a table), stating ``tiling``."""
    from tests._chain_fixtures import BESPOKE_MODELS, CLASSIFIER_SOURCE
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA, detection_images, fixture_data_dir, table_images,
    )

    where = fixture_data_dir(project, task)
    if task == "detection":
        source = {"builder": BESPOKE_DETECTION, "source_files": [BESPOKE_MODELS],
                  "task": "detection"}
        frames = {**detection_images(where, SCOPED_DATA["scope"], n=4), **SCOPED_DATA}
    else:
        source, frames = dict(CLASSIFIER_SOURCE), table_images(where, n=4)
    return training_config(
        source, {**frames, "tiling": tiling, "split": {"seed": 0, "val_ratio": 0.25}},
        evaluation=evaluation_block(selection_metric="loss"))


def _resolved_data(run_dir) -> dict:
    """The data section ``run_dir``'s launch record says its run resolved."""
    from tcip_mcp.experiments import observe

    return observe(run_dir).record["resolved"]["data"]


def test_an_untiled_runs_resolved_record_drops_the_requested_geometry(tmp_path):
    """The resolved data section is recorded whole, so an untiled run's record keeps no stale
    requested tile_size from the config it was launched with: a classification run, which no
    tiler serves, launched stating a tile geometry."""
    from tests._verified_checkpoint_fixtures import opened_run

    config = _base_config({"enabled": True, "tile_size": 640}, tmp_path, task="classification")
    run_dir = opened_run(tmp_path, config)

    data = _resolved_data(run_dir)
    assert data["tiling"] == {"enabled": False}
    assert data["images_dir"] == config["data"]["images_dir"]


def _trial(tmp_path, base_config):
    """One HPO trial of a sweep over ``base_config``, opened by the sweep's own trial producer;
    its run directory."""
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    return open_trial(opened_sweep(tmp_path, base_config), "0", {"optimizer.head_lr": 3e-4})


def test_an_hpo_trials_resolved_record_replaces_unrealized_tiling(tmp_path):
    """A trial that trained untiled must not leave the base config's requested tile_size in its
    resolved record, the record a later reader takes for the trial's geometry."""
    trial_dir = _trial(tmp_path, _base_config({"enabled": True, "tile_size": 999}, tmp_path,
                                              task="classification"))

    assert _resolved_data(trial_dir)["tiling"] == {"enabled": False}


def test_an_hpo_trials_resolved_record_carries_the_effective_tile_geometry(tmp_path):
    """A tiled trial records the tile edge its config stated and the overlap its resolution
    derived where the config stated none."""
    trial_dir = _trial(tmp_path, _base_config(
        {"enabled": True, "tile_size": 32, "sliver_frac": 0.5}, tmp_path))

    tiling = _resolved_data(trial_dir)["tiling"]
    assert tiling["enabled"] is True and tiling["tile_size"] == 32
    assert isinstance(tiling["overlap"], float)
