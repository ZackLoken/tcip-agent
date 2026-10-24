"""A run's served geometry and tiling block as its resolution records them.

What a run's train dataset serves (its object density, its native frame) is the resolution's own
geometry record, not an input key, a run's tiling block is the one its resolution wrote, and a
tiling stated for classification or semantic segmentation is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_CLASSIFIER,
    BESPOKE_DETECTION,
    BESPOKE_SEMANTIC_SEG,
    FRAME_STATISTICS,
    training_config,
)
from tests._producer_fixtures import dataset_over, registry_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    drawn_split,
    evaluation_block,
    tiling_block,
)

if TYPE_CHECKING:
    from pydantic import JsonValue
    from torch.utils.data import Dataset

    from tcip_mcp.pipelines.derivations import Region
    from tcip_mcp.pipelines.schemas import (
        StatedDetectorGeometry,
        TrainGeometry,
    )

torch = pytest.importorskip("torch")


class _TiledStub:
    """A stub tiler of tile size 96 whose one region, of area 96 by 96 px, holds two objects."""

    tile_size = 96

    @property
    def regions(self) -> list[Region]:
        from tests._verified_checkpoint_fixtures import objects_over

        return [objects_over([[0, 0, 4, 4], [10, 10, 14, 14]], 96 * 96)]


class _OpaqueStub:
    """A dataset stub with no regions and no source list."""


def _stated() -> StatedDetectorGeometry:
    """Return a detector geometry a model source states with its reason."""
    from tcip_mcp.pipelines.schemas import StatedDetectorGeometry
    from tests._chain_fixtures import stated_detector_geometry

    return StatedDetectorGeometry.model_validate(stated_detector_geometry())


def _geometry(
    data_cfg: dict,
    train_ds: Dataset | _TiledStub | _OpaqueStub,
    task: str = "detection",
    stated: StatedDetectorGeometry | None = None,
) -> TrainGeometry:
    """Return the geometry a ``task`` run reads off what ``train_ds`` serves.

    The run's data block is ``data_cfg`` and its model source states the detector geometry
    ``stated``; the geometry is read through ``generic_trainer.effective_data_geometry``.
    """
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.pipelines.training.generic_trainer import (
        effective_data_geometry,
    )

    return effective_data_geometry(
        task,
        DataSpec.model_validate({"split": drawn_split(), **data_cfg}),
        train_ds,
        stated,
    )


def test_stamp_tiled_run_records_its_tiles_density_and_no_native_frame() -> (
    None
):
    """A tiled run records its tiles' density, no native frame and a tile-edge detector geometry.

    The detector geometry is the one derived at its tile edge.
    """
    from tcip_mcp.pipelines.derivations import DETECTOR_GEOMETRY_DERIVATION

    tiling = tiling_block(tile_size=96, overlap=0.25)
    stamped = _geometry({"tiling": tiling}, _TiledStub())

    assert stamped.object_density == pytest.approx(2 / (96 * 96))
    assert stamped.native_size is None
    assert stamped.detector is not None
    assert (stamped.detector.min_size, stamped.detector.max_size) == (96, 96)
    assert stamped.detector.source == DETECTOR_GEOMETRY_DERIVATION


def test_a_detector_dataset_records_its_density_whoever_built_it() -> None:
    """A detector dataset records its density whoever built it.

    A bespoke builder's detector dataset exposing regions records their density and the
    geometry its model source states; a detector dataset exposing none refuses naming
    ``regions``, and a classification run over it records no density.
    """
    bespoke = {"dataset_source": {"builder": "my_module:build_ds"}}

    stamped = _geometry(bespoke, _TiledStub(), stated=_stated())
    assert stamped.object_density == pytest.approx(2 / (96 * 96))
    assert stamped.detector == _stated()
    with pytest.raises(ValueError, match="regions"):
        _geometry(bespoke, _OpaqueStub())
    assert (
        _geometry(bespoke, _OpaqueStub(), "classification").object_density
        is None
    )


def _detection_dataset(
    tmp_path: Path, sizes: list[tuple[int, int]]
) -> Dataset:
    """Return a real DetectionDataset over tiny images, one per (width, height) in ``sizes``."""
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    registry_over(tmp_path, SubjectRegistry((Subject("bud", ""),)))
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h)).save(images_dir / f"img{i}.png")
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
            w,
            h,
        )
    return dataset_over("detection", str(images_dir), subject="bud")


def test_stamp_untiled_uniform_frames_record_their_native_size(
    tmp_path: Path,
) -> None:
    """An untiled run of uniform frames records their shared native size as (width, height).

    It also records the object density counted over those frames.
    """
    stamped = _geometry(
        {"tiling": None}, _detection_dataset(tmp_path, [(64, 48), (64, 48)])
    )

    assert stamped.native_size == (64, 48)
    assert stamped.object_density == pytest.approx(1 / (64 * 48))


def test_stamp_untiled_mixed_frames_record_no_native_frame(
    tmp_path: Path,
) -> None:
    """An untiled run of mixed frame sizes records no native frame.

    Over frames of 64 by 48 and 32 by 32, no native size is recorded, and the detector takes the
    geometry its model source states.
    """
    stamped = _geometry(
        {"tiling": None},
        _detection_dataset(tmp_path, [(64, 48), (32, 32)]),
        stated=_stated(),
    )

    assert stamped.native_size is None
    assert stamped.detector == _stated()


def test_a_detector_resolving_no_frame_and_stating_no_geometry_refuses_naming_both(
    tmp_path: Path,
) -> None:
    """A detector resolving no frame and stating no geometry refuses naming both.

    With no one frame to derive at and no stated geometry, a detector's resolution refuses
    naming the tiling and the model source's statement.
    """
    with pytest.raises(ValueError) as refused:
        _geometry(
            {"tiling": None},
            _detection_dataset(tmp_path, [(64, 48), (32, 32)]),
        )

    assert "data.tiling" in str(refused.value)
    assert "model_source.detector_geometry" in str(refused.value)


def test_a_stated_geometry_beside_a_resolved_frame_refuses_naming_the_derivation(
    tmp_path: Path,
) -> None:
    """A stated detector geometry beside a resolved frame refuses naming the derivation.

    Beside two 64 by 48 frames, a stated geometry refuses naming the derivation and the field to
    drop.
    """
    with pytest.raises(ValueError) as refused:
        _geometry(
            {"tiling": None},
            _detection_dataset(tmp_path, [(64, 48), (64, 48)]),
            stated=_stated(),
        )

    assert "derive_detector_geometry" in str(refused.value)
    assert "model_source.detector_geometry" in str(refused.value)


def test_uniform_frames_derive_the_detector_geometry_at_their_own_frame(
    tmp_path: Path,
) -> None:
    """Untiled uniform frames derive the detector geometry at the frame they share.

    The geometry is the one ``derivations.derive_detector_geometry`` gives over the same
    regions.
    """
    from tcip_mcp.pipelines.derivations import derive_detector_geometry

    dataset = _detection_dataset(tmp_path, [(64, 48), (64, 48)])
    stamped = _geometry({"tiling": None}, dataset)

    assert stamped.detector == derive_detector_geometry(
        (64, 48), dataset.regions
    )
    assert (stamped.detector.min_size, stamped.detector.max_size) == (48, 64)


@pytest.mark.parametrize(
    "stated",
    [{"train_native_size": [64, 48]}, {"train_object_density": 0.001}],
)
def test_a_config_stating_a_resolved_geometry_key_refuses_at_the_door_naming_it(
    tmp_path: Path, stated: dict[str, list[int]] | dict[str, float]
) -> None:
    """A config stating a resolved geometry key refuses at the launch door naming it.

    A config stating the native frame or the object density fails preflight naming the key, and
    the same config without it opens a run whose resolution records the 64 by 48 frame its
    untiled frames share, the geometry the observed run reads back.
    """
    from tcip_mcp.experiments import observe_run
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import opened_run

    config = _base_config(None, tmp_path)
    (key,) = stated
    report = preflight_config(
        tmp_path, {**config, "data": {**config["data"], **stated}}
    )
    assert not report["valid"]
    assert any(key in issue for issue in report["issues"]), report["issues"]

    run_dir = opened_run(tmp_path, config)
    geometry = observe_run(run_dir).record.resolution.geometry
    assert geometry.native_size == (64, 48)
    assert observed_run(observe_run(run_dir)).geometry == geometry


# ── the run's resolved record, for a launched run and an HPO trial alike ──


def _base_config(
    tiling: dict | None,
    project: Path,
    *,
    task: str = "detection",
    refused: bool = False,
) -> dict:
    """Return a ``task`` run's config over frames of its own under ``project``, stating ``tiling``.

    A detection run's frames are labeled by documents, a classification run's by a table;
    ``tiling`` is ``None`` for a run that tiles nothing. The config normalizes by its own
    frames' statistics, or by pretrained weights when ``refused``.
    """
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        CLASSIFIER_SOURCE,
        weights_statistics,
    )
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA,
        detection_images,
        fixture_data_dir,
        table_images,
    )

    where = fixture_data_dir(project, task)
    if task == "detection":
        source = {
            "builder": BESPOKE_DETECTION,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        }
        frames = {
            **detection_images(where, SCOPED_DATA["scope"], n=4),
            **SCOPED_DATA,
        }
    else:
        source, frames = dict(CLASSIFIER_SOURCE), table_images(where, n=4)
    return training_config(
        project,
        weights_statistics(source) if refused else source,
        {
            **frames,
            "tiling": tiling,
            "split": drawn_split(val_ratio=0.25, grouping="tile_prefix"),
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )


def _resolved_data(run_dir: Path) -> dict[str, JsonValue]:
    """Return the data section ``run_dir``'s launch record says its run resolved."""
    from tcip_mcp.experiments import RUN_FILE, read_run_record

    record = read_run_record(run_dir / RUN_FILE)
    assert isinstance(record, dict)
    resolved = record["resolved"]
    assert isinstance(resolved, dict)
    data = resolved["data"]
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    "task, builder",
    [
        ("classification", BESPOKE_CLASSIFIER),
        ("semantic_seg", BESPOKE_SEMANTIC_SEG),
    ],
)
def test_a_tiling_stated_on_a_task_that_trains_whole_frames_refuses_at_the_door(
    task: str, builder: str
) -> None:
    """A tiling stated for classification or semantic segmentation is refused at validation.

    A classification or semantic segmentation run stating a ``data.tiling`` is refused when its
    config is validated, naming the task and ``data.tiling``.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import BESPOKE_MODELS, weights_statistics

    config = training_config(
        REPO_ROOT,
        weights_statistics(
            {
                "builder": builder,
                "source_files": [BESPOKE_MODELS],
                "task": task,
            }
        ),
        {
            "num_channels": 3,
            "tiling": tiling_block(tile_size=64),
            "split": drawn_split(),
        },
    )
    with pytest.raises(
        ValueError,
        match=rf"data\.tiling is stated for a run of task '{task}'",
    ):
        train_config(config)


def test_the_tiling_question_and_the_detector_question_each_read_their_own_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tiling question and the detector question each read their own task set.

    A task admitted to the tiling set alone tiles where a config's tiling is admitted, and is a
    detector nowhere a head's kind is asked: its validation states no confidence and its batches
    collate as tensors.
    """
    from tcip_mcp.pipelines import model_contract
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.collation import task_collate
    from tests import REPO_ROOT
    from tests._chain_fixtures import BESPOKE_MODELS, weights_statistics

    monkeypatch.setattr(
        model_contract,
        "TILING_TASKS",
        model_contract.TILING_TASKS | {"semantic_seg"},
    )
    tiling = tiling_block(tile_size=64)
    spec = train_config(
        training_config(
            REPO_ROOT,
            weights_statistics(
                {
                    "builder": BESPOKE_SEMANTIC_SEG,
                    "source_files": [BESPOKE_MODELS],
                    "task": "semantic_seg",
                }
            ),
            {"num_channels": 3, "tiling": tiling, "split": drawn_split()},
            evaluation=evaluation_block(conf_threshold=None),
        )
    )

    assert spec.data.tiling is not None and spec.data.tiling.tile_size == 64
    assert spec.trainer_reads().conf_threshold is None
    assert task_collate("semantic_seg") is task_collate("classification")


def test_an_untiled_runs_resolved_record_carries_no_tiling(
    tmp_path: Path,
) -> None:
    """An untiled run's resolved record carries ``tiling: None``; a tiled one carries its block.

    A classification run stating no tiling opens through the launcher's own producer, and the
    record it resolved, read raw and validated again through ``observed_run``, carries
    ``tiling: None``.
    """
    from tcip_mcp.experiments import observe_run
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    config = _base_config(None, tmp_path, task="classification")
    run_dir = opened_run(tmp_path, config)

    data = _resolved_data(run_dir)
    assert "tiling" in data and data["tiling"] is None
    assert data["images_dir"] == config["data"]["images_dir"]
    assert observed_run(observe_run(run_dir)).spec.data.tiling is None

    tiled = _resolved_data(
        opened_run(
            tmp_path,
            _base_config(
                tiling_block(tile_size=32, sliver_frac=0.5), tmp_path
            ),
        )
    )["tiling"]
    assert isinstance(tiled, dict)
    assert tiled["tile_size"] == 32 and tiled["rationale"]


def _trial(tmp_path: Path, base_config: dict) -> Path:
    """Open one HPO trial of a sweep over ``base_config`` and return its run directory.

    The trial is opened by the sweep's own trial producer.
    """
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    return open_trial(
        opened_sweep(tmp_path, base_config), "0", {"optimizer.head_lr": 3e-4}
    )


def test_a_sweep_over_a_tiling_its_task_never_reads_opens_nothing(
    tmp_path: Path,
) -> None:
    """A sweep stating a tiling its classification task does not read opens nothing.

    A sweep whose classification base config states a tiling is refused by name and opens no
    sweep directory; the same base config stating none opens a trial whose resolved record
    carries no tiling.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import open_sweep
    from tests._verified_checkpoint_fixtures import SWEEP_ARGUMENTS

    arguments = dict(SWEEP_ARGUMENTS)
    space = arguments.pop("param_space")
    refused = open_sweep(
        tmp_path,
        _base_config(
            tiling_block(tile_size=999),
            tmp_path,
            task="classification",
            refused=True,
        ),
        space,
        actor=None,
        **arguments,
    )

    assert isinstance(refused, dict)
    assert "data.tiling is stated for a run of task 'classification'" in str(
        refused
    ), refused
    assert experiments.sweep_dirs(tmp_path) == []
    trial_dir = _trial(
        tmp_path, _base_config(None, tmp_path, task="classification")
    )
    assert _resolved_data(trial_dir)["tiling"] is None


def test_an_hpo_trials_resolved_record_carries_the_effective_tile_geometry(
    tmp_path: Path,
) -> None:
    """An HPO trial's resolved record carries the effective tile geometry.

    A tiled trial records the 48 px edge its config stated and a float overlap where the config
    stated none.
    """
    trial_dir = _trial(
        tmp_path,
        _base_config(tiling_block(tile_size=48, sliver_frac=0.5), tmp_path),
    )

    tiling = _resolved_data(trial_dir)["tiling"]
    assert isinstance(tiling, dict)
    assert tiling["tile_size"] == 48
    assert isinstance(tiling["overlap"], float)
