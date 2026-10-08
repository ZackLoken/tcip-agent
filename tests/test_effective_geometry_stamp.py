"""The effective input-geometry stamp: what a run actually trained on (tile geometry or
native frame) lands in the persisted config, and a requested-but-unrealized tiling record
never survives an untiled run."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest
from tests._chain_fixtures import BESPOKE_DETECTION, training_config  # noqa: E402
from tests._producer_fixtures import dataset_over, registry_over  # noqa: E402
from tests._training_values import evaluation_block  # noqa: E402

torch = pytest.importorskip("torch")


class _TiledStub:
    tile_size = 224
    overlap = 0.2


class _OpaqueStub:
    """No tile geometry and no source list: nothing can be probed, nothing is stamped."""


def test_stamp_tiled_run_fills_effective_geometry_into_tiling():
    from tcip_mcp.pipelines.training.generic_trainer import stamp_effective_data_geometry

    data_cfg = {"tiling": {"enabled": True}}  # no tile_size: the dataset's default is the truth
    stamped = stamp_effective_data_geometry(data_cfg, _TiledStub())

    assert data_cfg["tiling"] == {"enabled": True, "tile_size": 224, "overlap": pytest.approx(0.2)}
    assert stamped["tiling_replaced"] is False
    assert stamped["train_native_size"] is None
    assert "train_native_size" not in data_cfg


def test_a_bespoke_run_records_no_geometry_at_all():
    """The stamp reads a dataset's own tile attributes and probes its sources, so a dataset the
    platform did not build is one it cannot measure, not one serving untiled frames. Recording
    ``{"enabled": False}`` there would put a geometry nobody measured onto the checkpoint every
    predictor and tiled-eval default reads back, so a bespoke run records nothing."""
    from tcip_mcp.pipelines.training.generic_trainer import stamp_effective_data_geometry

    data_cfg = {"dataset_source": {"builder": "my_module:build_ds"},
                "tiling": {"enabled": True, "tile_size": 640}}

    assert stamp_effective_data_geometry(data_cfg, _OpaqueStub()) is None
    assert data_cfg["tiling"] == {"enabled": True, "tile_size": 640}  # untouched, not replaced
    assert "train_native_size" not in data_cfg

    # A platform-built dataset in the same shape still stamps: absence is the bespoke fact alone.
    platform_cfg = {"tiling": {"enabled": True, "tile_size": 640}}
    assert stamp_effective_data_geometry(platform_cfg, _OpaqueStub()) is not None
    assert platform_cfg["tiling"] == {"enabled": False}


def test_stamp_untiled_run_replaces_tiling_record_wholesale():
    """An untiled run must never carry a requested tile_size into its persisted config: a
    reader would take it for the frame the model trained on."""
    from tcip_mcp.pipelines.training.generic_trainer import stamp_effective_data_geometry

    data_cfg = {"tiling": {"enabled": True, "tile_size": 640, "overlap": 0.3}}
    stamped = stamp_effective_data_geometry(data_cfg, _OpaqueStub())

    assert data_cfg["tiling"] == {"enabled": False}
    assert stamped["tiling_replaced"] is True
    assert stamped["train_native_size"] is None


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
    """With no tiling dict at all, the untiled stamp still runs: the tiling record states the
    untiled truth and the shared native frame is recorded as [width, height]."""
    from tcip_mcp.pipelines.training.generic_trainer import stamp_effective_data_geometry

    ds = _detection_dataset(tmp_path, [(64, 48), (64, 48)])
    data_cfg: dict = {}
    stamped = stamp_effective_data_geometry(data_cfg, ds)

    assert data_cfg["tiling"] == {"enabled": False}
    assert data_cfg["train_native_size"] == [64, 48]
    assert stamped["train_native_size"] == [64, 48]


def test_stamp_untiled_mixed_frames_record_nothing(tmp_path):
    """Mixed source sizes have no single native frame; stamping any one of them would be a
    guess, so no train_native_size is written."""
    from tcip_mcp.pipelines.training.generic_trainer import stamp_effective_data_geometry

    ds = _detection_dataset(tmp_path, [(64, 48), (32, 32)])
    data_cfg: dict = {}
    stamped = stamp_effective_data_geometry(data_cfg, ds)

    assert data_cfg["tiling"] == {"enabled": False}
    assert "train_native_size" not in data_cfg
    assert stamped["train_native_size"] is None


# ── the run's resolved record, for a launched run and an HPO trial alike ──


class _ServedDataset:
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return i


class _TiledServedDataset(_TiledStub, _ServedDataset):
    pass


def _serve(monkeypatch, train_ds):
    """The run's datasets resolved to ``train_ds`` with no validation side, so the resolved
    record carries only what the stamp wrote."""
    from tcip_mcp.pipelines.data import split_construction as sc

    monkeypatch.setattr(
        sc, "auto_train_val",
        lambda project, task, data_cfg, transforms, **_: (train_ds, None, {"samples": []}))


def _base_config(tiling, project: Path):
    return training_config(
        {"builder": BESPOKE_DETECTION, "task": "detection"},
        {"images_dir": str(project / "imgs"), "tiling": tiling,
         "split": {"seed": 0, "val_ratio": 0.15}},
        evaluation=evaluation_block(selection_metric="loss"))


def _resolved_data(run_dir) -> dict:
    """The data section ``run_dir``'s launch record says its run resolved."""
    from tcip_mcp.experiments import observe

    return observe(run_dir).record["resolved"]["data"]


def test_an_untiled_runs_resolved_record_drops_the_requested_geometry(monkeypatch, tmp_path):
    """The resolved data section is recorded whole, so an untiled run's record keeps no stale
    requested tile_size from the config it was launched with."""
    from tests._verified_checkpoint_fixtures import opened_run

    _serve(monkeypatch, _ServedDataset())
    run_dir = opened_run(tmp_path, _base_config({"enabled": True, "tile_size": 640}, tmp_path))

    data = _resolved_data(run_dir)
    assert data["tiling"] == {"enabled": False}
    assert data["images_dir"] == str(tmp_path.resolve() / "imgs")


def _trial(tmp_path, base_config):
    """One HPO trial of a sweep over ``base_config``, opened by the sweep's own trial producer;
    its run directory."""
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    Path(base_config["data"]["images_dir"]).mkdir(parents=True, exist_ok=True)
    return open_trial(opened_sweep(tmp_path, base_config), "0", {"optimizer.head_lr": 3e-4})


def test_an_hpo_trials_resolved_record_replaces_unrealized_tiling(monkeypatch, tmp_path):
    """A trial that trained untiled must not leave the base config's requested tile_size in its
    resolved record, the record a later reader takes for the trial's geometry."""
    _serve(monkeypatch, _ServedDataset())
    trial_dir = _trial(tmp_path, _base_config({"enabled": True, "tile_size": 999}, tmp_path))

    assert _resolved_data(trial_dir)["tiling"] == {"enabled": False}


def test_an_hpo_trials_resolved_record_carries_the_effective_tile_geometry(
        monkeypatch, tmp_path):
    _serve(monkeypatch, _TiledServedDataset())
    trial_dir = _trial(tmp_path, _base_config({"enabled": True}, tmp_path))

    tiling = _resolved_data(trial_dir)["tiling"]
    assert tiling == {"enabled": True, "tile_size": 224, "overlap": pytest.approx(0.2)}
