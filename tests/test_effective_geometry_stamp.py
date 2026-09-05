"""The effective input-geometry stamp: what a run actually trained on (tile geometry or
native frame) lands in the persisted config, and a requested-but-unrealized tiling record
never survives an untiled run."""

from __future__ import annotations

import pytest

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
    from pathlib import Path

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    write_registry(Path(tmp_path) / "classes.json", ClassRegistry((Subject("bud"),)))
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], w, h)
    return DetectionDataset(str(images_dir), str(labels_dir), subject="bud")


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


# ── the durable experiment record mirror ──────────────────────────────


def _write_experiment_config(tmp_path, monkeypatch, data):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_store as ts
    from tcip_mcp.experiments import config_key

    key = config_key("exp1")
    ts.replace(key, {"model_source": {"builder": "x:y"}, "data": data})
    return key


def test_patch_experiment_tiling_replace_drops_stale_geometry(tmp_path, monkeypatch):
    """update()-merging an untiled record would leave the stale requested tile_size in the
    durable experiment config; replace mode must not."""
    import tcip_store as ts

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_tiling

    key = _write_experiment_config(
        tmp_path, monkeypatch,
        {"images_dir": "img", "tiling": {"enabled": True, "tile_size": 640}})
    _patch_experiment_config_tiling("exp1", {"enabled": False}, replace=True,
                                    train_native_size=[64, 48])

    cfg = ts.read(key)
    assert cfg["data"]["tiling"] == {"enabled": False}
    assert cfg["data"]["train_native_size"] == [64, 48]
    assert cfg["data"]["images_dir"] == "img"  # a patch of the data section, not a rewrite
    assert cfg["model_source"] == {"builder": "x:y"}


def test_patch_experiment_tiling_default_still_merges(tmp_path, monkeypatch):
    import tcip_store as ts

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_tiling

    key = _write_experiment_config(
        tmp_path, monkeypatch, {"tiling": {"enabled": True, "sliver_frac": 0.4}})
    _patch_experiment_config_tiling("exp1", {"tile_size": 224, "overlap": 0.2})

    tiling = ts.read(key)["data"]["tiling"]
    assert tiling == {"enabled": True, "sliver_frac": 0.4, "tile_size": 224,
                      "overlap": pytest.approx(0.2)}


# ── the HPO trial records the same truth in its resolved-config snapshot ──


def _patch_trial_machinery(monkeypatch, train_ds):
    import torch.utils.data as tud

    from tcip_mcp.pipelines.data import samplers
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import generic_trainer as gt

    def fake_train(run, train_loader, val_loader, task="detection",
                   epoch_callback=None, resume_from=""):
        run.best_metric = 1.0
        run.status = "completed"
        return run

    monkeypatch.setattr(
        sc, "auto_train_val", lambda task, data_cfg, transforms: (train_ds, None, None))
    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.setattr(samplers, "build_sampler", lambda *a, **k: None)
    monkeypatch.setattr(tud, "DataLoader", lambda *a, **k: object())


def _base_config(tiling):
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": "imgs", "labels_dir": "lbls", "tiling": tiling},
        "training": {"batch_size": 2},
    }


def test_hpo_trial_resolved_config_replaces_unrealized_tiling(monkeypatch, tmp_path):
    """A trial that trained untiled must not leave the base config's requested tile_size in
    resolved_config, the record a later reader takes for the trial's geometry."""
    import tcip_store as ts

    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    class _UntiledTrialDataset:
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return i

    _patch_trial_machinery(monkeypatch, _UntiledTrialDataset())
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4}, [].append,
                   _base_config({"enabled": True, "tile_size": 999}), str(trial_dir))

    resolved = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))
    assert resolved["data"]["tiling"] == {"enabled": False}


def test_hpo_trial_resolved_config_records_effective_tile_geometry(monkeypatch, tmp_path):
    import tcip_store as ts

    from tcip_mcp.tools.training_tools import _run_hpo_trial, trial_config_key

    class _TiledTrialDataset(_TiledStub):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return i

    _patch_trial_machinery(monkeypatch, _TiledTrialDataset())
    trial_dir = tmp_path / "trial_0"
    _run_hpo_trial({"lr": 3e-4}, [].append, _base_config({"enabled": True}), str(trial_dir))

    tiling = ts.read(trial_config_key(trial_dir.parent, trial_dir.name))["data"]["tiling"]
    assert tiling == {"enabled": True, "tile_size": 224, "overlap": pytest.approx(0.2)}
