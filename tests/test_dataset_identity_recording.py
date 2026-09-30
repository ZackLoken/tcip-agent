"""A run records the dataset identity it launched against, once.

The content end of the reproduce-a-number chain: id + fingerprint are written into the run's
launch record when it is opened, which nothing rewrites, and compare_experiments surfaces whether
two runs share a dataset so a metric comparison across different data is not read as
apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.experiments import compare_experiments
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, opened_run

SUBJECT = "bud"


def _make_dataset(root: Path, *, shade: int = 0,
                  images: str = "images") -> tuple[Path, Path]:
    """Two labeled images of ``shade`` under one capture date, in an images directory named
    ``images`` (a name outside the dataset layout leaves them under no dataset root); answers
    ``(images_dir, labels_dir)``."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import subject_registry
    from tcip_mcp.subject_registry import SubjectRegistry, Subject

    images_dir, labels_dir = root / images / "2-11-26", root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        Image.new("RGB", (32, 32), color=(shade, 10 * i, 0)).save(images_dir / f"img_{i:03d}.jpg")
        json_io.write_annotations(str(labels_dir / f"img_{i:03d}.json"),
                                  [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))], 32, 32)
    subject_registry.write_registry(root / "subjects.json",
                                  SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    return images_dir, labels_dir


def _config(images_dir: Path, labels_dir: Path) -> dict:
    """A :data:`BUILT_DETECTOR` run's config over ``images_dir`` and ``labels_dir``."""
    return {"model_source": dict(BUILT_DETECTOR),
            "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                     "scope": {"subject": SUBJECT, "id_map": {SUBJECT: 0}}}}


def test_compare_experiments_surfaces_shared_fingerprint(tmp_path):
    first = _config(*_make_dataset(tmp_path / "first"))
    opened_run(tmp_path, first, experiment_id="a")
    opened_run(tmp_path, first, experiment_id="b")
    assert compare_experiments(["a", "b"], project=tmp_path)["same_dataset_fingerprint"] is True
    opened_run(tmp_path, _config(*_make_dataset(tmp_path / "second", shade=200)),
               experiment_id="c")
    assert compare_experiments(["a", "c"], project=tmp_path)["same_dataset_fingerprint"] is False


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(tmp_path):
    """One run with a known fingerprint compared against a run whose images sit under no dataset
    root (None) must report unknown identity, not a false apples-to-apples True: the two
    demonstrably did not train on the same (known) data."""
    opened_run(tmp_path, _config(*_make_dataset(tmp_path / "first")), experiment_id="a")
    opened_run(tmp_path, _config(*_make_dataset(tmp_path / "loose", images="frames")),
               experiment_id="b")
    assert compare_experiments(["a", "b"], project=tmp_path)["same_dataset_fingerprint"] is None


def test_dataset_identity_helper_registered_vs_bespoke(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    _make_dataset(tmp_path)
    reg = register_dataset(tmp_path, str(tmp_path), crop="currant")
    ds_id, fp = dataset_identity({"images_dir": str(tmp_path / "images" / "2-11-26")})
    assert ds_id == reg["id"] and fp == reg["fingerprint"]
    # bespoke / imageless run -> no fabricated identity
    assert dataset_identity({}) == (None, None)


def test_dataset_identity_raises_a_fingerprint_read_failure(tmp_path, monkeypatch):
    """A fingerprint read failure (a locked or removed image mid-scan) raises by name rather than
    reading as a run with no identity: the launch refuses before its directory exists."""
    import tcip_mcp.pipelines.data.dataset_fingerprint as dataset_fingerprint_mod
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.tools.project_tools import register_dataset

    _make_dataset(tmp_path)
    register_dataset(tmp_path, str(tmp_path), crop="currant")

    def _raise(_root):
        raise OSError("simulated I/O error mid-scan")

    # dataset_identity reads the fingerprint through its own module at call time, so it is
    # patched at the source module.
    monkeypatch.setattr(dataset_fingerprint_mod, "dataset_fingerprint", _raise)
    with pytest.raises(OSError, match="simulated I/O error"):
        dataset_identity({"images_dir": str(tmp_path / "images" / "2-11-26")})


def test_a_trial_over_a_data_axis_records_the_dataset_its_own_input_names(tmp_path, monkeypatch):
    """A sweep trial whose sampled point names another dataset records that dataset's identity,
    read off the trial's own input, never the identity of the sweep's base dataset."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from tcip_mcp.experiments import RUN_FILE, read_record, sweeps_dir
    from tcip_mcp.pipelines.training import subprocess_worker
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    base_images, base_labels = _make_dataset(tmp_path / "base")
    register_dataset(tmp_path / "base", str(tmp_path / "base"), crop="currant")
    other_images, other_labels = _make_dataset(tmp_path / "other", shade=200)
    other = register_dataset(tmp_path / "other", str(tmp_path / "other"), crop="currant")
    monkeypatch.setattr(subprocess_worker, "run_directory",
                        lambda *a, **k: SimpleNamespace(status="failed"))
    trial_dir = sweeps_dir(tmp_path) / "sweep" / "trial_a"
    trial_dir.parent.mkdir(parents=True)

    _run_hpo_trial({"data.images_dir": str(other_images), "data.labels_dir": str(other_labels)},
                   [].append, _config(base_images, base_labels), trial_dir, project=tmp_path,
                   objective={"selection_metric": "loss", "higher_is_better": False},
                   launched_by={"launcher": "process"})

    dataset = read_record(trial_dir / RUN_FILE)["dataset"]
    assert dataset == {"id": other["id"], "fingerprint": other["fingerprint"]}


def test_a_launch_records_the_identity_of_the_dataset_it_trains_on(tmp_path, monkeypatch):
    """The launch reads the registered dataset's identity once and writes it into the run's
    launch record before the child starts; the child's resolved partition names the samples
    beside it."""
    import subprocess

    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import launch_training

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    class _StubChild:
        pid = 4242

        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    images_dir, labels_dir = _make_dataset(tmp_path)
    registered = register_dataset(tmp_path, str(tmp_path), crop="currant")
    launched = launch_training(tmp_path, {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 64},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": SUBJECT}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}], "device": "cpu",
        "experiment_id": "exp-identity",
    })
    assert "error" not in launched, launched

    dataset = read_record(experiment_dir("exp-identity", project=tmp_path) / RUN_FILE)["dataset"]
    assert dataset == {"id": registered["id"], "fingerprint": registered["fingerprint"]}
