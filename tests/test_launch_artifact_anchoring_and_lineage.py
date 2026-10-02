"""What a launched run records about where it wrote and which data it trained on.

Two ends of the reproduce-a-number chain meet at ``launch_training``: the run directory a later
reader resolves, and the dataset identity its run record carries. The training body itself is a
separate process, so these tests stand in for the child and assert only what the parent resolves,
writes and hands to it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def recorded_children(monkeypatch):
    """Stand in for the training subprocess, recording the argv each launch hands it.

    The child's own training body is another module's contract; what the parent resolves, writes
    and passes across the process boundary is this module's.
    """
    children: list = []

    class _RecordedChild:
        def __init__(self, argv, **kwargs):
            self.argv = list(argv)
            self.pid = 4242
            children.append(self)

        def __class_getitem__(cls, item):
            # subprocess.Popen carries subscripted annotations elsewhere in the environment.
            return cls

    monkeypatch.setattr(subprocess, "Popen", _RecordedChild)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    return children


def _canonical_dataset(root: Path, date: str = "2-11-26") -> tuple[Path, Path]:
    """A small dataset in the canonical layout, the shape ``dataset_root_of`` resolves."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    for i in range(2):
        Image.new("RGB", (96, 64), color=(110, 120, 130)).save(images_dir / f"img_{i}.png")
        json_io.write_annotations(
            str(labels_dir / f"img_{i}.json"),
            [Annotation(subject="bud", geometry=BBox(8, 6, 40, 22))], 96, 64)
    return images_dir, labels_dir


def _detection_config(images_dir: Path, labels_dir: Path) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 96},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "bud"}, "auto_val": False},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
        "evaluation": {"selection_metric": "loss"},
    }


def _launch_record(res: dict) -> dict:
    """The launch record of the run ``res`` names."""
    from tcip_mcp.experiments import RUN_FILE, read_record

    return read_record(Path(res["output_dir"]) / RUN_FILE)


def test_the_run_directory_lies_under_its_project_not_the_process_cwd(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The run directory handed to the child and the run record written into it resolve under the
    project's experiments directory, never under the launching process's cwd."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, experiments_dir

    project = tmp_path / "project"
    server_cwd = tmp_path / "server_cwd"
    project.mkdir()
    server_cwd.mkdir()
    monkeypatch.chdir(server_cwd)

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(project, _detection_config(images_dir, labels_dir))

    run_dir = Path(res["output_dir"])
    assert run_dir == experiments_dir(project) / res["experiment_id"]
    assert (run_dir / RUN_FILE).is_file()
    assert list(server_cwd.iterdir()) == []

    [argv] = [c.argv for c in recorded_children if "--run-dir" in c.argv]
    assert argv[argv.index("--run-dir") + 1] == str(run_dir)


def test_launched_run_records_the_datasets_identity_in_its_run_record(
        tmp_path: Path, recorded_children) -> None:
    """The run record carries the identity of the data section's dataset, so the metric this run
    produces can be traced back to the exact content it trained on. A registered dataset's minted
    id and its recomputed fingerprint both land there, not None."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import register_dataset

    ds_root = tmp_path / "ds"
    images_dir, labels_dir = _canonical_dataset(ds_root)
    registered = register_dataset(tmp_path, str(ds_root), crop="currant")
    assert registered["id"] and registered["fingerprint"]

    res = training_tools_launch(tmp_path, _detection_config(images_dir, labels_dir))

    dataset = _launch_record(res)["dataset"]
    assert dataset == {"id": registered["id"], "fingerprint": registered["fingerprint"]}


def test_launch_records_what_the_smoke_contract_checked(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """launch_training records a model_contract (subject/gating/batch_source/dims/issues/
    gradient_magnitudes) on the run record, what the launch-time smoke actually checked, and the
    caller's own config dict is never the one mutated to carry it."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    launched_config = _detection_config(images_dir, labels_dir)
    res = training_tools_launch(project, launched_config)
    assert "model_contract" not in launched_config

    record = _launch_record(res)["model_contract"]
    assert record["subject"] == "the model as built at launch, before any training step"
    assert record["gating"] is True
    assert record["issues"] == []
    assert isinstance(record["gradient_magnitudes"], dict) and record["gradient_magnitudes"]


def test_launch_omitting_overfit_check_records_null(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The flag defaults off: the record carries overfit_check: null, never a missing key."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(project, _detection_config(images_dir, labels_dir))
    assert res["overfit_check"] is None

    assert _launch_record(res)["model_contract"]["overfit_check"] is None


def test_launch_with_overfit_check_records_the_rendered_report(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """``launch_training(overfit_check=True)`` runs the diagnostic on the contract's own batch
    and records the rendered report on both the returned dict and the persisted config."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools import training_tools

    project = tmp_path / "project"
    project.mkdir()

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools.launch_training(
        project, _detection_config(images_dir, labels_dir), overfit_check=True)
    assert "error" not in res, res
    assert res["overfit_check"] is not None
    assert "passed" in res["overfit_check"]

    record = _launch_record(res)["model_contract"]
    assert record["overfit_check"] == res["overfit_check"]


def test_launch_with_overfit_check_over_a_diverging_model_proceeds_with_a_json_safe_record(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """A model whose loss diverges to nan under the overfit diagnostic never blocks the launch,
    since the check is voluntary and non-gating: the run proceeds, and the persisted report
    renders the non-finite losses as null with the state named beside them, so the record this
    launch writes still passes check_json_value on what the diagnostic actually observed."""
    pytest.importorskip("torchvision")
    from tcip_store import check_json_value

    from tcip_mcp.tools import training_tools

    project = tmp_path / "project"
    project.mkdir()

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_diverging_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 96},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "bud"}, "auto_val": False},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
        "evaluation": {"selection_metric": "loss"},
    }
    res = training_tools.launch_training(project, config, overfit_check=True)
    assert "error" not in res, res

    record = _launch_record(res)["model_contract"]
    check_json_value(record, path="model_contract")
    report = record["overfit_check"]
    assert report["passed"] is False
    assert any(loss is None for loss in report["losses"])
    assert report.get("final_state") in ("nan", "positive_infinity", "negative_infinity")


def test_the_run_record_the_worker_reads_carries_the_seed_and_the_training_keys(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The writer is the real launch_training (Popen and TensorBoard stubbed by
    recorded_children so no subprocess actually spawns); the reader is the one the training
    child performs, the run directory's own ``run.json``. The trainer's keys sit at the config's
    top level and ``draw_seed_if_unset`` draws a seed into it before the record is written, so
    the config the worker reads carries both, in the directory the run's own id names."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(project, _detection_config(images_dir, labels_dir))

    record = _launch_record(res)
    config = record["config"]

    assert Path(res["output_dir"]).name == res["experiment_id"]
    assert isinstance(config["seed"], int)
    assert config["device"] == "cpu"
    assert "training" not in config
    assert "experiment_id" not in config


def training_tools_launch(project: Path, config: dict) -> dict:
    """Launch a run of ``project`` and assert the config was accepted, so a preflight refusal
    never reads as a provenance failure in the tests above."""
    from tcip_mcp.tools import training_tools

    res = training_tools.launch_training(project, config)
    assert "error" not in res, res
    return res
