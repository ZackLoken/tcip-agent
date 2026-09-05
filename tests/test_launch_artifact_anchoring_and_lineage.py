"""What a launched run records about where it wrote and which data it trained on.

Two ends of the reproduce-a-number chain meet at ``launch_training``: the artifact directory a
later reader resolves from the experiment record, and the dataset identity the immutable lineage
carries. The training body itself is a separate process, so these tests stand in for the child and
assert only what the parent resolves, writes and hands to it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import tcip_store as ts


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
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    class_registry.write_registry(root / "classes.json",
                                  ClassRegistry(subjects=(Subject(name="bud"),)))
    for i in range(2):
        Image.new("RGB", (96, 64), color=(110, 120, 130)).save(images_dir / f"img_{i}.png")
        json_io.write_annotations(
            str(labels_dir / f"img_{i}.json"),
            [Annotation(subject="bud", geometry=BBox(8, 6, 40, 22))], 96, 64)
    return images_dir, labels_dir


def _detection_config(images_dir: Path, labels_dir: Path) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 96},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "subject": "bud", "auto_val": False},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
    }


def test_a_relative_output_dir_anchors_to_the_platform_state_root_not_the_process_cwd(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """A relative ``output_dir`` names a directory inside the project. The run directory handed to
    the child, the launch config written into it, and the path stamped into the experiment's
    status.json all resolve under the platform state root, never under the launching process's
    cwd."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    server_cwd = tmp_path / "server_cwd"
    project.mkdir()
    server_cwd.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))
    monkeypatch.chdir(server_cwd)

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(_detection_config(images_dir, labels_dir), "runs/nightly")

    from tcip_mcp.tools.training_tools import launch_config_key

    run_dir = Path(res["output_dir"])
    assert run_dir == project / "runs" / "nightly" / res["run_id"]
    assert ts.exists(launch_config_key(run_dir))
    assert not (server_cwd / "runs").exists()

    argv = recorded_children[0].argv
    assert argv[argv.index("--output-dir") + 1] == str(run_dir)

    from tcip_mcp.experiments import status_key

    status = ts.read(status_key(res["experiment_id"]))
    assert status["output_dir"] == str(run_dir)


def test_an_absolute_output_dir_stays_the_callers_own_choice(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The anchoring never captures a path the caller already made explicit: an absolute
    ``output_dir`` outside the project is honored, with the run still nested under its run id."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    scratch = tmp_path / "scratch_volume"
    project.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(_detection_config(images_dir, labels_dir), str(scratch))

    from tcip_mcp.tools.training_tools import launch_config_key

    assert Path(res["output_dir"]) == scratch / res["run_id"]
    assert ts.exists(launch_config_key(scratch / res["run_id"]))
    # The experiment record itself still belongs to the project, wherever the weights go.
    from tcip_mcp.experiments import status_key

    assert ts.exists(status_key(res["experiment_id"], root=project))


def test_launched_run_records_the_datasets_identity_in_its_lineage(
        tmp_path: Path, recorded_children) -> None:
    """The lineage carries the identity of the data section's dataset, so the metric this run
    produces can be traced back to the exact content it trained on. A registered dataset's minted
    id and its recomputed fingerprint both land there, not None."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import register_dataset

    ds_root = tmp_path / "ds"
    images_dir, labels_dir = _canonical_dataset(ds_root)
    registered = register_dataset(str(ds_root), crop="currant")
    assert registered["id"] and registered["fingerprint"]

    res = training_tools_launch(_detection_config(images_dir, labels_dir), "")

    from tcip_mcp.experiments import lineage_key

    lineage = ts.read(lineage_key(res["experiment_id"]))
    assert lineage["dataset_id"] == registered["id"]
    assert lineage["dataset_fingerprint"] == registered["fingerprint"]


def test_launch_records_what_the_smoke_contract_checked(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """launch_training persists a model_contract record (subject/gating/batch_source/dims/issues/
    gradient_magnitudes) onto the config every checkpoint embeds, so config.json and
    launch_config.json both carry what the launch-time smoke actually checked, and the caller's
    own config dict is never the one mutated to carry it."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    launched_config = _detection_config(images_dir, labels_dir)
    res = training_tools_launch(launched_config, "")
    assert "model_contract" not in launched_config

    from tcip_mcp.experiments import config_key
    from tcip_mcp.tools.training_tools import launch_config_key

    config = ts.read(config_key(res["experiment_id"]))
    record = config["model_contract"]
    assert record["subject"] == "the model as built at launch, before any training step"
    assert record["gating"] is True
    assert record["issues"] == []
    assert isinstance(record["gradient_magnitudes"], dict) and record["gradient_magnitudes"]

    launch_config = ts.read(launch_config_key(Path(res["output_dir"])))
    assert launch_config["model_contract"] == record


def test_launch_omitting_overfit_check_records_null(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The flag defaults off: the record carries overfit_check: null, never a missing key."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(_detection_config(images_dir, labels_dir), "")
    assert res["overfit_check"] is None

    from tcip_mcp.experiments import config_key

    config = ts.read(config_key(res["experiment_id"]))
    assert config["model_contract"]["overfit_check"] is None


def test_launch_with_overfit_check_records_the_rendered_report(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """``launch_training(overfit_check=True)`` runs the diagnostic on the contract's own batch
    and records the rendered report on both the returned dict and the persisted config."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools import training_tools

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools.launch_training(
        _detection_config(images_dir, labels_dir), "", overfit_check=True)
    assert "error" not in res, res
    assert res["overfit_check"] is not None
    assert "passed" in res["overfit_check"]

    from tcip_mcp.experiments import config_key

    config = ts.read(config_key(res["experiment_id"]))
    assert config["model_contract"]["overfit_check"] == res["overfit_check"]


def test_launch_with_overfit_check_over_a_diverging_model_proceeds_with_a_json_safe_record(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """A model whose loss diverges to nan under the overfit diagnostic never blocks the launch,
    since the check is voluntary and non-gating: the run proceeds, and the persisted report
    renders the non-finite losses as null with the state named beside them, so the record this
    launch writes still passes check_json_value on what the diagnostic actually observed."""
    pytest.importorskip("torchvision")
    from tcip_store import check_json_value

    from tcip_mcp.experiments import config_key
    from tcip_mcp.tools import training_tools

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_diverging_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 96},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "subject": "bud", "auto_val": False},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
    }
    res = training_tools.launch_training(config, "", overfit_check=True)
    assert "error" not in res, res

    record = ts.read(config_key(res["experiment_id"]))["model_contract"]
    check_json_value(record, path="model_contract")
    report = record["overfit_check"]
    assert report["passed"] is False
    assert any(loss is None for loss in report["losses"])
    assert report.get("final_state") in ("nan", "positive_infinity", "negative_infinity")


def test_the_launch_record_the_worker_reads_carries_the_seed_and_the_hoisted_training_keys(
        tmp_path: Path, monkeypatch, recorded_children) -> None:
    """The writer is the real launch_training, driven the same way the tests above drive it
    (training_tools_launch over _canonical_dataset/_detection_config, Popen and TensorBoard
    stubbed by recorded_children so no subprocess actually spawns). The reader is the same read
    the training child performs: tcip_store.read(launch_config_key(output_dir)), the call
    subprocess_worker.run() makes, not a second parse of config.json. normalize_train_config
    (schemas.py) hoists every key set under config["training"] onto the top level when the top
    level doesn't already have it, and run_registry.create_run draws a seed into
    config["seed"] before this document is written, so the document the worker reads must carry
    both, plus the model_contract launch_training records and the resolved experiment_id: a
    document only the writer's own test has seen is one the worker's read could silently
    disagree with."""
    pytest.importorskip("torchvision")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project))

    images_dir, labels_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(_detection_config(images_dir, labels_dir), "")

    from tcip_mcp.tools.training_tools import launch_config_key

    launch_config = ts.read(launch_config_key(Path(res["output_dir"])))

    assert isinstance(launch_config["seed"], int)
    assert isinstance(launch_config["model_contract"], dict)
    assert launch_config["experiment_id"] == res["experiment_id"]
    # "device" is set only under config["training"] in _detection_config; normalize_train_config
    # hoists it to the top level, which is what this checks actually happened.
    assert launch_config["training"]["device"] == "cpu"
    assert launch_config["device"] == "cpu"


def training_tools_launch(config: dict, output_dir: str) -> dict:
    """Launch and assert the config was accepted, so a preflight refusal never reads as a
    provenance failure in the tests above."""
    from tcip_mcp.tools import training_tools

    res = training_tools.launch_training(config, output_dir=output_dir)
    assert "error" not in res, res
    return res
