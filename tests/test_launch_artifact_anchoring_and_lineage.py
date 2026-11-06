"""What a launched run records about where it writes and which dataset it names.

``launch_training`` writes the run directory and a run record carrying the dataset's identity.
The training body is a separate process, so these tests stub it and assert what the parent
resolves, writes and hands to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.tools.training_tools import RunLaunched
from tests._chain_fixtures import BESPOKE_DETECTION
from tests._producer_fixtures import fake_popen
from tests._training_values import drawn_split, evaluation_block


def _canonical_dataset(root: Path, date: str = "2-11-26") -> Path:
    """Write a small dataset in the canonical layout and return its image directory.

    Each frame holds its labeled object bright on a dim ground.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import image_dir
    from tests._producer_fixtures import (
        labeled_frame,
        registry_of,
        registry_over,
    )

    images_dir = image_dir(root, date)
    images_dir.mkdir(parents=True)
    registry_over(root, registry_of("bud"))
    for i in range(2):
        labeled_frame(
            images_dir / f"img_{i}.png",
            [Annotation(subject="bud", geometry=BBox(8, 6, 40, 22))],
            96,
            64,
            (110, 120, 130),
        )
    return images_dir


def _detection_config(
    project: Path,
    images_dir: Path | str,
    builder: str = BESPOKE_DETECTION,
    split: dict | None = None,
    **data: object,
) -> dict:
    """Return a one-image-per-batch ``training_config`` of the detection ``builder``.

    The config is of ``project`` over the admitted images of ``images_dir``, drawing ``split``
    (by default holding nothing out), with ``data``'s keys in its data section, normalizing by
    frame statistics and selecting on its loss.
    """
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        FRAME_STATISTICS,
        training_config,
    )

    return training_config(
        project,
        {
            "builder": builder,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": split or drawn_split(val_ratio=0.0),
            **data,
        },
        batch_size=1,
        evaluation=evaluation_block(selection_metric="loss"),
    )


def _launch_record(res: RunLaunched) -> dict:
    """Return the launch record of the run ``res`` names."""
    from tcip_mcp.experiments import RUN_FILE, read_record

    return read_record(Path(res.output_dir) / RUN_FILE)


def test_the_run_directory_lies_under_its_project_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run directory lies under its project, not under the process's cwd.

    The run directory handed to the child and the run record written into it resolve under the
    project's experiments directory, and the launching process's cwd stays empty.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, experiments_dir

    started: list[list[str]] = []
    fake_popen(monkeypatch, started)
    project = tmp_path / "project"
    server_cwd = tmp_path / "server_cwd"
    project.mkdir()
    server_cwd.mkdir()
    monkeypatch.chdir(server_cwd)

    images_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(
        project, _detection_config(project, images_dir)
    )

    run_dir = Path(res.output_dir)
    assert run_dir == experiments_dir(project) / res.experiment_id
    assert (run_dir / RUN_FILE).is_file()
    assert list(server_cwd.iterdir()) == []

    [argv] = [argv for argv in started if "--run-dir" in argv]
    assert argv[argv.index("--run-dir") + 1] == str(run_dir)


def test_launched_run_records_the_datasets_identity_in_its_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched run records its dataset's identity in its run record.

    A registered dataset's minted id and its recomputed fingerprint both land in the run record.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import register_dataset

    fake_popen(monkeypatch, [])

    ds_root = tmp_path / "ds"
    images_dir = _canonical_dataset(ds_root)
    registered = register_dataset(
        tmp_path, str(ds_root), crop="currant", actor=None
    )
    assert registered["id"] and registered["fingerprint"]

    res = training_tools_launch(
        tmp_path, _detection_config(tmp_path, images_dir)
    )

    dataset = _launch_record(res)["dataset"]
    assert dataset == {
        "id": registered["id"],
        "fingerprint": registered["fingerprint"],
    }


def test_launch_records_what_the_smoke_contract_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch records what its smoke contract checked, leaving the handed config alone.

    launch_training records a model_contract (subject, gating, issues, gradient magnitudes) on
    the run record and leaves the config dict it was handed without one.
    """
    pytest.importorskip("torchvision")
    fake_popen(monkeypatch, [])
    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    launched_config = _detection_config(project, images_dir)
    res = training_tools_launch(project, launched_config)
    assert "model_contract" not in launched_config

    record = _launch_record(res)["model_contract"]
    assert (
        record["subject"]
        == "the model as built at launch, before any training step"
    )
    assert record["gating"] is True
    assert record["issues"] == []
    assert (
        isinstance(record["gradient_magnitudes"], dict)
        and record["gradient_magnitudes"]
    )


def test_launch_omitting_overfit_check_records_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted overfit check records ``overfit_check`` as null in the answer and the record."""
    pytest.importorskip("torchvision")
    fake_popen(monkeypatch, [])
    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(
        project, _detection_config(project, images_dir)
    )
    assert res.overfit_check is None

    assert _launch_record(res)["model_contract"]["overfit_check"] is None


def test_launch_with_overfit_check_records_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch with the overfit check records the report its preflight answered.

    ``launch_training(overfit_check=True)`` records one report on both its answer and the run
    record's model contract.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools import training_tools

    fake_popen(monkeypatch, [])

    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    res = training_tools.launch_training(
        project,
        _detection_config(project, images_dir),
        overfit_check=True,
        actor=None,
    )
    assert isinstance(res, RunLaunched), res
    assert res.overfit_check is not None

    record = _launch_record(res)["model_contract"]
    assert record["overfit_check"] == res.overfit_check.model_dump(mode="json")


def test_launch_with_overfit_check_over_a_diverging_model_records_each_non_finite_loss_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overfit check over a diverging model records each non-finite loss as None and fails.

    A model whose loss diverges under the overfit diagnostic launches with no error. Its run
    record, read back through the run reader, carries the report with ``None`` at each position
    whose loss was not finite, the final loss among them, the issue naming it and the check
    failed by that issue alone: the stored report states neither a pass nor a state beside a
    null.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools import training_tools

    fake_popen(monkeypatch, [])
    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    config = _detection_config(
        project, images_dir, "bespoke_models:build_diverging_detection"
    )
    res = training_tools.launch_training(
        project, config, overfit_check=True, actor=None
    )
    assert isinstance(res, RunLaunched), res

    stored = _launch_record(res)["model_contract"]["overfit_check"]
    assert set(stored) == {"losses", "initial", "final", "issue"}, stored

    contract = read_run(Path(res.output_dir)).model_contract
    assert contract is not None
    report = contract["overfit_check"]
    assert report is not None
    assert report.issue == "loss became non-finite during overfitting"
    assert report.passed is False
    assert None in report.losses and report.final is None
    assert report.final == report.losses[-1]
    assert isinstance(report.initial, float), report
    assert report.initial == report.losses[0]


def test_the_run_record_the_worker_reads_carries_the_seed_and_the_training_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched run's record carries the stated seed and the trainer's keys at top level.

    Under the real launch_training with Popen and TensorBoard stubbed by ``fake_popen``, the
    run's ``run.json`` sits in the directory its id names, and its config carries the stated
    seed and the device, with no ``training`` section and no ``experiment_id``.
    """
    pytest.importorskip("torchvision")
    fake_popen(monkeypatch, [])
    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    stated = _detection_config(project, images_dir)
    res = training_tools_launch(project, stated)

    record = _launch_record(res)
    config = record["config"]

    assert Path(res.output_dir).name == res.experiment_id
    assert config["seed"] == stated["seed"]
    assert config["device"] == "cpu"
    assert "training" not in config
    assert "experiment_id" not in config


def training_tools_launch(project: Path, config: dict) -> RunLaunched:
    """Launch a run of ``project`` and return its answer, asserting the config was accepted."""
    from tcip_mcp.tools import training_tools

    res = training_tools.launch_training(project, config, actor=None)
    assert isinstance(res, RunLaunched), res
    return res
