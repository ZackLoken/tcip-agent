"""What a launched run records about where it wrote and which data it trained on.

Two ends of the reproduce-a-number chain meet at ``launch_training``: the run directory a later
reader resolves, and the dataset identity its run record carries. The training body itself is a
separate process, so these tests stand in for the child and assert only what the parent resolves,
writes and hands to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._chain_fixtures import BESPOKE_DETECTION
from tests._producer_fixtures import fake_popen
from tests._training_values import drawn_split, evaluation_block


def _canonical_dataset(root: Path, date: str = "2-11-26") -> Path:
    """A small dataset in the canonical layout, the shape ``dataset_root_of`` resolves, each
    frame holding its labeled object bright on a dim ground; its image directory.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import (
        BRIGHT,
        label_image,
        painted_frame,
        registry_over,
    )

    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    for i in range(2):
        painted_frame(
            96, 64, (110, 120, 130), [((8, 6, 40, 22), BRIGHT)]
        ).save(images_dir / f"img_{i}.png")
        label_image(
            images_dir / f"img_{i}.png",
            [Annotation(subject="bud", geometry=BBox(8, 6, 40, 22))],
            96,
            64,
        )
    return images_dir


def _detection_config(
    project: Path,
    images_dir: Path | str,
    builder: str = BESPOKE_DETECTION,
    split: dict | None = None,
    **data,
) -> dict:
    """A one-image-per-batch ``training_config`` of ``project`` of the detection ``builder``
    over the admitted images of ``images_dir``, drawing ``split`` (by default holding nothing
    out), with ``data``'s keys in its data section, at the statistics and geometry its frames
    and boxes derive, selecting on its loss.
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


def _launch_record(res: dict) -> dict:
    """The launch record of the run ``res`` names."""
    from tcip_mcp.experiments import RUN_FILE, read_record

    return read_record(Path(res["output_dir"]) / RUN_FILE)


def test_the_run_directory_lies_under_its_project_not_the_process_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """The run directory handed to the child and the run record written into it resolve under the
    project's experiments directory, never under the launching process's cwd.
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

    run_dir = Path(res["output_dir"])
    assert run_dir == experiments_dir(project) / res["experiment_id"]
    assert (run_dir / RUN_FILE).is_file()
    assert list(server_cwd.iterdir()) == []

    [argv] = [argv for argv in started if "--run-dir" in argv]
    assert argv[argv.index("--run-dir") + 1] == str(run_dir)


def test_launched_run_records_the_datasets_identity_in_its_run_record(
    tmp_path: Path, monkeypatch
) -> None:
    """The run record carries the identity of the data section's dataset, so the metric this run
    produces can be traced back to the exact content it trained on. A registered dataset's minted
    id and its recomputed fingerprint both land there, not None.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import register_dataset

    fake_popen(monkeypatch, [])

    ds_root = tmp_path / "ds"
    images_dir = _canonical_dataset(ds_root)
    registered = register_dataset(tmp_path, str(ds_root), crop="currant")
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
    tmp_path: Path, monkeypatch
) -> None:
    """launch_training records a model_contract (subject/gating/batch_source/dims/issues/
    gradient_magnitudes) on the run record, what the launch-time smoke actually checked, and the
    caller's own config dict is never the one mutated to carry it.
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
    tmp_path: Path, monkeypatch
) -> None:
    """The flag defaults off: the record carries overfit_check: null, never a missing key."""
    pytest.importorskip("torchvision")
    fake_popen(monkeypatch, [])
    project = tmp_path / "project"
    project.mkdir()

    images_dir = _canonical_dataset(project / "ds")
    res = training_tools_launch(
        project, _detection_config(project, images_dir)
    )
    assert res["overfit_check"] is None

    assert _launch_record(res)["model_contract"]["overfit_check"] is None


def test_launch_with_overfit_check_records_the_rendered_report(
    tmp_path: Path, monkeypatch
) -> None:
    """``launch_training(overfit_check=True)`` runs the diagnostic on the contract's own batch
    and records the rendered report on both the returned dict and the persisted config.
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
    assert "error" not in res, res
    assert res["overfit_check"] is not None
    assert "passed" in res["overfit_check"]

    record = _launch_record(res)["model_contract"]
    assert record["overfit_check"] == res["overfit_check"]


def test_launch_with_overfit_check_over_a_diverging_model_proceeds_with_a_json_safe_record(
    tmp_path: Path, monkeypatch
) -> None:
    """A model whose loss diverges to nan under the overfit diagnostic never blocks the launch,
    since the check is voluntary and non-gating: the run proceeds, and the persisted report
    renders the non-finite losses as null with the state named beside them, so the record this
    launch writes still passes check_json_value on what the diagnostic actually observed.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools import training_tools
    from tcip_store import check_json_value

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
    assert "error" not in res, res

    record = _launch_record(res)["model_contract"]
    check_json_value(record, path="model_contract")
    report = record["overfit_check"]
    assert report["passed"] is False
    assert any(loss is None for loss in report["losses"])
    assert report.get("final_state") in (
        "nan",
        "positive_infinity",
        "negative_infinity",
    )


def test_the_run_record_the_worker_reads_carries_the_seed_and_the_training_keys(
    tmp_path: Path, monkeypatch
) -> None:
    """The writer is the real launch_training (Popen and TensorBoard stubbed by ``fake_popen``
    so no subprocess actually spawns); the reader is the one the training child performs, the
    run directory's own ``run.json``. The trainer's keys and the seed the caller states sit at
    the config's top level, so the config the worker reads carries both, in the directory the
    run's own id names.
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

    assert Path(res["output_dir"]).name == res["experiment_id"]
    assert config["seed"] == stated["seed"]
    assert config["device"] == "cpu"
    assert "training" not in config
    assert "experiment_id" not in config


def training_tools_launch(project: Path, config: dict) -> dict:
    """Launch a run of ``project`` and assert the config was accepted, so a preflight refusal
    never reads as a provenance failure in the tests above.
    """
    from tcip_mcp.tools import training_tools

    res = training_tools.launch_training(project, config, actor=None)
    assert "error" not in res, res
    return res
