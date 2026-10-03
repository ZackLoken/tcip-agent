"""A run is a directory, and every state it reads as comes from what its own process wrote there.

The launched cases go through the real launcher (``launch_training``) and a real child process,
so the facts are the ones a run directory actually carries across the process boundary: the
launch record the parent writes once, the final status the child writes once, and the heartbeat
that stands in for a process nobody can ask. The HPO trial cases run the sweep's own trial body
(``_run_hpo_trial``) in this process, over the same launch writer and child entry a launched run
uses.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.fixture
def children(monkeypatch) -> list:
    """Every training child a launch in this test spawned (a process handed ``--run-dir``), each
    a real process."""
    spawned: list = []
    real_popen = subprocess.Popen

    class _Recorded(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if "--run-dir" in (args[0] if args else kwargs.get("args", ())):
                spawned.append(self)

    monkeypatch.setattr(subprocess, "Popen", _Recorded)
    return spawned


@pytest.fixture
def launch(tmp_path, monkeypatch, children):
    """``launch_training`` in the project ``tmp_path``, over a tiny regression dataset, with
    ``epochs`` stages and any other top-level keys; TensorBoard is not started."""
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.tools.training_tools import launch_training
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.0, 0.25, 0.5, 1.0], values=[0.1, 0.3, 0.5, 0.9])

    def _launch(epochs: int = 1, *, resume_from: str = "", **extra) -> dict:
        config = {
            "model_source": {
                "builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                "task": "regression"},
            "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
            "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": epochs}],
            "mixed_precision": False, "device": "cpu",
            "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}, **extra,
        }
        return launch_training(tmp_path, config, resume_from=resume_from, actor=None)

    return _launch


def _until(what: str, answer, deadline_s: float = 120):
    """``answer()`` once it is truthy, polled until ``deadline_s`` passes; the child's own state
    is what each caller waits on."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        value = answer()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"{what} did not happen within {deadline_s}s")


def _wait_final(run_dir: Path) -> dict:
    """The run's final status once its child has written one."""
    from tcip_mcp.experiments import observe

    return _until(f"{run_dir.name}'s final status", lambda: observe(run_dir).final)


def _contents(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def test_a_launched_run_completes_and_its_completion_is_its_registration(launch, tmp_path):
    from tcip_mcp.experiments import METRICS_FILE, RUN_FILE, observe, read_rows
    from tcip_mcp.model_registry import ModelRegistry, load_registered_checkpoint

    res = launch()
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])

    final = _wait_final(run_dir)
    assert final["state"] == "completed", final
    observation = observe(run_dir)
    assert observation.state == "completed"
    assert (run_dir / RUN_FILE).is_file()
    assert read_rows(run_dir / METRICS_FILE)[0]
    checkpoint = observation.checkpoint
    assert checkpoint is not None and Path(checkpoint["path"]).is_file()
    assert final["checkpoint"]["sha256"] == checkpoint["sha256"]
    (entry,) = [m for m in ModelRegistry(str(tmp_path)).list_models()
                if m["name"] == run_dir.name]
    assert entry["sha256"] == checkpoint["sha256"]
    loaded = load_registered_checkpoint(checkpoint["path"], project=tmp_path)
    assert loaded.producer["experiment_id"] == run_dir.name


def test_a_cancel_requested_by_id_ends_the_child_canceled(launch, tmp_path):
    """A cancel requested once the child is training (its first epoch row logged) ends it
    canceled, completing no checkpoint."""
    from tcip_mcp.experiments import METRICS_FILE, observe, read_rows
    from tcip_mcp.tools.training_tools import cancel_training

    res = launch(epochs=200)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    _until("the child's first epoch row", lambda: read_rows(run_dir / METRICS_FILE)[0])

    assert cancel_training(tmp_path, res["experiment_id"], actor=None)["cancel_requested"] is True
    final = _wait_final(run_dir)

    assert final["state"] == "canceled", final
    assert observe(run_dir).checkpoint is None


def test_a_resume_is_a_new_directory_that_leaves_its_source_untouched(launch, tmp_path):
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.registry_paths import stored_path

    first = launch(epochs=2, checkpoint_every_n_epochs=1)
    source = Path(first["output_dir"])
    assert _wait_final(source)["state"] == "completed"
    epoch_checkpoints = sorted(source.glob("checkpoint_epoch_*.pt"))
    assert epoch_checkpoints, sorted(p.name for p in source.iterdir())
    before = _contents(source)

    resumed = launch(epochs=3, resume_from=str(epoch_checkpoints[-1]))
    assert "error" not in resumed, resumed
    run_dir = Path(resumed["output_dir"])
    assert run_dir != source
    assert _wait_final(run_dir)["state"] == "completed"

    assert read_record(run_dir / RUN_FILE)["resume_from"] == stored_path(epoch_checkpoints[-1],
                                                                         tmp_path)
    assert (run_dir / "model_final.pt").is_file()
    assert _contents(source) == before


def test_a_child_killed_before_its_final_status_reads_running_then_interrupted(
        launch, children, monkeypatch):
    """Killed while it runs, the child writes no final status: once the process has exited the
    run reads running while its last sign of life is fresh, and interrupted once that sign of
    life is older than the heartbeat window."""
    from tcip_mcp import experiments

    res = launch(epochs=200)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    [child] = children
    child.kill()
    child.wait()

    assert experiments.observe(run_dir).final is None
    assert experiments.observe(run_dir).state == "running"
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert experiments.observe(run_dir).state == "interrupted"


def test_progress_logged_after_the_final_status_never_reopens_the_run(launch, tmp_path):
    """Once the child wrote its final status the run's metrics writer refuses a row, leaving the
    log and the final status as they were; the completed run's summary is the best selection over
    its own rows under its recorded objective."""
    from tcip_mcp.experiments import (
        FINAL_STATUS_FILE, METRICS_FILE, RunEnded, best_selection, observe, read_rows,
    )
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import TrainRun
    from tcip_mcp.tools.training_tools import monitor_training

    res = launch(epochs=2)
    run_dir = Path(res["output_dir"])
    assert _wait_final(run_dir)["state"] == "completed"
    before = {name: (run_dir / name).read_bytes() for name in (FINAL_STATUS_FILE, METRICS_FILE)}
    observation = observe(run_dir)
    run = TrainRun(id=run_dir.name, config=observation.record["config"],
                   objective=observation.record["resolved"]["objective"],
                   project=tmp_path, output_dir=str(run_dir))

    with pytest.raises(RunEnded):
        TrainContext(run=run, train_loader=None).log_metrics(
            99, {"loss": 1e-9, "selection_metric": "loss", "selection": 1e-9})

    assert {name: (run_dir / name).read_bytes() for name in before} == before
    rows = read_rows(observation.metrics_log)[0]
    summary = monitor_training(tmp_path, res["experiment_id"])
    assert summary["status"] == "completed"
    assert summary["best_metric"] is not None
    assert summary["best_metric"] == best_selection(
        rows, observation.record["resolved"]["objective"])


def test_a_reader_never_sees_a_final_status_before_its_bytes_are_whole(tmp_path, monkeypatch):
    """Read while the final status's bytes are being written, the run reads as it did before
    that write began: no final status and still running, never a partial record."""
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    seen: list = []
    real_publish = experiments.publish_once

    def observed_mid_write(path, write):
        def write_after_reading(handle):
            seen.append(experiments.observe(run_dir))
            write(handle)
        real_publish(path, write_after_reading)

    monkeypatch.setattr(experiments, "publish_once", observed_mid_write)
    experiments.write_final_status(run_dir, "failed", "stopped by hand")

    [during] = seen
    assert (during.final, during.state) == (None, "running")
    assert experiments.observe(run_dir).state == "failed"


def test_a_completed_run_missing_its_checkpoint_refuses_naming_it(tmp_path):
    """A completed run's checkpoint is read by whatever loads it; with that file gone the load
    refuses naming it rather than answering from anything else."""
    from tcip_mcp.model_registry import checkpoint_payload
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = Path(registered_checkpoint(tmp_path, experiment_id="exp-lost-weights"))
    named = observe(checkpoint.parent).checkpoint
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match=checkpoint.name):
        checkpoint_payload(named["path"], named["sha256"])


def test_a_launch_into_an_existing_directory_refuses_and_writes_nothing(launch, tmp_path):
    """Every launch is a new run: naming a directory that exists, a run's or one left holding no
    launch record at all, refuses by name and leaves it as it was."""
    from tcip_mcp.experiments import RUN_FILE, experiment_dir

    first = launch()
    run_dir = Path(first["output_dir"])
    _wait_final(run_dir)
    before = _contents(run_dir)

    again = launch(experiment_id=first["experiment_id"])
    assert "already exists" in again["error"], again
    assert _contents(run_dir) == before

    leftover = experiment_dir("leftover-run", project=tmp_path)
    leftover.mkdir(parents=True)
    (leftover / "notes.txt").write_text("left by hand", encoding="utf-8")

    refused = launch(experiment_id="leftover-run")
    assert "already exists" in refused["error"], refused
    assert not (leftover / RUN_FILE).exists()
    assert _contents(leftover) == {"notes.txt": b"left by hand"}


def _trial(tmp_path, monkeypatch, **extra) -> tuple[Path, list[float]]:
    """One HPO trial over a tiny regression dataset, run through the sweep's own trial body under
    the sweep's objective (``loss``, lower better); its run directory and every value it
    reported."""
    from tcip_mcp.experiments import sweeps_dir
    from tcip_mcp.tools.training_tools import _run_hpo_trial
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.0, 0.25, 0.5, 0.75, 1.0, 0.1],
        values=[0.1, 0.3, 0.5, 0.7, 0.9, 0.2])
    base_config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                 "split": {"val_ratio": 0.34, "seed": 3}},
        "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 2}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}, **extra,
    }
    trial_dir = sweeps_dir(tmp_path) / "hpo_study" / "trial_a"
    trial_dir.parent.mkdir(parents=True)
    reported: list[float] = []
    _run_hpo_trial({"lr": 0.01}, reported.append, base_config, trial_dir, project=tmp_path,
                   objective={"selection_metric": "loss", "higher_is_better": False})
    return trial_dir, reported


def test_an_hpo_trial_is_a_run_directory_reporting_the_sweeps_one_objective(
        tmp_path, monkeypatch):
    """A trial is a run directory like any launched run's, its sampled point and what it resolved
    on its launch record, and it reports the sweep's objective each epoch; its result is the best
    of what it reported, in the sweep's direction."""
    from tcip_mcp.experiments import RUN_FILE, observe, read_record
    from tcip_mcp.tools.training_tools import _trial_row

    trial_dir, reported = _trial(tmp_path, monkeypatch)

    record = read_record(trial_dir / RUN_FILE)
    objective = {"selection_metric": "loss", "higher_is_better": False}
    assert record["trial_params"] == {"lr": 0.01}
    assert record["resolved"]["partition"]["samples"]
    assert record["resolved"]["objective"] == objective
    assert observe(trial_dir).state == "completed"
    assert len(reported) == 2
    assert _trial_row(observe(trial_dir), objective)["value"] == min(reported)


def test_a_trial_whose_resolution_fails_is_a_directory_whose_final_status_names_it(
        tmp_path, monkeypatch):
    """A trial whose sampled point cannot resolve still opens its run directory, and its final
    status is ``failed`` naming why; the sweep's projection lists it."""
    from tcip_mcp.experiments import SWEEP_FILE, observe, write_once
    from tcip_mcp.tools.training_tools import read_sweep

    trial_dir, reported = _trial(tmp_path, monkeypatch, data={"images_dir": str(tmp_path / "gone"),
                                                             "labels_dir": str(tmp_path / "gone")})

    final = observe(trial_dir).final
    assert final["state"] == "failed" and final["error"]
    assert reported == []
    write_once(trial_dir.parent / SWEEP_FILE,
               {"objective": {"selection_metric": "loss", "higher_is_better": False},
                "input": {"split_draws": 1}})
    (row,) = read_sweep(observe(trial_dir.parent, SWEEP_FILE))["trials"]
    assert (row["trial_id"], row["status"], row["error"]) == ("a", "failed", final["error"])


def _reports_only(ctx):
    """A bespoke body that reports its objective through ``ctx.report_objective`` alone and saves
    a checkpoint carrying no metrics."""
    ctx.report_objective(0.25)
    ctx.save_checkpoint({"model_state_dict": ctx.build_model().state_dict()}, "model_final")


def test_a_report_only_trial_projects_the_value_it_reported(tmp_path, monkeypatch):
    """A bespoke trial reporting only through ``report_objective`` projects that value as its
    result, the same value the sweep's scheduler saw."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import _trial_row

    trial_dir, reported = _trial(tmp_path, monkeypatch,
                                 training_source=f"{__name__}:_reports_only")

    assert observe(trial_dir).state == "completed"
    assert reported == [0.25]
    row = _trial_row(observe(trial_dir), {"selection_metric": "loss", "higher_is_better": False})
    assert row["value"] == 0.25


def test_the_sweep_the_live_summary_and_a_trial_read_one_recorded_objective(
        tmp_path, monkeypatch, real_hpo_base_config):
    """The objective a sweep resolves once is the one its trials record and the one each trial's
    live summary names, even for a point whose own config states another metric: the three read
    one record. The trial's body is not run."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import observe, run_summary
    from tcip_mcp.pipelines.training import subprocess_worker

    monkeypatch.setattr(subprocess_worker, "run_directory", lambda *a, **k: None)

    def one_trial(**kw):
        kw["objective_fn"]({"evaluation.selection_metric": "loss"}, lambda value: None)
        return str(Path(kw["storage_path"]) / kw["study_name"])

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", one_trial)
    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config,
                                          n_trials=1, search_seed=0, auto_tensorboard=False)
    sweep = tt.sweep_observation(result["study_name"], project=tmp_path)
    (trial_dir,) = [d for d in sweep.directory.iterdir() if d.name.startswith("trial_")]
    trial = observe(trial_dir)

    assert sweep.record["objective"]["selection_metric"] != "loss"
    assert trial.record["resolved"]["objective"] == sweep.record["objective"]
    assert (run_summary(trial, [])["best_metric_name"]
            == sweep.record["objective"]["selection_metric"])


def test_a_context_for_an_ended_run_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    """A run whose final status is written takes no more writes: running it again refuses before
    anything is written, and a context built from its record refuses a save, leaving its
    directory byte-identical."""
    from tcip_mcp.experiments import RunEnded, observe
    from tcip_mcp.pipelines.training.subprocess_worker import prepare_run_context, run_directory
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-ended")
    before = _contents(run_dir)

    with pytest.raises(RunEnded):
        run_directory(run_dir)
    ctx = prepare_run_context(observe(run_dir))
    with pytest.raises(RunEnded):
        ctx.save_checkpoint({"model_state_dict": {}}, "after_terminal")

    assert _contents(run_dir) == before
