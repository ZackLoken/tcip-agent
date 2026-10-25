"""A run is a directory, and the state it reads as comes from what its own process wrote there.

The launched cases go through the real launcher (``launch_training``) and a real child process,
so the facts are the ones a run directory carries across the process boundary: the launch record
the parent writes, the final status the child writes, and the heartbeat. The HPO trial cases run
the sweep's own trial body (``_run_hpo_trial``) in this process.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.experiments import FinalStatus
    from tcip_mcp.pipelines.training.envelope import TrainContext

pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tests._training_values import (  # noqa: E402
    drawn_split,
    fifo_search,
    sweep_space,
)

ONE_RATE = {"optimizer.backbone_lr": 0.01, "optimizer.head_lr": 0.01}
"""A trial point setting the backbone and head rates to one value, 0.01."""


@pytest.fixture
def children(monkeypatch: pytest.MonkeyPatch) -> list:
    """Return the training children the launches spawned, each a real process.

    A training child is a process handed ``--run-dir``.
    """
    spawned: list = []
    real_popen = subprocess.Popen

    class _Recorded(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            if "--run-dir" in (args[0] if args else kwargs.get("args", ())):
                spawned.append(self)

    monkeypatch.setattr(subprocess, "Popen", _Recorded)
    return spawned


@pytest.fixture
def launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, children: list
) -> Callable[..., dict]:
    """Return a ``launch_training`` caller in the project ``tmp_path`` over a tiny dataset.

    Each launch is over the regression dataset written under ``tmp_path / "ds"``, drawing 0.15
    onto validation, at ``horizon`` (1 by default), with ``extra``'s keys passed to the config
    and ``resume_from`` to the launch; TensorBoard is not started.
    """
    from tcip_mcp.tools.training_tools import launch_training
    from tests._producer_fixtures import no_tensorboard
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    no_tensorboard(monkeypatch)
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds",
        intensities=[0.0, 0.25, 0.5, 1.0],
        values=[0.1, 0.3, 0.5, 0.9],
    )

    def _launch(
        horizon: int = 1, *, resume_from: str = "", **extra: object
    ) -> dict:
        config = regressor_config(
            horizon,
            data={
                "images_dir": str(images_dir),
                "labels_dir": str(csv_path),
                "split": drawn_split(val_ratio=0.15),
            },
            project=tmp_path,
            **extra,
        )
        return launch_training(
            tmp_path, config, resume_from=resume_from, actor=None
        )

    return _launch


def _until[T](
    what: str, answer: Callable[[], T | None], deadline_s: float = 120
) -> T:
    """Poll ``answer()`` every 0.1 s and return its first truthy value, raising past ``deadline_s``.

    The ``AssertionError`` raised names ``what``.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        value = answer()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"{what} did not happen within {deadline_s}s")


def _wait_final(run_dir: Path) -> FinalStatus:
    """Wait for the final status the run's child writes and return it."""
    from tcip_mcp.experiments import observe

    return _until(
        f"{run_dir.name}'s final status", lambda: observe(run_dir).final
    )


def _contents(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_a_launched_run_completes_and_its_completion_is_its_registration(
    launch: Callable[..., dict], tmp_path: Path
) -> None:
    from tcip_mcp.experiments import METRICS_FILE, RUN_FILE, observe, read_rows
    from tcip_mcp.model_registry import (
        ModelRegistry,
        load_registered_checkpoint,
    )

    res = launch()
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])

    final = _wait_final(run_dir)
    assert final.state == "completed", final
    observation = observe(run_dir)
    assert observation.state == "completed"
    assert (run_dir / RUN_FILE).is_file()
    assert read_rows(run_dir / METRICS_FILE)[0]
    checkpoint = observation.checkpoint
    assert checkpoint is not None and Path(checkpoint.path).is_file()
    assert final.checkpoint is not None
    assert final.checkpoint.sha256 == checkpoint.sha256
    (listed,) = [
        m
        for m in ModelRegistry(str(tmp_path)).list_models()
        if m.entry.name == run_dir.name
    ]
    assert listed.entry.sha256 == checkpoint.sha256
    loaded = load_registered_checkpoint(checkpoint.path, project=tmp_path)
    assert loaded.producer.experiment_id == run_dir.name


def test_a_cancel_requested_by_id_ends_the_child_canceled(
    launch: Callable[..., dict], tmp_path: Path
) -> None:
    """A cancel requested by id ends a training child canceled, completing no checkpoint.

    The cancel is requested after the child logs its first epoch row.
    """
    from tcip_mcp.experiments import METRICS_FILE, observe, read_rows
    from tcip_mcp.tools.training_tools import cancel_training

    res = launch(horizon=200)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    _until(
        "the child's first epoch row",
        lambda: read_rows(run_dir / METRICS_FILE)[0],
    )

    assert (
        cancel_training(tmp_path, res["experiment_id"], actor=None)[
            "cancel_requested"
        ]
        is True
    )
    final = _wait_final(run_dir)

    assert final.state == "canceled", final
    assert observe(run_dir).checkpoint is None


def test_a_resume_is_a_new_directory_that_leaves_its_source_untouched(
    launch: Callable[..., dict], tmp_path: Path
) -> None:
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.registry_paths import stored_path

    first = launch(horizon=2, checkpoint_every_n_epochs=1)
    source = Path(first["output_dir"])
    assert _wait_final(source).state == "completed"
    epoch_checkpoints = sorted(source.glob("checkpoint_epoch_*.pt"))
    assert epoch_checkpoints, sorted(p.name for p in source.iterdir())
    before = _contents(source)

    resumed = launch(horizon=3, resume_from=str(epoch_checkpoints[-1]))
    assert "error" not in resumed, resumed
    run_dir = Path(resumed["output_dir"])
    assert run_dir != source
    assert _wait_final(run_dir).state == "completed"

    stored = read_record(run_dir / RUN_FILE)
    assert isinstance(stored, dict)
    assert stored["resume_from"] == stored_path(
        epoch_checkpoints[-1], tmp_path
    )
    assert (run_dir / "model_final.pt").is_file()
    assert _contents(source) == before


def test_a_child_killed_before_its_final_status_reads_running_then_interrupted(
    launch: Callable[..., dict],
    children: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child killed while it runs reads running, then interrupted.

    Killed while it runs, the child writes no final status: after the process has exited the
    run reads running while its last sign of life is fresh, and interrupted when that sign of
    life is older than the heartbeat window.
    """
    from tcip_mcp import experiments

    res = launch(horizon=200)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    [child] = children
    child.kill()
    child.wait()

    assert experiments.observe(run_dir).final is None
    assert experiments.observe(run_dir).state == "running"
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert experiments.observe(run_dir).state == "interrupted"


def test_progress_logged_after_the_final_status_never_reopens_the_run(
    launch: Callable[..., dict], tmp_path: Path
) -> None:
    """A metrics row logged after the final status is refused and the run stays completed.

    After the child wrote its final status the run's metrics writer refuses a row, leaving the
    log and the final status as they were; the completed run's summary is the best selection over
    its own rows under its recorded objective.
    """
    from tcip_mcp.experiments import (
        FINAL_STATUS_FILE,
        METRICS_FILE,
        RunEndedError,
        best_selection,
        observe,
        read_rows,
    )
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import monitor_training

    res = launch(horizon=2)
    run_dir = Path(res["output_dir"])
    assert _wait_final(run_dir).state == "completed"
    before = {
        name: (run_dir / name).read_bytes()
        for name in (FINAL_STATUS_FILE, METRICS_FILE)
    }
    observation = observe(run_dir)
    run = observed_run(observation)

    with pytest.raises(RunEndedError):
        TrainContext(run=run, train_loader=None).log_metrics(
            99, {"loss": 1e-9, "selection_metric": "loss", "selection": 1e-9}
        )

    assert {name: (run_dir / name).read_bytes() for name in before} == before
    rows = read_rows(observation.metrics_log)[0]
    summary = monitor_training(tmp_path, res["experiment_id"])["run"]
    assert summary["state"] == "completed"
    assert summary["best_metric"] is not None
    assert observation.record.spec.evaluation.selection_metric == "loss"
    assert summary["best_metric"] == best_selection(rows, False)


def test_a_reader_never_sees_a_final_status_before_its_bytes_are_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader never sees a final status before its bytes are whole.

    Read by a spy inside the write of the final status, the run has no final status and reads
    running; after the write it reads failed.
    """
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    seen: list = []
    real_publish = experiments.publish_once

    def observed_mid_write(
        path: Path, write: Callable[[BinaryIO], object]
    ) -> None:
        def write_after_reading(handle: BinaryIO) -> None:
            seen.append(experiments.observe(run_dir))
            write(handle)

        real_publish(path, write_after_reading)

    monkeypatch.setattr(experiments, "publish_once", observed_mid_write)
    experiments.write_final_status(run_dir, "failed", "stopped by hand")

    [during] = seen
    assert (during.final, during.state) == (None, "running")
    assert experiments.observe(run_dir).state == "failed"


def _refusing_final_status(
    monkeypatch: pytest.MonkeyPatch, times: float, winerror: int | None = None
) -> list[Path]:
    """Make the next ``times`` final-status reads fail, and return the paths refused, in order.

    Each refused read raises a ``PermissionError`` carrying ``winerror`` (by default a Windows
    sharing violation).
    """
    from tcip_mcp.experiments import FINAL_STATUS_FILE
    from tcip_store.file_backend import SHARING_VIOLATION

    code = SHARING_VIOLATION if winerror is None else winerror

    refused: list[Path] = []
    real_read = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self.name == FINAL_STATUS_FILE and len(refused) < times:
            refused.append(self)
            exc = PermissionError(
                13, "The process cannot access the file", str(self)
            )
            exc.winerror = code  # type: ignore[misc]
            raise exc
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return refused


def test_a_final_status_a_sharing_violation_refuses_once_is_read_on_the_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final status refused once by a sharing violation is read on the retry.

    The run observes as completed after one refused read.
    """
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-shared-status")
    refused = _refusing_final_status(monkeypatch, times=1)

    assert observe(run_dir).state == "completed"
    assert len(refused) == 1


def test_a_final_status_a_sharing_violation_keeps_refusing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sharing violation that outlasts the retry budget is raised after more than one read."""
    import math

    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-locked-status")
    refused = _refusing_final_status(monkeypatch, times=math.inf)

    with pytest.raises(PermissionError):
        observe(run_dir)
    assert len(refused) > 1


def test_a_final_status_denied_for_another_reason_raises_without_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final-status read denied for a reason other than sharing raises without a retry.

    A read denied with Windows' access-denied code (5) raises on its first refusal.
    """
    import math

    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-denied-status")
    refused = _refusing_final_status(monkeypatch, times=math.inf, winerror=5)

    with pytest.raises(PermissionError):
        observe(run_dir)
    assert len(refused) == 1


def test_no_retry_starts_after_the_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retry of a denied action starts after the retry budget is spent.

    An action denied until past the 0.12 s budget starts at 0, 0.051 and 0.102 s only: the wait
    that wakes late, past the deadline, ends with the denial raised.
    """
    from tcip_store import file_backend

    clock = {"now": 0.0}

    def sleep_waking_late(seconds: float) -> None:
        clock["now"] += seconds + 0.001

    monkeypatch.setattr(file_backend.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(file_backend.time, "sleep", sleep_waking_late)
    monkeypatch.setattr(file_backend.random, "uniform", lambda lo, hi: 0.05)
    started: list[float] = []

    def action() -> str:
        started.append(clock["now"])
        if clock["now"] > 0.12:
            return "released"
        raise PermissionError(13, "denied")

    with pytest.raises(PermissionError):
        file_backend.retry_while_denied(action, 0.12)
    assert started == pytest.approx([0.0, 0.051, 0.102])


def test_a_completed_run_missing_its_checkpoint_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """Loading a completed run's checkpoint whose file is gone refuses naming the file.

    With the checkpoint file the run recorded deleted, ``checkpoint_payload`` raises
    ``FileNotFoundError`` naming it.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import checkpoint_payload
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = Path(
        registered_checkpoint(tmp_path, experiment_id="exp-lost-weights")
    )
    named = observe(checkpoint.parent).checkpoint
    assert named is not None
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match=checkpoint.name):
        checkpoint_payload(named.path, named.sha256)


def test_a_config_naming_its_run_refuses_naming_the_key_and_writes_nothing(
    launch: Callable[..., dict], tmp_path: Path
) -> None:
    """A config stating ``experiment_id`` is refused naming the key and makes no directory.

    A launch with no ``experiment_id`` names its directory by the id it answers; a config stating
    that id or a name of its own refuses naming the key, and no directory is made.
    """
    from tcip_mcp.experiments import run_dirs

    first = launch()
    run_dir = Path(first["output_dir"])
    assert run_dir.name == first["experiment_id"]
    _wait_final(run_dir)
    before = _contents(run_dir)

    for named in (first["experiment_id"], "a-name-of-my-own"):
        refused = launch(experiment_id=named)
        assert "config.experiment_id" in refused["error"], refused
    assert _contents(run_dir) == before
    assert run_dirs(tmp_path) == [run_dir]


def test_a_name_is_reserved_by_the_directory_that_takes_it_first(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    """A name is reserved by the directory that takes it first.

    A directory created with no record written already holds its name: a second creation under
    it refuses, and a run under a sweep's name refuses naming its directory. A run, a sweep and
    one of its trials are each found by name.
    """
    from tcip_mcp.experiments import (
        RunDirectoryExistsError,
        create_run_directory,
        experiment_dir,
        find_run,
        find_sweep,
        named_directory,
    )
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
        opened_sweep,
    )

    composing = create_run_directory(
        experiment_dir("composing", project=tmp_path)
    )
    with pytest.raises(
        RunDirectoryExistsError, match=re.escape(str(composing))
    ):
        create_run_directory(composing)

    sweep = opened_sweep(tmp_path, real_hpo_base_config)
    trial = open_trial(sweep, "a", {"optimizer.head_lr": 0.01})
    config = detection_config(tmp_path.parent / "run-data", project=tmp_path)
    with pytest.raises(RunDirectoryExistsError, match=re.escape(str(sweep))):
        opened_run(tmp_path, config, experiment_id=sweep.name)
    run = opened_run(tmp_path, config, experiment_id="a-free-name")

    assert (
        find_run(run.name, project=tmp_path),
        find_run(trial.name, project=tmp_path),
        find_sweep(sweep.name, project=tmp_path),
    ) == (run, trial, sweep)
    assert [
        named_directory(d.name, project=tmp_path) for d in (run, trial, sweep)
    ] == [run, trial, sweep]


def test_the_live_run_conflict_reads_the_projects_one_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live-run conflict reads the project's one training listing.

    A running run is the conflict the listing shows it as, and a listing showing nothing
    running is no conflict.
    """
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    opened_run(
        tmp_path,
        detection_config(tmp_path.parent / "run-data", project=tmp_path),
        experiment_id="live-run",
    )
    assert "live-run" in (experiments.live_run_conflict(tmp_path) or "")

    monkeypatch.setattr(
        experiments,
        "training_listing",
        lambda project: experiments.TrainingListing(runs=[], sweeps=[]),
    )
    assert experiments.live_run_conflict(tmp_path) is None


def _trial(
    tmp_path: Path, point: dict | None = None, **extra: object
) -> tuple[Path, list[float]]:
    """Run one HPO trial and return its run directory and the values it reported, in order.

    The trial ``a`` is of a sweep over a regression dataset written under ``tmp_path``, drawing
    0.34 onto validation at seed 3, at ``point`` (by default :data:`ONE_RATE`), run through the
    sweep's own trial body (``_run_hpo_trial``); the model source lists this module beside the
    regressor's files, and ``extra`` passes further config keys.
    """
    from tcip_mcp.experiments import run_dirs
    from tcip_mcp.tools.training_tools import _run_hpo_trial
    from tests._verified_checkpoint_fixtures import opened_sweep
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds",
        intensities=[0.0, 0.25, 0.5, 0.75, 1.0, 0.1],
        values=[0.1, 0.3, 0.5, 0.7, 0.9, 0.2],
    )
    base_config = regressor_config(
        2,
        data={
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.34, seed=3),
        },
        project=tmp_path,
        **extra,
    )
    base_config["model_source"]["source_files"].append(__file__)
    sweep = opened_sweep(tmp_path, base_config)
    reported: list[float] = []
    _run_hpo_trial(point or dict(ONE_RATE), reported.append, sweep, "a")
    (trial_dir,) = run_dirs(tmp_path)
    return trial_dir, reported


def _trial_row(project: Path, trial_dir: Path) -> dict:
    """Return the row the project's listing reads for ``trial_dir``, under its sweep."""
    from tcip_mcp.experiments import training_listing

    (sweep,) = training_listing(project).sweeps
    return next(
        row for row in sweep.trials if row.experiment_id == trial_dir.name
    ).model_dump()


def test_an_hpo_trial_is_a_run_directory_reporting_the_sweeps_one_selection(
    tmp_path: Path,
) -> None:
    """An HPO trial is a run directory reporting its config's one selection metric.

    A trial is a run directory like any launched run's, its sampled point and what it resolved
    on its launch record, and it reports its config's selection each epoch; its result is the
    best of what it reported, in that metric's direction (``loss``, lower better).
    """
    from tcip_mcp.experiments import observe, read_run

    trial_dir, reported = _trial(tmp_path)

    record = read_run(trial_dir)
    assert record.trial_params == ONE_RATE
    assert record.resolution.partition.samples
    assert record.resolution.trait_revision is None
    assert record.spec.evaluation.selection_metric == "loss"
    assert observe(trial_dir).state == "completed"
    assert len(reported) == 2
    assert _trial_row(tmp_path, trial_dir)["best_metric"] == min(reported)


def test_a_trial_whose_resolution_fails_is_a_directory_whose_final_status_names_it(
    tmp_path: Path,
) -> None:
    """A trial whose resolution fails is a directory whose final status names why.

    A trial whose sampled point cannot resolve still opens its run directory, and its final
    status is ``failed`` naming why; the sweep's listing reads it as a row with no best value
    and no metric name.
    """
    from tcip_mcp.experiments import observe_run

    trial_dir, reported = _trial(
        tmp_path, {"data.labels_dir": str(tmp_path / "gone")}
    )

    final = observe_run(trial_dir).final
    assert final is not None
    assert final.state == "failed" and final.status_error
    assert reported == []
    row = _trial_row(tmp_path, trial_dir)
    assert (
        row["state"],
        row["status_error"],
        row["best_metric"],
        row["best_metric_name"],
    ) == ("failed", final.status_error, None, None)
    with pytest.raises(ValueError, match="resolved to nothing"):
        observe_run(trial_dir).record.resolution.trait_revision


def _reports_only(ctx: TrainContext) -> None:
    """Report 0.25 through ``ctx.report_objective`` and save ``model_final``.

    Pass ``{STATE_DICT_KEY: ctx.build_model().state_dict()}`` to ``ctx.save_checkpoint`` under
    ``model_final``.
    """
    ctx.report_objective(0.25)
    ctx.save_checkpoint(
        {STATE_DICT_KEY: ctx.build_model().state_dict()}, "model_final"
    )


def test_a_report_only_trial_projects_the_value_it_reported(
    tmp_path: Path,
) -> None:
    """A trial reporting through ``report_objective`` alone projects that value as its result.

    The trial completes, reports 0.25, and its listing row's best metric is 0.25.
    """
    from tcip_mcp.experiments import observe

    trial_dir, reported = _trial(
        tmp_path, training_source=f"{Path(__file__).stem}:_reports_only"
    )

    assert observe(trial_dir).state == "completed"
    assert reported == [0.25]
    assert _trial_row(tmp_path, trial_dir)["best_metric"] == 0.25


RESOLVED_KEYS = {"data", "geometry", "partition", "trait_revision"}
"""What a run's launch record says its launch resolved: no selection metric, no direction."""


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_runs_and_a_sweeps_records_carry_no_metric_and_every_reader_answers_the_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_hpo_base_config: dict,
) -> None:
    """Run and sweep records carry no selection metric, and their readers answer the config's.

    A run opened through the launcher's own producer (``run_context``) and a sweep
    opened by ``open_sweep`` with its one trial record no selection metric and no direction:
    their resolved keys are the data, geometry, partition and trait revision, the sweep's record
    holds its creation time, trait revision and input, and the trial's revision is the sweep's.
    The run's summary and the selection row its body reports, the trial's summary, and the
    sweep's search direction and outcome each answer the config's own metric. The trial's body is
    not run.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import (
        METRICS_FILE,
        find_sweep,
        observe,
        observe_run,
        observe_sweep,
        read_rows,
        read_sweep,
        run_dirs,
        run_rows,
        run_summary,
    )
    from tcip_mcp.pipelines.schemas import selection_of
    from tcip_mcp.pipelines.training import subprocess_worker
    from tests import _trait_fixtures as fx
    from tests._training_values import evaluation_block
    from tests.tiny_trainer_fixtures import (
        TRAIN_INTENSITIES,
        VAL_INTENSITIES,
        opposed_regression_data,
        regressor_config,
        run_context,
    )

    single = tmp_path / "single"
    ctx = run_context(
        single,
        regressor_config(
            data=opposed_regression_data(
                single / "ds", TRAIN_INTENSITIES, VAL_INTENSITIES
            ),
            project=single,
            evaluation=evaluation_block(selection_metric="mae"),
        ),
    )
    assert (
        set(observe_run(ctx.run_dir).record.resolution.model_dump())
        == RESOLVED_KEYS
    )
    ctx.report_objective(0.25)
    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    assert rows[-1]["selection_metric"] == "mae"
    assert (
        run_summary(observe(ctx.run_dir), rows, None).best_metric_name == "mae"
    )

    monkeypatch.setattr(
        subprocess_worker, "run_directory", lambda *a, **k: None
    )
    modes: list[str] = []

    def one_trial(
        **kw: Callable[[dict, Callable[[float], None]], object]
        | dict
        | str
        | int
        | Path
        | None,
    ) -> None:
        modes.append(kw["mode"])
        # A newer revision confirmed once the sweep opened: the trial still records the sweep's.
        fx.propose_and_confirm(
            tmp_path,
            fx.with_fields(
                fx.latest("bud_opening", tmp_path),
                notes="A newer statement of the trait.",
            ),
        )
        kw["objective_fn"]({}, lambda value: None)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", one_trial
    )
    base_config = {
        **real_hpo_base_config,
        "evaluation": {
            **real_hpo_base_config["evaluation"],
            "trait": "bud_opening",
        },
    }
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=base_config,
        param_space=sweep_space(),
        n_trials=1,
        search_seed=0,
        **fifo_search(),
        auto_tensorboard=False,
        actor=None,
    )
    sweep_dir = find_sweep(result["sweep"]["sweep_id"], project=tmp_path)
    (trial_dir,) = run_dirs(tmp_path)
    trial = observe_run(trial_dir)
    metric, higher = selection_of(
        real_hpo_base_config["evaluation"]["selection_metric"]
    )

    sweep = observe_sweep(sweep_dir).record
    assert set(sweep.model_dump()) == {
        "created",
        "trait_revision",
        "input",
    }
    assert set(trial.record.resolution.model_dump()) == RESOLVED_KEYS
    swept = sweep.trait_revision
    assert swept is not None
    assert trial.record.resolution.trait_revision == swept
    assert run_summary(trial, [], None).best_metric_name == metric
    assert modes == ["max" if higher else "min"]
    assert (
        "objective"
        not in read_sweep(sweep_dir, run_rows(tmp_path)).model_dump()
    )


def test_a_context_for_an_ended_run_is_refused_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context for an ended run is refused and writes nothing.

    For a run whose final status is written, running it again refuses and a context built from
    its record refuses a save, leaving its directory byte-identical.
    """
    from tcip_mcp.experiments import RunEndedError, observe
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
        run_directory,
    )
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-ended")
    before = _contents(run_dir)

    with pytest.raises(RunEndedError):
        run_directory(run_dir)
    ctx = prepare_run_context(observe(run_dir))
    with pytest.raises(RunEndedError):
        ctx.save_checkpoint({STATE_DICT_KEY: {}}, "after_terminal")

    assert _contents(run_dir) == before
