"""Subprocess isolation tests for launch_training and resource visibility and caps."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.data.selection import ClassScope


# ── persist the training run's class space ───────────────────────────


def _write_classes_json(dataset_root, subject="bud", attribute=None, values=None):
    from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

    attrs = ()
    if attribute:
        attrs = (Attribute(name=attribute, type="categorical", values=tuple(values)),)
    registry_over(dataset_root, SubjectRegistry((Subject(subject, attributes=attrs),)))


def _document_dataset(root, subject="bud", attribute=None, values=None):
    """Two images and their own label documents, under a registry naming ``subject``."""
    from pathlib import Path

    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    root = Path(root)
    images_dir, labels_dir = root / "images", root / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    _write_classes_json(root, subject=subject, attribute=attribute, values=values)
    attrs = {attribute: values[0]} if attribute else {}
    for stem in ("a", "b"):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=subject, geometry=BBox(4, 4, 20, 20), attributes=attrs)],
            32, 32, keep_empty=True)
    return images_dir, labels_dir


def test_the_class_space_recorded_is_the_one_the_run_admitted(tmp_path):
    """The checkpoint records the vocabulary the run trained in: the producer writes the scope it
    admitted under onto the run's own data config, and that is what is read back, so a
    ``subjects.json`` whose declared order changed after the admission cannot restamp the run
    with a vocabulary it never trained in."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = tmp_path / "plain"
    images_dir, labels_dir = _document_dataset(root, subject="bud")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "scope": {"subject": "bud"}, "split": {"val_ratio": 0.5, "seed": 1}}
    auto_train_val(tmp_path, "detection", data_cfg, None)

    assert ClassScope.of(data_cfg) == ClassScope("bud", ())

    _write_classes_json(root, subject="bud", attribute="opening", values=["open", "closed"])
    assert ClassScope.of(data_cfg) == ClassScope("bud", ())


def test_a_run_over_an_attributed_subject_records_its_attributes_in_declared_order(tmp_path):
    """A run's class space is its subject and every attribute the registry declares for it, its
    values in declared order, admitted once."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.subject_registry import Attribute

    images_dir, labels_dir = _document_dataset(
        tmp_path / "scoped", subject="bud", attribute="opening", values=["closed", "open"])
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "scope": {"subject": "bud"}, "split": {"val_ratio": 0.5, "seed": 1}}
    auto_train_val(tmp_path, "detection", data_cfg, None)

    assert ClassScope.of(data_cfg) == ClassScope(
        "bud", (Attribute("opening", "categorical", ("closed", "open")),))


def test_a_run_whose_ground_truth_carries_its_own_classes_records_the_empty_scope(tmp_path):
    """A mask raster scopes no class space and no registry answers for one, so the empty scope is
    stamped and decode falls through to its own derivation rather than to fabricated
    attributes."""
    from PIL import Image

    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    root = tmp_path / "masks"
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(images_dir / f"{stem}.png")
        Image.new("L", (32, 32), 1).save(masks_dir / f"{stem}.png")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"val_ratio": 0.5, "seed": 1}}
    auto_train_val(tmp_path, "semantic_seg", data_cfg, None)

    assert ClassScope.of(data_cfg) == ClassScope()


def test_the_launch_record_carries_the_resolution_and_the_child_resolves_nothing(tmp_path,
                                                                                 monkeypatch):
    """The launcher resolves a run once and writes what it resolved (data section, partition,
    objective) into its launch record; the child builds its context from that record alone,
    resolving nothing again and writing nothing beside it."""
    from tcip_mcp.experiments import RUN_FILE, observe
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import subprocess_worker as worker
    from tests._verified_checkpoint_fixtures import detection_config, opened_run, partition_side

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    launch_bytes = (run_dir / RUN_FILE).read_bytes()
    resolved = observe(run_dir).record["resolved"]
    assert set(resolved) == {"data", "partition", "objective"}
    files_before = sorted(p.name for p in run_dir.iterdir())

    def resolves_again(*args, **kwargs):
        raise AssertionError("the child resolved its run again")

    monkeypatch.setattr(sc, "auto_train_val", resolves_again)
    monkeypatch.setattr(sc, "resolve_run", resolves_again)

    ctx = worker.prepare_run_context(observe(run_dir))

    assert ctx.run.objective == resolved["objective"]
    assert ctx.run.config["data"] == resolved["data"]
    train_ds = ctx.train_loader.dataset
    assert sorted({train_ds.sample_of(key).member for key in train_ds.stems}) == partition_side(
        resolved["partition"], "train")
    assert (run_dir / RUN_FILE).read_bytes() == launch_bytes
    assert sorted(p.name for p in run_dir.iterdir()) == files_before


def test_launch_training_child_receives_its_own_run_directory(tmp_path, monkeypatch):
    """The child is told only its run directory, whose ``run.json`` the parent wrote before
    spawning it; a second launch naming the same id refuses before anything spawns. Mocks
    subprocess.Popen to capture argv without spawning a real child; preflight_config's smoke check
    still runs for real in this process, so a real (tiny) bespoke model/dataset is needed."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import subprocess

    from tcip_mcp.experiments import experiment_dir, observe
    from tcip_mcp.tools import training_tools

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    captured_argv: list[list[str]] = []

    class _FakeProc:
        pid = 424242

    def _fake_popen(argv, **kwargs):
        captured_argv.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    from tests._producer_fixtures import seed_two_bud_images, small_detection_config

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    seed_two_bud_images(images_dir, labels_dir)

    def _cfg(experiment_id: str) -> dict:
        return small_detection_config(images_dir, labels_dir, experiment_id)

    def children() -> list[list[str]]:
        return [argv for argv in captured_argv if "--run-dir" in argv]

    res = training_tools.launch_training(tmp_path, _cfg("exp_fresh"), actor=None)
    run_dir = experiment_dir("exp_fresh", project=tmp_path)
    assert res["experiment_id"] == "exp_fresh"
    assert children()[-1][-2:] == ["--run-dir", str(run_dir)]
    assert observe(run_dir).record["config"]["data"]["images_dir"] == str(images_dir)

    again = training_tools.launch_training(tmp_path, _cfg("exp_fresh"), actor=None)
    assert "already exists" in again["error"]
    assert len(children()) == 1


# ── should_cancel() / the cancellation record ──────────────────────


def test_ctx_should_cancel_and_dispatch_classification_honor_the_cancellation(tmp_path):
    """A bespoke train(ctx) that calls ctx.should_cancel() (the taught pattern) sees a
    cancellation requested through the run directory's own record, and dispatch_train_body
    classifies the resulting run as 'canceled', not 'completed'."""
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.run_registry import TrainRun

    run = TrainRun(id="run_ctx_cancel",
                   config={"training_source":
                           "tests.test_training_subprocess_isolation:_bespoke_loop"},
                   objective={"selection_metric": "loss", "higher_is_better": False},
                   project=tmp_path, output_dir=str(tmp_path))
    request_cancel(tmp_path)

    ctx = TrainContext(run=run, train_loader=None)
    assert ctx.should_cancel() is True

    dispatch_train_body(ctx)
    assert run.status == "canceled"


def _bespoke_loop(ctx) -> None:
    """Referenced by dotted path in the test above: a minimal train(ctx) that only checks
    ctx.should_cancel(), the exact taught pattern this test is pinning."""
    if ctx.should_cancel():
        return
    raise AssertionError("should_cancel() did not see the requested cancellation")


# ── run_summary ─────────────


def test_run_summary_reads_the_run_directory(tmp_path):
    """A live run's summary names the objective its launch recorded and carries no best while
    no row stamps a ``selection``."""
    from tcip_mcp.experiments import observe, read_rows, run_summary
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    log_epoch(run_dir, 3, {"loss": 0.1})

    observation = observe(run_dir)
    result = run_summary(observation, read_rows(observation.metrics_log)[0])
    assert result["status"] == "running"
    assert result["current_epoch"] == 3
    assert result["output_dir"] == str(run_dir)
    assert result["best_metric"] is None
    assert result["best_metric_name"] == (
        observation.record["resolved"]["objective"]["selection_metric"])


@pytest.mark.parametrize("higher_is_better, best", [(True, 0.7), (False, 0.5)],
                         ids=["higher-is-better", "lower-is-better"])
def test_the_live_summary_folds_its_rows_in_the_recorded_objectives_direction(
        tmp_path, higher_is_better, best):
    """The live summary's best is folded in the direction the run's launch record states, never
    one the summary looks up from the metric's name: the same rows give the highest under a
    higher-is-better objective and the lowest under a lower-is-better one."""
    from tcip_mcp.experiments import RUN_FILE, observe, read_rows, run_summary
    from tcip_store import decode_value, encode_record
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    record = decode_value((run_dir / RUN_FILE).read_bytes())
    record["resolved"]["objective"] = {"selection_metric": "map50",
                                       "higher_is_better": higher_is_better}
    (run_dir / RUN_FILE).write_bytes(encode_record(record))
    log_epoch(run_dir, 1, {"selection": 0.5, "selection_metric": "map50"})
    log_epoch(run_dir, 2, {"selection": 0.7, "selection_metric": "map50"})
    log_epoch(run_dir, 3, {"selection": float("nan"), "selection_metric": "map50"})

    observation = observe(run_dir)
    result = run_summary(observation, read_rows(observation.metrics_log)[0])
    assert (result["best_metric_name"], result["best_metric"]) == ("map50", best)


def test_run_summary_surfaces_the_wall_clock_failure_the_child_wrote(tmp_path):
    """A run past its deadline stops at its next cancel poll and its own final status names the
    wall clock; no second writer is involved."""
    from tcip_mcp.experiments import observe, run_summary
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(
        tmp_path, training_source="tests.test_training_subprocess_isolation:_bespoke_loop",
        wall_clock_passed=True)

    result = run_summary(observe(run_dir), [])
    assert result["status"] == "failed"
    assert result["error"] == "exceeded max_wall_clock_seconds"


def test_a_canceled_run_reads_canceled_whatever_its_heartbeat(tmp_path, monkeypatch):
    """A final status is the answer once written: a canceled run never reads running or
    interrupted, however stale its heartbeat."""
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import finished_run

    canceled = finished_run(
        tmp_path, training_source="tests.test_training_subprocess_isolation:_bespoke_loop",
        cancel_requested=True)

    assert experiments.observe(canceled).state == "canceled"
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert experiments.observe(canceled).state == "canceled"


# ── monitor_training / list_experiments(launched_only=True) read the directory ─────


def test_monitor_training_reads_the_run_directory(tmp_path, monkeypatch):
    from tcip_mcp.tools.training_tools import monitor_training
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"),
                         experiment_id="run_delegated")
    log_epoch(run_dir, 7, {"loss": 0.2})

    result = monitor_training(tmp_path, "run_delegated")
    assert result["epoch"] == 7
    assert result["status"] == "running"


def test_launched_runs_view_lists_a_run_read_from_its_directory(tmp_path):
    """A run any process launched lists, read straight from its own directory, carrying the
    directory's last epoch and last sign of life."""
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(
        tmp_path / "data", model_source={"builder": "my_models:chestnut_burr_det",
                                         "task": "detection"}),
        experiment_id="exp-no-stamp")
    log_epoch(run_dir, 9, {"loss": 0.1})

    by_id = {r["experiment_id"]: r for r in list_experiments(tmp_path, launched_only=True)["runs"]}
    assert by_id["exp-no-stamp"]["status"] == "running"
    assert by_id["exp-no-stamp"]["current_epoch"] == 9
    assert by_id["exp-no-stamp"]["heartbeat"] is not None


# ── GPU device pinning ───────────────────────────────────────────────────


def test_gpu_device_pinning_round_robins(monkeypatch):
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    seen = set()
    for _ in range(4):
        env = training_tools._child_env_for_launch({})
        seen.add(env["CUDA_VISIBLE_DEVICES"])
    assert seen == {"0", "1"}


def test_gpu_pinning_skipped_when_device_explicit(monkeypatch):
    """CUDA_VISIBLE_DEVICES remaps indices inside the child: pinning it when the config already
    names an explicit device would ask for an ordinal invalid in the child's own remapped view.
    Must be a no-op whenever the config already knows which device it wants."""
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    env = training_tools._child_env_for_launch({"device": "cuda:1"})
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_gpu_pinning_noop_with_single_gpu(monkeypatch):
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    env = training_tools._child_env_for_launch({})
    assert "CUDA_VISIBLE_DEVICES" not in env


# ── wall-clock timeout ──────────────────────────────────────────────


def test_max_wall_clock_seconds_terminates_a_hung_child_one_heartbeat_window_late(monkeypatch):
    """A child that has not stopped itself one heartbeat window past its wall clock is hung: the
    watcher terminates it and writes nothing, so its directory reads interrupted."""
    import subprocess
    import sys
    import time

    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import _watch_wall_clock

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", 0.2)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _watch_wall_clock(proc, 0.2)
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "watcher did not terminate the hung process"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── inspect_compute_resources ────────────────────────────────────────────────────────


def test_inspect_compute_resources_degrades_without_psutil(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated absent psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)

    from tcip_mcp.tools.training_tools import inspect_compute_resources
    result = inspect_compute_resources(tmp_path)
    assert result["cpu"]["percent_used"] is None
    assert result["memory"]["total_bytes"] is None
    assert result["memory"]["available_bytes"] is None
    assert isinstance(result["gpus"], list)  # still populates via torch alone
    assert "active_training_runs" in result


def test_inspect_compute_resources_reports_gpu_free_memory(tmp_path, monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(idx):
            return (1_000, 4_000)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    from tcip_mcp.tools.training_tools import inspect_compute_resources
    result = inspect_compute_resources(tmp_path)
    assert result["gpus"] == [{"index": 0, "free_bytes": 1_000, "total_bytes": 4_000}]


def test_inspect_compute_resources_counts_a_running_run_directory(tmp_path):
    """A run whose directory is fresh counts as active, whichever process launched it: the
    directory alone is enough."""
    from tcip_mcp.tools.training_tools import inspect_compute_resources
    from tests._verified_checkpoint_fixtures import opened_run

    baseline = inspect_compute_resources(tmp_path)["active_training_runs"]

    from tests._verified_checkpoint_fixtures import detection_config

    opened_run(tmp_path, detection_config(tmp_path / "data"), experiment_id="run_other_process")

    assert inspect_compute_resources(tmp_path)["active_training_runs"] == baseline + 1


# ── HPO resource caps ─────────────────────────────────────────────────────────


def test_default_trial_resources_derives_fractional_gpu_share(monkeypatch):
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    assert hpo._default_trial_resources(max_concurrent=1) == {"cpu": 1.0, "gpu": 1.0}
    assert hpo._default_trial_resources(max_concurrent=4) == {"cpu": 1.0, "gpu": 0.5}


def test_default_trial_resources_no_gpu(monkeypatch):
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    assert hpo._default_trial_resources(max_concurrent=1) == {"cpu": 1.0, "gpu": 0.0}


@pytest.mark.ray_cluster
def test_tune_search_accepts_explicit_resources_per_trial(tmp_path):
    """A real, lightweight Ray sweep (a pure-math objective, no training) runs to its end with an
    explicit resources_per_trial, its log directory under the caller's storage_path."""
    from pathlib import Path

    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    def obj(config, report):
        report((config["x"] - 2.0) ** 2)

    logdir = tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        metric="objective", mode="min", num_samples=4,
        search_alg="random", scheduler="none",
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
        storage_path=str(tmp_path), seed=0, project=tmp_path
    )
    assert Path(logdir).is_dir()
    assert Path(logdir).is_relative_to(tmp_path)


@pytest.mark.ray_cluster
def test_tune_search_runs_despite_deprecated_ray_result_dir_variables(tmp_path, monkeypatch):
    """Ray Tune refuses to run while TUNE_RESULT_DIR or RAY_AIR_LOCAL_CACHE_DIR is set anywhere
    in the environment, even though storage_path alone decides where trial results land. A
    machine that still carries such a redirect must get a working sweep, stored under the
    caller's storage_path, with the variables left in the environment exactly as they were."""
    import os

    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    machine_scratch = str(tmp_path / "machine_scratch")
    monkeypatch.setenv("TUNE_RESULT_DIR", machine_scratch)
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", machine_scratch)

    def obj(config, report):
        report((config["x"] - 2.0) ** 2)

    logdir = tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        metric="objective", mode="min", num_samples=2,
        search_alg="random", scheduler="none",
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
        storage_path=str(tmp_path / "sweep_store"), seed=0, project=tmp_path
    )
    from pathlib import Path

    assert Path(logdir).is_relative_to(tmp_path / "sweep_store")
    assert os.environ["TUNE_RESULT_DIR"] == machine_scratch
    assert os.environ["RAY_AIR_LOCAL_CACHE_DIR"] == machine_scratch


def test_tune_search_refuses_to_run_without_a_storage_path(tmp_path):
    """Trial results land where the caller says; with no storage_path Ray would fall back to a
    home-directory default outside any project, so the call refuses and names the resolver."""
    from tcip_mcp.pipelines.training.hpo import tune_search

    with pytest.raises(ValueError, match="storage_path"):
        tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}}, seed=0,
            project=tmp_path,
        )
