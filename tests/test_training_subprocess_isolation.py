"""Subprocess isolation tests for launch_training and resource visibility and caps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.schemas import train_config
from tests import REPO_ROOT
from tests._chain_fixtures import (
    FRAME_STATISTICS,
    training_config,
    weights_statistics,
)
from tests._training_values import drawn_split, tune_arguments

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType
    from typing import NoReturn

    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.subject_registry import Attribute


def _config(project: Path, data: dict | None = None, **keys: object) -> dict:
    """Return a training config of ``project`` for a detection model.

    The config is a :func:`~tests._chain_fixtures.training_config` of ``unbuilt_source``'s
    detector with this module added to its source files, with ``keys`` in place of theirs:
    over the data section ``data``, or with no data over its weights' statistics
    (``_chain_fixtures.weights_statistics``).
    """
    from tests._verified_checkpoint_fixtures import unbuilt_source

    source = unbuilt_source("detection")
    declared = {**source, "source_files": [*source["source_files"], __file__]}
    return training_config(
        project,
        declared if data else weights_statistics(declared),
        data or {},
        **keys,
    )


# ── persist the training run's class space ───────────────────────────


def _document_dataset(
    root: Path, subject: str, *attributes: Attribute
) -> Path:
    """Write two images and their own label documents under a registry naming ``subject``.

    The registry declares ``attributes`` on ``subject``, and each image's one annotation carries
    each attribute's first value. Returns the images directory.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import (
        label_image,
        registry_of,
        registry_over,
        write_image,
    )

    images_dir = image_dir(root, UNDATED_BUCKET)
    registry_over(root, registry_of(subject, *attributes))
    attrs = {a.name: a.values[0] for a in attributes}
    for stem in ("a", "b"):
        label_image(
            write_image(images_dir / f"{stem}.png", (32, 32), (10, 20, 30)),
            [
                Annotation(
                    subject=subject,
                    geometry=BBox(4, 4, 20, 20),
                    attributes=attrs,
                )
            ],
            32,
            32,
            keep_empty=True,
        )
    return images_dir


def _resolved_data(project: Path, task: str, data_cfg: dict) -> DataSpec:
    """Return the data block a ``task`` run of ``project`` over ``data_cfg`` resolves.

    The block is resolved through ``split_construction.auto_train_val``.
    """
    from tests._producer_fixtures import train_val

    return train_val(project, task, data_cfg)[3]


def test_the_class_space_recorded_is_the_one_the_run_admitted(
    tmp_path: Path,
) -> None:
    """The resolved data block's recorded scope is the one the run admitted.

    The scope reads ``bud`` with no attributes, and still does after ``subjects.json`` is
    rewritten to declare an attribute.
    """
    root = tmp_path / "plain"
    images_dir = _document_dataset(root, "bud")
    resolved = _resolved_data(
        tmp_path,
        "detection",
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(seed=1),
        },
    )

    assert resolved.recorded_scope == ClassScope("bud", ())

    from tcip_mcp.subject_registry import Attribute
    from tests._producer_fixtures import registry_of, registry_over

    registry_over(
        root,
        registry_of(
            "bud", Attribute("opening", "categorical", ("open", "closed"))
        ),
    )
    assert resolved.recorded_scope == ClassScope("bud", ())


def test_a_run_over_an_attributed_subject_records_its_attributes_in_declared_order(
    tmp_path: Path,
) -> None:
    """A run over an attributed subject records its attributes in declared order.

    A run's class space is its subject and the attribute the registry declares for it, its
    values in declared order.
    """
    from tcip_mcp.subject_registry import Attribute

    opening = Attribute("opening", "categorical", ("closed", "open"))
    images_dir = _document_dataset(tmp_path / "scoped", "bud", opening)
    resolved = _resolved_data(
        tmp_path,
        "detection",
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(seed=1),
        },
    )

    assert resolved.recorded_scope == ClassScope("bud", (opening,))


def test_a_run_whose_ground_truth_carries_its_own_classes_records_the_empty_scope(
    tmp_path: Path,
) -> None:
    """A run whose ground truth carries its own classes records the empty scope.

    A mask raster scopes no class space and no registry answers for one, so the empty scope is
    stamped.
    """
    from PIL import Image

    from tests._producer_fixtures import write_image

    root = tmp_path / "masks"
    images_dir, masks_dir = image_dir(root, UNDATED_BUCKET), root / "masks"
    masks_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        write_image(images_dir / f"{stem}.png", (32, 32), (10, 20, 30))
        Image.new("L", (32, 32), 1).save(masks_dir / f"{stem}.png")
    resolved = _resolved_data(
        tmp_path,
        "semantic_seg",
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(seed=1),
        },
    )

    assert resolved.recorded_scope == ClassScope()


def test_the_launch_record_carries_the_resolution_and_the_child_resolves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch record carries the resolution, and the child resolves nothing.

    The launcher writes what it resolved (data section, geometry, partition, trait revision)
    into its launch record; the child's context carries those values with both resolvers
    patched to raise, and the run directory is unchanged.
    """
    from tcip_mcp.experiments import RUN_FILE, observe_run
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import subprocess_worker as worker
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
        partition_side,
    )

    run_dir = opened_run(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    launch_bytes = (run_dir / RUN_FILE).read_bytes()
    resolved = observe_run(run_dir).record.resolution
    assert set(resolved.model_dump()) == {
        "data",
        "geometry",
        "partition",
        "trait_revision",
    }
    files_before = sorted(p.name for p in run_dir.iterdir())

    def resolves_again(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("the child resolved its run again")

    monkeypatch.setattr(sc, "auto_train_val", resolves_again)
    monkeypatch.setattr(sc, "resolve_run", resolves_again)

    ctx = worker.prepare_run_context(observe_run(run_dir))

    assert ctx.run.trait_revision is None is resolved.trait_revision
    assert ctx.spec.data.record() == resolved.data.record()
    assert ctx.run.geometry == resolved.geometry
    train_ds = ctx.train_loader.dataset
    assert sorted(
        {train_ds.sample_of(key).member for key in train_ds.stems}
    ) == partition_side(resolved.partition, "train")
    assert (run_dir / RUN_FILE).read_bytes() == launch_bytes
    assert sorted(p.name for p in run_dir.iterdir()) == files_before


def test_launch_training_child_receives_its_own_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched training child receives its own run directory alone.

    One captured child argv ends with ``--run-dir`` and the run directory of the id the launch
    answers, and that run's record holds the launched images directory.
    """
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)

    from tcip_mcp.experiments import experiment_dir, read_run
    from tcip_mcp.tools import training_tools
    from tests._producer_fixtures import (
        fake_popen,
        seed_bud_images,
        small_detection_config,
    )

    captured_argv: list[list[str]] = []
    fake_popen(monkeypatch, captured_argv)

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    seed_bud_images(images_dir, n=2, size=64, box=(1, 1, 9, 9))

    res = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, project=tmp_path),
        actor=None,
    )
    assert isinstance(res, training_tools.RunLaunched), res
    run_dir = experiment_dir(res.experiment_id, project=tmp_path)
    children = [argv for argv in captured_argv if "--run-dir" in argv]
    assert children == [children[0]] and children[0][-2:] == [
        "--run-dir",
        str(run_dir),
    ]
    assert read_run(run_dir).spec.data.images_dir == str(images_dir)


# ── should_cancel() / the cancellation record ──────────────────────


def test_ctx_should_cancel_and_dispatch_classification_honor_the_cancellation(
    tmp_path: Path,
) -> None:
    """A bespoke body sees a requested cancellation and the run is classified canceled.

    A bespoke train(ctx) that calls ctx.should_cancel() sees a cancellation requested through
    the run directory's own record, and dispatch_train_body classifies the resulting run as
    'canceled', not 'completed'.
    """
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.pipelines.training.envelope import dispatch_train_body
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA,
        detection_images,
    )
    from tests.tiny_trainer_fixtures import run_context

    data = {
        **detection_images(tmp_path / "ds", SCOPED_DATA["scope"]),
        **SCOPED_DATA,
    }
    ctx = run_context(
        tmp_path, _config(tmp_path, data, training_source=BESPOKE_LOOP)
    )
    request_cancel(ctx.run_dir)

    assert ctx.should_cancel() is True

    dispatch_train_body(ctx)
    assert ctx.run.status == "canceled"


def _bespoke_loop(ctx: TrainContext) -> None:
    """Return when ctx.should_cancel() answers true, and raise AssertionError when it is false."""
    if ctx.should_cancel():
        return
    raise AssertionError(
        "should_cancel() did not see the requested cancellation"
    )


BESPOKE_LOOP = f"{Path(__file__).stem}:_bespoke_loop"
"""The ``training_source`` of :func:`_bespoke_loop`."""


# ── run_summary ─────────────


def test_run_summary_reads_the_run_directory(tmp_path: Path) -> None:
    """A live run's summary is read from its run directory.

    It names its config's selection metric and carries no best while no row stamps a
    ``selection``.
    """
    from tcip_mcp.experiments import observe_run, read_rows, run_summary
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        log_epoch,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    log_epoch(run_dir, 3, {"loss": 0.1})

    observation = observe_run(run_dir)
    result = run_summary(
        observation, read_rows(observation.metrics_log)[0], {}
    ).model_dump()
    assert result["state"] == "running"
    assert result["current_epoch"] == 3
    assert result["output_dir"] == str(run_dir)
    assert result["best_metric"] is None
    assert (
        result["best_metric_name"]
        == observation.record.spec.evaluation.selection_metric
    )


@pytest.mark.parametrize(
    "metric, best",
    [("map50", 0.7), ("loss", 0.5)],
    ids=["higher-is-better", "lower-is-better"],
)
def test_the_live_summary_folds_its_rows_in_its_configs_metric_direction(
    tmp_path: Path, metric: str, best: float
) -> None:
    """The live summary folds its rows in its config's metric direction.

    The live summary's best is folded in the declared direction of the metric the run's
    config selects on: the same rows give the highest under a higher-is-better metric and the
    lowest under a lower-is-better one.
    """
    from tcip_mcp.experiments import observe, read_rows, run_summary
    from tests._training_values import evaluation_block
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        log_epoch,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(
            tmp_path / "data",
            project=tmp_path,
            evaluation=evaluation_block(selection_metric=metric),
        ),
    )
    log_epoch(run_dir, 1, {"selection": 0.5, "selection_metric": metric})
    log_epoch(run_dir, 2, {"selection": 0.7, "selection_metric": metric})
    log_epoch(
        run_dir, 3, {"selection": float("nan"), "selection_metric": metric}
    )

    observation = observe(run_dir)
    result = run_summary(
        observation, read_rows(observation.metrics_log)[0], {}
    ).model_dump()
    assert (result["best_metric_name"], result["best_metric"]) == (
        metric,
        best,
    )


def test_run_summary_surfaces_the_wall_clock_failure_the_child_wrote(
    tmp_path: Path,
) -> None:
    """The run summary surfaces the wall-clock failure the child wrote.

    A run past its deadline ends failed, and its own final status names the wall clock.
    """
    from tcip_mcp.experiments import observe, run_summary
    from tests._verified_checkpoint_fixtures import (
        detector_declaring,
        finished_run,
    )

    run_dir = finished_run(
        tmp_path,
        model_source=detector_declaring(__file__),
        training_source=BESPOKE_LOOP,
        wall_clock_passed=True,
    )

    result = run_summary(observe(run_dir), [], {}).model_dump()
    assert result["state"] == "failed"
    assert result["status_error"] == "exceeded max_wall_clock_seconds"


def test_a_canceled_run_reads_canceled_whatever_its_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canceled run reads canceled with its heartbeat fresh and with it stale."""
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import (
        detector_declaring,
        finished_run,
    )

    canceled = finished_run(
        tmp_path,
        model_source=detector_declaring(__file__),
        training_source=BESPOKE_LOOP,
        cancel_requested=True,
    )

    assert experiments.observe(canceled).state == "canceled"
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert experiments.observe(canceled).state == "canceled"


# ── monitor_training / list_experiments read the directory ─────


def test_monitor_training_reads_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.training_tools import monitor_training
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        log_epoch,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="run_delegated",
    )
    log_epoch(run_dir, 7, {"loss": 0.2})

    result = monitor_training(tmp_path, "run_delegated")["run"]
    assert result["current_epoch"] == 7
    assert result["state"] == "running"


def test_launched_runs_view_lists_a_run_read_from_its_directory(
    tmp_path: Path,
) -> None:
    """An opened run lists, read straight from its own run directory.

    The row carries the directory's last epoch and last sign of life.
    """
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        log_epoch,
        opened_run,
    )

    models = tmp_path / "code" / "my_models.py"
    models.parent.mkdir()
    models.write_text("", encoding="utf-8")
    run_dir = opened_run(
        tmp_path,
        detection_config(
            tmp_path / "data",
            project=tmp_path,
            model_source={
                "builder": "my_models:chestnut_burr_det",
                "source_files": [str(models)],
                "task": "detection",
                "normalization": FRAME_STATISTICS,
            },
        ),
        experiment_id="exp-no-stamp",
    )
    log_epoch(run_dir, 9, {"loss": 0.1})

    by_id = {r["experiment_id"]: r for r in list_experiments(tmp_path)["runs"]}
    assert by_id["exp-no-stamp"]["state"] == "running"
    assert by_id["exp-no-stamp"]["current_epoch"] == 9
    assert by_id["exp-no-stamp"]["heartbeat"] is not None


# ── GPU device pinning ───────────────────────────────────────────────────


def test_gpu_device_pinning_round_robins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    seen = set()
    for _ in range(4):
        env = training_tools._child_env_for_launch(
            train_config(_config(REPO_ROOT, device=None))
        )
        seen.add(env["CUDA_VISIBLE_DEVICES"])
    assert seen == {"0", "1"}


def test_gpu_pinning_skipped_when_device_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config naming an explicit device sets no CUDA_VISIBLE_DEVICES for the child."""
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    env = training_tools._child_env_for_launch(
        train_config(_config(REPO_ROOT, device="cuda:1"))
    )
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_gpu_pinning_noop_with_single_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    env = training_tools._child_env_for_launch(
        train_config(_config(REPO_ROOT, device=None))
    )
    assert "CUDA_VISIBLE_DEVICES" not in env


# ── wall-clock timeout ──────────────────────────────────────────────


def test_max_wall_clock_seconds_terminates_a_hung_child_one_heartbeat_window_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wall-clock watcher terminates a child that outlives its wall clock.

    With a 0.2 s wall clock and a 0.2 s heartbeat window, a process sleeping 60 s is terminated
    within 10 s.
    """
    import subprocess
    import sys
    import time

    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import _watch_wall_clock

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", 0.2)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _watch_wall_clock(proc, 0.2)
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, (
            "watcher did not terminate the hung process"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── inspect_compute_resources ────────────────────────────────────────────────────────


def test_inspect_compute_resources_degrades_without_psutil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_psutil(
        name: str,
        *args: Mapping[str, object] | Sequence[str] | int | None,
        **kwargs: Mapping[str, object] | Sequence[str] | int | None,
    ) -> ModuleType:
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


def test_inspect_compute_resources_reports_gpu_free_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def mem_get_info(idx: int) -> tuple[int, int]:
            return (1_000, 4_000)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    from tcip_mcp.tools.training_tools import inspect_compute_resources

    result = inspect_compute_resources(tmp_path)
    assert result["gpus"] == [
        {"index": 0, "free_bytes": 1_000, "total_bytes": 4_000}
    ]


def test_inspect_compute_resources_counts_a_running_run_directory(
    tmp_path: Path,
) -> None:
    """A run whose directory is fresh counts as one more active training run.

    The directory alone is enough.
    """
    from tcip_mcp.tools.training_tools import inspect_compute_resources
    from tests._verified_checkpoint_fixtures import opened_run

    baseline = inspect_compute_resources(tmp_path)["active_training_runs"]

    from tests._verified_checkpoint_fixtures import detection_config

    opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="run_other_process",
    )

    assert (
        inspect_compute_resources(tmp_path)["active_training_runs"]
        == baseline + 1
    )


# ── HPO resource caps ─────────────────────────────────────────────────────────


def test_default_trial_resources_derives_fractional_gpu_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    assert hpo._default_trial_resources(max_concurrent=1) == {
        "cpu": 1.0,
        "gpu": 1.0,
    }
    assert hpo._default_trial_resources(max_concurrent=4) == {
        "cpu": 1.0,
        "gpu": 0.5,
    }


def test_default_trial_resources_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    assert hpo._default_trial_resources(max_concurrent=1) == {
        "cpu": 1.0,
        "gpu": 0.0,
    }


@pytest.mark.ray_cluster
def test_tune_search_accepts_explicit_resources_per_trial(
    tmp_path: Path,
) -> None:
    """A Ray sweep runs to its end with an explicit resources_per_trial.

    The sweep runs a pure-math objective over four samples, and its named sweep_dir exists
    afterward.
    """
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    def obj(config: dict, report: Callable[[float], None]) -> None:
        report((config["x"] - 2.0) ** 2)

    tune_search(
        objective_fn=obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        sweep_dir=tmp_path / "explicit",
        **tune_arguments(
            num_samples=4, resources_per_trial={"cpu": 1.0, "gpu": 0.0}
        ),
    )
    assert (tmp_path / "explicit").is_dir()


@pytest.mark.ray_cluster
def test_tune_search_runs_despite_deprecated_ray_result_dir_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep runs despite deprecated Ray result-directory variables in the environment.

    With TUNE_RESULT_DIR and RAY_AIR_LOCAL_CACHE_DIR set, the named sweep_dir exists afterward
    and both variables keep their values.
    """
    import os

    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    machine_scratch = str(tmp_path / "machine_scratch")
    monkeypatch.setenv("TUNE_RESULT_DIR", machine_scratch)
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", machine_scratch)

    def obj(config: dict, report: Callable[[float], None]) -> None:
        report((config["x"] - 2.0) ** 2)

    tune_search(
        objective_fn=obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        sweep_dir=tmp_path / "sweep_store" / "redirected",
        **tune_arguments(
            num_samples=2, resources_per_trial={"cpu": 1.0, "gpu": 0.0}
        ),
    )

    assert (tmp_path / "sweep_store" / "redirected").is_dir()
    assert os.environ["TUNE_RESULT_DIR"] == machine_scratch
    assert os.environ["RAY_AIR_LOCAL_CACHE_DIR"] == machine_scratch
