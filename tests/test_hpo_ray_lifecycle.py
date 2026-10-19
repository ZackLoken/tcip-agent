"""``tune_search``'s lifecycle of the Ray cluster its process holds.

``tune_search`` shuts Ray down when the last concurrent sweep leaves, leaves a cluster its
process did not start running, and sizes and seeds the cluster it starts. Ray is faked here,
and each fake ``Tuner.fit`` blocks until its release event is set.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class _FakeDaemonProcess:
    """A stand-in for a Ray daemon's ``Popen``, as ``Node.all_processes`` holds it.

    ``terminate`` raises ``OSError`` while ``ray.no_console`` is true on the fake module and
    marks the daemon dead while it is false.
    """

    def __init__(self, ray: ModuleType, pid: int) -> None:
        self._ray = ray
        self.pid = pid
        self._alive = True
        self.kill_calls = 0
        self.wait_calls = 0
        self.terminate_calls = 0

    def poll(self) -> int | None:
        return None if self._alive else 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._ray.no_console:
            raise OSError(22, "The handle is invalid")
        self._alive = False


def _install_fake_ray(
    monkeypatch: pytest.MonkeyPatch, entered: list, release: list
) -> ModuleType:
    """Put a counting, blocking stand-in for Ray on ``sys.modules``.

    The nth ``Tuner`` built sets ``entered[n]`` when its ``fit`` starts and returns when
    ``release[n]`` is set. ``ray.tune.schedulers`` carries the one class the sample sweep's
    scheduler names.

    ``ray._private.worker.global_worker.node.all_processes`` holds one fake daemon per Ray
    process type, and ``shutdown`` terminates whichever of those daemons are still alive.
    ``psutil.Process`` is patched to raise ``NoSuchProcess`` for any pid.
    """
    import psutil

    from tcip_mcp.pipelines.training.hpo import _SCHEDULERS
    from tests._training_values import tune_arguments

    def fake_psutil_process(pid: int) -> NoReturn:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", fake_psutil_process)

    ray = ModuleType("ray")
    ray.init_calls = 0
    ray.shutdown_calls = 0
    ray.live = False
    ray.init_kwargs = {}
    ray.no_console = False

    daemon_process_infos = {
        process_type: [
            SimpleNamespace(process=_FakeDaemonProcess(ray, pid=10_000 + i))
        ]
        for i, process_type in enumerate(
            ("raylet", "gcs_server", "log_monitor", "reaper")
        )
    }
    ray._private = SimpleNamespace(
        worker=SimpleNamespace(
            global_worker=SimpleNamespace(
                node=SimpleNamespace(all_processes=daemon_process_infos)
            )
        )
    )

    def is_initialized() -> bool:
        return ray.live

    def init(**kwargs: int | bool | dict[str, dict[str, str]]) -> None:
        ray.init_calls += 1
        ray.live = True
        ray.init_kwargs = dict(kwargs)

    def shutdown() -> None:
        ray.shutdown_calls += 1
        node = ray._private.worker.global_worker.node
        for process_infos in node.all_processes.values():
            for process_info in process_infos:
                if process_info.process.poll() is None:
                    process_info.process.terminate()
        ray.live = False

    ray.is_initialized = is_initialized
    ray.init = init
    ray.shutdown = shutdown

    order = itertools.count()
    order_lock = threading.Lock()

    class _Tuner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            with order_lock:
                self.index = next(order)

        def fit(self) -> None:
            entered[self.index].set()
            assert release[self.index].wait(timeout=30)

    tune = ModuleType("ray.tune")
    tune.loguniform = lambda low, high: (low, high)
    tune.uniform = lambda low, high: (low, high)
    tune.randint = lambda low, high: (low, high)
    tune.choice = lambda choices: list(choices)
    tune.grid_search = lambda values: list(values)
    tune.with_resources = lambda trainable, resources: trainable
    tune.report = lambda metrics: None
    tune.TuneConfig = lambda **kwargs: kwargs
    tune.RunConfig = lambda **kwargs: kwargs
    tune.Tuner = _Tuner
    tune.Callback = type("Callback", (), {})
    tune.Stopper = type("Stopper", (), {})

    search = ModuleType("ray.tune.search")

    class Searcher:
        pass

    class BasicVariantGenerator(Searcher):
        def __init__(self, **kwargs: list[dict] | int | bool | None) -> None:
            self.kwargs = kwargs

    basic_variant = ModuleType("ray.tune.search.basic_variant")
    basic_variant.BasicVariantGenerator = BasicVariantGenerator
    search.Searcher = Searcher
    search.basic_variant = basic_variant
    tune.search = search

    scheduler_class = _SCHEDULERS[tune_arguments()["scheduler"]["name"]]
    schedulers = ModuleType("ray.tune.schedulers")
    setattr(schedulers, scheduler_class, type(scheduler_class, (), {}))
    tune.schedulers = schedulers
    ray.tune = tune

    for name, module in (
        ("ray", ray),
        ("ray.tune", tune),
        ("ray.tune.search", search),
        ("ray.tune.search.basic_variant", basic_variant),
        ("ray.tune.schedulers", schedulers),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return ray


def _run_one_search(project: Path) -> None:
    from tcip_mcp.pipelines.training.hpo import tune_search
    from tests._training_values import sweep_space, tune_arguments

    tune_search(
        objective_fn=lambda config, report: None,
        param_space=sweep_space(),
        sweep_dir=project / ".tcip" / "experiments" / "sweep",
        **tune_arguments(resources_per_trial={"cpu": 1.0, "gpu": 0.0}),
    )


@pytest.fixture(autouse=True)
def _isolated_lifecycle_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reset the cluster lifecycle state to idle and unowned."""
    import tcip_mcp.pipelines.training.hpo as hpo

    monkeypatch.setattr(hpo, "_active_searches", 0)
    monkeypatch.setattr(hpo, "_ray_started", False)
    monkeypatch.setattr(hpo, "_ray_runtime_pythonpath", None)
    monkeypatch.setattr(hpo, "_external_cluster_warned", False, raising=False)


def test_finished_sweep_leaves_a_concurrent_sweep_s_cluster_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    ray = _install_fake_ray(monkeypatch, entered, release)

    threads = [
        threading.Thread(target=_run_one_search, args=(tmp_path,))
        for _ in range(2)
    ]
    threads[0].start()
    assert entered[0].wait(timeout=30)
    threads[1].start()
    assert entered[1].wait(timeout=30)

    release[0].set()
    threads[0].join(timeout=30)
    assert not threads[0].is_alive()

    assert ray.init_calls == 1, "the second sweep joined the running cluster"
    assert ray.shutdown_calls == 0, (
        "the second sweep is still inside Tuner.fit"
    )

    release[1].set()
    threads[1].join(timeout=30)
    assert not threads[1].is_alive()
    assert ray.shutdown_calls == 1


def test_a_cluster_this_process_did_not_start_is_never_shut_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.live = True  # something else in this process already brought Ray up

    _run_one_search(tmp_path)

    assert ray.init_calls == 0
    assert ray.shutdown_calls == 0


def test_ray_init_propagates_this_process_s_import_search_path_to_trial_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ray.init``'s runtime environment carries this process's import path, less layout roots.

    Each ``sys.path`` entry that is not a layout root appears in that environment's
    ``PYTHONPATH``, and no layout root does.
    """
    from tcip_mcp.pipelines.model_build import _LAYOUT_ROOTS
    from tcip_store import canonical_path

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search(tmp_path)

    env_vars = ray.init_kwargs["runtime_env"]["env_vars"]
    pythonpath_entries = env_vars["PYTHONPATH"].split(os.pathsep)
    for entry in (
        p for p in sys.path if p and canonical_path(p) not in _LAYOUT_ROOTS
    ):
        assert entry in pythonpath_entries
    assert not [
        p for p in pythonpath_entries if canonical_path(p) in _LAYOUT_ROOTS
    ]


def test_a_concurrent_sweep_warns_when_the_running_cluster_s_import_path_has_gone_stale(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A concurrent sweep warns when the running cluster's import path has gone stale.

    With ``sys.path`` gaining an entry while a first sweep's cluster is up, a second sweep
    entering logs a warning that the cluster keeps the import path captured at its start.
    """
    import logging

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    _install_fake_ray(monkeypatch, entered, release)

    first = threading.Thread(target=_run_one_search, args=(tmp_path,))
    first.start()
    assert entered[0].wait(timeout=30)

    monkeypatch.syspath_prepend(str(tmp_path / "bespoke_source"))

    with caplog.at_level(
        logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"
    ):
        second = threading.Thread(target=_run_one_search, args=(tmp_path,))
        second.start()
        assert entered[1].wait(timeout=30)
        release[1].set()
        second.join(timeout=30)

    release[0].set()
    first.join(timeout=30)

    assert any(
        "keep the import path captured at cluster start" in record.message
        for record in caplog.records
    )


def test_a_concurrent_sweep_does_not_warn_when_the_import_path_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A concurrent sweep raises no warning when the import path is unchanged.

    Entering a second sweep against a cluster this module started, with the import path
    unchanged, raises no warning.
    """
    import logging

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    _install_fake_ray(monkeypatch, entered, release)

    first = threading.Thread(target=_run_one_search, args=(tmp_path,))
    first.start()
    assert entered[0].wait(timeout=30)

    with caplog.at_level(
        logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"
    ):
        second = threading.Thread(target=_run_one_search, args=(tmp_path,))
        second.start()
        assert entered[1].wait(timeout=30)
        release[1].set()
        second.join(timeout=30)

    release[0].set()
    first.join(timeout=30)

    assert not any(
        "keep the import path captured at cluster start" in record.message
        for record in caplog.records
    )


def test_sequential_sweeps_each_start_and_stop_their_own_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    for event in release:
        event.set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search(tmp_path)
    _run_one_search(tmp_path)

    assert ray.init_calls == 2
    assert ray.shutdown_calls == 2


def test_a_console_free_exit_kills_ray_daemons_before_shutdown_signals_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A console-free exit kills Ray's daemons, leaving ``ray.shutdown()`` none to signal.

    With no console attached and daemons whose terminate raises, the sweep's exit raises
    nothing, each daemon records one kill and no terminate call, and ``ray.shutdown()`` still
    runs.
    """
    import tcip_mcp.pipelines.training.hpo as hpo

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.no_console = True
    monkeypatch.setattr(
        hpo, "_has_attached_console", lambda: False, raising=False
    )

    escaped: BaseException | None = None
    try:
        _run_one_search(tmp_path)
    except BaseException as exc:
        escaped = exc

    assert escaped is None, (
        f"the sweep's exit raised {escaped!r} instead of completing"
    )
    for (
        process_infos
    ) in ray._private.worker.global_worker.node.all_processes.values():
        for process_info in process_infos:
            assert process_info.process.kill_calls == 1
            assert process_info.process.terminate_calls == 0
    assert ray.shutdown_calls == 1


def test_a_process_with_a_console_lets_ray_shutdown_signal_its_daemons_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process with a console lets ``ray.shutdown()`` signal its daemons directly.

    Asserts nothing is killed ahead of ``ray.shutdown()``, which signals each daemon through the
    terminate path, and that ``ray.shutdown()`` still ran.
    """
    import tcip_mcp.pipelines.training.hpo as hpo

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.no_console = False
    monkeypatch.setattr(
        hpo, "_has_attached_console", lambda: True, raising=False
    )

    _run_one_search(tmp_path)

    for (
        process_infos
    ) in ray._private.worker.global_worker.node.all_processes.values():
        for process_info in process_infos:
            assert process_info.process.kill_calls == 0
            assert process_info.process.terminate_calls == 1
    assert ray.shutdown_calls == 1


def test_a_cluster_this_process_starts_is_sized_to_the_sweep_s_own_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cluster this process starts is sized to the sweep's own CPU request.

    Three concurrent half-CPU trials ask Ray for two whole CPUs, and the sample one-CPU,
    one-at-a-time sweep asks for one.
    """
    from tcip_mcp.pipelines.training.hpo import tune_search
    from tests._training_values import sweep_space, tune_arguments

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    for event in release:
        event.set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    tune_search(
        objective_fn=lambda config, report: None,
        param_space=sweep_space(),
        sweep_dir=tmp_path / ".tcip" / "experiments" / "sweep",
        **tune_arguments(
            max_concurrent=3, resources_per_trial={"cpu": 0.5, "gpu": 0.0}
        ),
    )
    assert ray.init_kwargs["num_cpus"] == 2

    _run_one_search(tmp_path)
    assert ray.init_calls == 2, (
        "the first sweep's cluster was shut down, so this one started its own"
    )
    assert ray.init_kwargs["num_cpus"] == 1
