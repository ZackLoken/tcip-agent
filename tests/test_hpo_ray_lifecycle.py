"""Ray is one cluster per process, shared by every sweep running in it.

``tune_search`` may therefore only shut Ray down when the last concurrent sweep leaves, and
only when the platform is what started it. Ray is faked here (``tune_search`` imports it
inside the function body) so the lifecycle can be driven deterministically without a real
cluster: each fake ``Tuner.fit`` blocks until the test releases it.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


class _FakeDaemonProcess:
    """Stands in for a Ray daemon's ``Popen``, as ``Node.all_processes`` holds it.

    ``terminate`` mirrors ``ConsolePopen``'s Windows console-signal path: it raises the
    ``OSError`` that path raises on a daemon started without a console (``ray.no_console``
    True on the fake module), and otherwise succeeds like a daemon that has a console to
    signal.
    """

    def __init__(self, ray: ModuleType, pid: int) -> None:
        self._ray = ray
        self.pid = pid
        self._alive = True
        self.kill_calls = 0
        self.wait_calls = 0
        self.terminate_calls = 0

    def poll(self):
        return None if self._alive else 0

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def wait(self, timeout=None) -> None:
        self.wait_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._ray.no_console:
            raise OSError(22, "The handle is invalid")
        self._alive = False


def _install_fake_ray(monkeypatch, entered: list, release: list) -> ModuleType:
    """Put a counting, blocking stand-in for Ray on ``sys.modules``.

    The nth ``Tuner`` built sets ``entered[n]`` when its ``fit`` starts and returns only
    once ``release[n]`` is set, so a test can hold a sweep open across another's exit.

    ``ray._private.worker.global_worker.node.all_processes`` mirrors the real attribute path
    ``_kill_ray_daemons_before_shutdown`` reads, holding one fake daemon per Ray process type;
    ``shutdown`` mirrors ``Node.kill_all_processes``, terminating whichever of those daemons
    are still alive (``ray.no_console`` decides whether that terminate call raises).
    ``_kill_ray_daemons_before_shutdown`` looks up each fake daemon's descendants through the
    real ``psutil``, so ``psutil.Process`` is patched here to report the fake pids as never
    having any: a real pid this test made up matching a process actually running on the host
    would otherwise be walked and its children killed.
    """
    import psutil

    def fake_psutil_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", fake_psutil_process)

    ray = ModuleType("ray")
    ray.init_calls = 0
    ray.shutdown_calls = 0
    ray.live = False
    ray.init_kwargs = {}
    ray.no_console = False

    daemon_process_infos = {
        process_type: [SimpleNamespace(process=_FakeDaemonProcess(ray, pid=10_000 + i))]
        for i, process_type in enumerate(("raylet", "gcs_server", "log_monitor", "reaper"))
    }
    ray._private = SimpleNamespace(
        worker=SimpleNamespace(
            global_worker=SimpleNamespace(node=SimpleNamespace(all_processes=daemon_process_infos))
        )
    )

    def is_initialized() -> bool:
        return ray.live

    def init(**kwargs):
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
        def __init__(self, *args, **kwargs) -> None:
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
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    basic_variant = ModuleType("ray.tune.search.basic_variant")
    basic_variant.BasicVariantGenerator = BasicVariantGenerator
    search.Searcher = Searcher
    search.basic_variant = basic_variant
    tune.search = search
    ray.tune = tune

    for name, module in (("ray", ray), ("ray.tune", tune), ("ray.tune.search", search),
                         ("ray.tune.search.basic_variant", basic_variant)):
        monkeypatch.setitem(sys.modules, name, module)
    return ray


def _run_one_search(project) -> None:
    from tcip_mcp.pipelines.training.hpo import tune_search
    from tests._training_values import sweep_space, tune_arguments

    tune_search(
        objective_fn=lambda config, report: None, param_space=sweep_space(),
        sweep_dir=project / ".tcip" / "experiments" / "sweep",
        **tune_arguments(resources_per_trial={"cpu": 1.0, "gpu": 0.0}))


@pytest.fixture(autouse=True)
def _isolated_lifecycle_state(tmp_path, monkeypatch):
    """Start every test from an idle, unowned cluster."""
    import tcip_mcp.pipelines.training.hpo as hpo

    monkeypatch.setattr(hpo, "_active_searches", 0)
    monkeypatch.setattr(hpo, "_ray_started", False)
    monkeypatch.setattr(hpo, "_ray_runtime_pythonpath", None)
    monkeypatch.setattr(hpo, "_external_cluster_warned", False, raising=False)


def test_finished_sweep_leaves_a_concurrent_sweep_s_cluster_running(tmp_path, monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    ray = _install_fake_ray(monkeypatch, entered, release)

    threads = [threading.Thread(target=_run_one_search, args=(tmp_path,)) for _ in range(2)]
    threads[0].start()
    assert entered[0].wait(timeout=30)
    threads[1].start()
    assert entered[1].wait(timeout=30)

    release[0].set()
    threads[0].join(timeout=30)
    assert not threads[0].is_alive()

    assert ray.init_calls == 1, "the second sweep joined the running cluster"
    assert ray.shutdown_calls == 0, "the second sweep is still inside Tuner.fit"

    release[1].set()
    threads[1].join(timeout=30)
    assert not threads[1].is_alive()
    assert ray.shutdown_calls == 1


def test_a_cluster_this_process_did_not_start_is_never_shut_down(tmp_path, monkeypatch):
    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.live = True  # something else in this process already brought Ray up

    _run_one_search(tmp_path)

    assert ray.init_calls == 0
    assert ray.shutdown_calls == 0


def test_ray_init_propagates_this_process_s_import_search_path_to_trial_workers(
    tmp_path, monkeypatch
):
    """A trial worker Ray spawns starts from its own defaults, not this interpreter's
    sys.path; every entry of it reaches the worker, the same guarantee the launch subprocess
    gets, except a layout root an import put there, which no worker inherits: a worker imports
    from the layout it binds itself."""
    from tcip_mcp.pipelines.model_build import _LAYOUT_ROOTS
    from tcip_store import canonical_path

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search(tmp_path)

    env_vars = ray.init_kwargs["runtime_env"]["env_vars"]
    pythonpath_entries = env_vars["PYTHONPATH"].split(os.pathsep)
    for entry in (p for p in sys.path if p and canonical_path(p) not in _LAYOUT_ROOTS):
        assert entry in pythonpath_entries
    assert not [p for p in pythonpath_entries if canonical_path(p) in _LAYOUT_ROOTS]


def test_a_concurrent_sweep_warns_when_the_running_cluster_s_import_path_has_gone_stale(
    monkeypatch, caplog, tmp_path,
):
    """A cluster this module started keeps the ``PYTHONPATH`` it was handed at ``ray.init``;
    a second sweep entering while that cluster is still up, after ``sys.path`` gained an entry
    the first sweep never saw, must be told its trial workers will not see that entry either."""
    import logging

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    _install_fake_ray(monkeypatch, entered, release)

    first = threading.Thread(target=_run_one_search, args=(tmp_path,))
    first.start()
    assert entered[0].wait(timeout=30)

    monkeypatch.syspath_prepend(str(tmp_path / "bespoke_source"))

    with caplog.at_level(logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"):
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
    tmp_path, monkeypatch, caplog,
):
    """The companion of the stale-path warning: entering a second sweep against a cluster
    this module started, with the import path unchanged, raises no warning."""
    import logging

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    _install_fake_ray(monkeypatch, entered, release)

    first = threading.Thread(target=_run_one_search, args=(tmp_path,))
    first.start()
    assert entered[0].wait(timeout=30)

    with caplog.at_level(logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"):
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


def test_sequential_sweeps_each_start_and_stop_their_own_cluster(tmp_path, monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    for event in release:
        event.set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search(tmp_path)
    _run_one_search(tmp_path)

    assert ray.init_calls == 2
    assert ray.shutdown_calls == 2


def test_a_console_free_exit_kills_ray_daemons_before_shutdown_signals_them(tmp_path, monkeypatch):
    """Windows' console-signal shutdown path raises ``OSError: [WinError 6] The handle is
    invalid`` on a daemon started without a console; a console-free process must kill every
    daemon itself first so ``ray.shutdown()`` finds each one already dead and never signals
    it. Asserts the sweep's exit raises nothing, that every daemon was killed rather than
    signaled, and that ``ray.shutdown()`` still ran.
    """
    import tcip_mcp.pipelines.training.hpo as hpo

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.no_console = True
    monkeypatch.setattr(hpo, "_has_attached_console", lambda: False, raising=False)

    escaped: BaseException | None = None
    try:
        _run_one_search(tmp_path)
    except BaseException as exc:
        escaped = exc

    assert escaped is None, f"the sweep's exit raised {escaped!r} instead of completing"
    for process_infos in ray._private.worker.global_worker.node.all_processes.values():
        for process_info in process_infos:
            assert process_info.process.kill_calls == 1
            assert process_info.process.terminate_calls == 0
    assert ray.shutdown_calls == 1


def test_a_process_with_a_console_lets_ray_shutdown_signal_its_daemons_directly(
    tmp_path, monkeypatch
):
    """Coverage of the attached-console path: asserts nothing is killed ahead of
    ``ray.shutdown()``, which signals each daemon through the terminate path, and that
    ``ray.shutdown()`` still ran."""
    import tcip_mcp.pipelines.training.hpo as hpo

    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.no_console = False
    monkeypatch.setattr(hpo, "_has_attached_console", lambda: True, raising=False)

    _run_one_search(tmp_path)

    for process_infos in ray._private.worker.global_worker.node.all_processes.values():
        for process_info in process_infos:
            assert process_info.process.kill_calls == 0
            assert process_info.process.terminate_calls == 1
    assert ray.shutdown_calls == 1


def test_a_cluster_this_process_starts_is_sized_to_the_sweep_s_own_request(tmp_path, monkeypatch):
    """The cluster asks Ray for the sweep's CPUs, every concurrent trial's together, never for
    the host's: three half-CPU trials at once need two whole CPUs, and the default one-CPU,
    one-at-a-time sweep needs one."""
    from tcip_mcp.pipelines.training.hpo import tune_search
    from tests._training_values import sweep_space, tune_arguments

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    for event in release:
        event.set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    tune_search(
        objective_fn=lambda config, report: None, param_space=sweep_space(),
        sweep_dir=tmp_path / ".tcip" / "experiments" / "sweep",
        **tune_arguments(max_concurrent=3, resources_per_trial={"cpu": 0.5, "gpu": 0.0}))
    assert ray.init_kwargs["num_cpus"] == 2

    _run_one_search(tmp_path)
    assert ray.init_calls == 2, (
        "the first sweep's cluster was shut down, so this one started its own"
    )
    assert ray.init_kwargs["num_cpus"] == 1
