"""Ray is one cluster per process, shared by every sweep running in it.

``tune_search`` may therefore only shut Ray down when the last concurrent sweep leaves, and
only when the platform is what started it. Ray is faked here (``tune_search`` imports it
inside the function body) so the lifecycle can be driven deterministically without a real
cluster: each fake ``Tuner.fit`` blocks until the test releases it.

The same lifecycle owns the dashboard URL a cluster is reachable at, which is written to
platform state so another process (the web backend) can serve it.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import threading
from datetime import datetime
from types import ModuleType, SimpleNamespace

import pytest

DASHBOARD_HOST_PORT = "127.0.0.1:8265"


class _Result:
    config = {"lr": 0.1}
    metrics = {"objective": 1.0, "training_iteration": 1}
    error = None


class _Results(list):
    def __init__(self) -> None:
        super().__init__([_Result()])

    def get_best_result(self, metric=None, mode=None):
        return self[0]


def _install_fake_ray(monkeypatch, entered: list, release: list) -> ModuleType:
    """Put a counting, blocking stand-in for Ray on ``sys.modules``.

    The nth ``Tuner`` built sets ``entered[n]`` when its ``fit`` starts and returns only
    once ``release[n]`` is set, so a test can hold a sweep open across another's exit.
    """
    ray = ModuleType("ray")
    ray.init_calls = 0
    ray.shutdown_calls = 0
    ray.live = False
    ray.init_kwargs = {}
    ray.dashboard_url = DASHBOARD_HOST_PORT

    def is_initialized() -> bool:
        return ray.live

    def init(**kwargs):
        ray.init_calls += 1
        ray.live = True
        ray.init_kwargs = dict(kwargs)
        return SimpleNamespace(dashboard_url=ray.dashboard_url)

    def shutdown() -> None:
        ray.shutdown_calls += 1
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

        def fit(self) -> _Results:
            entered[self.index].set()
            assert release[self.index].wait(timeout=30)
            return _Results()

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

    search = ModuleType("ray.tune.search")

    class Searcher:
        pass

    class BasicVariantGenerator(Searcher):
        pass

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


def _patch_aiohttp_availability(monkeypatch, available: bool) -> None:
    """Force whether the dashboard's own key dependency looks importable, regardless of
    whatever actually happens to be installed in the environment running this test."""
    import tcip_mcp.pipelines.training.hpo as hpo

    real_find_spec = hpo.find_spec

    def fake_find_spec(name):
        if name == "aiohttp":
            return object() if available else None
        return real_find_spec(name)

    monkeypatch.setattr(hpo, "find_spec", fake_find_spec)


def _run_one_search() -> dict:
    from pathlib import Path

    from tcip_mcp.pipelines.training.hpo import tune_search

    # The autouse fixture pins TCIP_STATE_ROOT to each test's own tmp dir; storing there
    # keeps every fake sweep inside the test's isolated platform state root.
    return tune_search(
        objective_fn=lambda config, report: None,
        param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        num_samples=1,
        search_alg="random",
        scheduler=None,
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
        storage_path=str(Path(os.environ["TCIP_STATE_ROOT"]) / "hpo"),
    )


@pytest.fixture(autouse=True)
def _isolated_lifecycle_state(monkeypatch, tmp_path):
    """Start every test from an idle, unowned cluster and its own platform state root."""
    import tcip_mcp.pipelines.training.hpo as hpo

    monkeypatch.setattr(hpo, "_active_searches", 0, raising=False)
    monkeypatch.setattr(hpo, "_ray_started_here", False, raising=False)
    monkeypatch.setattr(hpo, "_ray_runtime_pythonpath", None, raising=False)
    monkeypatch.setattr(hpo, "_external_cluster_warned", False, raising=False)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))


def test_finished_sweep_leaves_a_concurrent_sweep_s_cluster_running(monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    ray = _install_fake_ray(monkeypatch, entered, release)

    threads = [threading.Thread(target=_run_one_search) for _ in range(2)]
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


def test_a_cluster_this_process_did_not_start_is_never_shut_down(monkeypatch):
    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)
    ray.live = True  # something else in this process already brought Ray up

    _run_one_search()

    assert ray.init_calls == 0
    assert ray.shutdown_calls == 0


def test_the_dashboard_url_is_readable_while_the_cluster_is_up_and_gone_after(monkeypatch):
    from tcip_store import store

    from tcip_mcp.pipelines.training.hpo import ray_dashboard_key, read_ray_dashboard

    _patch_aiohttp_availability(monkeypatch, True)
    entered = [threading.Event()]
    release = [threading.Event()]
    ray = _install_fake_ray(monkeypatch, entered, release)

    sweep = threading.Thread(target=_run_one_search)
    sweep.start()
    assert entered[0].wait(timeout=30)

    assert ray.init_kwargs["include_dashboard"] is True
    assert ray.init_kwargs["dashboard_host"] == "127.0.0.1"

    published = read_ray_dashboard()
    assert published is not None
    # The frontend needs a URL it can fetch; ray.init reports a bare host:port.
    assert published["url"] == f"http://{DASHBOARD_HOST_PORT}"
    assert published["pid"] == os.getpid()
    assert datetime.fromisoformat(published["started_at"])

    release[0].set()
    sweep.join(timeout=30)
    assert not sweep.is_alive()

    assert not store.exists(ray_dashboard_key())
    assert read_ray_dashboard() is None


def test_a_dashboard_recorded_by_a_process_that_is_gone_is_not_served(monkeypatch):
    """A URL outlives the process that wrote it; the cluster it names does not."""
    from tcip_store import store

    from tcip_mcp.pipelines.training.hpo import ray_dashboard_key, read_ray_dashboard

    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait(timeout=60)

    state = {"url": f"http://{DASHBOARD_HOST_PORT}", "pid": dead.pid,
             "started_at": "2026-01-01T00:00:00+00:00"}
    store.replace(ray_dashboard_key(), state)
    assert read_ray_dashboard() is None

    store.replace(ray_dashboard_key(), {**state, "pid": os.getpid()})
    assert read_ray_dashboard() == {**state, "pid": os.getpid()}


def test_ray_init_propagates_this_process_s_import_search_path_to_trial_workers(monkeypatch):
    """A trial worker Ray spawns starts from its own defaults, not this interpreter's
    sys.path; a bespoke model_source/training_source/dataset_source importable here must stay
    importable there, the same guarantee the launch subprocess gets."""
    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search()

    env_vars = ray.init_kwargs["runtime_env"]["env_vars"]
    pythonpath_entries = env_vars["PYTHONPATH"].split(os.pathsep)
    for entry in (p for p in sys.path if p):
        assert entry in pythonpath_entries


def test_a_sweep_still_runs_when_the_dashboard_dependency_is_not_installed(monkeypatch):
    """A bare ray[tune] install (this package's own declared minimum) has no aiohttp; asking
    ray.init for a dashboard it can't serve raises there, which must not stop a sweep from
    running on that install."""
    from tcip_mcp.pipelines.training.hpo import read_ray_dashboard

    _patch_aiohttp_availability(monkeypatch, False)
    entered = [threading.Event()]
    release = [threading.Event()]
    release[0].set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search()

    assert ray.init_calls == 1
    assert ray.shutdown_calls == 1
    assert ray.init_kwargs["include_dashboard"] is False
    assert read_ray_dashboard() is None


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

    first = threading.Thread(target=_run_one_search)
    first.start()
    assert entered[0].wait(timeout=30)

    monkeypatch.syspath_prepend(str(tmp_path / "bespoke_source"))

    with caplog.at_level(logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"):
        second = threading.Thread(target=_run_one_search)
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
    monkeypatch, caplog,
):
    """The companion of the stale-path warning: entering a second sweep against a cluster
    this module started, with the import path unchanged, raises no warning."""
    import logging

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    _install_fake_ray(monkeypatch, entered, release)

    first = threading.Thread(target=_run_one_search)
    first.start()
    assert entered[0].wait(timeout=30)

    with caplog.at_level(logging.WARNING, logger="tcip_mcp.pipelines.training.hpo"):
        second = threading.Thread(target=_run_one_search)
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


def test_sequential_sweeps_each_start_and_stop_their_own_cluster(monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    for event in release:
        event.set()
    ray = _install_fake_ray(monkeypatch, entered, release)

    _run_one_search()
    _run_one_search()

    assert ray.init_calls == 2
    assert ray.shutdown_calls == 2
