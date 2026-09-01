"""A trial Ray's own machinery kills before its first report never reaches ``tune.report``,
so ``ray.air.result.Result.config`` (a property reading ``metrics.get("config")``) answers
``None`` for it. ``tune_search`` used to assert every result carried a config, which crashed
the whole sweep's result assembly the moment one trial died that way; it now records the
death as an honest ``ERROR`` row instead.

Ray is faked here the same way ``test_hpo_ray_lifecycle.py`` fakes it (``tune_search`` imports
Ray inside the function body, so a stand-in on ``sys.modules`` is enough), except the fake
``Tuner.fit`` returns a fixed, pre-built results grid rather than blocking on release events:
this file is not exercising the cluster lifecycle, only what ``tune_search`` does with the
grid it gets back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ray.air.result import Result


class _ReportedResult:
    """A trial that reported at least once: Ray's real ``Result`` behaves the same way, but a
    plain stand-in is enough here since only the never-reported trial needs to be Ray's own
    dataclass (its ``config is None`` behavior is what this change reacts to)."""

    def __init__(self, config: dict, metrics: dict, error: Exception | None = None) -> None:
        self.config = config
        self.metrics = metrics
        self.error = error


class _ResultsGrid(list):
    def __init__(self, results: list) -> None:
        super().__init__(results)
        self._best = results[0]

    def get_best_result(self, metric=None, mode=None):
        return self._best


def _install_fake_ray(monkeypatch, results: list) -> None:
    """Put a non-blocking stand-in for Ray on ``sys.modules`` whose ``Tuner.fit`` hands back
    the given results grid, so ``tune_search`` runs synchronously against fixed trial results."""
    ray = ModuleType("ray")
    ray.is_initialized = lambda: False
    ray.init = lambda **kwargs: SimpleNamespace(dashboard_url=None)
    ray.shutdown = lambda: None

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

    class _Tuner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self) -> _ResultsGrid:
            return _ResultsGrid(results)

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


@pytest.fixture(autouse=True)
def _isolated_lifecycle_state(monkeypatch):
    """Start every test from an idle, unowned Ray cluster, matching the lifecycle fixture in
    test_hpo_ray_lifecycle.py so a dirty module-global from another hpo test cannot leak in."""
    import tcip_mcp.pipelines.training.hpo as hpo

    monkeypatch.setattr(hpo, "_active_searches", 0, raising=False)
    monkeypatch.setattr(hpo, "_ray_started_here", False, raising=False)
    monkeypatch.setattr(hpo, "_ray_runtime_pythonpath", None, raising=False)
    monkeypatch.setattr(hpo, "_external_cluster_warned", False, raising=False)


def _run_search(param_space: dict) -> dict:
    from tcip_mcp.pipelines.training.hpo import tune_search

    # tmp_path is pinned as TCIP_STATE_ROOT by the conftest's autouse _pin_platform_root
    # fixture, the same convention test_hpo_ray_lifecycle.py's _run_one_search relies on.
    return tune_search(
        objective_fn=lambda config, report: None,
        param_space=param_space,
        num_samples=2,
        search_alg="random",
        scheduler=None,
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
        storage_path=str(Path(os.environ["TCIP_STATE_ROOT"]) / "hpo"),
    )


_SPACE = {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}


def test_a_never_reported_trial_becomes_an_error_row_not_a_crash(monkeypatch):
    reported = _ReportedResult(config={"lr": 0.05}, metrics={"objective": 0.3, "training_iteration": 4})
    never_reported = Result(
        metrics={}, checkpoint=None, error=RuntimeError("worker died"), path="/fake/trial_1",
    )
    assert never_reported.config is None  # the premise this change reacts to
    _install_fake_ray(monkeypatch, [reported, never_reported])

    result = _run_search(_SPACE)

    all_trials = result["all_trials"]
    assert len(all_trials) == 2

    dead = next(row for row in all_trials if row["params"] is None)
    assert dead == {
        "params": None, "value": None, "iterations": None,
        "state": "ERROR", "error": dead["error"],
    }
    assert "never reported" in dead["error"]
    assert "worker died" in dead["error"]

    assert result["best_params"] == {"lr": 0.05}


def test_every_trial_reporting_yields_the_unchanged_row_shape(monkeypatch):
    first = _ReportedResult(config={"lr": 0.01}, metrics={"objective": 0.2, "training_iteration": 2})
    second = _ReportedResult(config={"lr": 0.02}, metrics={"objective": 0.4, "training_iteration": 3})
    _install_fake_ray(monkeypatch, [first, second])

    result = _run_search(_SPACE)

    all_trials = result["all_trials"]
    assert len(all_trials) == 2
    for row in all_trials:
        assert row["state"] == "COMPLETE"
        assert "error" not in row
        assert set(row) == {"params", "value", "iterations", "state"}
    assert result["best_params"] == {"lr": 0.01}


def test_ray_s_config_property_is_none_only_when_metrics_never_carried_one():
    """The premise the fix relies on is Ray's own ``Result.config`` behavior, not an assumption
    this test suite invents: a result with no reported metrics has no config, and a result
    whose metrics carry one answers it back."""
    never_reported = Result(metrics={}, checkpoint=None, error=None, path="/fake/trial_0")
    reported = Result(
        metrics={"config": {"lr": 0.1}}, checkpoint=None, error=None, path="/fake/trial_1",
    )

    assert never_reported.config is None
    assert reported.config == {"lr": 0.1}
