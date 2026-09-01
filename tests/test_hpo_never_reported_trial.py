"""A trial whose actor never answered Ray's first bookkeeping call (it died during actor start,
or never received one) comes back from the grid with no ``config``, since
``ray.air.result.Result.config`` is a property reading ``metrics.get("config")`` and Ray fills
that key only once the actor answers. ``tune_search`` used to assert every result carried a
config, which crashed the whole sweep's result assembly the moment one trial died that way; it
now records the death as an honest ``ERROR`` row instead.

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


_NEVER_ANSWERED_METRICS = {
    "empty": {},
    # What a real grid hands back for a trial whose actor never answered: Ray's own trial id and
    # nothing else, since Trial.last_result always stamps it.
    "bookkeeping only": {"trial_id": "90999_00001"},
}


@pytest.mark.parametrize("metrics", list(_NEVER_ANSWERED_METRICS.values()),
                         ids=list(_NEVER_ANSWERED_METRICS))
def test_a_never_answered_trial_becomes_an_error_row_not_a_crash(monkeypatch, metrics):
    reported = _ReportedResult(config={"lr": 0.05}, metrics={"objective": 0.3, "training_iteration": 4})
    never_answered = Result(
        metrics=metrics, checkpoint=None, error=RuntimeError("worker died"), path="/fake/trial_1",
    )
    assert never_answered.config is None  # the premise this change reacts to
    _install_fake_ray(monkeypatch, [reported, never_answered])

    result = _run_search(_SPACE)

    all_trials = result["all_trials"]
    assert len(all_trials) == 2

    dead = next(row for row in all_trials if row["params"] is None)
    assert dead == {
        "params": None, "value": None, "iterations": None,
        "state": "ERROR", "error": dead["error"],
    }
    assert "never answered" in dead["error"]
    assert "RuntimeError: worker died" in dead["error"]
    assert "\n" not in dead["error"]

    assert result["best_params"] == {"lr": 0.05}


def test_a_reported_trial_that_failed_carries_its_cause_on_one_clean_line(monkeypatch):
    """Ray wraps a trial's exception in a RayTaskError whose text is the worker's whole coloured
    traceback; the durable row keeps the cause's own type and message, nothing more."""
    from ray.exceptions import RayTaskError

    traceback_text = (
        "\x1b[36mray::ImplicitFunc.train()\x1b[39m (pid=12700)\n"
        "  File \"C:\\somewhere\\trainable.py\", line 19, in trainable\n"
        "RuntimeError: worker exploded before reporting"
    )
    failed = _ReportedResult(
        config={"lr": 0.9}, metrics={"config": {"lr": 0.9}, "trial_id": "t1"},
        error=RayTaskError("trainable", traceback_text,
                           RuntimeError("worker exploded before reporting")),
    )
    reported = _ReportedResult(config={"lr": 0.05}, metrics={"objective": 0.3, "training_iteration": 4})
    _install_fake_ray(monkeypatch, [reported, failed])

    result = _run_search(_SPACE)

    row = next(row for row in result["all_trials"] if row["state"] == "ERROR")
    assert row["params"] == {"lr": 0.9}
    assert row["error"] == "RuntimeError: worker exploded before reporting"


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
    this test suite invents: a result with no metrics, or with only Ray's own bookkeeping and
    no ``config`` key, has no config; a result whose metrics carry one answers it back."""
    empty = Result(metrics={}, checkpoint=None, error=None, path="/fake/trial_0")
    bookkeeping_only = Result(
        metrics={"trial_id": "90999_00001"}, checkpoint=None, error=None, path="/fake/trial_1",
    )
    reported = Result(
        metrics={"config": {"lr": 0.1}}, checkpoint=None, error=None, path="/fake/trial_2",
    )

    assert empty.config is None
    assert bookkeeping_only.config is None
    assert reported.config == {"lr": 0.1}
