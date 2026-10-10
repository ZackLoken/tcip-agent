"""Cooperative sweep cancel: ``cancel_training`` over a sweep's id, a sweep's canceled final
status, ``_run_hpo_trial``'s entry check, and the sweep ``Stopper``'s two stop-all conditions. The
cancel is the sweep directory's one cancellation record (``experiments.request_cancel``), which
every trial of it polls.
"""

from __future__ import annotations

import pytest

from tests._training_values import fifo_search, sweep_space
from tests._verified_checkpoint_fixtures import opened_sweep


def _stub_search(monkeypatch, *, before=None, raises: Exception | None = None) -> list[dict]:
    """Replace Ray Tune's search with one answering the sweep's log directory, first calling
    ``before(sweep_root)``, or raising ``raises``; returns the calls it received."""
    calls: list[dict] = []

    def fake_search(**kw):
        calls.append(kw)
        sweep_root = kw["sweep_dir"]
        if before is not None:
            before(sweep_root)
        if raises is not None:
            raise raises

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    return calls


def test_a_cancel_naming_no_run_or_sweep_refuses(project) -> None:
    from tcip_mcp.experiments import experiment_dir
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(project, "hpo_totally_unknown", actor=None)
    assert "error" in result
    assert not experiment_dir("hpo_totally_unknown", project=project).exists()


def test_a_cancel_written_before_the_run_ends_the_sweep_canceled_before_its_first_trial(
    project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel the opened sweep's directory holds when its run starts (a relaunch canceled
    before its worker ran) ends the sweep canceled, its final status carrying the reason, and
    never starts the search."""
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import (
        _CANCEL_BEFORE_START_REASON, monitor_training, run_sweep,
    )

    calls = _stub_search(monkeypatch)
    opened = opened_sweep(project, real_hpo_base_config)
    experiments.request_cancel(opened)

    result = run_sweep(opened)

    assert (result.state, result.status_error) == ("canceled", _CANCEL_BEFORE_START_REASON)
    assert calls == []
    sweep = monitor_training(project, opened.name)["sweep"]
    assert (sweep["state"], sweep["status_error"]) == ("canceled", _CANCEL_BEFORE_START_REASON)


@pytest.mark.parametrize("raises", [None, RuntimeError("the Ray cluster was torn down mid-sweep")],
                         ids=["search-returns", "search-raises"])
def test_a_cancel_landing_mid_search_ends_the_sweep_canceled(
    project, real_hpo_base_config, monkeypatch, raises,
) -> None:
    """A cancel that lands while the search runs ends the sweep canceled whether the search
    returns or raises once it stops."""
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import (
        _CANCEL_DURING_RUN_REASON, monitor_training, run_hyperparameter_search,
    )

    _stub_search(monkeypatch, before=experiments.request_cancel, raises=raises)

    result = run_hyperparameter_search(project, base_config=real_hpo_base_config,
                                       param_space=sweep_space(), n_trials=1, **fifo_search(),
                                       search_seed=0)

    assert (result["sweep"]["state"], result["sweep"]["status_error"]) == (
        "canceled", _CANCEL_DURING_RUN_REASON)
    sweep = monitor_training(project, result["sweep"]["sweep_id"])["sweep"]
    assert (sweep["state"], sweep["status_error"]) == ("canceled", _CANCEL_DURING_RUN_REASON)


def test_a_cancel_after_the_sweep_ended_refuses_and_leaves_its_final_status_as_written(
    project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel arriving after the sweep ended refuses by name and writes nothing: its final
    status is the one it ended with."""
    from tcip_mcp.experiments import cancel_requested, experiment_dir
    from tcip_mcp.tools.training_tools import (
        cancel_training, monitor_training, run_hyperparameter_search,
    )

    _stub_search(monkeypatch)
    sweep_id = run_hyperparameter_search(project, base_config=real_hpo_base_config,
                                         param_space=sweep_space(), n_trials=1,
                                         **fifo_search(), search_seed=0)["sweep"]["sweep_id"]

    result = cancel_training(project, sweep_id, actor=None)

    assert "has ended" in result["error"]
    assert not cancel_requested(experiment_dir(sweep_id, project=project))
    assert monitor_training(project, sweep_id)["sweep"]["state"] == "completed"


def test_a_second_cancel_keeps_the_time_of_the_first(project, real_hpo_base_config) -> None:
    """The cancellation record is written once: a repeated request keeps the first request's
    time."""
    from tcip_mcp.experiments import request_cancel

    opened = opened_sweep(project, real_hpo_base_config)
    first = request_cancel(opened)
    second = request_cancel(opened)

    assert second == first


def test_a_sweep_cancel_reaches_only_a_sweep_of_the_project_it_is_handed(
    project, real_hpo_base_config, tmp_path_factory,
) -> None:
    """The cancel resolves the sweep under the project it is handed; another project holds no
    such sweep and refuses."""
    from tcip_mcp.experiments import cancel_requested, experiment_dir
    from tcip_mcp.tools.training_tools import cancel_training

    sweep_id = opened_sweep(project, real_hpo_base_config).name
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    result = cancel_training(project, sweep_id, actor=None)

    assert result["cancel_requested"] is True
    assert cancel_requested(experiment_dir(sweep_id, project=project))
    assert "error" in cancel_training(elsewhere, sweep_id, actor=None)


def test_run_hpo_trial_reports_nothing_and_opens_no_run_when_the_sweep_is_canceled(
    project, real_hpo_base_config,
) -> None:
    from tcip_mcp.experiments import request_cancel, run_dirs
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    opened = opened_sweep(project, real_hpo_base_config)
    request_cancel(opened)

    reported: list[float] = []
    _run_hpo_trial({}, reported.append, opened, "aaa00000")

    assert reported == []
    assert run_dirs(project) == []


class _FakeTrial:
    """Stands in for a ``ray.tune.experiment.Trial`` in the callback: only ``trial_id`` is read."""

    def __init__(self, trial_id: str) -> None:
        self.trial_id = trial_id


def test_sweep_stopper_stop_all_true_once_ray_no_longer_holds_any_trial_live(tmp_path) -> None:
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper

    stopper, callback = _build_sweep_stopper(tmp_path)
    assert stopper.stop_all() is False  # no cancel requested yet

    trial = _FakeTrial("running_trial")
    callback.on_trial_start(0, [], trial)
    request_cancel(tmp_path)
    assert stopper.stop_all() is False  # Ray still holds the trial live
    assert stopper("running_trial", {}) is True  # per-trial call always follows the request

    callback.on_trial_complete(0, [], trial)
    assert stopper.stop_all() is True  # Ray no longer holds any trial live


def test_sweep_stopper_stop_all_true_at_once_for_a_trial_ray_killed_outright(tmp_path) -> None:
    """A trial Ray kills outright never writes its final status, so a disk-based check alone
    would read it as running until the heartbeat window passed; the callback's own live set, fed
    by Ray's own ``on_trial_error`` report, says otherwise at once."""
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper

    request_cancel(tmp_path)
    stopper, callback = _build_sweep_stopper(tmp_path)
    trial = _FakeTrial("killed_trial")
    callback.on_trial_start(0, [], trial)
    assert stopper.stop_all() is False
    callback.on_trial_error(0, [], trial)

    assert stopper.stop_all() is True


def test_sweep_stopper_stop_all_true_after_the_stale_window_elapses_regardless_of_a_live_trial(
    tmp_path, monkeypatch
) -> None:
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper

    stopper, callback = _build_sweep_stopper(tmp_path)
    callback.on_trial_start(0, [], _FakeTrial("stuck_trial"))  # Ray never reports it finished

    experiments.request_cancel(tmp_path)
    assert stopper.stop_all() is False  # freshly requested: still inside the window

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert stopper.stop_all() is True  # the bounded fallback: Ray's own stop takes over


def test_run_hyperparameter_search_refuses_a_relaunched_from_naming_no_sweep_under_this_root(
    project, real_hpo_base_config, monkeypatch,
) -> None:
    """relaunched_from must name a sweep this root holds: a name that resolves to nothing is
    refused before any directory is made, rather than recorded as lineage nothing answers for;
    a launch naming none is untouched by the check."""
    from tcip_mcp.experiments import sweep_dirs
    from tcip_mcp.tools.training_tools import run_hyperparameter_search

    _stub_search(monkeypatch)
    result = run_hyperparameter_search(
        project, base_config=real_hpo_base_config, param_space=sweep_space(), n_trials=1,
        **fifo_search(), relaunched_from="hpo_does_not_exist", search_seed=0)
    assert "hpo_does_not_exist" in result["error"]
    assert sweep_dirs(project) == []

    admitted = run_hyperparameter_search(project, base_config=real_hpo_base_config,
                                         param_space=sweep_space(), n_trials=1, **fifo_search(),
                                         search_seed=0)
    assert "error" not in admitted, admitted
