"""Cooperative sweep cancel: ``cancel_hyperparameter_search``, a sweep's canceled final status,
``_run_hpo_trial``'s entry check, and the sweep ``Stopper``'s two stop-all conditions. The cancel
is the sweep directory's one cancellation record (``experiments.request_cancel``), which every
trial of it polls.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _stub_search(monkeypatch, *, before=None, raises: Exception | None = None) -> list[dict]:
    """Replace Ray Tune's search with one answering the sweep's log directory, first calling
    ``before(sweep_root)``, or raising ``raises``; returns the calls it received."""
    calls: list[dict] = []

    def fake_search(**kw):
        calls.append(kw)
        sweep_root = Path(kw["storage_path"]) / kw["study_name"]
        if before is not None:
            before(sweep_root)
        if raises is not None:
            raise raises
        return str(sweep_root)

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    return calls


def _opened(project: Path, base_config: dict, study_name: str):
    """A sweep of ``project`` over ``base_config`` whose input its own writer wrote
    (``open_sweep``), at the arguments ``run_hyperparameter_search`` defaults to, one trial and
    search seed 0."""
    from tcip_mcp.tools.training_tools import open_sweep

    opened = open_sweep(
        project, base_config, None, n_trials=1, search_alg="random", scheduler="asha", grace_period=5,
        reduction_factor=3, warm_start=False, baseline_params=None, max_concurrent=1,
        resources_per_trial=None, study_name=study_name, split_draws=1, split_draw_seeds=None,
        search_seed=0, trial_budget=None, relaunched_from=None)
    assert not isinstance(opened, dict), opened
    return opened


def test_cancel_hyperparameter_search_refuses_a_study_no_directory_holds(project) -> None:
    from tcip_mcp.tools.training_tools import cancel_hyperparameter_search, sweep_dir

    result = cancel_hyperparameter_search(project, "hpo_totally_unknown")
    assert "error" in result
    assert not sweep_dir("hpo_totally_unknown", project=project).exists()


def test_a_cancel_written_before_the_run_ends_the_sweep_canceled_before_its_first_trial(
    project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel the opened sweep's directory holds when its run starts (a relaunch the web route
    canceled before its worker ran) ends the sweep canceled, its final status carrying the
    reason, and never starts the search."""
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import (
        _CANCEL_BEFORE_START_REASON, monitor_training, run_sweep,
    )

    calls = _stub_search(monkeypatch)
    opened = _opened(project, real_hpo_base_config, "hpo_precancel1")
    experiments.request_cancel(opened.directory)

    result = run_sweep(opened, auto_tensorboard=False)

    assert (result["status"], result["error"]) == ("canceled", _CANCEL_BEFORE_START_REASON)
    assert calls == []
    sweep = monitor_training(project, sweep_id="hpo_precancel1")
    assert (sweep["status"], sweep["error"]) == ("canceled", _CANCEL_BEFORE_START_REASON)


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

    result = run_hyperparameter_search(project, base_config=real_hpo_base_config, n_trials=1,
                                       study_name="hpo_midcancel1", search_seed=0)

    assert (result["status"], result["error"]) == ("canceled", _CANCEL_DURING_RUN_REASON)
    sweep = monitor_training(project, sweep_id="hpo_midcancel1")
    assert (sweep["status"], sweep["error"]) == ("canceled", _CANCEL_DURING_RUN_REASON)


def test_a_cancel_after_the_sweep_ended_refuses_and_leaves_its_final_status_as_written(
    project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel arriving after the sweep ended refuses by name and writes nothing: its final
    status is the one it ended with."""
    from tcip_mcp.experiments import cancel_requested
    from tcip_mcp.tools.training_tools import (
        cancel_hyperparameter_search, monitor_training, run_hyperparameter_search, sweep_dir,
    )

    _stub_search(monkeypatch)
    run_hyperparameter_search(project, base_config=real_hpo_base_config, n_trials=1,
                              study_name="hpo_alreadydone1", search_seed=0)

    result = cancel_hyperparameter_search(project, "hpo_alreadydone1")

    assert "has ended" in result["error"]
    assert not cancel_requested(sweep_dir("hpo_alreadydone1", project=project))
    assert monitor_training(project, sweep_id="hpo_alreadydone1")["status"] == "completed"


def test_a_second_cancel_keeps_the_time_of_the_first(project, real_hpo_base_config) -> None:
    """The cancellation record is written once: a repeated request keeps the first request's
    time."""
    from tcip_mcp.experiments import request_cancel

    opened = _opened(project, real_hpo_base_config, "hpo_twicecancel1")
    first = request_cancel(opened.directory)
    second = request_cancel(opened.directory)

    assert second == first


def test_cancel_hyperparameter_search_reaches_only_a_sweep_of_the_project_it_is_handed(
    project, real_hpo_base_config, tmp_path_factory,
) -> None:
    """The cancel resolves the sweep under the project it is handed; another project holds no
    such sweep and refuses."""
    from tcip_mcp.experiments import cancel_requested
    from tcip_mcp.tools.training_tools import cancel_hyperparameter_search, sweep_dir

    _opened(project, real_hpo_base_config, "hpo_other01")
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    result = cancel_hyperparameter_search(project, "hpo_other01")

    assert result["cancel_requested"] is True
    assert cancel_requested(sweep_dir("hpo_other01", project=project))
    assert "error" in cancel_hyperparameter_search(elsewhere, "hpo_other01")


def test_run_hpo_trial_reports_nothing_and_opens_no_run_when_the_sweep_is_canceled(
    project, real_hpo_base_config,
) -> None:
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    opened = _opened(project, real_hpo_base_config, "hpo_trialcancel1")
    request_cancel(opened.directory)
    trial_dir = opened.directory / "trial_aaa00000"

    reported: list[float] = []
    _run_hpo_trial({}, reported.append, real_hpo_base_config, trial_dir, project=project,
                   objective={"selection_metric": "loss", "higher_is_better": False},
                   launched_by={"launcher": "process"})

    assert reported == []
    assert not trial_dir.exists()


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
    from tcip_mcp.tools.training_tools import run_hyperparameter_search, sweep_dir

    _stub_search(monkeypatch)
    result = run_hyperparameter_search(
        project, base_config=real_hpo_base_config, n_trials=1, study_name="hpo_refused_relaunch1",
        relaunched_from="hpo_does_not_exist", search_seed=0)
    assert "hpo_does_not_exist" in result["error"]
    assert not sweep_dir("hpo_refused_relaunch1", project=project).exists()

    admitted = run_hyperparameter_search(project, base_config=real_hpo_base_config, n_trials=1,
                                         study_name="hpo_norelaunch1", search_seed=0)
    assert "error" not in admitted, admitted
