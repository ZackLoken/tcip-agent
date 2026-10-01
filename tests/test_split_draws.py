"""Split sensitivity as a run_hyperparameter_search sweep factor: split_draws pairs data.split.seed as a grid
axis with every sampled point through Ray's own BasicVariantGenerator(constant_grid_search=
True); run_hyperparameter_search groups the resulting trials by point and picks the best by mean over each
point's draws. tune_search is faked throughout the door's own tests here (as test_hpo_durable.py
does), while the door's own trial_budget count imports Ray and counts a real search space
whenever a bound is read, so a test above one draw or naming a trial_budget pays that import and
no training; the module's real-Ray sweep tests (below, under "a real Ray sweep") keep running
real trials as before.
"""

from __future__ import annotations

import pytest


def _never_search(ran: list):
    """A ``tune_search`` stand-in that records it ran, for a refusal test to assert on: a
    refused sweep must never reach the search at all."""
    def fake_search(**kw):
        ran.append(1)
        return {"study_name": kw.get("study_name")}
    return fake_search


def test_real_hpo_base_config_is_admitted_by_preflight(tmp_path, real_hpo_base_config):
    """Holds the fixture's own claim: preflight_config, the sweep door's own admissibility
    check, reports the fixture valid with no issues."""
    from tcip_mcp.tools.training_tools import preflight_config

    result = preflight_config(tmp_path, real_hpo_base_config)
    assert result["issues"] == []
    assert result["valid"] is True


# -- the door's own split_draws argument and trial_budget guards --------------------


@pytest.mark.parametrize("value", [0, -3, False], ids=["zero", "negative", "the-bool-False"])
def test_run_hyperparameter_search_refuses_split_draws_below_one(
    tmp_path, real_hpo_base_config, monkeypatch, value,
):
    """guard. split_draws at or below one (zero, a negative int, or the bool False, which reads
    as the integer it names) refuses by name at the door's own argument leg, before the search is
    ever reached, with no seed axis in play to confuse the reason."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=value, search_seed=0
    )

    assert "error" in result and "pairs no partition" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_non_integer_split_draws_by_direct_call(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. A fractional split_draws (reachable by a direct Python call alone, since the MCP
    boundary rejects it) refuses as not a draw count rather than raising out of range(2.5)."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2.5, search_seed=0
    )

    assert "error" in result and "is not a draw count" in result["error"]
    assert not ran


@pytest.mark.parametrize("value", ["2", None, [2]])
def test_run_hyperparameter_search_refuses_a_split_draws_of_another_type_by_direct_call(
    tmp_path, real_hpo_base_config, monkeypatch, value,
):
    """guard. A split_draws of a type that does not order against an integer (reachable by a
    direct Python call alone) refuses as not a draw count, rather than raising TypeError out of
    the audited door where the comparison deciding whether a bound is read would meet it."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=value, search_seed=0
    )

    assert "error" in result and "is not a draw count" in result["error"]
    assert not ran


@pytest.mark.parametrize("value", [0, -1])
def test_run_hyperparameter_search_refuses_n_trials_not_a_count_when_a_bound_is_read(
    tmp_path, real_hpo_base_config, monkeypatch, value,
):
    """guard. n_trials that is not a positive count refuses at the budget leg on a call that
    reads a bound (here, a paired launch with no stated trial_budget), the search never reached:
    a negative n_trials counts zero through Ray's own generator while Tuner.fit runs it
    unbounded, and a zero n_trials launches nothing, so the bound the leg exists to check would
    admit exactly what it should refuse without this clause."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=value,
        scheduler="none", split_draws=2, search_seed=0
    )

    assert "error" in result and "is not a count of trials" in result["error"]
    assert not ran


@pytest.mark.parametrize("value", [0, -1])
def test_run_hyperparameter_search_admits_n_trials_not_a_count_at_one_draw_with_no_budget(
    tmp_path, real_hpo_base_config, monkeypatch, value,
):
    """coverage. n_trials=0 or -1 at one draw with no trial_budget reads no bound at all, so the
    budget leg is never called and the call behaves exactly as it did before this family: the
    search is reached and n_trials is passed through unread by the door."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": kw["num_samples"],
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=value, search_seed=0
    )

    assert "error" not in result, result
    assert captured["num_samples"] == value


def test_run_hyperparameter_search_refuses_split_draws_above_one_with_no_trial_budget(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """guard. A launch (never a relaunch) above one draw that states no trial_budget refuses
    naming the budget Ray's own variant count over the built space would admit, the search never
    reached: the paired grid multiplies a total the call never stated."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2, search_seed=0
    )

    assert "error" in result and "trial_budget=2" in result["error"]
    assert "would launch 2 trials" in result["error"]
    assert not ran


@pytest.mark.parametrize("value", [0, -1, True, "5"])
def test_run_hyperparameter_search_refuses_a_trial_budget_not_a_count(
    tmp_path, real_hpo_base_config, monkeypatch, value,
):
    """coverage. trial_budget that is not a positive int (a bool included, by name) refuses as
    not a count of trials, at one draw where the keyword is new."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", trial_budget=value, search_seed=0
    )

    assert "error" in result and "is not a count of trials" in result["error"]
    assert not ran


# -- refusals --------------------------------------------------------------------


def test_run_hyperparameter_search_refuses_split_draws_when_a_bound_selection_would_starve_a_side(
    tmp_path, monkeypatch,
):
    """A selection whose train-plus-val members hold one foreground group refuses the sweep
    before minting, the same distinct-groups check preflight_config runs for one config."""
    import tcip_mcp.tools.training_tools as tt

    from tests.test_selection_binding import one_foreground_group_selection

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    _root, selection_dir = one_foreground_group_selection(tmp_path)
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"split": {"selection_dir": str(selection_dir)}},
    }
    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, search_seed=0)

    assert "error" in result and "would starve" in result["error"]
    assert not ran
    assert list(tmp_path.glob("hpo_*")) == []


def test_run_hyperparameter_search_refuses_split_draws_when_auto_val_is_off(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "auto_val": False}
    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, search_seed=0)

    assert "error" in result and "auto_val" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_with_a_non_native_search_alg(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                        search_alg="bayesopt", scheduler="none", split_draws=2, search_seed=0)

    assert "error" in result and "search_alg" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_with_a_pruning_scheduler(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                        scheduler="asha", split_draws=2, search_seed=0)

    assert "error" in result and "scheduler" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draw_seeds_of_the_wrong_length(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                        scheduler="none", split_draws=3, split_draw_seeds=[1, 2], search_seed=0)

    assert "error" in result and "split_draw_seeds" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_param_space_already_sweeps_the_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config,
        param_space={"data.split.seed": {"type": "categorical", "choices": [1, 2]}},
        n_trials=1, scheduler="none", split_draws=2, search_seed=0
    )

    assert "error" in result and "data.split.seed" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_warm_start_baseline_names_the_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2, warm_start=True,
        baseline_params={"lr": 0.01, "data.split.seed": 7}, search_seed=0
    )

    assert "error" in result and "baseline_params" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_param_space_sweeps_another_data_axis(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config,
        param_space={"data.split.val_ratio": {"type": "uniform", "low": 0.1, "high": 0.3}},
        n_trials=1, scheduler="none", split_draws=2, search_seed=0
    )

    assert "error" in result and "data.split.val_ratio" in result["error"]
    assert not ran


def test_tune_search_refuses_split_draws_without_the_axis_in_param_space(tmp_path):
    """tune_search itself, called directly (not through run_hyperparameter_search's own axis-adding), raises
    rather than pairing nothing: a caller building the space by hand must include the axis."""
    from tcip_mcp.pipelines.training.hpo import tune_search

    with pytest.raises(ValueError, match="data.split.seed"):
        tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space={"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}},
            storage_path=str(tmp_path), split_draws=2, seed=0, project=tmp_path
        )


# -- a caller-supplied seed axis at one draw ----------------------------------------


@pytest.mark.parametrize("case", [
    pytest.param(
        {"param_space": {"data.split.seed": {"type": "categorical", "choices": [1, 2]}}},
        id="sampled-under-random",
    ),
    pytest.param(
        {"search_alg": "grid", "param_space": {
            "data.split.seed": {"type": "categorical", "choices": [1, 2]},
            "lr": {"type": "categorical", "choices": [0.1, 0.2]},
        }},
        id="gridded-beside-a-discrete-axis",
    ),
    pytest.param(
        {"param_space": {"data.split.seed": {"type": "uniform", "low": 1, "high": 100}}},
        id="a-float-range",
    ),
])
def test_run_hyperparameter_search_refuses_a_caller_split_seed_axis_at_one_draw(
    tmp_path, real_hpo_base_config, monkeypatch, case,
):
    """A caller-supplied data.split.seed axis with split_draws at 1 (unset) refuses whatever
    the sampler and however the axis is typed; tune_search must never be reached, and the error
    dict carries the remedy."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", search_seed=0, **case,
    )

    assert "error" in result and "cannot be replayed as recorded" in result["error"]
    assert "drop data.split.seed from param_space" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_zero_before_the_seed_axis_leg(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """guard. The door's own argument leg refuses split_draws=0 by name before the seed-axis leg
    ever runs: a caller-supplied data.split.seed axis at split_draws=0 hears the below-one
    reason, never the seed-axis one, since the argument leg runs first for every call."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=0,
        param_space={"data.split.seed": {"type": "categorical", "choices": [1, 2]}}, search_seed=0
    )

    assert "error" in result and "pairs no partition" in result["error"]
    assert "cannot be replayed as recorded" not in result["error"]
    assert not ran


def test_caller_split_seed_refusal_tolerates_an_infinite_split_draws_value():
    """A JSON Infinity literal decodes to float("inf") through the store's own plain
    json.loads even though its encode refuses to write one, so a manifest of unknown
    provenance can carry it; int(float("inf")) raises OverflowError. It is unreadable as a draw
    count and refuses nothing, the same tolerance an unreadable string already has."""
    import tcip_mcp.tools.training_tools as tt

    assert tt.caller_split_seed_refusal(
        {"data.split.seed": {"type": "categorical", "choices": [1, 2]}}, float("inf"),
    ) is None


def test_run_hyperparameter_search_refuses_repeated_split_draw_seeds(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """split_draw_seeds naming the same seed twice is not a spread over distinct partitions,
    and could read complete and eligible on one seed twice against group_split_draws's own
    rule; refused by the repeated value's own name."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2, split_draw_seeds=[7, 7], search_seed=0
    )

    assert "error" in result and "[7]" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_distinct_split_draw_seeds_beside_the_repeated_case(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The same call with distinct seeds is admitted, in the same module as the repeated-seed
    refusal above, so the refusal is proven to reject only the repeat, not the pair."""
    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2, split_draw_seeds=[7, 8], trial_budget=2, search_seed=0
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_admits_the_paired_path_with_a_param_space_beside_distinct_seeds(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """split_draws=2 with distinct split_draw_seeds admits a param_space too (the producer's
    own get_default_space(), which names no data.* axis), beside the existing case that omits
    param_space entirely."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.pipelines.training.hpo import get_default_space

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space=get_default_space(),
        n_trials=1,
        scheduler="none", split_draws=2, split_draw_seeds=[7, 8], trial_budget=2, search_seed=0
    )

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"]["choices"] == [7, 8]


def test_run_hyperparameter_search_admits_the_default_space_at_one_draw(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The platform's own get_default_space() names no data.split.seed axis, so it is admitted
    at split_draws=1 (unset), tune_search is reached, and the space passes through unaltered."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.pipelines.training.hpo import get_default_space

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"study_name": kw["study_name"]}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    space = get_default_space()
    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space=space,
        n_trials=1, search_seed=0
    )

    assert "error" not in result, result
    assert captured["param_space"] == space


def test_run_hyperparameter_search_reports_the_seed_axis_refusal_over_a_preflight_failure(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """A call that trips both this refusal and preflight (an unimportable builder here) answers
    this refusal: caller_split_seed_refusal is checked before preflight runs, and every branch
    ahead of that call is total, so the insertion point is always reached first."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["model_source"] = {**cfg["model_source"], "builder": "tests.no_such_module:no_such_builder"}

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=cfg,
        param_space={"data.split.seed": {"type": "categorical", "choices": [1, 2]}},
        n_trials=1, scheduler="none", search_seed=0
    )

    assert "error" in result and "cannot be replayed as recorded" in result["error"]
    assert "preflight" not in result["error"]
    assert not ran


# -- admits valid work -------------------------------------------------------------


def _bound_hpo_config(selection_dir, *, auto_val: bool | None = None) -> dict:
    data: dict = {"split": {"selection_dir": str(selection_dir)}}
    if auto_val is not None:
        data["auto_val"] = auto_val
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": data,
    }


def test_run_hyperparameter_search_admits_split_draws_bound_to_a_selection_and_sets_the_redraw_flag(
    tmp_path, monkeypatch,
):
    """A base_config bound to a selection is no longer refused: run_hyperparameter_search sets
    data.split.redraw_within_selection on its own copy, so every trial redraws train and val
    inside the selection's own members instead of its one recorded partition, and the recorded
    sweep manifest's own base_config carries the claim."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import draw_splits

    from tests.test_selection_binding import SUBJECT, _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    selection_dir = tmp_path / "m"
    make_result = draw_splits(tmp_path, str(root), output_path=str(selection_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in make_result, make_result

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    cfg = _bound_hpo_config(selection_dir)
    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical", "choices": [42, 43],
    }
    assert cfg["data"]["split"] == {"selection_dir": str(selection_dir)}  # the caller's own copy

    manifest = tt.monitor_training(tmp_path, sweep_id=result["study_name"])["input"]
    recorded_split = manifest["base_config"]["data"]["split"]
    assert recorded_split["redraw_within_selection"] is True
    assert recorded_split["seed"] == 42


def test_run_hyperparameter_search_admits_split_draws_bound_with_auto_val_false(tmp_path, monkeypatch):
    """A bound base_config does not read auto_val (the selection branch binds ahead of it), so
    auto_val=False no longer refuses it the way it refuses a drawn config."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import draw_splits

    from tests.test_selection_binding import SUBJECT, _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    selection_dir = tmp_path / "m"
    make_result = draw_splits(tmp_path, str(root), output_path=str(selection_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in make_result, make_result

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    cfg = _bound_hpo_config(selection_dir, auto_val=False)
    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result, result


def test_run_hyperparameter_search_admits_split_draws_and_derives_seeds_from_the_base_config(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The default draw seeds are the base config's own data.split.seed (else 42) plus the
    draw index, and the space Ray actually searches carries the paired grid axis."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {"lr": 0.1}, "best_value": 0.2, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=2,
                        scheduler="none", split_draws=3, trial_budget=6, search_seed=0)

    assert "error" not in result
    assert captured["split_draws"] == 3
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical", "choices": [42, 43, 44],
    }

    manifest = tt.monitor_training(tmp_path, sweep_id=result["study_name"])["input"]
    assert manifest["split_draws"] == 3
    assert manifest["split_draw_seeds"] == [42, 43, 44]
    assert "data.split.seed" not in manifest["param_space"]  # the caller's own axes, unaugmented


def test_run_hyperparameter_search_admits_split_draws_with_explicit_seeds(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                        scheduler="none", split_draws=2, split_draw_seeds=[7, 99], trial_budget=2, search_seed=0)

    assert "error" not in result
    assert captured["param_space"]["data.split.seed"]["choices"] == [7, 99]


@pytest.mark.parametrize("search_alg, budget", [("grid", 6), ("variant_generator", 2)])
def test_run_hyperparameter_search_admits_split_draws_with_a_native_search_alg(
    tmp_path, real_hpo_base_config, monkeypatch, search_alg, budget,
):
    """Every native search_alg (random, grid, variant_generator) builds the paired
    BasicVariantGenerator; only a backend searcher is refused. Each budget is Ray's own count
    over the default space at two draws: three batch_size values under grid, a sampled space
    under variant_generator."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                        search_alg=search_alg, scheduler="none", split_draws=2, trial_budget=budget, search_seed=0)

    assert "error" not in result
    assert captured["search_alg"] == search_alg


def test_run_hyperparameter_search_admits_split_draws_for_instance_seg(tmp_path, real_hpo_base_config, monkeypatch):
    """instance_seg over polygon ground truth is admitted to split_draws the way detection is."""
    import tcip_mcp.tools.training_tools as tt
    from tests._verified_checkpoint_fixtures import detection_images

    cfg = dict(real_hpo_base_config)
    cfg["model_source"] = {**cfg["model_source"], "task": "instance_seg"}
    scope = cfg["data"]["scope"]
    cfg["data"] = {**detection_images(tmp_path / "polygons", scope, polygons=True),
                   "scope": scope}

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result


def test_run_hyperparameter_search_admits_split_draws_with_explicit_auto_val_true(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """data.auto_val=True explicitly stated (not merely defaulted) admits the same as omitting
    it."""
    import tcip_mcp.tools.training_tools as tt

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "auto_val": True}

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result


def test_run_hyperparameter_search_admits_split_draws_with_a_warm_start_not_naming_the_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """warm_start's baseline_params is only refused when it names data.split.seed itself; a
    baseline over the ordinary space stays admitted."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1,
        scheduler="none", split_draws=2, warm_start=True, baseline_params={"lr": 0.01},
        trial_budget=2, search_seed=0
    )

    assert "error" not in result
    assert captured["warm_start"] is True


def test_run_hyperparameter_search_admits_a_bare_seed_axis_beside_split_draws(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """The bare 'seed' axis (the run seed) is a different key from data.split.seed and stays
    admitted beside the draws."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config,
        param_space={"seed": {"type": "int", "low": 1, "high": 3}},
        n_trials=1, scheduler="none", split_draws=2, trial_budget=2, search_seed=0
    )

    assert "error" not in result
    assert "seed" in captured["param_space"]
    assert "data.split.seed" in captured["param_space"]


# -- the trial_budget bound ----------------------------------------------------------


def test_run_hyperparameter_search_refuses_a_budget_that_admits_fewer_than_every_draw(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. A stated trial_budget the count exceeds, but not at one draw, refuses
    naming the count, the budget and how many draws it admits: Ray's own count over the default
    space at n_trials=4, split_draws=3, random is 12 (4 per draw), which trial_budget=10 does
    not fit whole, admitting 2 draws."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=4,
        scheduler="none", search_alg="random", split_draws=3, trial_budget=10, search_seed=0
    )

    assert "error" in result
    assert "12" in result["error"] and "trial_budget=10" in result["error"]
    assert "4" in result["error"] and "2 draw" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_budget_at_the_quotient_boundary(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. Ray's own count over the default space at n_trials=2, split_draws=3, grid
    is 18 (6 per draw); trial_budget=6 is not above the per-draw count, so this is the quotient
    branch (1 admitted draw), never the exceeds-at-one-draw branch."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=2,
        scheduler="none", search_alg="grid", split_draws=3, trial_budget=6, search_seed=0
    )

    assert "error" in result
    assert "18" in result["error"] and "trial_budget=6" in result["error"]
    assert "1 draw" in result["error"]
    assert "exceeds" not in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_budget_a_single_draw_already_exceeds(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. Ray's own count over the default space at n_trials=2, split_draws=2, grid
    is 12 (6 per draw); trial_budget=5 is below even one draw's own count, so the sweep exceeds
    the budget at one draw, the exceeds branch, never a positive admitted-draws quotient."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=2,
        scheduler="none", search_alg="grid", split_draws=2, trial_budget=5, search_seed=0
    )

    assert "error" in result
    assert "12" in result["error"] and "trial_budget=5" in result["error"]
    assert "exceeds at one draw" in result["error"] and "6 trials per draw" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_a_budget_the_count_fits_and_records_it(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. n_trials=2, split_draws=3, random over the default space counts 6, which
    trial_budget=6 admits exactly; the manifest records trial_budget verbatim."""
    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": kw["num_samples"],
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=2,
        scheduler="none", search_alg="random", split_draws=3, trial_budget=6, search_seed=0
    )

    assert "error" not in result, result

    manifest = tt.monitor_training(tmp_path, sweep_id=result["study_name"])["input"]
    assert manifest["trial_budget"] == 6


def test_run_hyperparameter_search_refuses_a_one_draw_launch_a_stated_budget_exceeds(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. At one draw a caller need state nothing new, but a stated trial_budget is
    still checked: n_trials=5 over the default space (random, the default alg) counts 5, which
    trial_budget=3 does not admit, the exceeds branch since split_draws is 1."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=5,
        scheduler="none", trial_budget=3, search_seed=0
    )

    assert "error" in result
    assert "5" in result["error"] and "trial_budget=3" in result["error"]
    assert "exceeds at one draw" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_a_one_draw_launch_with_no_stated_budget(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. One draw with no trial_budget reads no bound at all: the manifest still
    gains the key, null, so a relaunch has something to read back."""
    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1, scheduler="none", search_seed=0
    )

    assert "error" not in result, result

    manifest = tt.monitor_training(tmp_path, sweep_id=result["study_name"])["input"]
    assert manifest["trial_budget"] is None


def _record_source_sweep(tmp_path, study_name: str, base_config: dict, monkeypatch) -> None:
    """A finished sweep named ``study_name``, recorded by ``run_hyperparameter_search`` itself
    over a stubbed search: only its directory's ``sweep.json`` is checked by relaunched_from."""
    import tcip_mcp.tools.training_tools as tt

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", lambda **kw: {
        "best_params": {}, "best_value": 0.1, "n_trials": 1, "study_name": kw["study_name"],
        "all_trials": []})
    source = tt.run_hyperparameter_search(tmp_path, base_config=base_config, n_trials=1,
                                          scheduler="none", study_name=study_name, search_seed=0)
    assert "error" not in source, source


def test_run_hyperparameter_search_relaunch_with_no_budget_replays_as_recorded(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. A relaunch (relaunched_from given) with no trial_budget reads no bound at
    all, whatever split_draws says: the door counts nothing and the search is reached."""
    import tcip_mcp.tools.training_tools as tt

    _record_source_sweep(tmp_path, "hpo_relaunch_src1", real_hpo_base_config, monkeypatch)

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": kw["num_samples"],
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1, scheduler="none",
        split_draws=2, relaunched_from="hpo_relaunch_src1", search_seed=0
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_relaunch_with_a_budget_is_still_checked(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. A relaunch that does name a trial_budget is checked exactly as a launch is:
    the count (2, one draw's worth times two draws under random) exceeds trial_budget=1."""
    import tcip_mcp.tools.training_tools as tt

    _record_source_sweep(tmp_path, "hpo_relaunch_src2", real_hpo_base_config, monkeypatch)

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1, scheduler="none",
        split_draws=2, relaunched_from="hpo_relaunch_src2", trial_budget=1, search_seed=0
    )

    assert "error" in result and "2" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_categorical_with_no_choices_as_uncountable(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. A param_space naming a categorical axis with no choices key fails the
    count's own space-building (KeyError), before minting: today the same call mints a manifest
    and fails inside tune_search after the fact; this refusal is earlier, not new in kind."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space={"x": {"type": "categorical"}},
        n_trials=1, scheduler="none", split_draws=2, trial_budget=5, search_seed=0
    )

    assert "error" in result and "cannot be counted" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_the_same_uncountable_space_at_one_draw_with_no_budget(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. The identical categorical-with-no-choices space reads no bound at one draw with
    no trial_budget, so the count never runs and the call reaches the faked search unchanged,
    exactly as it does today."""
    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space={"x": {"type": "categorical"}},
        n_trials=1, scheduler="none", search_seed=0
    )

    assert "error" not in result, result


@pytest.mark.parametrize("search_alg", ["grid", "random"])
def test_run_hyperparameter_search_refuses_an_inverted_int_bound_as_uncountable(
    tmp_path, real_hpo_base_config, monkeypatch, search_alg,
):
    """coverage. An int axis whose low exceeds high turns into an empty grid list; the
    count's own generator counts it silently with no exception (0 under grid, a wrong nonzero
    under random), but the one trial the count draws to validate the space raises (IndexError
    under grid, ValueError under random), so the door still refuses before minting."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config,
        param_space={"batch_size": {"type": "int", "low": 8, "high": 2}},
        n_trials=1, scheduler="none", search_alg=search_alg,
        split_draws=2, trial_budget=5, search_seed=0
    )

    assert "error" in result and "cannot be counted" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_huge_int_span_as_uncountable(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. An int axis spanning more than a Python list can index raises OverflowError
    out of _to_tune_space's own list(range(...)), before any generator is built."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config,
        param_space={"x": {"type": "int", "low": 0, "high": 10**30}},
        n_trials=1, scheduler="none", split_draws=2, trial_budget=5,
        search_seed=0,
    )

    assert "error" in result and "cannot be counted" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_an_empty_param_space_at_one_draw_proving_the_substitution(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. An empty param_space at one draw substitutes the default space
    (get_default_space's own three batch_size choices), the same substitution tune_search's own
    _to_tune_space makes, carried verbatim by split_draw_search_space: trial_budget=5 does not
    admit the default space's own count of 6 under grid at n_trials=2."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space={}, n_trials=2,
        scheduler="none", search_alg="grid", trial_budget=5, search_seed=0
    )

    assert "error" in result and "6" in result["error"] and "trial_budget=5" in result["error"]
    assert not ran


def test_run_hyperparameter_search_an_empty_param_space_above_one_draw_counts_the_seed_axis_alone(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """coverage. Above one draw, an empty param_space runs the seed axis alone (the fact
    stated in The fact), never the substituted default space: at n_trials=1, split_draws=2, grid,
    the count is 2 (one per draw), so trial_budget=1 refuses it and trial_budget=2 admits it."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space={}, n_trials=1,
        scheduler="none", search_alg="grid", split_draws=2, trial_budget=1, search_seed=0
    )

    assert "error" in result and "2" in result["error"] and "1 per draw" in result["error"]
    assert not ran

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": kw["num_samples"],
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space={}, n_trials=1,
        scheduler="none", search_alg="grid", split_draws=2, trial_budget=2, search_seed=0
    )

    assert "error" not in result, result


# -- grouping and the best-by-mean -------------------------------------------------


def _trial(lr: float, seed: int | None, value: float | None, status: str = "completed") -> dict:
    """One trial as the sweep projections read it (``training_tools._trial_row``'s shape)."""
    params: dict = {"lr": lr} if seed is None else {"lr": lr, "data.split.seed": seed}
    return {"trial_id": f"{lr}_{seed}", "status": status,
            "error": "boom" if status == "failed" else None, "has_metrics": True,
            "params": params, "value": value}


def _two_draw_input(project, real_hpo_base_config) -> dict:
    """The input a two-draw sweep over ``real_hpo_base_config`` records, written by its own
    writer (``open_sweep``): its objective is lower=better and its seeds are 42 and 43."""
    import tcip_mcp.tools.training_tools as tt

    opened = tt.open_sweep(
        project, real_hpo_base_config, {"lr": {"type": "categorical", "choices": [0.1, 0.2]}},
        n_trials=2, search_alg="random", scheduler="none", grace_period=5, reduction_factor=3,
        warm_start=False, baseline_params=None, max_concurrent=1, resources_per_trial=None,
        study_name=None, split_draws=2, split_draw_seeds=[42, 43], search_seed=0,
        trial_budget=4, relaunched_from=None)
    assert not isinstance(opened, dict), opened
    return opened.record


def test_a_sweeps_outcome_groups_trials_by_point_and_picks_the_best_by_mean(
    tmp_path, real_hpo_base_config,
):
    """Two points, two draws each; real_hpo_base_config's own metric is lower=better, so the
    point with the lower mean wins, and every point keeps its own block."""
    import tcip_mcp.tools.training_tools as tt

    trials = [
        _trial(0.1, 42, 0.5), _trial(0.1, 43, 0.7),   # mean 0.6
        _trial(0.2, 42, 0.2), _trial(0.2, 43, 0.4),   # mean 0.3 (best)
    ]

    outcome = tt.sweep_outcome(trials, _two_draw_input(tmp_path, real_hpo_base_config))

    assert outcome["best_params"] == {"lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.3)
    spread = outcome["best_value_spread"]
    assert spread["n"] == 2 and spread["n_complete"] == 2
    assert spread["seeds_complete"] == [42, 43]
    assert spread["values"] == [0.2, 0.4]
    assert spread["std"] == pytest.approx(0.14142135623730951)
    points = {tuple(sorted(g["point"].items())) for g in outcome["split_sensitivity"]}
    assert points == {(("lr", 0.1),), (("lr", 0.2),)}


def test_a_sweeps_outcome_marks_a_group_with_a_failed_draw_ineligible(
    tmp_path, real_hpo_base_config,
):
    import tcip_mcp.tools.training_tools as tt

    trials = [
        _trial(0.1, 42, 0.9), _trial(0.1, 43, None, status="failed"),  # ineligible
        _trial(0.2, 42, 0.4), _trial(0.2, 43, 0.5),   # both complete, eligible
    ]

    outcome = tt.sweep_outcome(trials, _two_draw_input(tmp_path, real_hpo_base_config))

    # The lr=0.1 point never became eligible, so lr=0.2 wins even though 0.9 alone looked worse.
    assert outcome["best_params"] == {"lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.45)


def test_a_sweeps_outcome_never_picks_a_repeated_point_that_never_completed_every_planned_seed(
    tmp_path, real_hpo_base_config,
):
    """Ray can repeat a point across grid cells (grid search repeats every point per sample; a
    categorical-only random space collides), landing two complete trials under the same seed
    while the point's other planned seed fails. A count of complete trials alone would call
    that eligible on a mean over one seed twice, and its low value (0.5) would beat the fully
    completed point (0.85) under this fixture's lower-is-better metric; grouping by the planned
    seeds themselves keeps it ineligible."""
    import tcip_mcp.tools.training_tools as tt

    trials = [
        _trial(0.1, 42, 0.5), _trial(0.1, 42, 0.6), _trial(0.1, 43, None, status="failed"),
        _trial(0.2, 42, 0.9), _trial(0.2, 43, 0.8),
    ]

    outcome = tt.sweep_outcome(trials, _two_draw_input(tmp_path, real_hpo_base_config))

    assert outcome["best_params"] == {"lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.85)


def test_a_sweeps_outcome_is_a_null_best_when_no_point_is_eligible(
    tmp_path, real_hpo_base_config,
):
    import tcip_mcp.tools.training_tools as tt

    trials = [_trial(0.1, 42, 0.9), _trial(0.1, 43, None, status="failed")]

    outcome = tt.sweep_outcome(trials, _two_draw_input(tmp_path, real_hpo_base_config))

    assert outcome["best_params"] is None
    assert outcome["best_value"] is None
    assert "best_value_state" not in outcome


# -- group_split_draws direct coverage ----------------------------------------------


def test_group_split_draws_with_no_planned_seeds_groups_a_sweep_without_the_axis():
    """Trials with no data.split.seed axis at all (a sweep that never asked for draws) still
    group cleanly with planned_seeds=[]: one trivially-complete block per distinct point, and a
    point whose trial failed is not eligible."""
    from tcip_mcp.tools.training_tools import group_split_draws

    trials = [_trial(0.1, None, 0.4), _trial(0.2, None, 0.6),
              _trial(0.3, None, None, status="failed")]

    groups = group_split_draws(trials, [])

    assert len(groups) == 3
    by_point = {tuple(sorted(g["point"].items())): g for g in groups}
    assert by_point[(("lr", 0.1),)]["eligible"] is True
    assert by_point[(("lr", 0.1),)]["block"]["n"] == 1
    assert by_point[(("lr", 0.1),)]["block"]["n_complete"] == 1
    assert by_point[(("lr", 0.1),)]["block"]["seeds_complete"] == [None]
    assert by_point[(("lr", 0.1),)]["block"]["std"] is None
    assert by_point[(("lr", 0.3),)]["eligible"] is False


def test_group_split_draws_ineligible_when_a_repeated_point_never_completes_every_planned_seed():
    """A point Ray sampled twice under the same seed (grid search repeats every point per
    sample; a categorical-only random space collides) with its other planned seed failing:
    two complete values reach the split count, but only one of the two planned seeds ever
    completed, so the point is not eligible on a mean over that one seed twice."""
    from tcip_mcp.tools.training_tools import group_split_draws

    trials = [_trial(0.1, 42, 0.5), _trial(0.1, 42, 0.6), _trial(0.1, 43, None, status="failed")]

    groups = group_split_draws(trials, [42, 43])

    assert len(groups) == 1
    group = groups[0]
    assert group["point"] == {"lr": 0.1}
    assert group["block"]["n"] == 3
    assert group["block"]["n_complete"] == 2
    assert group["block"]["seeds_complete"] == [42]
    assert group["eligible"] is False


# -- end to end: a real Ray sweep ----------------------------------------------------


@pytest.mark.ray_cluster
def test_tune_search_split_draws_end_to_end_pairs_every_point_with_every_seed(tmp_path, monkeypatch):
    """A real Ray sweep over a trivial objective (the shape test_hpo_ray_detached_exit.py's
    subprocess script uses: one value reported, resources_per_trial={"cpu": 1}, storage_path
    under tmp_path): split_draws=2 pairs the seed grid with every sampled point through
    BasicVariantGenerator(constant_grid_search=True), so each of the two sampled lr points
    trains once per seed, each trial marking the point it was handed in a file of its own, and
    group_split_draws groups them into two eligible points.

    Not restricted to one platform: test_imbalance_aug_hpo.py's own
    test_tune_search_warm_start_and_optimizes already starts a real Ray cluster on every
    platform through this identical call (only pytest.importorskip("ray")), so this one does
    too, rather than carrying test_hpo_ray_detached_exit.py's Windows-only skip, which guards a
    console-signal exit path this test never touches.
    """
    pytest.importorskip("ray")

    from tcip_mcp.pipelines.training.hpo import tune_search
    from tcip_mcp.tools.training_tools import group_split_draws

    marks = tmp_path / "marks"
    marks.mkdir()

    def objective_fn(config, report):
        (marks / f"{config['lr']!r}__{config['data.split.seed']}").touch()
        report(1.0)

    tune_search(
        objective_fn=objective_fn,
        param_space={
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "data.split.seed": {"type": "categorical", "choices": [42, 43]},
        },
        num_samples=2,
        search_alg="random",
        scheduler=None,
        resources_per_trial={"cpu": 1},
        storage_path=str(tmp_path),
        split_draws=2, seed=0, project=tmp_path
    )

    trials = []
    seeds_by_lr: dict[str, set[int]] = {}
    for mark in marks.iterdir():
        lr, seed = mark.name.split("__")
        seeds_by_lr.setdefault(lr, set()).add(int(seed))
        trials.append({"status": "completed", "value": 1.0,
                       "params": {"lr": lr, "data.split.seed": int(seed)}})

    assert len(trials) == 4  # 2 sampled points x 2 draws
    assert len(seeds_by_lr) == 2
    for seeds in seeds_by_lr.values():
        assert seeds == {42, 43}

    groups = group_split_draws(trials, [42, 43])
    eligible = [g for g in groups if g["eligible"]]
    assert len(eligible) == 2


# -- the single-source spatial-strip leg -----------------------------------------


def _one_source_tiled_cfg(images_dir, labels_dir) -> dict:
    """A base config wrapping a one-source tiled mosaic, the shape ``auto_train_val`` takes
    into ``spatial_single_source_split`` when tiling is on and fewer than two stems are
    admitted, given a ``model_source`` block the way ``real_hpo_base_config`` gives its own."""
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {
            "images_dir": str(images_dir), "labels_dir": str(labels_dir),
            "scope": {"subject": "bud"},
            "auto_val": True, "split": {"val_ratio": 0.2, "test_ratio": 0.1},
            # sliver_frac stated: a fixture this small derives no box-size spread.
            "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2, "sliver_frac": 0.5},
        },
    }


def test_run_hyperparameter_search_refuses_split_draws_over_a_single_source_spatial_split(
    tmp_path, monkeypatch,
):
    """One admitted source under a built-in detection config with tiling on takes the
    single-source spatial-strip path, whose partition no draw of data.split.seed varies:
    split_draws above 1 refuses, naming the one admitted source and the path, and never
    reaches the search."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _big_single_source

    images_dir, labels_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = _one_source_tiled_cfg(images_dir, labels_dir)

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, search_seed=0)

    assert "error" in result
    assert "one trainable source" in result["error"]
    assert "spatial strip path" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_a_single_source_spatial_config_at_one_draw(
    tmp_path, monkeypatch,
):
    """The leg sits behind split_draws's own <= 1 return: the identical single-source config
    that refuses above 1 is admitted at split_draws=1, since split_draws governs nothing there."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _big_single_source

    images_dir, labels_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = _one_source_tiled_cfg(images_dir, labels_dir)

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=1, search_seed=0)

    assert "error" not in result


def test_run_hyperparameter_search_admits_split_draws_over_a_two_source_tiled_config(
    tmp_path, monkeypatch,
):
    """An unbound, built-in detection config with tiling on that admits two or more sources never reaches this leg's own single-source branch, so
    split_draws above 1 mints the sweep."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _detection_dataset

    images_dir, labels_dir, _stems = _detection_dataset(tmp_path / "ds")
    cfg = _one_source_tiled_cfg(images_dir, labels_dir)

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result


def test_run_hyperparameter_search_admits_split_draws_over_a_bespoke_dataset_source(
    tmp_path, monkeypatch,
):
    """A bespoke dataset_source is excluded from this leg outright: the single-source config the
    built-in leg refuses when tiled is admitted with a builder named (which takes no platform
    tiling), because the spatial-strip route serves datasets the platform builds itself, so a
    bespoke run never takes it however few sources it admits. Its own samples still come from
    the platform's producer, so every draw varies the same admitted set."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _big_single_source

    images_dir, labels_dir, _stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    cfg = _one_source_tiled_cfg(images_dir, labels_dir)
    untiled = {k: v for k, v in cfg["data"].items() if k != "tiling"}
    cfg["data"] = {**untiled, "dataset_source": {
        "builder": "tests.test_dataset_source_seam:build_bespoke_ds"}}
    # One source holds nothing out, so the run selects on its training loss.
    cfg["evaluation"] = {"selection_metric": "loss"}

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    assert "error" not in result, result


def test_run_hyperparameter_search_reads_an_earlier_legs_reason_before_this_one(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """A config an earlier leg already refuses (here, ``data.auto_val`` off) reads
    that leg's own reason, never this one's: the spatial-strip leg runs last and is never
    reached for a config an earlier leg already rejected."""
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "auto_val": False,
                   "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2}}
    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, search_seed=0)

    assert "error" in result and "auto_val" in result["error"]
    assert "spatial strip path" not in result["error"]
    assert not ran


def test_a_misrouted_coco_is_named_by_the_producer_not_by_the_single_source_leg(
    tmp_path, monkeypatch,
):
    """A dataset-level COCO document misrouted into data.labels_dir, at an image's label path, is
    the one reader's refusal, raised where the run admits and reported at preflight in those same
    words. This leg adds nothing of its own: neither its single-source message nor a spatial-strip
    claim about a membership nothing could admit."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import json

    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _save_png

    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "detect"
    labels_dir.mkdir(parents=True)
    _save_png(images_dir / "img0.png")
    (labels_dir / "img0.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "img0.png"}], "annotations": [], "categories": []}))
    cfg = _one_source_tiled_cfg(images_dir, labels_dir)

    preflight = tt.preflight_config(tmp_path, cfg)
    assert any("dataset-level COCO" in issue for issue in preflight["issues"]), preflight
    assert not any("one trainable source" in issue or "spatial strip path" in issue
                   for issue in preflight["issues"])
    assert not any("stem(s) admitted" in warning for warning in preflight["warnings"])

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1,
                        scheduler="none", split_draws=2, trial_budget=2, search_seed=0)

    # The sweep refuses on the admission's own message, and searches nothing over data no run
    # could admit.
    assert any("dataset-level COCO" in issue for issue in result["issues"]), result
    assert not ran
