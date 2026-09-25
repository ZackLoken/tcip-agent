"""planned_trial_count answers the trial count run_hyperparameter_search's own budget door checks
against: Ray's own variant count over the identical specification _search_space_and_points
prepares for tune_search, so the two can never diverge, and no Ray session is started. The last
section drains the searcher tune_search itself builds to exhaustion and checks that
planned_trial_count's answer is the number of trials it really yields, the property the door's
bound rests on."""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.training.hpo import (
    _search_space_and_points,
    get_default_space,
    planned_trial_count,
    split_draw_search_space,
)


def _paired_space(draws: int) -> dict:
    space, _seeds = split_draw_search_space(
        get_default_space(), {"data": {"split": {"seed": 42}}}, draws, None)
    return space


# -- the measured counts over the default space -------------------------------------


@pytest.mark.parametrize("draws, expected", [(1, 5), (2, 10), (3, 15)])
def test_planned_trial_count_random_over_the_default_space(draws, expected):
    """random over the default space, no warm start: n_trials draws per point, times the seed
    draws paired above one (the seed axis is a grid factor of every sample whether or not the
    sweep's own search_alg is random)."""
    space = get_default_space() if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "random", draws, False, None) == expected


@pytest.mark.parametrize("draws, expected", [(1, 15), (2, 30), (3, 45)])
def test_planned_trial_count_grid_over_the_default_space(draws, expected):
    """grid over the default space, no warm start: n_trials times the three batch_size choices,
    times the draws."""
    space = get_default_space() if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "grid", draws, False, None) == expected


@pytest.mark.parametrize("draws, expected", [(1, 13), (2, 26), (3, 39)])
def test_planned_trial_count_grid_with_the_default_warm_start(draws, expected):
    """The default warm start pins batch_size to one sample per draw and leaves the other
    n_trials-1 samples to yield three each, over the default space's default baseline."""
    space = get_default_space() if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "grid", draws, True, None) == expected


def test_planned_trial_count_random_at_two_draws_with_the_warm_start():
    """The warm start's preset never names the seed axis, so its own sample still yields every
    draw: random's count is unmoved by warm_start."""
    space = _paired_space(2)
    assert planned_trial_count(space, 5, "random", 2, True, None) == 10


def test_planned_trial_count_a_uniform_axis_under_grid_contributes_one():
    """A continuous axis (loguniform/uniform) stays sampled even under grid, so it contributes a
    factor of one to the grid count: the total is exactly num_samples."""
    space = {"x": {"type": "uniform", "low": 0.0, "high": 1.0}}
    assert planned_trial_count(space, 3, "grid", 1, False, None) == 3


def test_planned_trial_count_normalizes_the_search_alg_name():
    """search_alg='Grid' (a recorded value this family has met before) is normalized the same
    way tune_search's own case-insensitive read does, in the one shared helper."""
    assert planned_trial_count(get_default_space(), 5, "Grid", 1, False, None) == 15


def test_planned_trial_count_over_an_empty_param_space_at_one_draw():
    """An empty (or None) param_space substitutes the default space, _to_tune_space's own
    substitution carried verbatim by _search_space_and_points, proving the helper carries it."""
    assert planned_trial_count({}, 2, "grid", 1, False, None) == 6


def test_planned_trial_count_over_an_empty_param_space_run_through_split_draw_search_space():
    """Above one draw, an empty param_space run through split_draw_search_space first carries
    only the paired seed axis: the paired sweep runs the seed axis alone, never the substituted
    default space (the fact "The fact" states about today's own behavior)."""
    space, _seeds = split_draw_search_space({}, {"data": {"split": {"seed": 42}}}, 2, None)
    assert space == {"data.split.seed": {"type": "categorical", "choices": [42, 43]}}
    assert planned_trial_count(space, 1, "grid", 2, False, None) == 2


def test_planned_trial_count_over_a_backend_search_alg_name():
    """A backend name (bayesopt) at one draw over a sampled space counts n_trials: the
    generator's own count, never a claim about how many points the backend searcher itself would
    draw (build_search_alg's own refusal of an uninstalled backend lives inside tune_search, not
    here)."""
    space = {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}
    assert planned_trial_count(space, 4, "bayesopt", 1, False, None) == 4


def test_planned_trial_count_num_samples_negative_one_counts_zero():
    """Ray's own generator counts a negative num_samples as zero rather than negative or
    unbounded, the fact the door's n_trials argument clause rests on: Tuner.fit() itself would
    convert -1 to an unbounded sweep, so the bound must refuse it by name rather than trust this
    count to catch it."""
    assert planned_trial_count(get_default_space(), -1, "random", 1, False, None) == 0
    assert planned_trial_count(get_default_space(), -1, "grid", 1, False, None) == 0


# -- uncountable shapes --------------------------------------------------------------


def test_planned_trial_count_raises_on_a_categorical_with_no_choices():
    """A categorical spec naming no choices key fails _to_tune_space's own subscript read
    (KeyError), before any generator is built."""
    with pytest.raises(KeyError):
        planned_trial_count({"x": {"type": "categorical"}}, 2, "grid", 1, False, None)


@pytest.mark.parametrize("search_alg", ["grid", "random"])
def test_planned_trial_count_raises_on_an_int_axis_whose_low_exceeds_high(search_alg):
    """An int axis with low above high names no value to train, gridded or sampled:
    _to_tune_space refuses it by name, before anything is counted."""
    space = {"batch_size": {"type": "int", "low": 8, "high": 2}}
    with pytest.raises(ValueError, match="'batch_size'.*exceeds high"):
        planned_trial_count(space, 2, search_alg, 1, False, None)


def test_planned_trial_count_raises_on_a_categorical_with_an_empty_choice_list():
    space = {"x": {"type": "categorical", "choices": []}}
    with pytest.raises(ValueError, match="'x'.*no choices"):
        planned_trial_count(space, 2, "grid", 1, False, None)


def test_planned_trial_count_raises_overflowerror_on_a_huge_int_span():
    """An int axis spanning more than a Python list can index makes _to_tune_space's own
    list(range(...)) raise OverflowError, before any generator is built."""
    space = {"x": {"type": "int", "low": 0, "high": 10**30}}
    with pytest.raises(OverflowError):
        planned_trial_count(space, 2, "grid", 1, False, None)


# -- the shared derivation, and where the count lands --------------------------------


def test_tune_search_and_planned_trial_count_share_one_search_space_derivation(tmp_path, monkeypatch):
    """Both tune_search and planned_trial_count call _search_space_and_points, never a second
    derivation that could disagree with the first: patched to record its own arguments and raise
    before either function does anything Ray-heavy, each call records the identical arguments."""
    import tcip_mcp.pipelines.training.hpo as hpo

    calls: list[tuple] = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("stop before Ray")

    monkeypatch.setattr(hpo, "_search_space_and_points", recorder)
    space = {"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}}

    with pytest.raises(RuntimeError, match="stop before Ray"):
        hpo.planned_trial_count(space, 3, "random", 1, False, None)
    with pytest.raises(RuntimeError, match="stop before Ray"):
        hpo.tune_search(
            objective_fn=lambda config, report: report(0.0), param_space=space,
            num_samples=3, search_alg="random", scheduler=None, storage_path=str(tmp_path), seed=0
        )

    assert len(calls) == 2
    assert calls[0] == calls[1] == ((space, "random", 1, False, None), {})


def test_split_draw_search_space_at_one_draw_returns_param_space_itself():
    """At split_draws of one or below, split_draw_search_space is a no-op: the same object
    back, and no seeds resolved."""
    space = {"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}}
    result, seeds = split_draw_search_space(space, {}, 1, None)
    assert result is space
    assert seeds is None


def test_planned_trial_count_leaves_nothing_under_home_or_the_project_root(tmp_path, monkeypatch):
    """planned_trial_count builds no searcher and no experiment, so a monkeypatched home
    directory and project root both stay empty after a real count."""
    home = tmp_path / "home"
    home.mkdir()
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project_root)

    count = planned_trial_count(get_default_space(), 5, "random", 1, False, None)

    assert count == 5
    assert list(home.iterdir()) == []
    assert list(project_root.iterdir()) == []


# -- the tally against the trials Ray really yields -----------------------------------


def _trials_ray_yields(
    param_space: dict, num_samples: int, search_alg: str | None, draws: int,
    warm_start: bool, baseline_params: dict | None, storage_path,
) -> tuple[int, int]:
    """Build the searcher tune_search launches for this sweep and drain it, returning the number
    of trials next_trial() actually yielded and the total_samples tally read back afterwards."""
    from ray.tune.experiment import Experiment

    from tcip_mcp.pipelines.training.hpo import build_search_alg

    space, points, alg = _search_space_and_points(
        param_space, search_alg, draws, warm_start, baseline_params)
    generator = build_search_alg(alg, seed=0, points_to_evaluate=points,
                                 constant_grid_search=draws > 1)
    generator.add_configurations(Experiment(
        name="drained_count", run=lambda config: None, config=space,
        num_samples=num_samples, storage_path=str(storage_path),
    ))
    yielded = 0
    while generator.next_trial() is not None:
        yielded += 1
        if yielded > 1000:
            raise AssertionError("the generator did not exhaust within 1000 trials")
    return yielded, generator.total_samples


@pytest.mark.parametrize("search_alg, draws, warm_start", [
    ("random", 1, False),
    ("random", 3, False),
    ("grid", 1, False),
    ("grid", 2, False),
    ("grid", 3, True),
    ("random", 2, True),
])
def test_planned_trial_count_equals_the_trials_the_generator_yields(
        search_alg, draws, warm_start, tmp_path):
    """Coverage that the tally the door reads is the launched count, not merely a number the same
    function computes twice: a separately built generator is drained to exhaustion, and the trials
    it yielded, its own total_samples, and planned_trial_count's answer are all one number."""
    param_space = get_default_space() if draws == 1 else _paired_space(draws)
    baseline_params = {"lr": 0.001} if warm_start else None

    yielded, tally = _trials_ray_yields(
        param_space, 5, search_alg, draws, warm_start, baseline_params, tmp_path)

    assert yielded == tally
    assert planned_trial_count(
        param_space, 5, search_alg, draws, warm_start, baseline_params) == yielded


@pytest.mark.parametrize("draws", [2, 3, 4])
def test_the_counted_total_divides_evenly_into_the_per_draw_count(draws):
    """Coverage of the invariant the budget refusal's per-draw arithmetic rests on: the seed axis
    is a grid factor of every sample, so the counted total is an exact multiple of split_draws and
    the count // split_draws the refusal reports is at least one, never a floor to zero."""
    space = _paired_space(draws)

    for search_alg in ("random", "grid"):
        count = planned_trial_count(space, 5, search_alg, draws, False, None)
        assert count % draws == 0
        assert count // draws >= 1
