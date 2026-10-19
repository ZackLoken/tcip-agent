"""planned_trial_count answers the trial count the sweep's budget door checks against.

The count run_hyperparameter_search's own budget door checks is Ray's own variant count over the
specification _search_space_and_points prepares for tune_search. A searcher built as tune_search
builds it, drained to exhaustion, yields as many trials as planned_trial_count answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tcip_mcp.pipelines.schemas import DrawnSplit
from tcip_mcp.pipelines.training.hpo import (
    _search_space_and_points,
    planned_trial_count,
    resolved_draw_seeds,
    split_draw_search_space,
)
from tests._training_values import drawn_split, tune_arguments

SPACE = {
    "optimizer.head_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
    "optimizer.weight_decay": {
        "type": "loguniform",
        "low": 1e-5,
        "high": 1e-2,
    },
}
"""A sample space: two sampled axes and one three-choice categorical axis."""

BASELINE = {
    "optimizer.head_lr": 3e-4,
    "batch_size": 4,
    "optimizer.weight_decay": 1e-4,
}
"""A sample baseline point over :data:`SPACE`'s own axes."""

SEEDED_SPLIT = DrawnSplit.model_validate(drawn_split(seed=42))
"""A base config's ``data.split`` stating seed 42."""


def _paired_space(draws: int, param_space: dict = SPACE) -> dict:
    return split_draw_search_space(
        param_space, resolved_draw_seeds(SEEDED_SPLIT, draws, None)
    )


# -- the measured counts over the sample space --------------------------------------


@pytest.mark.parametrize("draws, expected", [(1, 5), (2, 10), (3, 15)])
def test_planned_trial_count_random_over_the_sample_space(
    draws: int, expected: int
) -> None:
    """Random over the sample space counts n_trials per point times the paired seed draws.

    With no baseline, five draws per point at one, two and three paired seed draws count 5, 10
    and 15 trials.
    """
    space = SPACE if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "random", draws, None) == expected


@pytest.mark.parametrize("draws, expected", [(1, 15), (2, 30), (3, 45)])
def test_planned_trial_count_grid_over_the_sample_space(
    draws: int, expected: int
) -> None:
    """Grid with no baseline counts n_trials times the three batch_size choices, times the draws."""
    space = SPACE if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "grid", draws, None) == expected


@pytest.mark.parametrize("draws, expected", [(1, 13), (2, 26), (3, 39)])
def test_planned_trial_count_grid_with_a_baseline(
    draws: int, expected: int
) -> None:
    """A baseline under grid pins batch_size to one sample per draw.

    A baseline pins batch_size to one sample per draw and leaves the other n_trials-1 samples
    to yield three each.
    """
    space = SPACE if draws == 1 else _paired_space(draws)
    assert planned_trial_count(space, 5, "grid", draws, BASELINE) == expected


def test_planned_trial_count_random_at_two_draws_with_a_baseline() -> None:
    """A baseline leaves random's count at two draws unmoved."""
    space = _paired_space(2)
    assert planned_trial_count(space, 5, "random", 2, BASELINE) == 10


def test_planned_trial_count_a_uniform_axis_under_grid_contributes_one() -> (
    None
):
    """A continuous axis under grid contributes a factor of one.

    With one uniform axis, three samples under grid count three trials.
    """
    space = {"x": {"type": "uniform", "low": 0.0, "high": 1.0}}
    assert planned_trial_count(space, 3, "grid", 1, None) == 3


def test_planned_trial_count_normalizes_the_search_alg_name() -> None:
    """planned_trial_count counts ``search_alg='Grid'`` as a grid search."""
    assert planned_trial_count(SPACE, 5, "Grid", 1, None) == 15


def test_planned_trial_count_over_an_empty_param_space_at_one_draw() -> None:
    """An empty param_space at one draw counts num_samples trials of the one point it holds."""
    assert planned_trial_count({}, 2, "grid", 1, None) == 2


def test_planned_trial_count_over_an_empty_param_space_run_through_split_draw_search_space() -> (
    None
):
    """Above one draw, an empty param_space holds the paired seed axis alone, one trial a seed."""
    space = _paired_space(2, {})
    assert space == {
        "data.split.seed": {"type": "categorical", "choices": [42, 43]}
    }
    assert planned_trial_count(space, 1, "grid", 2, None) == 2


def test_planned_trial_count_over_a_backend_search_alg_name() -> None:
    """A backend search algorithm (bayesopt) at one draw over a sampled space counts n_trials."""
    space = {
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}
    }
    assert planned_trial_count(space, 4, "bayesopt", 1, None) == 4


# -- uncountable shapes --------------------------------------------------------------


def test_planned_trial_count_raises_on_a_categorical_with_no_choices() -> None:
    """A categorical axis with no choices key raises ``KeyError``."""
    with pytest.raises(KeyError):
        planned_trial_count({"x": {"type": "categorical"}}, 2, "grid", 1, None)


@pytest.mark.parametrize("search_alg", ["grid", "random"])
def test_planned_trial_count_raises_on_an_int_axis_whose_low_exceeds_high(
    search_alg: str,
) -> None:
    """An int axis whose low exceeds high refuses naming the axis, gridded or sampled."""
    space = {"batch_size": {"type": "int", "low": 8, "high": 2}}
    with pytest.raises(ValueError, match="'batch_size'.*exceeds high"):
        planned_trial_count(space, 2, search_alg, 1, None)


def test_planned_trial_count_raises_on_an_unknown_type_naming_the_axis() -> (
    None
):
    with pytest.raises(ValueError, match="'x'.*'beta'"):
        planned_trial_count({"x": {"type": "beta"}}, 2, "random", 1, None)


def test_planned_trial_count_raises_on_a_categorical_with_an_empty_choice_list() -> (
    None
):
    space = {"x": {"type": "categorical", "choices": []}}
    with pytest.raises(ValueError, match="'x'.*no choices"):
        planned_trial_count(space, 2, "grid", 1, None)


def test_planned_trial_count_counts_a_large_sampled_int_span_and_refuses_it_gridded() -> (
    None
):
    """A large sampled int span counts its samples, and gridded it refuses by name.

    Sampled, the count is the samples drawn; gridded, an axis past the bound a grid axis may
    enumerate refuses naming the axis.
    """
    space = {"x": {"type": "int", "low": 0, "high": 10**6}}
    assert planned_trial_count(space, 2, "random", 1, None) == 2
    with pytest.raises(ValueError, match="'x'.*past the"):
        planned_trial_count(space, 2, "grid", 1, None)


def test_planned_trial_count_refuses_an_int_axis_past_the_samplers_int64_range() -> (
    None
):
    with pytest.raises(ValueError, match="'x'.*int64"):
        planned_trial_count(
            {"x": {"type": "int", "low": 0, "high": 10**30}},
            2,
            "random",
            1,
            None,
        )


# -- the shared derivation, and where the count lands --------------------------------


def test_tune_search_and_planned_trial_count_share_one_search_space_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tune_search and planned_trial_count share one search-space derivation.

    With _search_space_and_points patched to record its arguments and raise, tune_search and
    planned_trial_count each call it with the identical arguments.
    """
    import tcip_mcp.pipelines.training.hpo as hpo

    calls: list[tuple] = []

    def recorder(
        *args: dict | str | int | None, **kwargs: dict | str | int | None
    ) -> NoReturn:
        calls.append((args, kwargs))
        raise RuntimeError("stop before Ray")

    monkeypatch.setattr(hpo, "_search_space_and_points", recorder)
    space = {
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}
    }

    with pytest.raises(RuntimeError, match="stop before Ray"):
        hpo.planned_trial_count(space, 3, "random", 1, None)
    with pytest.raises(RuntimeError, match="stop before Ray"):
        hpo.tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space=space,
            sweep_dir=tmp_path / "sweep",
            **tune_arguments(num_samples=3),
        )

    assert len(calls) == 2
    assert calls[0] == calls[1] == ((space, "random", 1, None), {})


def test_split_draw_search_space_at_one_draw_returns_param_space_itself() -> (
    None
):
    """At one draw split_draw_search_space returns param_space itself.

    At split_draws of one or below no seeds resolve, and split_draw_search_space is then a
    no-op: the same object back.
    """
    space = {
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}
    }
    seeds = resolved_draw_seeds(
        DrawnSplit.model_validate(drawn_split()), 1, None
    )
    assert seeds is None
    assert split_draw_search_space(space, seeds) is space


def test_planned_trial_count_leaves_nothing_under_home_or_the_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """planned_trial_count writes nothing under home or the project root.

    A monkeypatched home directory and project root both stay empty after a real count.
    """
    home = tmp_path / "home"
    home.mkdir()
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project_root)

    count = planned_trial_count(SPACE, 5, "random", 1, None)

    assert count == 5
    assert list(home.iterdir()) == []
    assert list(project_root.iterdir()) == []


# -- the tally against the trials Ray really yields -----------------------------------


def _trials_ray_yields(
    param_space: dict,
    num_samples: int,
    search_alg: str,
    draws: int,
    baseline_params: dict | None,
    storage_path: Path,
) -> tuple[int, int]:
    """Build and drain a searcher over this sweep's space and starting points.

    Return the number of trials next_trial() yielded and the total_samples tally read back
    afterwards.
    """
    from ray.tune.experiment import Experiment

    from tcip_mcp.pipelines.training.hpo import build_search_alg

    space, points, alg = _search_space_and_points(
        param_space, search_alg, draws, baseline_params
    )
    generator = build_search_alg(
        alg, seed=0, points_to_evaluate=points, constant_grid_search=draws > 1
    )
    generator.add_configurations(
        Experiment(
            name="drained_count",
            run=lambda config: None,
            config=space,
            num_samples=num_samples,
            storage_path=str(storage_path),
        )
    )
    yielded = 0
    while generator.next_trial() is not None:
        yielded += 1
        if yielded > 1000:
            raise AssertionError(
                "the generator did not exhaust within 1000 trials"
            )
    return yielded, generator.total_samples


@pytest.mark.parametrize(
    "search_alg, draws, baseline_params",
    [
        ("random", 1, None),
        ("random", 3, None),
        ("grid", 1, None),
        ("grid", 2, None),
        ("grid", 3, {"optimizer.head_lr": 0.001}),
        ("random", 2, {"optimizer.head_lr": 0.001}),
    ],
)
def test_planned_trial_count_equals_the_trials_the_generator_yields(
    search_alg: str,
    draws: int,
    baseline_params: dict[str, float] | None,
    tmp_path: Path,
) -> None:
    """planned_trial_count equals the trials a drained generator yields.

    A separately built generator is drained to exhaustion, and the trials it yielded, its own
    total_samples, and planned_trial_count's answer are one number.
    """
    param_space = SPACE if draws == 1 else _paired_space(draws)

    yielded, tally = _trials_ray_yields(
        param_space, 5, search_alg, draws, baseline_params, tmp_path
    )

    assert yielded == tally
    assert (
        planned_trial_count(param_space, 5, search_alg, draws, baseline_params)
        == yielded
    )


@pytest.mark.parametrize("draws", [2, 3, 4])
def test_the_counted_total_divides_evenly_into_the_per_draw_count(
    draws: int,
) -> None:
    """The counted total divides evenly into the per-draw count.

    Under random and grid search alike, the counted total is an exact multiple of split_draws
    and ``count // split_draws`` is at least one.
    """
    space = _paired_space(draws)

    for search_alg in ("random", "grid"):
        count = planned_trial_count(space, 5, search_alg, draws, None)
        assert count % draws == 0
        assert count // draws >= 1
