"""Split sensitivity as a sweep factor.

split_draws pairs data.split.seed as a grid axis with every sampled point through Ray's own
BasicVariantGenerator(constant_grid_search=True), and the sweep's outcome groups the trials by
point and picks the best by mean over each point's draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._chain_fixtures import (
    DETECTION_SOURCE,
    training_config,
    weights_statistics,
)
from tests._server_calls import called_through_the_server
from tests._training_values import (
    asha_scheduler,
    drawn_split,
    evaluation_block,
    fifo_scheduler,
    fifo_search,
    sweep_space,
    tiling_block,
    tune_arguments,
)
from tests._verified_checkpoint_fixtures import opened_sweep

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tcip_mcp.experiments import RunRow
    from tcip_mcp.pipelines.data.selection import Selection
    from tcip_mcp.workspace import BoundProject

THREE_CHOICE_SPACE = {
    **sweep_space(),
    "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
}
"""A sample space whose one discrete axis holds three choices, so a grid count triples."""


def _never_search(ran: list) -> Callable[..., None]:
    """Return a ``tune_search`` stand-in that records into ``ran`` that it ran."""

    def fake_search(**kw: object) -> None:
        ran.append(1)

    return fake_search


def _search(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace ``tune_search`` with a search that records its keywords and runs no trial.

    The keywords it ran with are recorded into the returned dict.
    """
    captured: dict = {}

    def fake_search(**kw: object) -> None:
        captured.update(kw)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", fake_search
    )
    return captured


def test_real_hpo_base_config_is_admitted_by_preflight(
    tmp_path: Path, real_hpo_base_config: dict
) -> None:
    """The ``real_hpo_base_config`` fixture is admitted by preflight with no issues.

    preflight_config, the sweep door's own admissibility check, reports the fixture valid with
    no issues.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    result = preflight_config(tmp_path, real_hpo_base_config)
    assert result["issues"] == []
    assert result["valid"] is True


# -- the door's own split_draws argument and trial_budget guards --------------------


@pytest.mark.parametrize("name", ["split_draws", "n_trials", "trial_budget"])
def test_run_hyperparameter_search_takes_each_count_as_a_whole_number_of_at_least_one(
    name: str,
    bound: BoundProject,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep door takes each count only as a whole number of at least one.

    The MCP call refuses a count that is not a whole number of at least one by the
    parameter's own type, naming it; stated as 1 on a server bound to a project, the sweep
    opens and reaches its search once, one sample at one draw.
    """
    arguments = {
        "base_config": real_hpo_base_config,
        "param_space": sweep_space(),
        "n_trials": 1,
        **fifo_search(),
        "search_seed": 0,
        "auto_tensorboard": False,
    }
    refused = [0, -3, False, True, 2.5, "2"]
    if name != "trial_budget":
        refused.append(None)
    for value in refused:
        error = called_through_the_server(
            "run_hyperparameter_search", {**arguments, name: value}
        )
        assert name in error, value

    searched: list[dict] = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: searched.append(kw),
    )
    answer = called_through_the_server(
        "run_hyperparameter_search", {**arguments, name: 1}, bound=bound
    )

    assert "error" not in answer, answer
    (search,) = searched
    assert (search["num_samples"], search["split_draws"]) == (1, 1)


def test_run_hyperparameter_search_refuses_split_draws_above_one_with_no_trial_budget(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch above one draw stating no trial_budget is refused, naming the budget.

    A launch at two draws that states no trial_budget refuses naming ``trial_budget=2`` and the
    two trials it would launch, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "trial_budget=2" in result["error"]
    assert "would launch 2 trials" in result["error"]
    assert not ran


# -- refusals --------------------------------------------------------------------


def test_run_hyperparameter_search_refuses_split_draws_when_a_bound_selection_would_starve_a_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split-draws sweep over a selection holding one foreground group is refused.

    The refusal names the shortfall, the search is not reached, and no sweep directory is
    minted.
    """
    import tcip_mcp.tools.training_tools as tt
    from tests.test_selection_binding import one_foreground_group_selection

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    _root, selection_dir = one_foreground_group_selection(tmp_path)
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {"split": {"selection_dir": str(selection_dir), "seed": 0}},
    )
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert any(
        "fewer than the 2 the requested sides need" in issue
        for issue in result["issues"]
    ), result
    assert not ran
    assert tt.experiments.sweep_dirs(tmp_path) == []


def test_run_hyperparameter_search_refuses_split_draws_over_a_zero_validation_share(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    cfg = _unheld(tmp_path, BUILT_DETECTOR, real_hpo_base_config["data"])
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "val_ratio zero" in result["error"]
    assert not ran


def _unheld(
    project: Path, model_source: dict, data: dict, **keys: object
) -> dict:
    """Return ``training_config`` of ``project`` and ``model_source`` over a merged data section.

    The data section is ``data`` with its ``split`` given ``val_ratio`` 0.0, then ``keys``
    merged last.
    """
    return training_config(
        project,
        model_source,
        {**data, "split": {**data["split"], "val_ratio": 0.0}, **keys},
    )


def test_run_hyperparameter_search_refuses_split_draws_with_a_non_native_search_alg(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        search_alg="bayesopt",
        scheduler=fifo_scheduler(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "search_alg" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_with_a_pruning_scheduler(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        search_alg="random",
        scheduler=asha_scheduler(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "scheduler" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draw_seeds_of_the_wrong_length(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=3,
        split_draw_seeds=[1, 2],
        search_seed=0,
    )

    assert "error" in result and "split_draw_seeds" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_param_space_already_sweeps_the_seed(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={
            "data.split.seed": {"type": "categorical", "choices": [1, 2]}
        },
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "data.split.seed" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_the_baseline_names_the_seed(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        baseline_params={"optimizer.head_lr": 0.01, "data.split.seed": 7},
        search_seed=0,
    )

    assert "error" in result and "baseline_params" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_split_draws_when_param_space_sweeps_another_data_axis(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={
            "data.split.val_ratio": {
                "type": "uniform",
                "low": 0.1,
                "high": 0.3,
            }
        },
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "data.split.val_ratio" in result["error"]
    assert not ran


def test_tune_search_refuses_split_draws_without_the_axis_in_param_space(
    tmp_path: Path,
) -> None:
    """tune_search called directly refuses split_draws without the seed axis in its space.

    tune_search called directly with split_draws=2 and a space holding no data.split.seed axis
    raises naming the axis.
    """
    from tcip_mcp.pipelines.training.hpo import tune_search

    with pytest.raises(ValueError, match="data.split.seed"):
        tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space=sweep_space(),
            sweep_dir=tmp_path / "sweep",
            **tune_arguments(split_draws=2),
        )


# -- a caller-supplied seed axis at one draw ----------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "param_space": {
                    "data.split.seed": {
                        "type": "categorical",
                        "choices": [1, 2],
                    }
                }
            },
            id="sampled-under-random",
        ),
        pytest.param(
            {
                "search_alg": "grid",
                "param_space": {
                    "data.split.seed": {
                        "type": "categorical",
                        "choices": [1, 2],
                    },
                    "optimizer.head_lr": {
                        "type": "categorical",
                        "choices": [0.1, 0.2],
                    },
                },
            },
            id="gridded-beside-a-discrete-axis",
        ),
        pytest.param(
            {
                "param_space": {
                    "data.split.seed": {
                        "type": "uniform",
                        "low": 1,
                        "high": 100,
                    }
                }
            },
            id="a-float-range",
        ),
    ],
)
def test_run_hyperparameter_search_refuses_a_caller_split_seed_axis_at_one_draw(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    case: dict,
) -> None:
    """A caller-supplied split seed axis at one draw is refused, naming the remedy.

    A caller-supplied data.split.seed axis with split_draws at 1 (unset) refuses under the random
    and the grid sampler and as a categorical or a float axis; tune_search is not reached, and
    the error dict carries the remedy.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        n_trials=1,
        scheduler=fifo_scheduler(),
        search_seed=0,
        **{"search_alg": "random", **case},
    )

    assert (
        "error" in result
        and "cannot be replayed as recorded" in result["error"]
    )
    assert "drop data.split.seed from param_space" in result["error"]
    assert not ran


def test_caller_split_seed_refusal_tolerates_an_infinite_split_draws_value() -> (
    None
):
    """``caller_split_seed_refusal`` reads an infinite split_draws as no draw count.

    A seed axis at a split_draws of float("inf"), the value a JSON Infinity literal decodes
    to, answers no refusal.
    """
    import tcip_mcp.tools.training_tools as tt

    assert (
        tt.caller_split_seed_refusal(
            {"data.split.seed": {"type": "categorical", "choices": [1, 2]}},
            float("inf"),
        )
        is None
    )


def test_run_hyperparameter_search_refuses_repeated_split_draw_seeds(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated seed in split_draw_seeds is refused by the repeated value's own name.

    split_draw_seeds naming the same seed twice is not a spread over distinct partitions; the
    search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=3,
        split_draw_seeds=[3, 7, 7],
        search_seed=0,
    )

    assert "split_draw_seeds repeats [7]:" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_distinct_split_draw_seeds_beside_the_repeated_case(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep at split_draws=2 stating distinct seeds [7, 8] and trial_budget 2 is admitted."""
    import tcip_mcp.tools.training_tools as tt

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        split_draw_seeds=[7, 8],
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_admits_the_paired_path_with_a_param_space_beside_distinct_seeds(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct split_draw_seeds are added as the seed axis beside a param_space.

    split_draws=2 with distinct split_draw_seeds admits a param_space naming no data.* axis,
    the seed axis added beside it.
    """
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        split_draw_seeds=[7, 8],
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"]["choices"] == [7, 8]


def test_run_hyperparameter_search_admits_a_space_naming_no_seed_axis_at_one_draw(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A space naming no seed axis at one draw reaches the search unaltered.

    A space naming no data.split.seed axis is admitted at split_draws=1 (unset),
    tune_search is reached, and the space passes through unaltered.
    """
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    space = sweep_space()
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=space,
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["param_space"] == space


def test_run_hyperparameter_search_reports_the_seed_axis_refusal_over_a_preflight_failure(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call tripping the seed-axis refusal and preflight answers the seed-axis refusal.

    A seed axis over a base config with an unimportable builder answers the seed-axis refusal,
    the error carries no preflight finding, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    cfg = dict(real_hpo_base_config)
    cfg["model_source"] = {
        **cfg["model_source"],
        "builder": "tests.no_such_module:no_such_builder",
    }

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space={
            "data.split.seed": {"type": "categorical", "choices": [1, 2]}
        },
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert (
        "error" in result
        and "cannot be replayed as recorded" in result["error"]
    )
    assert "preflight" not in result["error"]
    assert not ran


def test_the_seed_axis_refusal_answers_before_a_later_points_schema_error(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep with a seed axis and a schema-refused point answers the seed-axis refusal.

    A space carrying a caller-supplied seed axis and a batch size of 0 answers the seed-axis
    refusal, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={
            "data.split.seed": {"type": "categorical", "choices": [1, 2]},
            "batch_size": {"type": "categorical", "choices": [2, 0]},
        },
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert (
        "error" in result
        and "cannot be replayed as recorded" in result["error"]
    ), result
    assert not ran


# -- admits valid work -------------------------------------------------------------


def _bound_hpo_config(project: Path, selection_dir: Path) -> dict:
    """Return a detection config bound to the selection at ``selection_dir``, redrawn at seed 42."""
    return training_config(
        project,
        DETECTION_SOURCE,
        {"split": {"selection_dir": str(selection_dir), "seed": 42}},
    )


def test_run_hyperparameter_search_admits_split_draws_bound_to_a_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split-draws sweep over a base_config bound to a selection is admitted.

    The seed axis holds the bound split's seed and the next (42, 43); the caller's split keeps
    its ``selection_dir`` and seed 42, and the recorded base_config's split holds exactly
    ``selection_dir`` and ``seed``, with seed 42.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import draw_splits
    from tests.test_selection_binding import (
        SUBJECT,
        _two_subject_two_date_dataset,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    selection_dir = tmp_path / "m"
    make_result = draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection_dir),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
    )
    assert "error" not in make_result, make_result

    captured = _search(monkeypatch)

    cfg = _bound_hpo_config(tmp_path, selection_dir)
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical",
        "choices": [42, 43],
    }
    assert cfg["data"]["split"] == {
        "selection_dir": str(selection_dir),
        "seed": 42,
    }  # the caller's own copy

    manifest = result["sweep"]["input"]
    recorded_split = manifest["base_config"]["data"]["split"]
    assert set(recorded_split) == {"selection_dir", "seed"}
    assert recorded_split["seed"] == 42


def test_a_bound_sweep_reads_and_redraws_its_selection_once_on_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a bound sweep reads its selection once and redraws it once."""
    import tcip_mcp.pipelines.data.selection as selection_mod
    import tcip_mcp.pipelines.data.split_construction as split_mod
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import draw_splits
    from tests.test_selection_binding import (
        SUBJECT,
        _two_subject_two_date_dataset,
    )

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    selection_dir = tmp_path / "m"
    assert "error" not in draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection_dir),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.25,
            calibration_ratio=0.125,
            holdout_ratio=0.125,
            grouping="tile_prefix",
        ),
    )
    config = _bound_hpo_config(tmp_path, selection_dir)
    calls = {"read": 0, "redraw": 0}
    real_read, real_redraw = (
        selection_mod.read_selection,
        split_mod.redrawn_selection,
    )

    def read(*args: str | Path, **kwargs: str | Path) -> Selection:
        calls["read"] += 1
        return real_read(*args, **kwargs)

    def redraw(
        *args: Selection | str | int, **kwargs: Selection | str | int
    ) -> Selection:
        calls["redraw"] += 1
        return real_redraw(*args, **kwargs)

    monkeypatch.setattr(selection_mod, "read_selection", read)
    monkeypatch.setattr(split_mod, "redrawn_selection", redraw)
    opened = tt.open_sweep(
        tmp_path,
        config,
        sweep_space(),
        n_trials=1,
        **fifo_search(),
        baseline_params=None,
        max_concurrent=1,
        resources_per_trial=None,
        split_draws=2,
        split_draw_seeds=None,
        search_seed=0,
        trial_budget=2,
        relaunched_from=None,
        actor=None,
    )

    assert not isinstance(opened, dict), opened
    assert calls == {"read": 1, "redraw": 1}


def test_run_hyperparameter_search_admits_split_draws_and_derives_seeds_from_the_base_config(
    tmp_path: Path,
    hpo_base_config_at: Callable[[int], dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default draw seeds are the base config's split seed plus the draw index.

    The default draw seeds are the base config's own data.split.seed plus the draw index, and
    the space handed to the search carries the paired grid axis.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import SweepInput

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=hpo_base_config_at(42),
        param_space=sweep_space(),
        n_trials=2,
        **fifo_search(),
        split_draws=3,
        trial_budget=6,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["split_draws"] == 3
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical",
        "choices": [42, 43, 44],
    }

    manifest = SweepInput.model_validate(result["sweep"]["input"])
    assert manifest.split_draws == 3
    assert manifest.split_draw_seeds == [42, 43, 44]
    assert (
        "data.split.seed" not in manifest.param_space
    )  # the caller's own axes, unaugmented


def test_run_hyperparameter_search_admits_split_draws_with_explicit_seeds(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        split_draw_seeds=[7, 99],
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"]["choices"] == [7, 99]


@pytest.mark.parametrize("search_alg, budget", [("grid", 6), ("random", 2)])
def test_run_hyperparameter_search_admits_split_draws_with_a_native_search_alg(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    search_alg: str,
    budget: int,
) -> None:
    """A split-draws sweep is admitted under the grid and random search algorithms.

    Over the sample space at one trial and two draws, grid at a trial budget of 6 and random at
    2 are each admitted, and the search receives the stated algorithm.
    """
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        search_alg=search_alg,
        scheduler=fifo_scheduler(),
        split_draws=2,
        trial_budget=budget,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["search_alg"] == search_alg


def test_run_hyperparameter_search_admits_split_draws_for_instance_seg(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split-draws sweep over an instance_seg config on polygon ground truth is admitted."""
    import tcip_mcp.tools.training_tools as tt
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        detection_images,
    )

    scope = real_hpo_base_config["data"]["scope"]
    cfg = training_config(
        tmp_path,
        {**BUILT_DETECTOR, "task": "instance_seg"},
        {
            **detection_images(tmp_path / "polygons", scope, polygons=True),
            "scope": scope,
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_admits_split_draws_with_a_baseline_not_naming_the_seed(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split-draws sweep admits a baseline that does not name the split seed.

    A baseline over the ordinary space is admitted and handed to the search as stated.
    """
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        baseline_params={"optimizer.head_lr": 0.01},
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result
    assert captured["baseline_params"] == {"optimizer.head_lr": 0.01}


def test_run_hyperparameter_search_admits_a_bare_seed_axis_beside_split_draws(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare run-seed axis stays admitted beside split draws.

    The bare 'seed' axis (the run seed) is a different key from data.split.seed and stays
    admitted beside the draws.
    """
    import tcip_mcp.tools.training_tools as tt

    captured = _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={"seed": {"type": "int", "low": 1, "high": 3}},
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result
    assert "seed" in captured["param_space"]
    assert "data.split.seed" in captured["param_space"]


# -- the trial_budget bound ----------------------------------------------------------


def test_run_hyperparameter_search_refuses_a_budget_that_admits_fewer_than_every_draw(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial_budget admitting fewer draws than requested is refused, naming the draws it admits.

    A stated trial_budget the count exceeds, but not at one draw, refuses
    naming the count, the budget and how many draws it admits: Ray's own count over the sample
    space at n_trials=4, split_draws=3, random is 12 (4 per draw), which trial_budget=10 does
    not fit whole, admitting 2 draws.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=4,
        **fifo_search(),
        split_draws=3,
        trial_budget=10,
        search_seed=0,
    )

    assert "error" in result
    assert "12" in result["error"] and "trial_budget=10" in result["error"]
    assert "4" in result["error"] and "2 draw" in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_budget_at_the_quotient_boundary(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial_budget at the per-draw count is refused, naming the one draw it admits.

    Ray's own count over a three-choice space at n_trials=2, split_draws=3, grid is 18 (6 per
    draw); trial_budget=6 refuses naming 18, the budget and one admitted draw, with no mention
    of exceeding, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=THREE_CHOICE_SPACE,
        n_trials=2,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        split_draws=3,
        trial_budget=6,
        search_seed=0,
    )

    assert "error" in result
    assert "18" in result["error"] and "trial_budget=6" in result["error"]
    assert "1 draw" in result["error"]
    assert "exceeds" not in result["error"]
    assert not ran


def test_run_hyperparameter_search_refuses_a_budget_a_single_draw_already_exceeds(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial_budget a single draw already exceeds is refused as exceeding at one draw.

    Ray's own count over a three-choice space at n_trials=2, split_draws=2, grid is 12 (6 per
    draw); trial_budget=5 refuses naming 12, the budget, six trials per draw and that it exceeds
    at one draw, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=THREE_CHOICE_SPACE,
        n_trials=2,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        split_draws=2,
        trial_budget=5,
        search_seed=0,
    )

    assert "error" in result
    assert "12" in result["error"] and "trial_budget=5" in result["error"]
    assert (
        "exceeds at one draw" in result["error"]
        and "6 trials per draw" in result["error"]
    )
    assert not ran


def test_run_hyperparameter_search_admits_a_budget_the_count_fits_and_records_it(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial_budget the count fits is admitted and recorded verbatim.

    n_trials=2, split_draws=3, random over the sample space counts 6, which trial_budget=6
    admits exactly; the manifest records trial_budget verbatim.
    """
    import tcip_mcp.tools.training_tools as tt

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=2,
        **fifo_search(),
        split_draws=3,
        trial_budget=6,
        search_seed=0,
    )

    assert "error" not in result, result

    manifest = result["sweep"]["input"]
    assert manifest["trial_budget"] == 6


def test_run_hyperparameter_search_refuses_a_one_draw_launch_a_stated_budget_exceeds(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-draw launch a stated trial_budget does not fit is refused.

    At one draw a stated trial_budget is checked: n_trials=5 over the sample space (random
    search) counts 5, which trial_budget=3 does not admit.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=5,
        **fifo_search(),
        trial_budget=3,
        search_seed=0,
    )

    assert "error" in result
    assert "5" in result["error"] and "trial_budget=3" in result["error"]
    assert "exceeds at one draw" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_a_one_draw_launch_with_no_stated_budget(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-draw launch with no stated budget is admitted and records a null budget."""
    import tcip_mcp.tools.training_tools as tt

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )

    assert "error" not in result, result

    manifest = result["sweep"]["input"]
    assert manifest["trial_budget"] is None


def test_run_hyperparameter_search_relaunch_with_no_budget_replays_as_recorded(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relaunch (``relaunched_from`` given) at two draws stating no trial_budget is admitted."""
    import tcip_mcp.tools.training_tools as tt

    source = opened_sweep(tmp_path, real_hpo_base_config).name

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        relaunched_from=source,
        search_seed=0,
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_relaunch_with_a_budget_is_still_checked(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relaunch that names a trial_budget its count exceeds is refused.

    The count (2, one draw's worth times two draws under random) exceeds trial_budget=1.
    """
    import tcip_mcp.tools.training_tools as tt

    source = opened_sweep(tmp_path, real_hpo_base_config).name

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        relaunched_from=source,
        trial_budget=1,
        search_seed=0,
    )

    assert "error" in result and "2" in result["error"]
    assert not ran


@pytest.mark.parametrize("split_draws, trial_budget", [(2, 5), (1, None)])
def test_run_hyperparameter_search_refuses_a_categorical_with_no_choices_before_minting(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    split_draws: int,
    trial_budget: int | None,
) -> None:
    """A categorical axis with no choices is refused, and the search is not reached.

    A param_space naming a categorical axis with no choices key is refused as not a search
    space, whether or not a trial budget is read.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={"x": {"type": "categorical"}},
        n_trials=1,
        **fifo_search(),
        split_draws=split_draws,
        trial_budget=trial_budget,
        search_seed=0,
    )

    assert "error" in result and "not a search space" in result["error"]
    assert not ran


@pytest.mark.parametrize("search_alg", ["grid", "random"])
def test_run_hyperparameter_search_refuses_an_inverted_int_bound_before_minting(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    search_alg: str,
) -> None:
    """An int axis whose low exceeds its high is refused by name, and the search is not reached.

    Grid and random search alike refuse it as not a search space whose low exceeds high.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={"batch_size": {"type": "int", "low": 8, "high": 2}},
        n_trials=1,
        scheduler=fifo_scheduler(),
        search_alg=search_alg,
        split_draws=2,
        trial_budget=5,
        search_seed=0,
    )

    assert "error" in result and "not a search space" in result["error"]
    assert "exceeds high" in result["error"]
    assert not ran


def test_a_sampled_int_axis_over_a_large_range_is_drawn_from_its_bounds() -> (
    None
):
    """A sampled int axis over a million-wide range draws a value within its bounds."""
    from tcip_mcp.pipelines.training.hpo import _to_tune_space

    drawn = _to_tune_space({"x": {"type": "int", "low": 0, "high": 10**6}})[
        "x"
    ].sample(random_state=0)

    assert 0 <= drawn <= 10**6


@pytest.mark.parametrize("end", [-(2**63), 2**63 - 1])
def test_a_sampled_int_axis_reaches_each_end_of_the_int64_values(
    end: int,
) -> None:
    """A sampled int axis reaches each end of the int64 values and refuses past it.

    An axis holding a single end of the int64 values draws that end, and an axis one past it
    refuses naming the axis.
    """
    from tcip_mcp.pipelines.training.hpo import _to_tune_space

    assert (
        _to_tune_space({"x": {"type": "int", "low": end, "high": end}})[
            "x"
        ].sample(random_state=0)
        == end
    )
    past = (
        {"low": end - 1, "high": end}
        if end < 0
        else {"low": end, "high": end + 1}
    )
    with pytest.raises(ValueError, match="'x'.*int64"):
        _to_tune_space({"x": {"type": "int", **past}})


def test_run_hyperparameter_search_refuses_a_sampled_int_axis_the_sampler_cannot_draw(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An int axis past the int64 range refuses by name, and the search is not reached."""
    import tcip_mcp.tools.training_tools as tt

    ran: list = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={"x": {"type": "int", "low": 0, "high": 10**30}},
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=5,
        search_seed=0,
    )

    assert "'x'" in result.get("error", "") and "int64" in result["error"], (
        result
    )
    assert not ran


@pytest.mark.parametrize("split_draws, trial_budget", [(2, 5), (1, None)])
def test_run_hyperparameter_search_refuses_a_grid_axis_past_its_bound_by_name(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    split_draws: int,
    trial_budget: int | None,
) -> None:
    """A grid axis past its enumeration bound is refused by name, and the search is not reached.

    A gridded int axis of a million values refuses naming the axis, its bounds and its span,
    whether or not a trial budget is read.
    """
    import tcip_mcp.tools.training_tools as tt

    ran: list = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={"x": {"type": "int", "low": 0, "high": 10**6 - 1}},
        n_trials=1,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        split_draws=split_draws,
        trial_budget=trial_budget,
        search_seed=0,
    )

    assert "error" in result, result
    assert all(
        part in result["error"]
        for part in ("'x'", f"[0, {10**6 - 1}]", f"{10**6} values")
    ), result
    assert not ran


def test_run_hyperparameter_search_counts_an_empty_param_space_at_one_draw_as_itself(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty param_space at one draw is counted as itself, no space substituted.

    n_trials=2 under grid counts 2, which trial_budget=1 does not admit.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={},
        n_trials=2,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        trial_budget=1,
        search_seed=0,
    )

    assert "error" in result and "2 trials" in result["error"]
    assert "trial_budget=1" in result["error"]
    assert not ran


def test_run_hyperparameter_search_an_empty_param_space_above_one_draw_counts_the_seed_axis_alone(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty param_space above one draw counts the seed axis alone.

    At n_trials=1, split_draws=2, grid, the count is 2 (one per draw), so trial_budget=1 refuses
    it and trial_budget=2 admits it.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={},
        n_trials=1,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        split_draws=2,
        trial_budget=1,
        search_seed=0,
    )

    assert (
        "error" in result
        and "2" in result["error"]
        and "1 per draw" in result["error"]
    )
    assert not ran

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space={},
        n_trials=1,
        scheduler=fifo_scheduler(),
        search_alg="grid",
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result


# -- grouping and the best-by-mean -------------------------------------------------


def _trial_rows(
    sweep: Path, trials: list[tuple[float, int | None, float | None]]
) -> list[RunRow]:
    """Open each ``(lr, seed, value)`` trial under ``sweep`` and return its listing rows.

    Each trial is opened by its own producer (``training_tools.open_trial``), ``value`` logged as
    its ``selection`` through the envelope's own sink and the trial ended ``completed``, or
    ``failed`` with nothing logged when ``value`` is ``None``; the rows the project's listing
    reads for them are returned in that order.
    """
    from tcip_mcp.experiments import (
        project_of_run,
        run_rows,
        write_final_status,
    )
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import log_epoch

    names = []
    for i, (lr, seed, value) in enumerate(trials):
        point: dict = {"optimizer.head_lr": lr}
        if seed is not None:
            point["data.split.seed"] = seed
        trial = open_trial(sweep, f"t{i}", point)
        if value is not None:
            log_epoch(trial, 1, {"selection": value})
        write_final_status(
            trial,
            "completed" if value is not None else "failed",
            None if value is not None else "boom",
            checkpoint=None,
        )
        names.append(trial.name)
    by_name = {
        row.experiment_id: row for row in run_rows(project_of_run(sweep))
    }
    return [by_name[name] for name in names]


def _two_draw_sweep(project: Path, real_hpo_base_config: dict) -> Path:
    """Open a two-draw sweep over ``real_hpo_base_config`` and return its directory.

    The sweep is opened by its own writer (``open_sweep``) over ``optimizer.head_lr`` choices
    0.1 and 0.2 at two trials, split draws at seeds 42 and 43, and a trial budget of 4.
    """
    from tests._verified_checkpoint_fixtures import opened_sweep

    return opened_sweep(
        project,
        real_hpo_base_config,
        param_space={
            "optimizer.head_lr": {"type": "categorical", "choices": [0.1, 0.2]}
        },
        n_trials=2,
        split_draws=2,
        split_draw_seeds=[42, 43],
        trial_budget=4,
    )


def _outcome(
    project: Path,
    real_hpo_base_config: dict,
    trials: list[tuple[float, int | None, float | None]],
) -> dict:
    """Return what ``trials`` (:func:`_trial_rows`) of a :func:`_two_draw_sweep` amount to."""
    from tcip_mcp.experiments import observe_sweep, sweep_outcome

    sweep = _two_draw_sweep(project, real_hpo_base_config)
    observed = observe_sweep(sweep)
    return sweep_outcome(_trial_rows(sweep, trials), observed.record.input)


def test_a_sweeps_outcome_groups_trials_by_point_and_picks_the_best_by_mean(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    """A sweep's outcome groups trials by point and picks the best by mean.

    Two points, two draws each; real_hpo_base_config's own metric is lower=better, so the
    point with the lower mean wins with its spread, and both points appear in the split
    sensitivity.
    """
    outcome = _outcome(
        tmp_path,
        real_hpo_base_config,
        [
            (0.1, 42, 0.5),
            (0.1, 43, 0.7),  # mean 0.6
            (0.2, 42, 0.2),
            (0.2, 43, 0.4),  # mean 0.3 (best)
        ],
    )

    assert outcome["best_params"] == {"optimizer.head_lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.3)
    spread = outcome["best_value_spread"]
    assert spread["n"] == 2 and spread["n_complete"] == 2
    assert spread["seeds_complete"] == [42, 43]
    assert spread["values"] == [0.2, 0.4]
    assert spread["std"] == pytest.approx(0.14142135623730951)
    points = {
        tuple(sorted(g["point"].items())) for g in outcome["split_sensitivity"]
    }
    assert points == {
        (("optimizer.head_lr", 0.1),),
        (("optimizer.head_lr", 0.2),),
    }


def test_a_sweeps_outcome_marks_a_group_with_a_failed_draw_ineligible(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    outcome = _outcome(
        tmp_path,
        real_hpo_base_config,
        [
            (0.1, 42, 0.9),
            (0.1, 43, None),  # ineligible
            (0.2, 42, 0.4),
            (0.2, 43, 0.5),  # both complete, eligible
        ],
    )

    # The lr=0.1 point never became eligible, so lr=0.2 wins even though 0.9 alone looked worse.
    assert outcome["best_params"] == {"optimizer.head_lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.45)


def test_a_sweeps_outcome_never_picks_a_repeated_point_that_never_completed_every_planned_seed(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    """A sweep's outcome does not pick a repeated point missing a planned seed.

    A point with two complete trials under seed 42 and a failed one under its other planned seed
    is not picked; the point complete under both seeds, at a mean of 0.85, is the best.
    """
    outcome = _outcome(
        tmp_path,
        real_hpo_base_config,
        [
            (0.1, 42, 0.5),
            (0.1, 42, 0.6),
            (0.1, 43, None),
            (0.2, 42, 0.9),
            (0.2, 43, 0.8),
        ],
    )

    assert outcome["best_params"] == {"optimizer.head_lr": 0.2}
    assert outcome["best_value"] == pytest.approx(0.85)


def test_a_sweeps_outcome_is_a_null_best_when_no_point_is_eligible(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    outcome = _outcome(
        tmp_path, real_hpo_base_config, [(0.1, 42, 0.9), (0.1, 43, None)]
    )

    assert outcome["best_params"] is None
    assert outcome["best_value"] is None
    assert "best_value_state" not in outcome


# -- group_split_draws direct coverage ----------------------------------------------


def test_group_split_draws_with_no_planned_seeds_groups_a_sweep_without_the_axis(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    """``group_split_draws`` with no planned seeds groups a sweep without the seed axis.

    Trials with no data.split.seed axis group with planned_seeds=[] into one complete block per
    distinct point, and a point whose trial failed is not eligible.
    """
    from tcip_mcp.experiments import group_split_draws
    from tests._verified_checkpoint_fixtures import opened_sweep

    trials = _trial_rows(
        opened_sweep(tmp_path, real_hpo_base_config),
        [(0.1, None, 0.4), (0.2, None, 0.6), (0.3, None, None)],
    )

    groups = group_split_draws(trials, [])

    assert len(groups) == 3
    by_point = {tuple(sorted(g["point"].items())): g for g in groups}
    assert by_point[(("optimizer.head_lr", 0.1),)]["eligible"] is True
    assert by_point[(("optimizer.head_lr", 0.1),)]["block"]["n"] == 1
    assert by_point[(("optimizer.head_lr", 0.1),)]["block"]["n_complete"] == 1
    assert by_point[(("optimizer.head_lr", 0.1),)]["block"][
        "seeds_complete"
    ] == [None]
    assert by_point[(("optimizer.head_lr", 0.1),)]["block"]["std"] is None
    assert by_point[(("optimizer.head_lr", 0.3),)]["eligible"] is False


def test_group_split_draws_ineligible_when_a_repeated_point_never_completes_every_planned_seed(
    tmp_path: Path,
    real_hpo_base_config: dict,
) -> None:
    """``group_split_draws`` marks a repeated point missing a planned seed ineligible.

    A point with two complete trials under seed 42 and a failed one under its other planned
    seed 43 counts three trials, two complete, with seed 42 alone complete, and is not eligible.
    """
    from tcip_mcp.experiments import group_split_draws

    trials = _trial_rows(
        _two_draw_sweep(tmp_path, real_hpo_base_config),
        [(0.1, 42, 0.5), (0.1, 42, 0.6), (0.1, 43, None)],
    )

    groups = group_split_draws(trials, [42, 43])

    assert len(groups) == 1
    group = groups[0]
    assert group["point"] == {"optimizer.head_lr": 0.1}
    assert group["block"]["n"] == 3
    assert group["block"]["n_complete"] == 2
    assert group["block"]["seeds_complete"] == [42]
    assert group["eligible"] is False


# -- end to end: a real Ray sweep ----------------------------------------------------


@pytest.mark.ray_cluster
def test_tune_search_split_draws_end_to_end_pairs_every_point_with_every_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real Ray sweep at split_draws=2 pairs every sampled point with every seed.

    A real Ray sweep over a trivial objective: split_draws=2 pairs the seed grid with every
    sampled point, and each trial marks the point it was handed in a file of its own, leaving
    four marks over two sampled lr points, each carrying both seeds.
    """
    pytest.importorskip("ray")

    from tcip_mcp.pipelines.training.hpo import tune_search

    marks = tmp_path / "marks"
    marks.mkdir()

    def objective_fn(config: dict, report: Callable[[float], None]) -> None:
        (marks / f"{config['lr']!r}__{config['data.split.seed']}").touch()
        report(1.0)

    tune_search(
        objective_fn=objective_fn,
        param_space={
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "data.split.seed": {"type": "categorical", "choices": [42, 43]},
        },
        sweep_dir=tmp_path / "sweep",
        **tune_arguments(
            num_samples=2, resources_per_trial={"cpu": 1}, split_draws=2
        ),
    )

    seeds_by_lr: dict[str, set[int]] = {}
    for mark in marks.iterdir():
        lr, seed = mark.name.split("__")
        seeds_by_lr.setdefault(lr, set()).add(int(seed))

    assert len(list(marks.iterdir())) == 4  # 2 sampled points x 2 draws
    assert len(seeds_by_lr) == 2
    for seeds in seeds_by_lr.values():
        assert seeds == {42, 43}


# -- the single-source spatial-strip leg -----------------------------------------


def _tiled_cfg(
    project: Path, images_dir: Path, task: str, **split: object
) -> dict:
    """Return a tiled ``task`` config of ``project`` over the ``bud`` images of ``images_dir``.

    The config uses the bespoke detector's builder, tiled at 128 px with 0.2 overlap and a
    stated sliver fraction, its split drawing a fifth onto validation at seed 1 grouped by tile
    prefix with ``split``'s shares beside it, at the statistics of the frames that split places
    on its training side (``_chain_fixtures.training_config``).
    """
    return training_config(
        project,
        {**DETECTION_SOURCE, "task": task},
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(
                val_ratio=0.2, seed=1, grouping="tile_prefix", **split
            ),
            # sliver_frac stated: a fixture this small derives no box-size spread.
            "tiling": tiling_block(
                tile_size=128, overlap=0.2, sliver_frac=0.5
            ),
        },
    )


@pytest.mark.parametrize("task", ["detection", "instance_seg"])
def test_run_hyperparameter_search_refuses_split_draws_over_a_single_source_spatial_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    """Split draws over a single-source spatial split are refused, naming the source.

    One admitted source under a built-in config of a task that tiles, with tiling on, at two
    draws refuses naming the one admitted source and its split over its own pixels, and the
    search is not reached.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests._producer_fixtures import big_single_source

    images_dir, _stem = big_single_source(
        tmp_path / "ds", 4000, 3000, polygons=task == "instance_seg"
    )
    # A holdout strip of a tenth, which this mosaic's tile lattice admits.
    cfg = _tiled_cfg(tmp_path, images_dir, task, holdout_ratio=0.1)

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "one admitted source" in result["error"]
    assert "over its own pixels" in result["error"]
    assert not ran


def test_run_hyperparameter_search_admits_a_single_source_spatial_config_at_one_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-source tiled config is admitted at one draw."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests._producer_fixtures import big_single_source

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    # A holdout strip of a tenth, which this mosaic's tile lattice admits.
    cfg = _tiled_cfg(tmp_path, images_dir, "detection", holdout_ratio=0.1)

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=1,
        search_seed=0,
    )

    assert "error" not in result, result


def test_run_hyperparameter_search_admits_split_draws_over_a_two_source_tiled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split draws over a two-source tiled config are admitted.

    An unbound, built-in detection config with tiling on that admits two sources answers no
    error at two draws.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests.test_training_autoval import _detection_dataset

    images_dir, _stems = _detection_dataset(tmp_path / "ds")
    cfg = _tiled_cfg(tmp_path, images_dir, "detection")

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert "error" not in result, result


def test_split_draws_over_one_bespoke_source_refuse_the_validation_they_cannot_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split draws over one bespoke source are refused for the validation they cannot draw.

    One admitted source behind a bespoke dataset_source leaves nothing to hold a validation side
    out of, and the sweep's issues name every share zero.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tcip_mcp.tools.training_tools as tt
    from tests._producer_fixtures import big_single_source
    from tests.test_dataset_source_seam import BESPOKE_DS, BESPOKE_DS_FILE

    images_dir, _stem = big_single_source(tmp_path / "ds", 4000, 3000)
    # A config no run resolves: its statistics are the weights' it names. One source holds
    # nothing out, so the run selects on its training loss.
    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(
                val_ratio=0.2, seed=1, grouping="tile_prefix"
            ),
            "dataset_source": {
                "builder": BESPOKE_DS,
                "source_files": [BESPOKE_DS_FILE],
            },
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )

    _search(monkeypatch)

    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        trial_budget=2,
        search_seed=0,
    )

    assert any("every share zero" in issue for issue in result["issues"]), (
        result
    )


def test_run_hyperparameter_search_reads_an_earlier_legs_reason_before_this_one(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config an earlier check refuses answers that check's reason, not the spatial split's.

    A tiled config with a zero ``data.split.val_ratio`` answers the zero-ratio reason, the error
    carries no spatial-split reason, and the search is not reached.
    """
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran)
    )

    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    # A config no run resolves: its statistics are the weights' it names.
    cfg = _unheld(
        tmp_path,
        weights_statistics(BUILT_DETECTOR),
        real_hpo_base_config["data"],
        tiling=tiling_block(tile_size=128, overlap=0.2),
    )
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=cfg,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        split_draws=2,
        search_seed=0,
    )

    assert "error" in result and "val_ratio zero" in result["error"]
    assert "over its own pixels" not in result["error"]
    assert not ran
