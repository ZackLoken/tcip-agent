"""Split sensitivity as a run_hpo sweep factor: split_draws pairs data.split.seed as a grid
axis with every sampled point through Ray's own BasicVariantGenerator(constant_grid_search=
True); run_hpo groups the resulting trials by point and picks the best by mean over each
point's draws. tune_search is faked throughout run_hpo's own tests (as test_hpo_durable.py
does), so these exercise run_hpo's own refusal, seed-derivation and grouping logic without a
real Ray/training cost; one test calls tune_search's own guard directly, before Ray is touched.
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


# -- refusals --------------------------------------------------------------------


def test_run_hpo_refuses_split_draws_when_a_bound_manifest_would_starve_a_side(
    tmp_path, monkeypatch,
):
    """A manifest whose date's train-plus-val members resolve to only one group under its own
    recorded grouping refuses the sweep before minting, the same distinct-groups check
    preflight_config runs for one config."""
    import tcip_store as ts
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import make_splits, split_manifest_key

    from tests.test_split_manifest_binding import (
        DATES, SUBJECT, _collapse_date_to_one_group, _two_subject_two_date_dataset,
    )

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest_dir = tmp_path / "m"
    make_result = make_splits(str(root), output_path=str(manifest_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in make_result, make_result
    manifest = ts.read(split_manifest_key(manifest_dir))
    ts.replace(split_manifest_key(manifest_dir), _collapse_date_to_one_group(manifest, DATES[0]))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {
            "images_dir": str(root / "images" / DATES[0]),
            "labels_dir": str(root / "annotations" / DATES[0]),
            "subject": SUBJECT, "split": {"manifest_dir": str(manifest_dir)},
        },
    }
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" in result and "would starve" in result["error"]
    assert not ran
    assert list(tmp_path.glob("hpo_*")) == []


def test_run_hpo_refuses_split_draws_with_val_images_dir(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "val_images_dir": str(tmp_path / "val")}
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" in result and "val_images_dir" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_when_auto_val_is_off(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "auto_val": False}
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" in result and "auto_val" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_for_a_task_outside_the_drawn_path(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["model_source"] = {**cfg["model_source"], "task": "ordinal"}
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" in result and "ordinal" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_with_a_non_native_search_alg(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        search_alg="bayesopt", scheduler="none", split_draws=2)

    assert "error" in result and "search_alg" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_with_a_pruning_scheduler(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        scheduler="asha", split_draws=2)

    assert "error" in result and "scheduler" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draw_seeds_of_the_wrong_length(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=3, split_draw_seeds=[1, 2])

    assert "error" in result and "split_draw_seeds" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_when_param_space_already_sweeps_the_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(
        base_config=real_hpo_base_config,
        param_space={"data.split.seed": {"type": "categorical", "choices": [1, 2]}},
        n_trials=1, output_dir=str(tmp_path), scheduler="none", split_draws=2,
    )

    assert "error" in result and "data.split.seed" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_when_warm_start_baseline_names_the_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(
        base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
        scheduler="none", split_draws=2, warm_start=True,
        baseline_params={"lr": 0.01, "data.split.seed": 7},
    )

    assert "error" in result and "baseline_params" in result["error"]
    assert not ran


def test_run_hpo_refuses_split_draws_when_param_space_sweeps_another_data_axis(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    result = tt.run_hpo(
        base_config=real_hpo_base_config,
        param_space={"data.split.val_ratio": {"type": "uniform", "low": 0.1, "high": 0.3}},
        n_trials=1, output_dir=str(tmp_path), scheduler="none", split_draws=2,
    )

    assert "error" in result and "data.split.val_ratio" in result["error"]
    assert not ran


def test_tune_search_refuses_split_draws_without_the_axis_in_param_space(tmp_path):
    """tune_search itself, called directly (not through run_hpo's own axis-adding), raises
    rather than pairing nothing: a caller building the space by hand must include the axis."""
    from tcip_mcp.pipelines.training.hpo import tune_search

    with pytest.raises(ValueError, match="data.split.seed"):
        tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space={"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}},
            storage_path=str(tmp_path), split_draws=2,
        )


# -- admits valid work -------------------------------------------------------------


def _bound_hpo_config(root, manifest_dir, date, subject, *, auto_val: bool | None = None) -> dict:
    data: dict = {
        "images_dir": str(root / "images" / date), "labels_dir": str(root / "annotations" / date),
        "subject": subject, "split": {"manifest_dir": str(manifest_dir)},
    }
    if auto_val is not None:
        data["auto_val"] = auto_val
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": data,
    }


def test_run_hpo_admits_split_draws_bound_to_a_manifest_and_sets_the_redraw_flag(
    tmp_path, monkeypatch,
):
    """A base_config bound to a split manifest is no longer refused: run_hpo sets
    data.split.redraw_within_manifest on its own copy, so every trial redraws train and val
    inside the manifest's own members instead of the manifest's one recorded partition, and the
    recorded sweep manifest's own base_config carries the claim."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import make_splits

    from tests.test_split_manifest_binding import DATES, SUBJECT, _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest_dir = tmp_path / "m"
    make_result = make_splits(str(root), output_path=str(manifest_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in make_result, make_result

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    cfg = _bound_hpo_config(root, manifest_dir, DATES[0], SUBJECT)
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" not in result, result
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical", "choices": [42, 43],
    }
    assert cfg["data"]["split"] == {"manifest_dir": str(manifest_dir)}  # the caller's own copy

    import tcip_store as ts
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    recorded_split = manifest["base_config"]["data"]["split"]
    assert recorded_split["redraw_within_manifest"] is True
    assert recorded_split["seed"] == 42


def test_run_hpo_admits_split_draws_bound_with_auto_val_false(tmp_path, monkeypatch):
    """A bound base_config reads neither val_images_dir nor auto_val (the manifest branch binds
    ahead of both), so auto_val=False no longer refuses it the way it refuses a drawn config."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.tools.data_tools import make_splits

    from tests.test_split_manifest_binding import DATES, SUBJECT, _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    manifest_dir = tmp_path / "m"
    make_result = make_splits(str(root), output_path=str(manifest_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in make_result, make_result

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    cfg = _bound_hpo_config(root, manifest_dir, DATES[0], SUBJECT, auto_val=False)
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" not in result, result


def test_run_hpo_admits_split_draws_and_derives_seeds_from_the_base_config(
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

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=2, output_dir=str(tmp_path),
                        scheduler="none", split_draws=3)

    assert "error" not in result
    assert captured["split_draws"] == 3
    assert captured["param_space"]["data.split.seed"] == {
        "type": "categorical", "choices": [42, 43, 44],
    }

    import tcip_store as ts
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["split_draws"] == 3
    assert manifest["split_draw_seeds"] == [42, 43, 44]
    assert "data.split.seed" not in manifest["param_space"]  # the caller's own axes, unaugmented


def test_run_hpo_admits_split_draws_with_explicit_seeds(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2, split_draw_seeds=[7, 99])

    assert "error" not in result
    assert captured["param_space"]["data.split.seed"]["choices"] == [7, 99]


@pytest.mark.parametrize("search_alg", ["grid", "variant_generator"])
def test_run_hpo_admits_split_draws_with_a_native_search_alg(
    tmp_path, real_hpo_base_config, monkeypatch, search_alg,
):
    """Every native search_alg (random, grid, variant_generator) builds the paired
    BasicVariantGenerator; only a backend searcher is refused."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        search_alg=search_alg, scheduler="none", split_draws=2)

    assert "error" not in result
    assert captured["search_alg"] == search_alg


def test_run_hpo_admits_split_draws_for_instance_seg(tmp_path, real_hpo_base_config, monkeypatch):
    """instance_seg sits in STEM_TASKS beside detection; split_draws admits it the same way."""
    import tcip_mcp.tools.training_tools as tt

    cfg = dict(real_hpo_base_config)
    cfg["model_source"] = {**cfg["model_source"], "task": "instance_seg"}

    def fake_search(**kw):
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": []}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" not in result


def test_run_hpo_admits_split_draws_with_explicit_auto_val_true(
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

    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" not in result


def test_run_hpo_admits_split_draws_with_a_warm_start_not_naming_the_seed(
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

    result = tt.run_hpo(
        base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
        scheduler="none", split_draws=2, warm_start=True, baseline_params={"lr": 0.01},
    )

    assert "error" not in result
    assert captured["warm_start"] is True


def test_run_hpo_admits_a_bare_seed_axis_beside_split_draws(
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

    result = tt.run_hpo(
        base_config=real_hpo_base_config,
        param_space={"seed": {"type": "int", "low": 1, "high": 3}},
        n_trials=1, output_dir=str(tmp_path), scheduler="none", split_draws=2,
    )

    assert "error" not in result
    assert "seed" in captured["param_space"]
    assert "data.split.seed" in captured["param_space"]


def test_run_hpo_admits_a_sampled_split_seed_axis_when_split_draws_is_1(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """With split_draws at 1 (the default), a caller-sampled data.split.seed axis is untouched,
    exactly as it is today: the refusal only fires above 1."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])

    result = tt.run_hpo(
        base_config=real_hpo_base_config,
        param_space={"data.split.seed": {"type": "categorical", "choices": [1, 2]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" not in result
    assert captured["param_space"] == {"data.split.seed": {"type": "categorical", "choices": [1, 2]}}


# -- grouping and the best-by-mean -------------------------------------------------


def _all_trials_row(lr: float, seed: int, value: float | None, state: str = "COMPLETE") -> dict:
    row = {"params": {"lr": lr, "data.split.seed": seed}, "value": value,
          "iterations": 1, "state": state}
    if state == "ERROR":
        row["error"] = "boom"
    return row


def test_run_hpo_groups_trials_by_point_and_picks_the_best_by_mean(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """Two points, two draws each; real_hpo_base_config's own metric is lower=better (mode
    'min'), so the point with the lower mean wins."""
    import tcip_mcp.tools.training_tools as tt

    all_trials = [
        _all_trials_row(0.1, 42, 0.5), _all_trials_row(0.1, 43, 0.7),   # mean 0.6
        _all_trials_row(0.2, 42, 0.2), _all_trials_row(0.2, 43, 0.4),   # mean 0.3 (best)
    ]

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.5, "n_trials": 2,
                "study_name": kw["study_name"], "all_trials": all_trials}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=2, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert result["best_params"] == {"lr": 0.2}
    assert result["best_value"] == pytest.approx(0.3)
    spread = result["best_value_spread"]
    assert spread["n"] == 2 and spread["n_complete"] == 2
    assert spread["seeds_complete"] == [42, 43]
    assert spread["values"] == [0.2, 0.4]
    assert spread["std"] == pytest.approx(0.14142135623730951)
    assert result["n_points"] == 2
    assert result["split_draws"] == 2
    assert len(result["split_sensitivity"]) == 2

    import tcip_store as ts
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["result"]["best_value_spread"] == spread
    assert manifest["result"]["split_sensitivity"] == result["split_sensitivity"]
    assert manifest["result"]["n_points"] == 2
    assert manifest["result"]["split_draws"] == 2


def test_run_hpo_marks_a_group_with_an_errored_draw_ineligible(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    all_trials = [
        _all_trials_row(0.1, 42, 0.9), _all_trials_row(0.1, 43, None, state="ERROR"),  # ineligible
        _all_trials_row(0.2, 42, 0.4), _all_trials_row(0.2, 43, 0.5),   # both complete, eligible
    ]

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.9, "n_trials": 2,
                "study_name": kw["study_name"], "all_trials": all_trials}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=2, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    # The lr=0.1 point never became eligible, so lr=0.2 wins even though 0.9 alone looked worse.
    assert result["best_params"] == {"lr": 0.2}
    assert result["best_value"] == pytest.approx(0.45)


def test_run_hpo_never_picks_a_repeated_point_that_never_completed_every_planned_seed(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """Ray can repeat a point across grid cells (grid search repeats every point per sample; a
    categorical-only random space collides), landing two complete rows under the same seed
    while the point's other planned seed errors. A count of complete rows alone would call
    that eligible on a mean over one seed twice, and its low value (0.5) would beat the fully
    completed point (0.85) under this fixture's lower-is-better metric; grouping by the planned
    seeds themselves keeps it ineligible."""
    import tcip_mcp.tools.training_tools as tt

    all_trials = [
        _all_trials_row(0.1, 42, 0.5), _all_trials_row(0.1, 42, 0.6),
        _all_trials_row(0.1, 43, None, state="ERROR"),
        _all_trials_row(0.2, 42, 0.9), _all_trials_row(0.2, 43, 0.8),
    ]

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.5, "n_trials": 3,
                "study_name": kw["study_name"], "all_trials": all_trials}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=2, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert result["best_params"] == {"lr": 0.2}
    assert result["best_value"] == pytest.approx(0.85)


def test_run_hpo_records_a_null_best_when_no_point_is_eligible(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    all_trials = [
        _all_trials_row(0.1, 42, 0.9), _all_trials_row(0.1, 43, None, state="ERROR"),
    ]

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.9, "n_trials": 1,
                "study_name": kw["study_name"], "all_trials": all_trials}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert result["best_params"] is None
    assert result["best_value"] is None
    assert "no eligible point" in result["best_value_state"]


def test_run_hpo_result_and_manifest_explain_their_own_point_and_draw_counts(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """split_sensitivity keeps every point's own block, not only the winner's, and n_points/
    split_draws land beside Ray's own n_trials, on both the result and the sweep manifest's
    own result, so the JSON explains its own counts."""
    import tcip_mcp.tools.training_tools as tt

    all_trials = [
        _all_trials_row(0.1, 42, 0.5), _all_trials_row(0.1, 43, 0.7),
        _all_trials_row(0.2, 42, 0.2), _all_trials_row(0.2, 43, 0.4),
    ]

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.5, "n_trials": 4,
                "study_name": kw["study_name"], "all_trials": all_trials}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=2, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert result["n_points"] == 2
    assert result["split_draws"] == 2
    points = {tuple(sorted(g["point"].items())) for g in result["split_sensitivity"]}
    assert points == {(("lr", 0.1),), (("lr", 0.2),)}

    import tcip_store as ts
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["result"]["n_points"] == 2
    assert manifest["result"]["split_draws"] == 2
    assert manifest["result"]["split_sensitivity"] == result["split_sensitivity"]


# -- group_split_draws direct coverage ----------------------------------------------


def test_group_split_draws_with_no_planned_seeds_still_enriches_a_result_without_the_axis():
    """A result dict with no data.split.seed axis at all (a sweep that never asked for draws,
    or an older result read back) still groups cleanly with planned_seeds=[]: one
    trivially-complete block per distinct point."""
    from tcip_mcp.tools.training_tools import group_split_draws

    all_trials = [
        {"params": {"lr": 0.1}, "value": 0.4, "iterations": 1, "state": "COMPLETE"},
        {"params": {"lr": 0.2}, "value": 0.6, "iterations": 1, "state": "COMPLETE"},
        {"params": None, "value": None, "iterations": None, "state": "ERROR", "error": "dead"},
    ]

    groups = group_split_draws(all_trials, [])

    assert len(groups) == 3
    by_point = {tuple(sorted(g["point"].items())) if g["point"] else None: g for g in groups}
    assert by_point[(("lr", 0.1),)]["eligible"] is True
    assert by_point[(("lr", 0.1),)]["block"]["n"] == 1
    assert by_point[(("lr", 0.1),)]["block"]["n_complete"] == 1
    assert by_point[(("lr", 0.1),)]["block"]["seeds_complete"] == [None]
    assert by_point[(("lr", 0.1),)]["block"]["std"] is None
    assert by_point[None]["eligible"] is False
    assert by_point[None]["point"] is None


def test_group_split_draws_ineligible_when_a_repeated_point_never_completes_every_planned_seed():
    """A point Ray sampled twice under the same seed (grid search repeats every point per
    sample; a categorical-only random space collides) with its other planned seed erroring:
    two complete values reach the split count, but only one of the two planned seeds ever
    completed, so the point is not eligible on a mean over that one seed twice."""
    from tcip_mcp.tools.training_tools import group_split_draws

    all_trials = [
        {"params": {"lr": 0.1, "data.split.seed": 42}, "value": 0.5, "iterations": 1,
         "state": "COMPLETE"},
        {"params": {"lr": 0.1, "data.split.seed": 42}, "value": 0.6, "iterations": 1,
         "state": "COMPLETE"},
        {"params": {"lr": 0.1, "data.split.seed": 43}, "value": None, "iterations": 1,
         "state": "ERROR", "error": "boom"},
    ]

    groups = group_split_draws(all_trials, [42, 43])

    assert len(groups) == 1
    group = groups[0]
    assert group["point"] == {"lr": 0.1}
    assert group["block"]["n"] == 3
    assert group["block"]["n_complete"] == 2
    assert group["block"]["seeds_complete"] == [42]
    assert group["eligible"] is False


# -- end to end: a real Ray sweep ----------------------------------------------------


def test_tune_search_split_draws_end_to_end_pairs_every_point_with_every_seed(tmp_path, monkeypatch):
    """A real Ray sweep over a trivial objective (the shape test_hpo_ray_detached_exit.py's
    subprocess script uses: one value reported, resources_per_trial={"cpu": 1}, storage_path
    under tmp_path): split_draws=2 pairs the seed grid with every sampled point through
    BasicVariantGenerator(constant_grid_search=True), so each of the two sampled lr points
    trains once per seed and group_split_draws groups them into two eligible points.

    Not restricted to one platform: test_imbalance_aug_hpo.py's own
    test_tune_search_warm_start_and_optimizes already starts a real Ray cluster on every
    platform through this identical call (only pytest.importorskip("ray")), so this one does
    too, rather than carrying test_hpo_ray_detached_exit.py's Windows-only skip, which guards a
    console-signal exit path this test never touches.
    """
    pytest.importorskip("ray")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path))

    from tcip_mcp.pipelines.training.hpo import tune_search
    from tcip_mcp.tools.training_tools import group_split_draws

    def objective_fn(config, report):
        report(1.0)

    result = tune_search(
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
        split_draws=2,
    )

    assert result["n_trials"] == 4  # 2 sampled points x 2 draws

    seeds_by_lr: dict[float, set[int]] = {}
    for row in result["all_trials"]:
        assert row["state"] == "COMPLETE", row
        params = row["params"]
        assert "data.split.seed" in params
        seeds_by_lr.setdefault(params["lr"], set()).add(params["data.split.seed"])

    assert len(seeds_by_lr) == 2
    for seeds in seeds_by_lr.values():
        assert seeds == {42, 43}

    groups = group_split_draws(result["all_trials"], [42, 43])
    eligible = [g for g in groups if g["eligible"]]
    assert len(eligible) == 2
