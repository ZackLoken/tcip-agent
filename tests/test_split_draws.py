"""Split sensitivity as a run_hpo sweep factor: split_draws pairs data.split.seed as a grid
axis with every sampled point through Ray's own BasicVariantGenerator(constant_grid_search=
True); run_hpo groups the resulting trials by point and picks the best by mean over each
point's draws. tune_search is faked throughout (as test_hpo_durable.py does) so these exercise
run_hpo's own refusal, seed-derivation and grouping logic without a real Ray/training cost.
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


def test_run_hpo_refuses_split_draws_bound_to_a_manifest(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    ran = []
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", _never_search(ran))

    cfg = dict(real_hpo_base_config)
    cfg["data"] = {**cfg["data"], "split": {"manifest_dir": str(tmp_path / "m")}}
    result = tt.run_hpo(base_config=cfg, n_trials=1, output_dir=str(tmp_path),
                        scheduler="none", split_draws=2)

    assert "error" in result and "manifest" in result["error"]
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


# -- admits valid work -------------------------------------------------------------


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
    assert spread["n"] == 2 and "n_complete" not in spread
    assert spread["values"] == [0.2, 0.4]
    assert spread["std"] == pytest.approx(0.14142135623730951)

    import tcip_store as ts
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["result"]["best_value_spread"] == spread


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


# -- group_split_draws direct coverage ----------------------------------------------


def test_group_split_draws_at_split_draws_1_still_enriches_a_pre_family_result():
    """A result dict shaped the way tune_search returned before this family (no data.split.seed
    axis at all) still groups cleanly under split_draws=1: one trivially-complete block per
    distinct point."""
    from tcip_mcp.tools.training_tools import group_split_draws

    all_trials = [
        {"params": {"lr": 0.1}, "value": 0.4, "iterations": 1, "state": "COMPLETE"},
        {"params": {"lr": 0.2}, "value": 0.6, "iterations": 1, "state": "COMPLETE"},
        {"params": None, "value": None, "iterations": None, "state": "ERROR", "error": "dead"},
    ]

    groups = group_split_draws(all_trials, 1)

    assert len(groups) == 3
    by_point = {tuple(sorted(g["point"].items())) if g["point"] else None: g for g in groups}
    assert by_point[(("lr", 0.1),)]["eligible"] is True
    assert by_point[(("lr", 0.1),)]["block"]["n"] == 1
    assert by_point[(("lr", 0.1),)]["block"]["std"] is None
    assert by_point[None]["eligible"] is False
    assert by_point[None]["point"] is None
