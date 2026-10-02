"""run_hyperparameter_search records a sweep as its own directory: its input written once before the
first trial, a heartbeat while it runs, one run directory per trial, and a final status. The Ray
Tune search itself is faked so the test doesn't init Ray or train."""

from __future__ import annotations

from pathlib import Path

import pytest


def _stub_search(monkeypatch, *, during=None) -> dict:
    """Replace Ray Tune's search with one that first calls ``during(**kw)`` and answers the
    sweep's log directory; returns the keywords it was last called with."""
    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        if during is not None:
            during(**kw)
        return str(Path(kw["storage_path"]) / kw["study_name"])

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    return captured


def test_run_hyperparameter_search_threads_the_sweeps_directory_and_records_its_result(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1, search_seed=0)

    assert captured["study_name"].startswith("hpo_")
    assert Path(captured["storage_path"]) == tmp_path / ".tcip" / "hpo"
    assert captured["mode"] == "min"  # the composite objective is lower=better
    assert captured["metric"] == "objective"
    sweep = tt.monitor_training(tmp_path, sweep_id=captured["study_name"])
    assert sweep["status"] == "completed"
    assert sweep["trials"] == [] and sweep["outcome"]["best_params"] is None


def test_run_hyperparameter_search_resolves_direction_from_the_top_level_evaluation_block(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """The sweep's mode comes from the config's one ``evaluation`` block, the same one the
    trainer trains on; a nested ``training`` placement is refused at the door by name."""
    import tcip_mcp.tools.training_tools as tt

    cfg = dict(real_hpo_base_config)
    cfg["evaluation"] = {"selection_metric": "f1"}  # higher-is-better -> mode "max"

    nested = {**cfg, "training": {"evaluation": {"selection_metric": "loss"}}}
    refused = tt.run_hyperparameter_search(tmp_path, base_config=nested, n_trials=1, search_seed=0)
    assert any("'training' is not a config section" in issue for issue in refused["issues"])

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=cfg, n_trials=1, search_seed=0)

    assert captured["metric"] == "objective"
    assert captured["mode"] == "max"


def test_a_sweep_is_on_disk_while_it_runs_and_its_trials_are_its_own_run_directories(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A sweep is on disk from the moment it starts, with its trials under its own directory:
    an agent-launched sweep has no other way to be seen while it runs."""
    import tcip_mcp.tools.training_tools as tt

    observed: dict = {}

    def fake_trial(point, report, base_config, trial_dir, **kwargs):
        observed["trial_dir"] = trial_dir
        report(0.25)

    def during(**kw):
        observed["while_running"] = tt.monitor_training(tmp_path, sweep_id=kw["study_name"])
        kw["objective_fn"]({"lr": 0.1}, lambda value: None)

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    _stub_search(monkeypatch, during=during)

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                                          search_seed=0)
    study = result["study_name"]

    running = observed["while_running"]
    assert running["status"] == "running"
    assert running["input"]["n_trials"] == 1
    assert running["input"]["param_space"]  # the space the sweep is searching
    assert observed["trial_dir"].parent == tt.sweep_dir(study, project=tmp_path)
    assert observed["trial_dir"].name.startswith("trial_")

    assert tt.monitor_training(tmp_path, sweep_id=study)["status"] == "completed"


def test_run_hyperparameter_search_honors_a_callers_study_name(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """A caller that already minted its own sweep id (the Tuning route's relaunch, so its own
    job and every sweep route agree on it) has the sweep's directory written under that id."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    given_id = "hpo-caller-supplied-id"
    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                                          study_name=given_id, search_seed=0)

    assert result["study_name"] == given_id
    assert tt.monitor_training(tmp_path, sweep_id=given_id)["status"] == "completed"


def test_a_sweep_whose_search_raises_ends_failed_naming_the_error(
    tmp_path, real_hpo_base_config, monkeypatch
):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def exploding_search(**kw):
        captured["study_name"] = kw["study_name"]
        raise RuntimeError("no ray here")

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", exploding_search)

    with pytest.raises(RuntimeError):
        tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                                     search_seed=0)

    sweep = tt.monitor_training(tmp_path, sweep_id=captured["study_name"])
    assert sweep["status"] == "failed"
    assert "no ray here" in sweep["error"]


def test_a_second_launch_under_one_study_name_refuses_and_leaves_the_first_as_written(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A sweep's directory is written by one launch: a second launch naming it refuses before
    its search starts, and the first sweep's input and final status are what it wrote."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=1,
                                 study_name="hpo_once0001", search_seed=0)
    first = tt.monitor_training(tmp_path, sweep_id="hpo_once0001")

    again = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=3,
                                         study_name="hpo_once0001", search_seed=0)

    assert "already exists" in again["error"]
    assert tt.monitor_training(tmp_path, sweep_id="hpo_once0001") == first


def test_run_hyperparameter_search_refuses_before_minting_when_the_base_config_fails_preflight(
    tmp_path, monkeypatch
):
    """An unimportable builder refuses at the door, making no sweep directory, rather than
    failing in every trial."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={"model_source": {"builder": "not.a:real_builder", "task": "detection"}},
        n_trials=1, search_seed=0)

    assert "error" in result
    assert not captured
    assert not (tmp_path / ".tcip" / "hpo").exists()


def test_run_hyperparameter_search_checks_a_swept_placeholder_axis_at_its_resolved_value(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A param_space sampling the builder itself resolves a real one before the door checks it,
    so a base config carrying only a placeholder there is not refused for that reason."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "task": "detection"},
                     "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {
            "type": "categorical", "choices": ["tests.bespoke_models:build_bespoke_detection"]}},
        n_trials=1, search_seed=0)

    assert "error" not in result


@pytest.mark.parametrize("choices", [["still:bad"],
                                     ["tests.bespoke_models:build_bespoke_detection", "still:bad"]],
                         ids=["every-choice-fails", "a-later-choice-fails"])
def test_run_hyperparameter_search_refuses_a_swept_axis_any_of_whose_choices_fails(
    tmp_path, real_hpo_base_config, monkeypatch, choices
):
    """The whole space is checked, not only the first sampled corner."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "task": "detection"},
                     "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {"type": "categorical", "choices": choices}},
        n_trials=1, search_seed=0)

    assert "'still'" in " ".join([result["error"], *result["issues"]])
    assert not captured


def test_run_hyperparameter_search_admits_a_swept_axis_whose_every_choice_resolves(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """Admits valid work: every categorical choice checked, and every one of them importable."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "task": "detection"},
                     "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {
            "type": "categorical",
            "choices": ["tests.bespoke_models:build_bespoke_detection",
                        "tests.bespoke_models:build_bare_score_thresh_detector"]}},
        n_trials=1, search_seed=0)

    assert "error" not in result


def test_a_non_finite_best_value_names_why_in_the_outcome():
    """A completed trial whose selection is non-finite still names why in the sweep's outcome,
    rather than a bare null with no reason a served sweep can show."""
    import tcip_mcp.tools.training_tools as tt

    trials = [{"trial_id": "a", "status": "completed", "error": None, "has_metrics": True,
               "params": {"lr": 0.01}, "value": float("nan")}]
    outcome = tt.sweep_outcome(trials, {"objective": {"higher_is_better": False},
                                        "input": {"split_draws": 1}})

    assert outcome["best_value"] is None
    assert outcome["best_value_state"] == "nan"


def test_run_hyperparameter_search_passes_agent_search_and_scheduler_choices(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """search_alg + scheduler are the agent's choice and reach tune_search verbatim."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config, n_trials=3,
                                 search_alg="bayesopt", scheduler="median", max_concurrent=2,
                                 search_seed=0)

    assert (captured["search_alg"], captured["scheduler"], captured["max_concurrent"],
            captured["num_samples"]) == ("bayesopt", "median", 2, 3)
