"""run_hyperparameter_search records a sweep as its own directory: its input written once before the
first trial, a heartbeat while it runs, one run directory per trial, and a final status. The Ray
Tune search itself is faked so the test doesn't init Ray or train."""

from __future__ import annotations

import pytest

from tests._chain_fixtures import (
    BARE_SCORE_THRESH_DETECTOR, BESPOKE_DETECTION, BESPOKE_MODELS, training_config,
)
from tests._training_values import asha_scheduler, evaluation_block, fifo_search, sweep_space


def _stub_search(monkeypatch, *, during=None) -> dict:
    """Replace Ray Tune's search with one that calls ``during(**kw)``; returns the keywords it
    was last called with."""
    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        if during is not None:
            during(**kw)

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    return captured


def test_run_hyperparameter_search_threads_the_sweeps_directory_and_records_its_result(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space=sweep_space(), n_trials=1,
        **fifo_search(), search_seed=0)

    assert captured["sweep_dir"].name.startswith("hpo_")
    assert captured["sweep_dir"].parent == tmp_path / ".tcip" / "experiments"
    assert captured["mode"] == "min"  # the composite objective is lower=better
    assert captured["metric"] == "objective"
    sweep = tt.monitor_training(tmp_path, captured["sweep_dir"].name)["sweep"]
    assert sweep["state"] == "completed"
    assert sweep["trials"] == [] and sweep["outcome"]["best_params"] is None


def test_run_hyperparameter_search_resolves_direction_from_the_top_level_evaluation_block(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """The sweep's mode comes from the config's one ``evaluation`` block, the same one the
    trainer trains on; a nested ``training`` placement is refused at the door by name."""
    import tcip_mcp.tools.training_tools as tt

    cfg = dict(real_hpo_base_config)
    cfg["evaluation"] = evaluation_block(selection_metric="f1")  # higher-is-better -> "max"

    nested = {**cfg, "training": {"evaluation": {"selection_metric": "loss"}}}
    refused = tt.run_hyperparameter_search(tmp_path, base_config=nested,
                                           param_space=sweep_space(), n_trials=1,
                                           **fifo_search(), search_seed=0)
    assert any("'training' is not a config section" in issue for issue in refused["issues"])

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=cfg, param_space=sweep_space(),
                                 n_trials=1, **fifo_search(), search_seed=0)

    assert captured["metric"] == "objective"
    assert captured["mode"] == "max"


def test_a_sweep_is_on_disk_while_it_runs_and_its_trials_are_its_own_run_directories(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A sweep is on disk from the moment it starts, with its trials under its own directory:
    an agent-launched sweep has no other way to be seen while it runs."""
    import tcip_mcp.tools.training_tools as tt

    observed: dict = {}

    def fake_trial(point, report, sweep, trial_id):
        observed["trial"] = tt.open_trial(sweep, trial_id, point)
        report(0.25)

    def during(**kw):
        observed["while_running"] = tt.monitor_training(tmp_path, kw["sweep_dir"].name)["sweep"]
        kw["objective_fn"]({"optimizer.head_lr": 0.1}, lambda value: None)

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    _stub_search(monkeypatch, during=during)

    result = tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config,
                                          param_space=sweep_space(), n_trials=1,
                                          **fifo_search(), search_seed=0)
    sweep_id = result["sweep"]["sweep_id"]

    running = observed["while_running"]
    assert running["state"] == "running"
    assert running["input"]["n_trials"] == 1
    assert running["input"]["param_space"]  # the space the sweep is searching
    assert observed["trial"].parent == tt.experiments.experiment_dir(sweep_id, project=tmp_path)

    sweep = tt.monitor_training(tmp_path, sweep_id)["sweep"]
    assert sweep["state"] == "completed"
    assert [t["experiment_id"] for t in sweep["trials"]] == [observed["trial"].name]


def test_a_sweep_whose_search_raises_ends_failed_naming_the_error(
    tmp_path, real_hpo_base_config, monkeypatch
):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def exploding_search(**kw):
        captured["sweep_id"] = kw["sweep_dir"].name
        raise RuntimeError("no ray here")

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", exploding_search)

    with pytest.raises(RuntimeError):
        tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config,
                                     param_space=sweep_space(), n_trials=1, **fifo_search(),
                                     search_seed=0)

    sweep = tt.monitor_training(tmp_path, captured["sweep_id"])["sweep"]
    assert sweep["state"] == "failed"
    assert "no ray here" in sweep["status_error"]


def test_run_hyperparameter_search_refuses_before_minting_when_the_base_config_fails_preflight(
    tmp_path, monkeypatch
):
    """An unimportable builder refuses at the door, making no sweep directory, rather than
    failing in every trial."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=training_config(
            {"builder": "not.a:real_builder", "task": "detection"}, {}),
        param_space=sweep_space(), n_trials=1, **fifo_search(), search_seed=0)

    assert "error" in result
    assert any("not.a" in issue for issue in result["issues"]), result
    assert not captured
    assert not tt.experiments.experiments_dir(tmp_path).exists()


def test_run_hyperparameter_search_checks_a_swept_placeholder_axis_at_its_resolved_value(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A param_space sampling the builder itself resolves a real one before the door checks it,
    so a base config carrying only a placeholder there is not refused for that reason."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={**real_hpo_base_config,
                               "model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                                "source_files": [BESPOKE_MODELS],
                                                "task": "detection"}},
        param_space={"model_source.builder": {
            "type": "categorical", "choices": [BESPOKE_DETECTION]}},
        n_trials=1, **fifo_search(), search_seed=0)

    assert "error" not in result, result


@pytest.mark.parametrize("choices", [["still:bad"],
                                     [BESPOKE_DETECTION, "still:bad"]],
                         ids=["every-choice-fails", "a-later-choice-fails"])
def test_run_hyperparameter_search_refuses_a_swept_axis_any_of_whose_choices_fails(
    tmp_path, real_hpo_base_config, monkeypatch, choices
):
    """The whole space is checked, not only the first sampled corner."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={**real_hpo_base_config,
                               "model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                                "source_files": [BESPOKE_MODELS],
                                                "task": "detection"}},
        param_space={"model_source.builder": {"type": "categorical", "choices": choices}},
        n_trials=1, **fifo_search(), search_seed=0)

    assert "'still'" in " ".join([result["error"], *result["issues"]])
    assert not captured


def test_run_hyperparameter_search_admits_a_swept_axis_whose_every_choice_resolves(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """Admits valid work: every categorical choice checked, and every one of them importable."""
    import tcip_mcp.tools.training_tools as tt

    _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config={**real_hpo_base_config,
                               "model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                                "source_files": [BESPOKE_MODELS],
                                                "task": "detection"}},
        param_space={"model_source.builder": {
            "type": "categorical",
            "choices": [BESPOKE_DETECTION,
                        BARE_SCORE_THRESH_DETECTOR]}},
        n_trials=1, **fifo_search(), search_seed=0)

    assert "error" not in result, result


def test_run_hyperparameter_search_passes_agent_search_and_scheduler_choices(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """search_alg + scheduler are the agent's choice and reach tune_search verbatim."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config,
                                 param_space=sweep_space(), n_trials=3, search_alg="bayesopt",
                                 scheduler=asha_scheduler(), max_concurrent=2, search_seed=0)

    assert (captured["search_alg"], captured["scheduler"], captured["max_concurrent"],
            captured["num_samples"]) == ("bayesopt", asha_scheduler(), 2, 3)


@pytest.mark.parametrize("unstated", ["n_trials", "search_alg", "scheduler"])
def test_run_hyperparameter_search_refuses_an_unstated_search_setting_by_name(
    tmp_path, real_hpo_base_config, monkeypatch, unstated,
):
    """guard. The trial count, the search algorithm and the scheduler block have no default:
    a call leaving one unstated refuses naming it, and no search runs."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    stated = {"n_trials": 1, **fifo_search()}
    del stated[unstated]
    with pytest.raises(TypeError, match=f"'{unstated}'"):
        tt.run_hyperparameter_search(tmp_path, base_config=real_hpo_base_config,
                                     param_space=sweep_space(), search_seed=0, **stated)
    assert not captured


@pytest.mark.parametrize("scheduler, named", [
    ({k: v for k, v in asha_scheduler().items() if k != "max_t"}, "unstated ['max_t']"),
    ({**asha_scheduler(), "hard_stop": True}, "not taken ['hard_stop']"),
    ({"name": "fifo", "max_t": 5}, "not taken ['max_t']"),
], ids=["asha-without-max_t", "asha-with-a-setting-it-does-not-take", "fifo-with-max_t"])
def test_run_hyperparameter_search_refuses_a_scheduler_setting_unstated_or_not_taken(
    tmp_path, real_hpo_base_config, monkeypatch, scheduler, named,
):
    """guard. A scheduler block leaving a setting of its Ray class unstated, or stating one the
    class does not take, refuses at the door naming it, before anything is minted."""
    import tcip_mcp.tools.training_tools as tt

    captured = _stub_search(monkeypatch)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space=sweep_space(), n_trials=1,
        search_alg="random", scheduler=scheduler, search_seed=0)

    assert named in result["error"], result
    assert not captured
    assert not tt.experiments.experiments_dir(tmp_path).exists()


def test_a_sweeps_recorded_scheduler_is_the_one_ray_tunes_under(
    tmp_path, real_hpo_base_config, monkeypatch,
):
    """guard. The scheduler block a sweep states is recorded whole on its input, and Ray's own
    TuneConfig receives the scheduler built from that record: an ASHA scheduler capped at the
    stated max_t, never at Ray's default."""
    import contextlib

    pytest.importorskip("ray")
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.pipelines.training import hpo

    tuned: dict = {}

    class _RecordingTuner:
        def __init__(self, trainable, *, param_space, tune_config, run_config):
            tuned["scheduler"] = tune_config.scheduler

        def fit(self) -> None:
            return None

    monkeypatch.setattr(tune, "Tuner", _RecordingTuner)
    monkeypatch.setattr(hpo, "_ray_session", lambda ray, num_cpus: contextlib.nullcontext())
    scheduler = asha_scheduler(max_t=4)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, param_space=sweep_space(), n_trials=1,
        search_alg="random", scheduler=scheduler, search_seed=0)

    sweep = tt.monitor_training(tmp_path, result["sweep"]["sweep_id"])["sweep"]
    assert sweep["input"]["scheduler"] == scheduler
    assert isinstance(tuned["scheduler"], AsyncHyperBandScheduler)
    assert tuned["scheduler"]._max_t == 4
