"""run_hpo persists trial results to Ray's storage_path + a durable result file, and stamps a
manifest under the sweep's own directory. The Ray Tune search itself is faked so the test
doesn't init Ray or train."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts


def _real_base_config(tmp_path: Path) -> dict:
    """A base config the door's own structural preflight admits: an importable builder and a
    data section whose directories exist, so these tests exercise the sweep search itself
    rather than the door's refusal."""
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir(exist_ok=True)
    lbls.mkdir(exist_ok=True)
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
    }


def test_run_hpo_threads_storage_path_and_writes_result(tmp_path, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {"lr": 0.01}, "best_value": 0.5, "n_trials": 1,
                "study_name": kw.get("study_name")}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    tt.run_hpo(base_config=_real_base_config(tmp_path), n_trials=1, output_dir=str(tmp_path))

    # A unique study name + storage_path under output_dir were threaded into the search.
    assert captured["study_name"].startswith("hpo_")
    assert captured["storage_path"] == str(tmp_path)
    assert captured["mode"] == "min"  # the composite objective is lower=better
    assert captured["metric"] == "objective"

    # The result was persisted to a durable record alongside Ray's trial store.
    result = ts.read(tt.study_result_key(captured["study_name"], str(tmp_path)))
    assert result["best_params"] == {"lr": 0.01}


def test_run_hpo_defaults_storage_to_platform_root_when_no_output_dir(tmp_path, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])
    # Pin the platform root so the store lands under this tmp dir, not the real repo.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))

    tt.run_hpo(base_config=_real_base_config(tmp_path), n_trials=1)  # no output_dir
    assert (tmp_path / ".tcip" / "hpo").as_posix() in captured["storage_path"].replace("\\", "/")


def test_run_hpo_stamps_a_running_manifest_and_namespaces_trial_dirs(tmp_path, monkeypatch):
    """A sweep is on disk from the moment it starts, with its trials under its own
    directory: an agent-launched sweep has no other way to be seen while it runs."""
    import tcip_mcp.tools.training_tools as tt

    observed: dict = {}

    def fake_trial(config, report, base_config, trial_dir):
        observed["trial_dir"] = trial_dir

    def fake_search(**kw):
        study = kw["study_name"]
        observed["while_running"] = ts.read(tt.sweep_manifest_key(study, str(tmp_path)))
        kw["objective_fn"]({"lr": 0.1}, lambda value: None)
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": study}

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=_real_base_config(tmp_path), n_trials=1,
                        output_dir=str(tmp_path))
    study = result["study_name"]

    running = observed["while_running"]
    assert running["status"] == "running"
    assert running["study_name"] == study
    assert running["n_trials"] == 1
    assert running["param_space"]  # the space the sweep is searching

    # Trials land under the sweep's own directory, not loose in the shared hpo root.
    assert Path(observed["trial_dir"]).parent == tmp_path / study
    assert Path(observed["trial_dir"]).name.startswith("trial_")

    finished = ts.read(tt.sweep_manifest_key(study, str(tmp_path)))
    assert finished["status"] == "completed"
    assert finished["finished_at"]
    assert finished["result"]["best_params"] == {"lr": 0.1}


def test_run_hpo_marks_the_manifest_failed_when_the_search_raises(tmp_path, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def exploding_search(**kw):
        captured["study_name"] = kw["study_name"]
        raise RuntimeError("no ray here")

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", exploding_search)

    with pytest.raises(RuntimeError):
        tt.run_hpo(base_config=_real_base_config(tmp_path), n_trials=1, output_dir=str(tmp_path))

    manifest = ts.read(tt.sweep_manifest_key(captured["study_name"], str(tmp_path)))
    assert manifest["status"] == "failed"
    assert "no ray here" in manifest["error"]


def test_run_hpo_refuses_before_minting_when_the_base_config_fails_preflight(tmp_path, monkeypatch):
    """An unimportable builder refuses at the door, writing no manifest, rather than reporting
    as the losing side in every trial."""
    import tcip_mcp.tools.training_tools as tt

    ran = []

    def fake_search(**kw):
        ran.append(1)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config={"model_source": {"builder": "not.a:real_builder"}},
                        n_trials=1, output_dir=str(tmp_path))

    assert "error" in result
    assert not ran
    # No sweep directory, no manifest, no result record: nothing was minted.
    assert not (tmp_path / "hpo_manifest.json").exists()
    assert list(tmp_path.glob("hpo_*")) == []


def test_run_hpo_checks_a_swept_placeholder_axis_at_its_resolved_value(tmp_path, monkeypatch):
    """A param_space sampling the builder itself resolves a real one before the door checks it,
    so a base config carrying only a placeholder there is not refused for that reason."""
    import tcip_mcp.tools.training_tools as tt

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {}, "best_value": 0.1, "n_trials": 1},
    )

    result = tt.run_hpo(
        base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "builder_kwargs": {"num_classes": 1}, "task": "detection"},
                    "data": _real_base_config(tmp_path)["data"]},
        param_space={"model_source.builder": {
            "type": "categorical", "choices": ["tests.bespoke_models:build_bespoke_detection"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" not in result


def test_run_hpo_refuses_when_every_sampled_value_of_a_swept_axis_still_fails(
    tmp_path, monkeypatch
):
    """The same swept-builder shape, but no sampled value resolves to anything importable: the
    door still refuses, rather than admitting a sweep with nothing legitimate to search."""
    import tcip_mcp.tools.training_tools as tt

    ran = []

    def fake_search(**kw):
        ran.append(1)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(
        base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "builder_kwargs": {"num_classes": 1}, "task": "detection"},
                    "data": _real_base_config(tmp_path)["data"]},
        param_space={"model_source.builder": {"type": "categorical", "choices": ["still:bad"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" in result
    assert not ran


def test_run_hpo_passes_agent_search_and_scheduler_choices(tmp_path, monkeypatch):
    """search_alg + scheduler are the agent's choice and reach tune_search verbatim."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])

    tt.run_hpo(base_config=_real_base_config(tmp_path), n_trials=3,
               output_dir=str(tmp_path), search_alg="bayesopt", scheduler="median",
               max_concurrent=2)
    assert captured["search_alg"] == "bayesopt"
    assert captured["scheduler"] == "median"
    assert captured["max_concurrent"] == 2
    assert captured["num_samples"] == 3
