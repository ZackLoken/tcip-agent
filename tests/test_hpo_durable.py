"""run_hpo persists trial results to Ray's storage_path + a durable result file, and stamps a
manifest under the sweep's own directory. The Ray Tune search itself is faked so the test
doesn't init Ray or train."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts


def test_run_hpo_threads_storage_path_and_writes_result(tmp_path, real_hpo_base_config, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {"lr": 0.01}, "best_value": 0.5, "n_trials": 1,
                "study_name": kw.get("study_name")}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))

    # A unique study name + storage_path under output_dir were threaded into the search.
    assert captured["study_name"].startswith("hpo_")
    assert captured["storage_path"] == str(tmp_path)
    assert captured["mode"] == "min"  # the composite objective is lower=better
    assert captured["metric"] == "objective"

    # The result was persisted to a durable record alongside Ray's trial store.
    result = ts.read(tt.study_result_key(captured["study_name"], str(tmp_path)))
    assert result["best_params"] == {"lr": 0.01}


def test_run_hpo_defaults_storage_to_platform_root_when_no_output_dir(
    tmp_path, real_hpo_base_config, monkeypatch
):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])
    # Pin the platform root so the store lands under this tmp dir, not the real repo.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    tt.run_hpo(base_config=real_hpo_base_config, n_trials=1)  # no output_dir
    assert (tmp_path / ".tcip" / "hpo").as_posix() in captured["storage_path"].replace("\\", "/")


def test_run_hpo_stamps_a_running_manifest_and_namespaces_trial_dirs(
    tmp_path, real_hpo_base_config, monkeypatch
):
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

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1,
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


def test_run_hpo_honours_a_callers_study_name(tmp_path, real_hpo_base_config, monkeypatch):
    """A caller that already minted its own sweep id (the Tuning route's launch, so its own
    registry entry, manifest and every sweep route agree on it) must have the real search,
    not a stand-in, write the manifest and the sweep directory under that same id."""
    import tcip_mcp.tools.training_tools as tt

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                     "study_name": kw["study_name"]},
    )

    given_id = "hpo-caller-supplied-id"
    result = tt.run_hpo(
        base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
        study_name=given_id,
    )

    assert result["study_name"] == given_id
    manifest = ts.read(tt.sweep_manifest_key(given_id, str(tmp_path)))
    assert manifest["study_name"] == given_id
    assert Path(manifest["sweep_dir"]) == tmp_path / given_id


def test_run_hpo_marks_the_manifest_failed_when_the_search_raises(
    tmp_path, real_hpo_base_config, monkeypatch
):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def exploding_search(**kw):
        captured["study_name"] = kw["study_name"]
        raise RuntimeError("no ray here")

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", exploding_search)

    with pytest.raises(RuntimeError):
        tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))

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


def test_run_hpo_checks_a_swept_placeholder_axis_at_its_resolved_value(
    tmp_path, real_hpo_base_config, monkeypatch
):
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
                    "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {
            "type": "categorical", "choices": ["tests.bespoke_models:build_bespoke_detection"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" not in result


def test_run_hpo_refuses_when_every_sampled_value_of_a_swept_axis_still_fails(
    tmp_path, real_hpo_base_config, monkeypatch
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
                    "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {"type": "categorical", "choices": ["still:bad"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" in result
    assert not ran


def test_run_hpo_refuses_when_a_non_first_swept_choice_fails_preflight(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """The whole space is checked, not only the first sampled corner: a second choice that fails
    to import must still be caught, even though the first choice the old, narrower check saw
    is a real builder."""
    import tcip_mcp.tools.training_tools as tt

    ran = []

    def fake_search(**kw):
        ran.append(1)
        return {"best_params": {}, "best_value": 0.1, "n_trials": 1}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(
        base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "builder_kwargs": {"num_classes": 1}, "task": "detection"},
                    "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {
            "type": "categorical",
            "choices": ["tests.bespoke_models:build_bespoke_detection", "still:bad"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" in result
    assert "still:bad" in result["error"]
    assert not ran


def test_run_hpo_admits_a_swept_axis_whose_every_choice_resolves(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """Admits valid work: every categorical choice checked, and every one of them importable, so
    the sweep runs rather than being refused for having more than one choice."""
    import tcip_mcp.tools.training_tools as tt

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {}, "best_value": 0.1, "n_trials": 1},
    )

    result = tt.run_hpo(
        base_config={"model_source": {"builder": "PLACEHOLDER:PLACEHOLDER",
                                      "builder_kwargs": {"num_classes": 1}, "task": "detection"},
                    "data": real_hpo_base_config["data"]},
        param_space={"model_source.builder": {
            "type": "categorical",
            "choices": ["tests.bespoke_models:build_bespoke_detection",
                        "tests.bespoke_models:build_bare_score_thresh_detector"]}},
        n_trials=1, output_dir=str(tmp_path),
    )

    assert "error" not in result


def test_run_hpo_carries_best_value_state_into_the_completed_manifest(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """A non-finite best_value still names why in the manifest, rather than a bare null with
    no reason a served sweep can show."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_store import stored_number

    def fake_search(**kw):
        return {"best_params": {"lr": 0.01}, "n_trials": 1, "study_name": kw["study_name"],
                **stored_number("best_value", float("nan"))}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))

    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["result"]["best_value"] is None
    assert manifest["result"]["best_value_state"] == "nan"


def test_run_hpo_passes_agent_search_and_scheduler_choices(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """search_alg + scheduler are the agent's choice and reach tune_search verbatim."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])

    tt.run_hpo(base_config=real_hpo_base_config, n_trials=3,
               output_dir=str(tmp_path), search_alg="bayesopt", scheduler="median",
               max_concurrent=2)
    assert captured["search_alg"] == "bayesopt"
    assert captured["scheduler"] == "median"
    assert captured["max_concurrent"] == 2
    assert captured["num_samples"] == 3
