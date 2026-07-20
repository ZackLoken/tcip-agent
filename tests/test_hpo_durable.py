"""run_hpo persists trial results to Ray's storage_path + a durable result file. The Ray Tune
search itself is faked so the test doesn't init Ray or train."""

from __future__ import annotations

import json


def test_run_hpo_threads_storage_path_and_writes_result(tmp_path, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {"lr": 0.01}, "best_value": 0.5, "n_trials": 1,
                "study_name": kw.get("study_name")}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    tt.run_hpo(base_config={"model_source": {"builder": "x:y"}}, n_trials=1,
               output_dir=str(tmp_path))

    # A unique study name + storage_path under output_dir were threaded into the search.
    assert captured["study_name"].startswith("hpo_")
    assert captured["storage_path"] == str(tmp_path)
    assert captured["mode"] == "min"  # the composite objective is lower=better
    assert captured["metric"] == "objective"

    # The result was persisted to a durable json alongside Ray's trial store.
    result_file = tmp_path / f"{captured['study_name']}.json"
    assert result_file.is_file()
    assert json.loads(result_file.read_text())["best_params"] == {"lr": 0.01}


def test_run_hpo_defaults_storage_to_platform_root_when_no_output_dir(tmp_path, monkeypatch):
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])
    # Pin the platform root so the store lands under this tmp dir, not the real repo.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))

    tt.run_hpo(base_config={"model_source": {"builder": "x:y"}}, n_trials=1)  # no output_dir
    assert (tmp_path / ".tcip" / "hpo").as_posix() in captured["storage_path"].replace("\\", "/")


def test_run_hpo_passes_agent_search_and_scheduler_choices(tmp_path, monkeypatch):
    """search_alg + scheduler are the agent's choice and reach tune_search verbatim."""
    import tcip_mcp.tools.training_tools as tt

    captured: dict = {}
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])

    tt.run_hpo(base_config={"model_source": {"builder": "x:y"}}, n_trials=3,
               output_dir=str(tmp_path), search_alg="bayesopt", scheduler="median",
               max_concurrent=2)
    assert captured["search_alg"] == "bayesopt"
    assert captured["scheduler"] == "median"
    assert captured["max_concurrent"] == 2
    assert captured["num_samples"] == 3
