"""L16 — run_hpo persists the Optuna study (sqlite) + a result file, not an ephemeral in-memory
study. The search itself is faked so the test doesn't train."""

from __future__ import annotations

import json


def test_run_hpo_threads_durable_storage_and_writes_result(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.hpo as hpo_mod

    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return {"best_params": {"head_lr": 0.01}, "best_value": 0.5, "n_trials": 1,
                "study_name": kw.get("study_name")}

    monkeypatch.setattr(hpo_mod, "optuna_search", fake_search)
    from tcip_mcp.tools.training_tools import run_hpo

    res = run_hpo(base_config={"model_spec": {}}, n_trials=1, output_dir=str(tmp_path))

    # A unique study name + a sqlite storage URL (as_posix'd) were threaded into the search.
    assert captured["study_name"].startswith("hpo_")
    assert captured["storage"].startswith("sqlite:///") and "hpo.db" in captured["storage"]
    assert res["storage"] == captured["storage"]

    # The result was persisted to a durable json alongside the study db.
    result_file = tmp_path / f"{captured['study_name']}.json"
    assert result_file.is_file()
    assert json.loads(result_file.read_text())["best_params"] == {"head_lr": 0.01}


def test_run_hpo_defaults_storage_to_platform_root_when_no_output_dir(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.hpo as hpo_mod

    captured: dict = {}
    monkeypatch.setattr(hpo_mod, "optuna_search",
                        lambda **kw: (captured.update(kw), {"study_name": kw["study_name"]})[1])
    # Pin the platform root so the sqlite lands under this tmp dir, not the real repo.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.tools.training_tools import run_hpo

    run_hpo(base_config={"model_spec": {}}, n_trials=1)  # no output_dir
    assert (tmp_path / ".tcip" / "hpo").as_posix() in captured["storage"]
