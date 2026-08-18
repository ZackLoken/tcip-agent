"""One sweep has one identity and one address.

A sweep is discovered on disk by whatever a later reader finds in its manifest: the study name and
the directory that name resolves to. If the manifest's recorded directory is not the one holding
it, that reader lists a sweep whose trials it can never open.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts


def _stub_sweep(monkeypatch, observed: dict):
    """Run the sweep body without Ray: the search stub drives one trial through the objective."""
    import tcip_mcp.tools.training_tools as tt

    def fake_trial(config, report, base_config, trial_dir):
        observed["trial_dir"] = trial_dir

    def fake_search(**kw):
        kw["objective_fn"]({"lr": 0.1}, lambda value: None)
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr(tt, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    return tt


def test_a_sweeps_manifest_records_the_directory_that_holds_it(tmp_path: Path, monkeypatch) -> None:
    """The manifest's ``sweep_dir`` is the sweep's own directory, not the shared root several
    sweeps share, and the trials of that sweep live under exactly that recorded directory."""
    observed: dict = {}
    tt = _stub_sweep(monkeypatch, observed)
    sweeps_root = tmp_path / "sweeps"

    result = tt.run_hpo(base_config={"model_source": {"builder": "x:y"}}, n_trials=1,
                        output_dir=str(sweeps_root))
    study_name = result["study_name"]

    manifest = ts.read(tt.sweep_manifest_key(study_name, str(sweeps_root)))
    recorded_dir = Path(manifest["sweep_dir"])
    assert recorded_dir == tt.sweep_dir(study_name, str(sweeps_root))
    assert recorded_dir.name == manifest["study_name"]
    assert Path(observed["trial_dir"]).parent == recorded_dir
    assert ts.exists(tt.study_result_key(study_name, str(sweeps_root)))


def test_a_sweep_launched_without_an_output_dir_is_addressed_the_same_way(
        tmp_path: Path, monkeypatch) -> None:
    """The default store under the project root addresses a sweep exactly as an explicit
    ``output_dir`` does, so a reader needs no second convention for agent-launched sweeps."""
    observed: dict = {}
    tt = _stub_sweep(monkeypatch, observed)

    result = tt.run_hpo(base_config={"model_source": {"builder": "x:y"}}, n_trials=1)

    study_name = result["study_name"]
    manifest = ts.read(tt.sweep_manifest_key(study_name))
    expected_dir = tmp_path / ".tcip" / "hpo" / study_name
    assert Path(manifest["sweep_dir"]) == expected_dir
    assert manifest["status"] == "completed"
    assert Path(observed["trial_dir"]).parent == expected_dir
