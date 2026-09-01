"""Cooperative sweep cancel: ``cancel_hpo``, ``run_hpo``'s cancelled-manifest exits,
``_run_hpo_trial``'s entry check, and the sweep ``Stopper``'s two stop-all conditions.
"""

from __future__ import annotations




def test_cancel_hpo_refuses_a_study_neither_manifest_nor_registry_names(tmp_path) -> None:
    from tcip_mcp.tools.training_tools import cancel_hpo

    result = cancel_hpo("hpo_totally_unknown", root=str(tmp_path))
    assert "error" in result


def test_cancel_hpo_under_a_non_default_root(tmp_path) -> None:
    import tcip_store

    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, cancel_hpo, sweep_dir, sweep_manifest_key,
    )

    other_root = tmp_path / "other_project"
    sweep_root = sweep_dir("hpo_other01", root=other_root)
    sweep_root.mkdir(parents=True)
    tcip_store.replace(sweep_manifest_key("hpo_other01", root=other_root),
                       {"study_name": "hpo_other01", "status": "running", "n_trials": 1})

    result = cancel_hpo("hpo_other01", root=str(other_root))
    assert result == {"study_name": "hpo_other01", "status": "running", "cancel_requested": True}
    assert (sweep_root / SWEEP_CANCEL_SENTINEL).exists()


def test_run_hpo_records_a_cancelled_manifest_when_the_file_exists_before_preflight_creates_it(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    """A cancel requested against a study_name a caller already minted (before run_hpo's own
    manifest write) records a cancelled manifest rather than being refused."""
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL, run_hpo, sweep_dir

    study_name = "hpo_precancel1"
    sweep_root = sweep_dir(study_name, str(tmp_path))
    sweep_root.mkdir(parents=True)
    (sweep_root / SWEEP_CANCEL_SENTINEL).touch()

    called = []

    def fake_search(**kw):
        called.append(kw)
        return {}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name=study_name)
    assert result == {"status": "cancelled", "study_name": study_name}
    assert called == []  # the search never started

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"


def test_run_hpo_records_a_cancelled_manifest_when_tune_search_returns_after_a_cancel(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, run_hpo, sweep_dir, sweep_manifest_key, study_result_key,
    )

    study_name = "hpo_cancelret1"
    sweep_root = sweep_dir(study_name, str(tmp_path))

    def fake_search(**kw):
        # The cancel lands mid-search, from the sweep's own perspective: the file appears
        # only once the (fake) search is already under way.
        (sweep_root / SWEEP_CANCEL_SENTINEL).touch()
        return {"best_params": {}, "best_value": 0.0, "n_trials": 1, "study_name": study_name,
                "all_trials": [], "search_alg": "random", "scheduler": "asha",
                "warm_start": False, "baseline_params": None}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name=study_name)
    assert result == {"status": "cancelled", "study_name": study_name}

    import tcip_store
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"
    assert not tcip_store.exists(study_result_key(study_name, str(tmp_path)))


def test_run_hpo_records_a_cancelled_manifest_when_tune_search_raises_after_a_cancel(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL, run_hpo, sweep_dir

    study_name = "hpo_cancelraise1"
    sweep_root = sweep_dir(study_name, str(tmp_path))

    def fake_search(**kw):
        (sweep_root / SWEEP_CANCEL_SENTINEL).touch()
        raise RuntimeError("the Ray cluster was torn down mid-sweep")

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name=study_name)
    assert result == {"status": "cancelled", "study_name": study_name}

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"


def test_run_hpo_trial_reports_the_losing_side_without_training_when_the_sweep_is_cancelled(
    tmp_path, real_hpo_base_config
) -> None:
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, _run_hpo_trial, trial_config_key,
    )

    sweep_root = tmp_path / "sweep"
    trial_dir = sweep_root / "trial_aaa00000"
    sweep_root.mkdir(parents=True)
    (sweep_root / SWEEP_CANCEL_SENTINEL).touch()

    reported = []
    _run_hpo_trial({}, reported.append, real_hpo_base_config, str(trial_dir))

    assert len(reported) == 1
    assert reported[0] == float("inf") or reported[0] == float("-inf")
    import tcip_store
    assert not tcip_store.exists(trial_config_key(sweep_root, trial_dir.name))


def test_sweep_stopper_stop_all_true_once_no_trial_still_looks_unfinished(tmp_path) -> None:
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL, trial_config_key

    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    running_trial = sweep_root / "trial_running0"
    running_trial.mkdir()
    finished_trial = sweep_root / "trial_done0000"
    finished_trial.mkdir()

    import tcip_store
    tcip_store.replace(trial_config_key(sweep_root, finished_trial.name), {"trial_params": {}})

    stopper = _build_sweep_stopper(lambda: (sweep_root / SWEEP_CANCEL_SENTINEL).exists(), sweep_root)
    assert stopper.stop_all() is False  # the file does not exist yet

    (sweep_root / SWEEP_CANCEL_SENTINEL).touch()
    assert stopper.stop_all() is False  # running_trial has no resolved config yet
    assert stopper("running_trial", {}) is True  # per-trial call always follows the file

    tcip_store.replace(trial_config_key(sweep_root, running_trial.name), {"trial_params": {}})
    assert stopper.stop_all() is True  # every trial directory now looks finished


def test_sweep_stopper_stop_all_true_after_the_stale_window_elapses_regardless_of_running_trials(
    tmp_path, monkeypatch
) -> None:
    from tcip_mcp.pipelines.training import hpo as hpo_module
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    monkeypatch.setattr("tcip_mcp.tools.training_tools.TCIP_HEARTBEAT_STALE_SECONDS", 0.05)
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    (sweep_root / "trial_stuck0000").mkdir()  # never writes its resolved config: stuck

    sentinel = sweep_root / SWEEP_CANCEL_SENTINEL
    sentinel.touch()
    stopper = hpo_module._build_sweep_stopper(lambda: sentinel.exists(), sweep_root)
    assert stopper.stop_all() is False  # freshly written: still inside the window

    import os
    import time
    stale = time.time() - 10
    os.utime(sentinel, (stale, stale))
    assert stopper.stop_all() is True  # the bounded fallback: Ray's own stop takes over
