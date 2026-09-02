"""Cooperative sweep cancel: ``cancel_hpo``, ``run_hpo``'s cancelled-manifest exits,
``_run_hpo_trial``'s entry check, and the sweep ``Stopper``'s two stop-all conditions.
"""

from __future__ import annotations

import pytest

def test_cancel_hpo_refuses_a_study_neither_manifest_nor_registry_names(tmp_path) -> None:
    from tcip_mcp.tools.training_tools import cancel_hpo

    result = cancel_hpo("hpo_totally_unknown", root=str(tmp_path))
    assert "error" in result


def test_mark_sweep_launching_lets_cancel_hpo_find_a_study_with_no_manifest_yet(tmp_path) -> None:
    """The pre-manifest window: a study a caller has marked as launching, but ``run_hpo`` has
    not yet reached its first manifest write, is a sweep ``cancel_hpo`` can find, not a study
    it refuses as unknown."""
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, cancel_hpo, mark_sweep_launching, sweep_dir,
    )

    study_name = "hpo_marked001"
    mark_sweep_launching(study_name, str(tmp_path))

    result = cancel_hpo(study_name, str(tmp_path))
    assert "error" not in result
    assert result == {"study_name": study_name, "status": "running", "cancel_requested": True}
    assert (sweep_dir(study_name, str(tmp_path)) / SWEEP_CANCEL_SENTINEL).exists()


def test_run_hpo_records_a_cancelled_manifest_for_a_cancel_that_landed_before_it_was_called(
    tmp_path, real_hpo_base_config
) -> None:
    """A cancel that reaches the sweep in the window the web relaunch route marks before
    starting its worker (mark, then a cancel, then ``run_hpo`` itself) records the same
    before-start cancellation ``run_hpo``'s own pre-existing sentinel check already handles;
    today's refusal is at ``cancel_hpo``, proven separately, above."""
    from tcip_mcp.tools.training_tools import (
        _CANCEL_BEFORE_START_REASON, cancel_hpo, mark_sweep_launching, run_hpo, sweep_manifest_key,
    )

    study_name = "hpo_racedcancel1"
    mark_sweep_launching(study_name, str(tmp_path))
    cancelled = cancel_hpo(study_name, str(tmp_path))
    assert "error" not in cancelled

    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name=study_name)
    assert result["status"] == "cancelled"
    assert result["error"] == _CANCEL_BEFORE_START_REASON

    import tcip_store
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"


def test_mark_sweep_launching_is_discarded_once_run_hpo_reaches_its_first_manifest_write(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    from tcip_mcp.tools.training_tools import (
        _sweep_launching, mark_sweep_launching, run_hpo, sweep_dir,
    )

    study_name = "hpo_markclear1"
    resolved_root = sweep_dir(study_name, str(tmp_path)).resolve()
    mark_sweep_launching(study_name, str(tmp_path))
    assert _sweep_launching(study_name, resolved_root) is True

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                     "study_name": kw["study_name"]},
    )
    run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
           study_name=study_name)

    assert _sweep_launching(study_name, resolved_root) is False


def test_mark_sweep_launching_is_discarded_even_when_preflight_refuses(tmp_path) -> None:
    """The mark is scoped to the window before run_hpo's first manifest write, refusal
    included: a caller's mark for a study whose config fails preflight must not linger forever
    once run_hpo has already answered for it."""
    from tcip_mcp.tools.training_tools import _sweep_launching, mark_sweep_launching, run_hpo, sweep_dir

    study_name = "hpo_markrefuse1"
    resolved_root = sweep_dir(study_name, str(tmp_path)).resolve()
    mark_sweep_launching(study_name, str(tmp_path))

    result = run_hpo(base_config={"model_source": {"builder": "not.a:real_builder"}},
                     n_trials=1, output_dir=str(tmp_path), study_name=study_name)
    assert "error" in result
    assert _sweep_launching(study_name, resolved_root) is False


def test_mark_sweep_launching_is_discarded_when_check_json_value_refuses(tmp_path) -> None:
    """The two ``check_json_value`` refusals sit inside ``run_hpo``'s own try/finally now, not
    ahead of it, so a caller's launch mark is discarded on that exit too."""
    from tcip_mcp.tools.training_tools import _sweep_launching, mark_sweep_launching, run_hpo, sweep_dir

    study_name = "hpo_badjson1"
    resolved_root = sweep_dir(study_name, str(tmp_path)).resolve()
    mark_sweep_launching(study_name, str(tmp_path))

    with pytest.raises(TypeError):
        run_hpo(base_config={"data": {"not_json": {1, 2, 3}}}, n_trials=1,
               output_dir=str(tmp_path), study_name=study_name)

    assert _sweep_launching(study_name, resolved_root) is False


def test_cancel_hpo_under_a_mismatched_root_is_refused_despite_the_launch_mark(tmp_path) -> None:
    """A mark recorded under one resolved root does not let a cancel under another root count
    the study as found: a cancel that lands where run_hpo will never look is refused, not
    honoured with a sentinel nothing will ever poll."""
    from tcip_mcp.tools.training_tools import cancel_hpo, mark_sweep_launching, sweep_dir

    study_name = "hpo_wrongroot1"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    mark_sweep_launching(study_name, root=root_a)

    result = cancel_hpo(study_name, root=str(root_b))
    assert "error" in result
    assert not sweep_dir(study_name, root=root_b).exists()


def test_cancel_hpo_under_a_non_default_root(tmp_path) -> None:
    import tcip_store
    from datetime import datetime, timezone

    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, cancel_hpo, sweep_dir, sweep_manifest_key,
    )

    other_root = tmp_path / "other_project"
    sweep_root = sweep_dir("hpo_other01", root=other_root)
    sweep_root.mkdir(parents=True)
    tcip_store.replace(sweep_manifest_key("hpo_other01", root=other_root),
                       {"study_name": "hpo_other01", "status": "running", "n_trials": 1,
                        "heartbeat": datetime.now(timezone.utc).isoformat()})

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
    assert result["status"] == "cancelled"
    assert result["error"]  # the reason travels with the result, not only the manifest
    assert called == []  # the search never started

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"
    assert manifest["error"] == result["error"]


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
    assert result["status"] == "cancelled"
    assert result["error"]

    import tcip_store
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"
    assert manifest["error"] == result["error"]
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
    assert result["status"] == "cancelled"
    assert result["error"]

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"
    assert manifest["error"] == result["error"]


def test_a_terminal_cancelled_manifest_keeps_cancel_requested(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    """run_hpo re-derives cancel_requested from the sweep's own stop file on every manifest
    write, including its terminal one, rather than writing its stale in-memory copy (which
    never carries a field cancel_hpo set on disk out of band) back over it."""
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, run_hpo, sweep_dir, sweep_manifest_key,
    )

    study_name = "hpo_keepflag1"
    sweep_root = sweep_dir(study_name, str(tmp_path))

    def fake_search(**kw):
        (sweep_root / SWEEP_CANCEL_SENTINEL).touch()
        return {"best_params": {}, "best_value": 0.0, "n_trials": 1, "study_name": study_name}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
           study_name=study_name)

    import tcip_store
    manifest = tcip_store.read(sweep_manifest_key(study_name, str(tmp_path)))
    assert manifest["status"] == "cancelled"
    assert manifest["cancel_requested"] is True


def test_cancel_hpo_leaves_a_terminal_manifest_terminal(tmp_path) -> None:
    """cancel_hpo never writes over a manifest already in a terminal status: the sentinel
    files still get written (a stray straggler process still stops), but the manifest's own
    status and content are not disturbed by a cancel arriving after the sweep already ended."""
    import tcip_store
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, cancel_hpo, sweep_dir, sweep_manifest_key,
    )

    study_name = "hpo_alreadydone1"
    sweep_root = sweep_dir(study_name, root=tmp_path)
    sweep_root.mkdir(parents=True)
    tcip_store.replace(
        sweep_manifest_key(study_name, root=tmp_path),
        {"study_name": study_name, "status": "completed", "n_trials": 1,
         "result": {"best_value": 0.1}},
    )

    result = cancel_hpo(study_name, root=str(tmp_path))
    assert result["status"] == "completed"
    assert (sweep_root / SWEEP_CANCEL_SENTINEL).exists()  # the sentinel is still written

    manifest = tcip_store.read(sweep_manifest_key(study_name, root=tmp_path))
    assert manifest["status"] == "completed"
    assert "cancel_requested" not in manifest  # untouched: never written over a terminal manifest


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


def test_cancel_hpo_writes_every_sentinel_beside_a_trial_whose_record_will_not_decode(
    tmp_path
) -> None:
    """A trial whose resolved-config record exists but will not decode means the trial wrote
    it, so it reads as not-running (the same call ``cancel_hpo`` and the Tune ``Stopper`` share)
    rather than raising and half-applying the cancel to every other trial."""
    import tcip_store
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper
    from tcip_mcp.pipelines.training.run_registry import CANCEL_SENTINEL
    from tcip_mcp.tools.training_tools import (
        SWEEP_CANCEL_SENTINEL, cancel_hpo, sweep_dir, sweep_manifest_key, trial_config_key,
    )
    from tests._record_damage_fixtures import damage_record

    study_name = "hpo_undecodable1"
    sweep_root = sweep_dir(study_name, root=tmp_path)
    sweep_root.mkdir(parents=True)
    good_trial = sweep_root / "trial_good00000"
    good_trial.mkdir()
    corrupt_trial = sweep_root / "trial_corrupt00"
    corrupt_trial.mkdir()
    running_trial = sweep_root / "trial_running00"
    running_trial.mkdir()

    tcip_store.replace(trial_config_key(sweep_root, good_trial.name), {"trial_params": {}})
    corrupt_key = trial_config_key(sweep_root, corrupt_trial.name)
    tcip_store.replace(corrupt_key, {"trial_params": {}})
    damage_record(corrupt_key, b"{not json at all")
    tcip_store.replace(
        sweep_manifest_key(study_name, root=tmp_path),
        {"study_name": study_name, "status": "running", "n_trials": 3},
    )

    result = cancel_hpo(study_name, root=str(tmp_path))
    assert result["cancel_requested"] is True
    assert (sweep_root / SWEEP_CANCEL_SENTINEL).exists()
    assert (running_trial / CANCEL_SENTINEL).exists()
    assert not (good_trial / CANCEL_SENTINEL).exists()  # already finished before the cancel
    assert not (corrupt_trial / CANCEL_SENTINEL).exists()  # undecodable means finished too

    stopper, callback = _build_sweep_stopper(
        lambda: (sweep_root / SWEEP_CANCEL_SENTINEL).exists(), sweep_root)
    assert stopper.stop_all() is True  # Ray never reported any trial live: the set is empty


class _FakeTrial:
    """Stands in for a ``ray.tune.experiment.Trial`` in the callback: only ``trial_id`` is read."""

    def __init__(self, trial_id: str) -> None:
        self.trial_id = trial_id


def test_sweep_stopper_stop_all_true_once_ray_no_longer_holds_any_trial_live(tmp_path) -> None:
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    sentinel = sweep_root / SWEEP_CANCEL_SENTINEL

    stopper, callback = _build_sweep_stopper(lambda: sentinel.exists(), sweep_root)
    assert stopper.stop_all() is False  # the file does not exist yet

    trial = _FakeTrial("running_trial")
    callback.on_trial_start(0, [], trial)
    sentinel.touch()
    assert stopper.stop_all() is False  # Ray still holds the trial live
    assert stopper("running_trial", {}) is True  # per-trial call always follows the file

    callback.on_trial_complete(0, [], trial)
    assert stopper.stop_all() is True  # Ray no longer holds any trial live


def test_sweep_stopper_stop_all_true_at_once_for_a_trial_ray_killed_outright(tmp_path) -> None:
    """A trial Ray kills outright may never reach ``_run_hpo_trial``'s own ``finally``, so its
    resolved-config record never gets written; the old disk-based check read that trial as
    still running until the heartbeat stale window passed regardless. The callback's own live
    set, fed by Ray's own ``on_trial_error`` report, says otherwise at once."""
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    sweep_root = tmp_path / "sweep"
    killed_trial = sweep_root / "trial_killed00"
    killed_trial.mkdir(parents=True)  # never writes its resolved config
    sentinel = sweep_root / SWEEP_CANCEL_SENTINEL
    sentinel.touch()

    stopper, callback = _build_sweep_stopper(lambda: sentinel.exists(), sweep_root)
    trial = _FakeTrial("killed_trial")
    callback.on_trial_start(0, [], trial)
    callback.on_trial_error(0, [], trial)

    assert stopper.stop_all() is True  # Ray reported it gone; the disk record never has to exist


def test_sweep_stopper_stop_all_treats_an_errored_trial_as_no_longer_live(tmp_path) -> None:
    from tcip_mcp.pipelines.training.hpo import _build_sweep_stopper
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    sentinel = sweep_root / SWEEP_CANCEL_SENTINEL
    sentinel.touch()

    stopper, callback = _build_sweep_stopper(lambda: sentinel.exists(), sweep_root)
    trial = _FakeTrial("errored_trial")
    callback.on_trial_start(0, [], trial)
    assert stopper.stop_all() is False

    callback.on_trial_error(0, [], trial)
    assert stopper.stop_all() is True


def test_sweep_stopper_stop_all_true_after_the_stale_window_elapses_regardless_of_a_live_trial(
    tmp_path, monkeypatch
) -> None:
    from tcip_mcp.pipelines.training import hpo as hpo_module
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    monkeypatch.setattr("tcip_mcp.tools.training_tools.TCIP_HEARTBEAT_STALE_SECONDS", 0.05)
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()

    sentinel = sweep_root / SWEEP_CANCEL_SENTINEL
    stopper, callback = hpo_module._build_sweep_stopper(lambda: sentinel.exists(), sweep_root)
    callback.on_trial_start(0, [], _FakeTrial("stuck_trial"))  # Ray never reports it finished

    sentinel.touch()
    assert stopper.stop_all() is False  # freshly written: still inside the window

    import os
    import time
    stale = time.time() - 10
    os.utime(sentinel, (stale, stale))
    assert stopper.stop_all() is True  # the bounded fallback: Ray's own stop takes over


def test_run_hpo_refuses_a_relaunched_from_naming_no_sweep_manifest_under_this_root(
    tmp_path, real_hpo_base_config
) -> None:
    """relaunched_from must name a sweep this root actually holds a manifest for: a name that
    resolves to nothing is refused before anything is minted, rather than recorded verbatim as
    fabricated lineage on a frozen record."""
    import tcip_store
    from tcip_mcp.tools.training_tools import run_hpo, sweep_manifest_key

    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name="hpo_refused_relaunch1", relaunched_from="hpo_does_not_exist")
    assert "error" in result
    assert "hpo_does_not_exist" in result["error"]
    assert not tcip_store.exists(sweep_manifest_key("hpo_refused_relaunch1", str(tmp_path)))


def test_run_hpo_with_no_relaunched_from_is_unaffected_by_the_new_check(
    tmp_path, real_hpo_base_config, monkeypatch
) -> None:
    """Admits valid work: a direct run_hpo call that never names relaunched_from is not
    touched by the refusal added for the case where it is given."""
    from tcip_mcp.tools.training_tools import run_hpo

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                     "study_name": kw["study_name"]},
    )
    result = run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path),
                     study_name="hpo_norelaunch1")
    assert "error" not in result
