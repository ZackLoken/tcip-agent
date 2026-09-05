"""Tests for the Tuning routes: launching a sweep, reading sweeps/trials off disk, and the
live-monitoring surfaces (Ray's dashboard, per-sweep and per-trial TensorBoards)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _worker_join_bound() -> float:
    """How long a test here waits on a sweep worker.

    A worker's slowest single act is one store write, bounded by the seam's own lock wait.
    """
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    return 2 * DEFAULT_LOCK_TIMEOUT_S


@pytest.fixture(autouse=True)
def _join_sweep_workers():
    """Wait out any sweep a test here launched before that test returns.

    A worker writes through the storage backend the fixtures close on the way out, so a test
    that returns while its worker is still writing leaves the close racing a live statement.
    """
    yield
    from tcip_web.routes import tuning

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()


@pytest.fixture
def hpo_root(tmp_path, monkeypatch) -> Path:
    """Point the platform state root at a tmp dir so the routes read this test's sweeps."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = tmp_path / ".tcip" / "hpo"
    root.mkdir(parents=True)
    return root


_RELAUNCH_FIELD_DEFAULTS = {
    "search_alg": "random", "scheduler": "asha", "grace_period": 5, "reduction_factor": 3,
    "max_concurrent": 1, "warm_start": False, "baseline_params": None, "resources_per_trial": None,
    "param_space": {},
}
"""Every ``run_hyperparameter_search`` argument beside ``n_trials``/``base_config`` a manifest carries, at the
values ``run_hyperparameter_search`` itself defaults to; :func:`_write_sweep` folds these in so a hand-written
test manifest is complete (as a real one always is) unless a test overrides a field, or omits
``base_config`` itself, to exercise a genuinely incomplete one."""


def _write_sweep(root: Path, study: str, **manifest_fields) -> Path:
    import tcip_store
    from datetime import datetime, timezone
    from tcip_mcp.tools.training_tools import sweep_manifest_key

    sweep = root / study
    sweep.mkdir(parents=True, exist_ok=True)
    manifest = {"study_name": study, "status": "running",
                "heartbeat": datetime.now(timezone.utc).isoformat(), "n_trials": 2,
                **_RELAUNCH_FIELD_DEFAULTS, **manifest_fields}
    tcip_store.replace(sweep_manifest_key(study), manifest)
    return sweep


def _write_trial(sweep: Path, trial_id: str, *, metrics: list[dict] | None = None,
                 params: dict | None = None) -> Path:
    import tcip_store
    from tcip_mcp.tools.training_tools import trial_config_key, trial_metrics_key

    trial = sweep / f"trial_{trial_id}"
    trial.mkdir(parents=True, exist_ok=True)
    if params is not None:
        tcip_store.replace(trial_config_key(sweep, trial.name),
                           {"training": {}, "trial_params": params, "unconsumed_params": []})
    if metrics is not None:
        for row in metrics:
            tcip_store.append(trial_metrics_key(sweep, trial.name), row)
    return trial


def test_list_sweeps_empty(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps")
    assert resp.status_code == 200
    body = resp.json()
    assert "sweeps" in body


def test_get_sweep_404(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps/nope")
    assert resp.status_code == 404


def test_relaunch_creates_sweep_then_listed(client: TestClient, hpo_root) -> None:
    _write_sweep(hpo_root, "hpo_seed00001",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 param_space={"training.batch_size": [2, 4]})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00001"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert sweep_id.startswith("hpo-")
    # The background thread will fail fast because the seeded config names no real model; we
    # only assert the registry got the entry.
    listing = client.get("/api/tuning/sweeps").json()
    assert any(s["sweep_id"] == sweep_id for s in listing["sweeps"])


def test_no_sweep_worker_outlives_the_wait_seam(client: TestClient, hpo_root) -> None:
    """A launch is joinable: the module hands a caller a real wait rather than a sleep.

    The sweep itself fails fast on a config that names no model, which is the point: what is
    asserted is that nothing the launch spawned is still writing once the wait returns, and
    that the worker persisted through the key its launch resolved, under this test's own root.
    """
    from tcip_web import jobstore
    from tcip_web.routes import tuning

    _write_sweep(hpo_root, "hpo_seed00002",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 param_space={"training.batch_size": [2, 4]})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00002"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert not any(t.is_alive() for t in tuning._workers.values())
    assert any(s["sweep_id"] == sweep_id for s in jobstore.load(tuning.HPO_REGISTRY))


def test_list_sweeps_finds_a_sweep_that_only_exists_on_disk(client, hpo_root) -> None:
    """An agent-launched sweep never touches this process, so its manifest is the listing."""
    _write_sweep(hpo_root, "hpo_abc12345")

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in sweeps if s["sweep_id"] == "hpo_abc12345")
    assert entry["status"] == "running"
    assert entry["external"] is True


def test_disk_sweep_with_a_stale_heartbeat_lists_interrupted_not_running(client, hpo_root) -> None:
    """A manifest still saying "running" with no process left to restamp its heartbeat is a
    dead sweep, not a live one: the listing (and get_sweep) must say so rather than trusting
    the manifest's own recorded status verbatim."""
    from datetime import datetime, timedelta, timezone

    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS

    stale = (datetime.now(timezone.utc)
             - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS + 60)).isoformat()
    _write_sweep(hpo_root, "hpo_stale0001", heartbeat=stale)

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in sweeps if s["sweep_id"] == "hpo_stale0001")
    assert entry["status"] == "interrupted"

    body = client.get("/api/tuning/sweeps/hpo_stale0001").json()
    assert body["status"] == "interrupted"


def test_disk_sweep_with_a_fresh_heartbeat_lists_running(client, hpo_root) -> None:
    """Admits valid work: a manifest whose heartbeat a live driver just restamped still lists
    (and reads back through get_sweep) as running."""
    _write_sweep(hpo_root, "hpo_fresh0001")  # _write_sweep's own default heartbeat is fresh

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in sweeps if s["sweep_id"] == "hpo_fresh0001")
    assert entry["status"] == "running"

    body = client.get("/api/tuning/sweeps/hpo_fresh0001").json()
    assert body["status"] == "running"


def test_a_job_the_registry_recorded_done_lists_that_state_despite_a_stale_manifest(
    client, hpo_root, monkeypatch
) -> None:
    """A job the registry itself recorded completed (``run_hyperparameter_search`` tolerates its own terminal
    manifest write failing) must not read interrupted off a manifest still saying running with
    a stale heartbeat: the registry's own recorded done state wins over the manifest's."""
    from datetime import datetime, timedelta, timezone

    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS
    from tcip_web.routes import tuning

    stale = (datetime.now(timezone.utc)
             - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS + 60)).isoformat()
    _write_sweep(hpo_root, "hpo_donestale1", heartbeat=stale)  # manifest still says "running"
    monkeypatch.setitem(tuning._registry.jobs, "hpo_donestale1",
                        tuning.HPOJob(sweep_id="hpo_donestale1", status="completed",
                                     result={"best_value": 0.1}))

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in sweeps if s["sweep_id"] == "hpo_donestale1")
    assert entry["status"] == "completed"

    body = client.get("/api/tuning/sweeps/hpo_donestale1").json()
    assert body["status"] == "completed"


def test_a_live_sweep_is_not_listed_twice_by_its_own_manifest(client, hpo_root, monkeypatch) -> None:
    """A sweep id with both a live registry entry and an on-disk manifest is listed once, from
    the live entry (``_summary``, which carries ``platform_root``) rather than the disk-only
    scan (``_manifest_summary``, which does not)."""
    from tcip_web.routes import tuning

    _write_sweep(hpo_root, "hpo_dup00001", status="completed")
    monkeypatch.setitem(tuning._registry.jobs, "hpo_dup00001",
                        tuning.HPOJob(sweep_id="hpo_dup00001", status="running"))

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    matching = [s for s in sweeps if s["sweep_id"] == "hpo_dup00001"]
    assert len(matching) == 1
    assert "platform_root" in matching[0]  # the live row won, not the disk-only one


def test_a_live_sweeps_manifest_is_read_only_once_by_one_listing(
    client, hpo_root, monkeypatch,
) -> None:
    """The disk scan must skip a sweep id a live registry entry already covers: ``_summary``
    already reads that sweep's manifest fresh, so a second read in ``_disk_sweeps`` only to be
    discarded by the live/disk merge is work the listing does not need."""
    from tcip_web.routes import tuning

    _write_sweep(hpo_root, "hpo_dup00002", status="completed")
    monkeypatch.setitem(tuning._registry.jobs, "hpo_dup00002",
                        tuning.HPOJob(sweep_id="hpo_dup00002", status="running"))

    reads: list[str] = []
    original_read_manifest = tuning._read_manifest

    def _counting_read_manifest(sweep_id, **kwargs):
        reads.append(sweep_id)
        return original_read_manifest(sweep_id, **kwargs)

    monkeypatch.setattr(tuning, "_read_manifest", _counting_read_manifest)

    client.get("/api/tuning/sweeps")

    assert reads.count("hpo_dup00002") == 1


def test_get_sweep_reads_the_manifest_when_the_sweep_is_not_in_memory(client, hpo_root) -> None:
    _write_sweep(hpo_root, "hpo_done0001", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})

    body = client.get("/api/tuning/sweeps/hpo_done0001").json()
    assert body["status"] == "completed"
    assert body["result"]["best_params"] == {"lr": 0.01}
    assert body["manifest"]["study_name"] == "hpo_done0001"


def test_has_manifest_is_false_pre_manifest_and_true_for_a_disk_sweep(
    client, hpo_root, monkeypatch
) -> None:
    """A live job with no manifest yet answers ``has_manifest: false`` on both the listing row
    and the detail, the fact a relaunch's pre-manifest window keys on rather than a 404 the
    route never produces; a disk-only sweep (a manifest by definition) always answers true."""
    from tcip_web.routes import tuning

    monkeypatch.setitem(tuning._registry.jobs, "hpo-premanifest1",
                        tuning.HPOJob(sweep_id="hpo-premanifest1", status="running"))

    listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    row = next(s for s in listing if s["sweep_id"] == "hpo-premanifest1")
    assert row["has_manifest"] is False
    detail = client.get("/api/tuning/sweeps/hpo-premanifest1").json()
    assert detail["has_manifest"] is False

    _write_sweep(hpo_root, "hpo_disk0001", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})
    disk_body = client.get("/api/tuning/sweeps/hpo_disk0001").json()
    assert disk_body["has_manifest"] is True
    disk_listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    disk_row = next(s for s in disk_listing if s["sweep_id"] == "hpo_disk0001")
    assert disk_row["has_manifest"] is True


def test_relaunched_from_names_the_parent_before_the_manifest_exists(
    client, hpo_root, monkeypatch
) -> None:
    """The source sweep is known at launch, in the job's own in-memory record, and the
    listing row and the detail must name it in the pre-manifest window rather than only once
    a manifest exists to read it back off."""
    from tcip_web.routes import tuning

    monkeypatch.setitem(tuning._registry.jobs, "hpo-relaunch-pre1",
                        tuning.HPOJob(sweep_id="hpo-relaunch-pre1", status="running",
                                     relaunched_from="hpo_source0001"))

    listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    row = next(s for s in listing if s["sweep_id"] == "hpo-relaunch-pre1")
    assert row["relaunched_from"] == "hpo_source0001"
    detail = client.get("/api/tuning/sweeps/hpo-relaunch-pre1").json()
    assert detail["relaunched_from"] == "hpo_source0001"


def _write_study_result(study: str, **fields) -> None:
    import tcip_store
    from tcip_mcp.tools.training_tools import study_result_key

    tcip_store.replace(study_result_key(study), {"study_name": study, **fields})


def test_get_sweep_carries_the_study_results_own_fields_for_a_completed_disk_sweep(
    client, hpo_root
) -> None:
    """The manifest's own completion projection is only best_params/best_value/n_trials; the
    study result record is the only place all_trials, the searcher and scheduler that ran, and
    warm-start provenance live."""
    _write_sweep(hpo_root, "hpo_study0001", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})
    _write_study_result(
        "hpo_study0001",
        best_params={"lr": 0.01}, best_value=0.2, n_trials=2,
        all_trials=[{"params": {"lr": 0.01}, "value": 0.2, "iterations": 3, "state": "COMPLETE"}],
        search_alg="bayesopt", scheduler="median", warm_start=True,
        baseline_params={"lr": 0.02},
    )

    body = client.get("/api/tuning/sweeps/hpo_study0001").json()
    assert body["result"]["all_trials"][0]["state"] == "COMPLETE"
    assert body["result"]["search_alg"] == "bayesopt"
    assert body["result"]["scheduler"] == "median"
    assert body["result"]["warm_start"] is True
    assert body["result"]["baseline_params"] == {"lr": 0.02}
    assert body["result"]["best_params"] == {"lr": 0.01}  # the manifest's own fields still serve


def test_get_sweep_serves_a_never_reported_error_row_with_its_error_text_untouched(
    client, hpo_root
) -> None:
    """The route layers ``all_trials`` on unmodified; a never-reported trial's row (no params,
    no value, an error string) has to reach the client exactly as the study result recorded it,
    the route holding no per-row knowledge of what a trial's state means."""
    _write_sweep(hpo_root, "hpo_error0001", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})
    dead_row = {"params": None, "value": None, "iterations": None, "state": "ERROR",
                "error": "the trial never answered Ray: its actor died during start"}
    _write_study_result(
        "hpo_error0001",
        best_params={"lr": 0.01}, best_value=0.2, n_trials=1, all_trials=[dead_row],
        search_alg="random", scheduler="none", warm_start=False, baseline_params=None,
    )

    body = client.get("/api/tuning/sweeps/hpo_error0001").json()
    assert body["result"]["all_trials"] == [dead_row]


def test_get_sweep_serves_the_manifest_alone_when_the_study_result_is_absent(
    client, hpo_root
) -> None:
    """An old or failed sweep never gets a study result record; the manifest's own result still
    answers rather than the route failing or fabricating the missing fields."""
    _write_sweep(hpo_root, "hpo_nostudy1", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})

    body = client.get("/api/tuning/sweeps/hpo_nostudy1").json()
    assert body["result"] == {"best_params": {"lr": 0.01}, "best_value": 0.2}
    assert "all_trials" not in body["result"]


def test_get_sweep_serves_the_manifest_result_for_a_rehydrated_completed_sweep(
    client, hpo_root
) -> None:
    """The live registry a restart rehydrates carries status only, never a trial result
    (:func:`tuning.rehydrate_for_current_root` never sets one); the disk manifest is what
    survived the restart, and the live branch must fall back to it rather than serving the
    still-registered job's empty result."""
    from tcip_web.routes import tuning

    _write_sweep(hpo_root, "hpo_rehydr01", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": None,
                         "best_value_state": "nan", "n_trials": 2})

    job = tuning.HPOJob(sweep_id="hpo_rehydr01", status="completed")
    with tuning._lock:
        tuning._registry.jobs[job.sweep_id] = job
    tuning._persist()
    tuning._registry.jobs.clear()
    tuning.rehydrate_for_current_root()

    try:
        # the rehydrated entry itself is bare
        assert tuning._registry.jobs["hpo_rehydr01"].result == {}
        body = client.get("/api/tuning/sweeps/hpo_rehydr01").json()
        assert body["result"]["best_params"] == {"lr": 0.01}
        assert body["result"]["best_value_state"] == "nan"
    finally:
        tuning._registry.jobs.clear()


def test_list_trials_reports_platform_trial_dirs_only(client, hpo_root) -> None:
    sweep = _write_sweep(hpo_root, "hpo_trials01")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}], params={"lr": 0.01})
    _write_trial(sweep, "bbb_00001", params={"lr": 0.5})
    (sweep / "trainable_ccc_00002_0_lr=0.1_2026-01-01_00-00-00").mkdir()

    trials = client.get("/api/tuning/sweeps/hpo_trials01/trials").json()["trials"]
    by_id = {t["trial_id"]: t for t in trials}
    assert set(by_id) == {"aaa_00000", "bbb_00001"}  # Ray's own trial dir is not one of ours
    assert by_id["aaa_00000"]["has_metrics"] is True
    assert by_id["aaa_00000"]["params"] == {"lr": 0.01}
    assert by_id["bbb_00001"]["has_metrics"] is False


def test_get_trial_metrics_returns_every_row(client, hpo_root) -> None:
    sweep = _write_sweep(hpo_root, "hpo_metrics1")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1, "val_loss": 0.5},
                                              {"epoch": 2, "val_loss": 0.4}])

    body = client.get("/api/tuning/sweeps/hpo_metrics1/trials/aaa_00000/metrics").json()
    assert body["exists"] is True
    assert [row["val_loss"] for row in body["metrics"]] == [0.5, 0.4]


def test_get_trial_metrics_reports_version_refused_rows_separately_from_corrupt(
    client, hpo_root
) -> None:
    """A row at a schema_version this reader does not accept is excluded from the served
    rows, the same as a corrupt one, but the endpoint still reports the trial as having
    metrics: the run wrote something, this reader just cannot show it back."""
    import tcip_store
    from tcip_store.file_backend import FileBackend
    from tcip_mcp.tools.training_tools import trial_metrics_key

    tcip_store.bind(FileBackend())
    sweep = _write_sweep(hpo_root, "hpo_metrics_version_refused")
    trial = sweep / "trial_aaa_00000"
    trial.mkdir(parents=True)
    key = trial_metrics_key(sweep, trial.name)
    tcip_store.append(key, {"epoch": 1, "val_loss": 0.5})
    poisoned = tcip_store.get_descriptor(key.store).codec.encode(
        {"epoch": 2, "schema_version": 99})
    with open(FileBackend().path_for(key), "ab") as handle:
        handle.write(poisoned + b"\n")

    body = client.get(
        "/api/tuning/sweeps/hpo_metrics_version_refused/trials/aaa_00000/metrics"
    ).json()
    assert body["exists"] is True
    assert [row["epoch"] for row in body["metrics"]] == [1]


def test_list_trials_has_metrics_true_for_a_trial_holding_only_a_version_refused_row(
    client, hpo_root
) -> None:
    import tcip_store
    from tcip_store.file_backend import FileBackend
    from tcip_mcp.tools.training_tools import trial_metrics_key

    tcip_store.bind(FileBackend())
    sweep = _write_sweep(hpo_root, "hpo_trials_version_refused")
    trial = sweep / "trial_aaa_00000"
    trial.mkdir(parents=True)
    key = trial_metrics_key(sweep, trial.name)
    poisoned = tcip_store.get_descriptor(key.store).codec.encode(
        {"epoch": 1, "schema_version": 99})
    FileBackend().path_for(key).parent.mkdir(parents=True, exist_ok=True)
    with open(FileBackend().path_for(key), "ab") as handle:
        handle.write(poisoned + b"\n")

    trials = client.get("/api/tuning/sweeps/hpo_trials_version_refused/trials").json()["trials"]
    by_id = {t["trial_id"]: t for t in trials}
    assert by_id["aaa_00000"]["has_metrics"] is True


def test_trials_of_an_unknown_sweep_are_a_404(client, hpo_root) -> None:
    assert client.get("/api/tuning/sweeps/hpo_missing/trials").status_code == 404


def test_a_trial_id_cannot_walk_out_of_its_sweep(client, hpo_root) -> None:
    """A trial id arrives as a path segment, so a name carrying a separator is refused rather
    than resolved into a log somewhere else. The ordinary id is still served."""
    sweep = _write_sweep(hpo_root, "hpo_walk0001")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1, "val_loss": 0.5}])
    outside = hpo_root.parent / "elsewhere.jsonl"
    outside.write_text(json.dumps({"epoch": 99}) + "\n", encoding="utf-8")

    from fastapi import HTTPException

    from tcip_web.routes import tuning

    with pytest.raises(HTTPException) as exc:
        tuning.get_trial_metrics("hpo_walk0001", "../../elsewhere")
    assert exc.value.status_code == 400

    served = client.get("/api/tuning/sweeps/hpo_walk0001/trials/aaa_00000/metrics").json()
    assert served["exists"] is True
    assert [row["epoch"] for row in served["metrics"]] == [1]


def test_a_sweep_id_cannot_walk_out_of_the_hpo_root(hpo_root) -> None:
    from fastapi import HTTPException

    from tcip_web.routes import tuning

    with pytest.raises(HTTPException) as exc:
        tuning._read_manifest("../../elsewhere")
    assert exc.value.status_code == 400


@pytest.fixture
def tb_launches(monkeypatch) -> list[tuple[str, str]]:
    """Record what the routes hand ``launch_tensorboard`` instead of starting a real one."""
    calls: list[tuple[str, str]] = []

    def fake_launch(logdir: str, run_id: str | None = None) -> dict:
        calls.append((logdir, run_id or ""))
        return {"url": "http://localhost:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )
    return calls


def test_ray_dashboard_is_null_when_no_cluster_is_up(client, hpo_root) -> None:
    assert client.get("/api/tuning/ray-dashboard").json() == {"url": None}


def test_ray_dashboard_serves_the_url_the_process_that_started_ray_wrote(
    client, hpo_root, tmp_path
) -> None:
    """The sweep's own process records the URL; this one only reads it back."""
    from tcip_store import store

    from tcip_mcp.pipelines.training.hpo import ray_dashboard_key

    store.replace(ray_dashboard_key(),
                  {"url": "http://127.0.0.1:8265", "pid": os.getpid(),
                   "started_at": "2026-01-01T00:00:00+00:00"})

    assert client.get("/api/tuning/ray-dashboard").json() == {"url": "http://127.0.0.1:8265"}


def test_sweep_tensorboard_launches_over_a_clean_named_trial_view(
    client, hpo_root, tb_launches
) -> None:
    """Not the sweep directory itself: TensorBoard would read each trial's run name off its
    nested ``trial_<id>/tensorboard`` leaf otherwise, showing ``trial_<id>\\tensorboard`` in its
    own run picker instead of ``trial_<id>``."""
    sweep = _write_sweep(hpo_root, "hpo_tbsweep1")
    trial = _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}])
    (trial / "tensorboard").mkdir()
    (trial / "tensorboard" / "marker.txt").write_text("x", encoding="utf-8")

    resp = client.post("/api/tuning/sweeps/hpo_tbsweep1/tensorboard", json={})
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    (logdir, key), = tb_launches
    view = Path(logdir)
    assert view != sweep.resolve()
    assert key == "sweep_hpo_tbsweep1"

    link = view / "trial_aaa_00000"
    assert link.is_dir()
    assert (link / "marker.txt").is_file()  # resolves through to the real trial's own dir


def test_sweep_tensorboard_view_gains_a_link_for_a_trial_that_appears_later(
    client, hpo_root, tb_launches
) -> None:
    """A sweep still running writes new trial dirs after the first launch call; the view has
    to pick each one up rather than being built once and left stale."""
    sweep = _write_sweep(hpo_root, "hpo_tbsweep2")
    trial_a = _write_trial(sweep, "aaa_00000")
    (trial_a / "tensorboard").mkdir()

    resp = client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard", json={})
    view = Path(resp.json()["logdir"])
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000"]

    trial_b = _write_trial(sweep, "bbb_00001")
    (trial_b / "tensorboard").mkdir()
    client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard", json={})
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000", "trial_bbb_00001"]


def test_trial_tensorboard_launches_over_that_trial_s_own_logdir(
    client, hpo_root, tb_launches
) -> None:
    sweep = _write_sweep(hpo_root, "hpo_tbtrial1")
    trial = _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}])

    resp = client.post("/api/tuning/sweeps/hpo_tbtrial1/trials/aaa_00000/tensorboard", json={})
    assert resp.status_code == 200
    (logdir, key), = tb_launches
    assert Path(logdir) == (trial / "tensorboard").resolve()
    assert key == "sweep_hpo_tbtrial1_trial_aaa_00000"


def test_stopping_a_trial_tensorboard_uses_that_trial_s_own_key(
    client, hpo_root, monkeypatch
) -> None:
    """Trials share a bounded port range, so each one's TensorBoard must be stoppable alone."""
    stopped: list[str] = []

    def fake_stop(run_id: str | None = None, logdir: str | None = None) -> dict:
        stopped.append(run_id or "")
        return {"status": "stopped", "pid": 7}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.stop_tensorboard", fake_stop
    )
    _write_trial(_write_sweep(hpo_root, "hpo_tbstop01"), "aaa_00000")

    resp = client.post("/api/tuning/sweeps/hpo_tbstop01/trials/aaa_00000/tensorboard/stop", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert stopped == ["sweep_hpo_tbstop01_trial_aaa_00000"]


def test_tensorboard_for_an_unknown_sweep_is_a_404(client, hpo_root, tb_launches) -> None:
    for url in ("/api/tuning/sweeps/hpo_missing/tensorboard",
                "/api/tuning/sweeps/hpo_missing/trials/aaa_00000/tensorboard"):
        resp = client.post(url, json={})
        assert resp.status_code == 404
        assert "hpo_missing" in resp.json()["detail"]


def test_a_launched_sweep_stays_reachable_after_the_backend_repins(
    client, hpo_root, tmp_path, monkeypatch, tb_launches
) -> None:
    """A sweep this process launched is addressed by the root it launched under: its detail
    and its TensorBoard route keep answering after this process repins to another project,
    rather than looking for its manifest under wherever the platform root has since moved,
    and the TensorBoard link farm itself lands beside the sweep's own trials, not inside
    whatever project this process has since adopted."""
    from tcip_mcp import workspace
    from tcip_web.routes import tuning

    launch_root = tmp_path

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        sweep = hpo_root / study_name
        tb_dir = sweep / "trial_aaa_00000" / "tensorboard"
        tb_dir.mkdir(parents=True)
        (tb_dir / "marker.txt").write_text("x", encoding="utf-8")
        tcip_store.replace(
            sweep_manifest_key(study_name),
            {"study_name": study_name, "status": "completed", "n_trials": 1},
        )
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_seed00003",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 param_space={"training.batch_size": [2, 4]})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00003"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert tuning._registry.jobs[sweep_id].platform_root == str(launch_root)

    other_proj = workspace.project_path("chestnut_burr_other")
    (other_proj / ".tcip").mkdir(parents=True)
    workspace.activate_project("chestnut_burr_other")

    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert body["sweep_id"] == sweep_id
    assert body["status"] == "completed"

    tb_resp = client.post(f"/api/tuning/sweeps/{sweep_id}/tensorboard", json={})
    assert tb_resp.status_code == 200
    (logdir, _), = tb_launches
    expected_view = launch_root / ".tcip" / "state" / "tensorboard_views" / sweep_id
    assert Path(logdir) == expected_view  # the link farm sits under the launch root, not chestnut_burr_other
    link = Path(logdir) / "trial_aaa_00000"
    assert link.is_dir()
    assert (link / "marker.txt").is_file()


def test_a_web_launched_sweep_runs_only_the_routes_own_tensorboard(
    client, hpo_root, real_hpo_base_config, monkeypatch, tb_launches
) -> None:
    """The real run_hyperparameter_search, driven through the launch route, must not auto-launch a TensorBoard
    of its own: the route serves its own per-sweep view on demand (the sweep-detail call
    below), so a second, unaddressable process over the same trials would be pure waste."""
    from tcip_web.routes import tuning

    def fake_search(**kw):
        study_name = kw["study_name"]
        sweep = Path(kw["storage_path"]) / study_name
        trial = sweep / "trial_aaa_00000"
        (trial / "tensorboard").mkdir(parents=True)
        (trial / "tensorboard" / "marker.txt").write_text("x", encoding="utf-8")
        return {
            "best_params": {}, "best_value": 0.0, "n_trials": 1,
            "study_name": study_name, "tensorboard_logdir": str(sweep),
        }

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    _write_sweep(hpo_root, "hpo_seed00004", base_config=real_hpo_base_config,
                 param_space={"training.batch_size": [2, 4]})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00004"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert tb_launches == []  # run_hyperparameter_search's own auto-launch stayed off for this caller

    tb_resp = client.post(f"/api/tuning/sweeps/{sweep_id}/tensorboard", json={})
    assert tb_resp.status_code == 200

    assert len(tb_launches) == 1
    (_, run_id), = tb_launches
    assert run_id == f"sweep_{sweep_id}"


def test_launch_route_no_longer_exists(client: TestClient, hpo_root) -> None:
    """The raw launch door retired with the config picker: a sweep starts only from a
    recorded manifest, through ``/api/tuning/sweeps``, never from a client-submitted
    base_config/param_space again."""
    resp = client.post(
        "/api/tuning/launch",
        json={"base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
              "param_space": {}, "n_trials": 1, "output_dir": "",
              "search_alg": "random", "scheduler": "asha"},
    )
    assert resp.status_code == 404


def test_relaunch_route_404s_for_an_unknown_sweep(client: TestClient, hpo_root) -> None:
    resp = client.post("/api/tuning/sweeps", json={"study_name": "nope"})
    assert resp.status_code == 404


def test_relaunch_route_409s_when_the_manifest_holds_no_base_config(client: TestClient, hpo_root) -> None:
    _write_sweep(hpo_root, "hpo_nobase01")  # a manifest without base_config
    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_nobase01"})
    assert resp.status_code == 409


def test_relaunch_route_409s_naming_a_field_missing_from_the_manifest_rather_than_defaulting_it(
    client: TestClient, hpo_root
) -> None:
    """A manifest carrying base_config but missing another run_hyperparameter_search argument (an old manifest
    from before this field existed) refuses by name, rather than silently substituting a
    default that was never the sweep's own choice."""
    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key

    manifest = {
        "study_name": "hpo_partial001", "status": "running", "n_trials": 2,
        "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
        "param_space": {}, "search_alg": "random", "grace_period": 5, "reduction_factor": 3,
        "max_concurrent": 1, "warm_start": False, "baseline_params": None,
        "resources_per_trial": None,
        # scheduler deliberately omitted: an old manifest from before run_hyperparameter_search recorded it.
    }
    (hpo_root / "hpo_partial001").mkdir(parents=True)
    tcip_store.replace(sweep_manifest_key("hpo_partial001"), manifest)

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_partial001"})
    assert resp.status_code == 409
    assert "scheduler" in resp.json()["detail"]


def test_relaunch_reads_the_source_manifest_under_the_sweeps_own_launch_root(
    client, hpo_root, tmp_path, monkeypatch
) -> None:
    """A relaunch resolves the source manifest under the sweep's own launch root
    (``_sweep_launch_root``), the same resolution the cancel route uses, so a live sweep stays
    relaunchable after this process repins away from the root it launched under."""
    from tcip_mcp import workspace
    from tcip_web.routes import tuning

    launch_root = tmp_path

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        tcip_store.replace(
            sweep_manifest_key(study_name),
            {"study_name": study_name, "status": "completed", "n_trials": 1,
             **_RELAUNCH_FIELD_DEFAULTS,
             "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}},
        )
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_launchroot1",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_launchroot1"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert tuning._registry.jobs[sweep_id].platform_root == str(launch_root)

    other_proj = workspace.project_path("chestnut_burr_other_relaunch")
    (other_proj / ".tcip").mkdir(parents=True)
    workspace.activate_project("chestnut_burr_other_relaunch")

    resp2 = client.post("/api/tuning/sweeps", json={"study_name": sweep_id})
    assert resp2.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()


def test_relaunch_ignores_any_path_the_manifest_itself_carries(
    client: TestClient, hpo_root, monkeypatch, tmp_path
) -> None:
    """``output_dir`` is always this request thread's own ``hpo_root()``, never a path read
    from the manifest: an absolute path in a file is not a path this process should follow."""
    from tcip_web.routes import tuning

    captured: dict = {}

    def fake_run_hyperparameter_search(*, output_dir, **kwargs):
        captured["output_dir"] = output_dir
        return {"study_name": kwargs["study_name"]}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    elsewhere = tmp_path / "elsewhere"
    _write_sweep(hpo_root, "hpo_path0001",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 sweep_dir=str(elsewhere))

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_path0001"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert captured["output_dir"] == str(hpo_root)


def test_relaunch_replays_every_manifest_field_run_hyperparameter_search_was_given(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """A relaunch reads every run_hyperparameter_search argument the manifest holds, not only base_config, so a
    sweep started with a non-default search shape replays that shape exactly."""
    from tcip_web.routes import tuning

    captured: dict = {}

    def fake_run_hyperparameter_search(**kwargs):
        captured.update(kwargs)
        return {"study_name": kwargs["study_name"]}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    base_config = {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}
    _write_sweep(hpo_root, "hpo_fields001", base_config=base_config,
                 param_space={"lr": {"type": "loguniform", "low": 1e-6, "high": 1e-1}},
                 n_trials=7, search_alg="bayesopt", scheduler="median",
                 grace_period=3, reduction_factor=4, max_concurrent=2,
                 warm_start=True, baseline_params={"lr": 0.05},
                 resources_per_trial={"cpu": 2.0, "gpu": 0.5})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_fields001"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    assert captured["base_config"] == base_config
    assert captured["param_space"] == {"lr": {"type": "loguniform", "low": 1e-6, "high": 1e-1}}
    assert captured["n_trials"] == 7
    assert captured["search_alg"] == "bayesopt"
    assert captured["scheduler"] == "median"
    assert captured["grace_period"] == 3
    assert captured["reduction_factor"] == 4
    assert captured["max_concurrent"] == 2
    assert captured["warm_start"] is True
    assert captured["baseline_params"] == {"lr": 0.05}
    assert captured["resources_per_trial"] == {"cpu": 2.0, "gpu": 0.5}
    assert captured["auto_tensorboard"] is False
    assert captured["relaunched_from"] == "hpo_fields001"


def test_relaunch_of_a_manifest_predating_split_draws_still_relaunches(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """A manifest without the field carries neither split_draws nor split_draw_seeds; the
    relaunch route reads them as run_hyperparameter_search's own defaults (1, None) rather than refusing."""
    from tcip_web.routes import tuning

    captured: dict = {}

    def fake_run_hyperparameter_search(**kwargs):
        captured.update(kwargs)
        return {"study_name": kwargs["study_name"]}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    base_config = {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}
    _write_sweep(hpo_root, "hpo_predraws01", base_config=base_config)

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key

    manifest = tcip_store.read(sweep_manifest_key("hpo_predraws01", str(hpo_root)))
    assert "split_draws" not in manifest and "split_draw_seeds" not in manifest

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_predraws01"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    assert captured["split_draws"] == 1
    assert captured["split_draw_seeds"] is None


def test_relaunch_passes_through_a_manifests_own_split_draws(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    from tcip_web.routes import tuning

    captured: dict = {}

    def fake_run_hyperparameter_search(**kwargs):
        captured.update(kwargs)
        return {"study_name": kwargs["study_name"]}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    base_config = {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}
    _write_sweep(hpo_root, "hpo_draws001", base_config=base_config,
                 split_draws=3, split_draw_seeds=[1, 2, 3])

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_draws001"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    assert captured["split_draws"] == 3
    assert captured["split_draw_seeds"] == [1, 2, 3]


def test_relaunch_records_the_source_study_as_relaunched_from_on_the_new_manifest(
    client: TestClient, hpo_root, real_hpo_base_config, monkeypatch
) -> None:
    """A relaunch through the route, driving the platform's own run_hyperparameter_search (only the Ray Tune
    search itself faked, run_hyperparameter_search unstubbed), leaves a manifest whose relaunched_from names the
    sweep it replayed."""
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search",
        lambda **kw: {"best_params": {}, "best_value": 0.1, "n_trials": 1,
                     "study_name": kw["study_name"]},
    )
    _write_sweep(hpo_root, "hpo_relsrc001", base_config=real_hpo_base_config,
                param_space={"training.batch_size": [2, 4]})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_relsrc001"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]

    from tcip_web.routes import tuning

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key

    manifest = tcip_store.read(sweep_manifest_key(sweep_id))
    assert manifest["relaunched_from"] == "hpo_relsrc001"


def test_relaunch_records_the_source_manifests_own_study_name_not_the_requests(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """relaunched_from is the source manifest's own recorded study_name, never the request
    body's echoed string: proven by handing back a manifest whose study_name differs from what
    was requested, the shape a case-differing lookup on the file backend produces."""
    from tcip_web.routes import tuning

    captured: dict = {}

    def fake_run_hyperparameter_search(**kwargs):
        captured.update(kwargs)
        return {"study_name": kwargs["study_name"]}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)

    base_config = {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}
    source_manifest = {
        "study_name": "hpo_source_actual1", "status": "completed", "n_trials": 1,
        "base_config": base_config, **_RELAUNCH_FIELD_DEFAULTS,
    }
    monkeypatch.setattr(tuning, "_read_manifest", lambda sweep_id, root=None: source_manifest)

    resp = client.post("/api/tuning/sweeps", json={"study_name": "HPO_SOURCE_ACTUAL1"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    assert captured["relaunched_from"] == "hpo_source_actual1"


def test_manifest_fields_projects_relaunched_from_none_when_the_manifest_lacks_the_key() -> None:
    """An older manifest, from before this field existed, projects relaunched_from as None the
    same as a manifest that genuinely was not a relaunch, rather than being treated as missing
    something a caller must supply."""
    from tcip_web.routes.tuning import _manifest_fields

    manifest = {"study_name": "hpo_old1", "status": "completed", "base_config": {}}
    assert "relaunched_from" not in manifest
    assert _manifest_fields(manifest)["relaunched_from"] is None


def test_manifest_fields_projects_redraws_within_manifest_from_the_base_config() -> None:
    """The recorded base_config's own data.split.redraw_within_manifest is the source of truth
    for the sweep listing's redraws_within_manifest field: false for a manifest predating the
    flag or one that never set it, true only when the base config actually carries it."""
    from tcip_web.routes.tuning import _manifest_fields

    without_flag = {"study_name": "hpo_old2", "base_config": {
        "data": {"split": {"manifest_dir": "m"}},
    }}
    assert _manifest_fields(without_flag)["redraws_within_manifest"] is False

    with_flag = {"study_name": "hpo_redraw2", "base_config": {
        "data": {"split": {"manifest_dir": "m", "redraw_within_manifest": True, "seed": 5}},
    }}
    assert _manifest_fields(with_flag)["redraws_within_manifest"] is True


def test_relaunch_of_an_older_manifest_missing_relaunched_from_still_succeeds(
    client: TestClient, hpo_root
) -> None:
    """_missing_relaunch_fields does not require relaunched_from: a manifest from before the
    field existed (_write_sweep's own default, the same shape a real pre-family manifest has)
    relaunches exactly as any other manifest does."""
    from tcip_web.routes.tuning import _RELAUNCH_FIELDS, _missing_relaunch_fields

    assert "relaunched_from" not in _RELAUNCH_FIELDS
    _write_sweep(hpo_root, "hpo_old_relaunch1",
                base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    import tcip_store
    from tcip_mcp.tools.training_tools import sweep_manifest_key

    manifest = tcip_store.read(sweep_manifest_key("hpo_old_relaunch1"))
    assert "relaunched_from" not in manifest
    assert _missing_relaunch_fields(manifest) == []

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_old_relaunch1"})
    assert resp.status_code == 200

    from tcip_web.routes import tuning

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()


def test_cancel_route_404s_for_an_unknown_sweep(client: TestClient, hpo_root) -> None:
    resp = client.post("/api/tuning/sweeps/nope/cancel", json={})
    assert resp.status_code == 404


def test_cancel_route_writes_the_sweep_and_run_level_sentinels(client: TestClient, hpo_root) -> None:
    from tcip_mcp.pipelines.training.run_registry import CANCEL_SENTINEL
    from tcip_mcp.tools.training_tools import SWEEP_CANCEL_SENTINEL

    sweep = _write_sweep(hpo_root, "hpo_cancel01",
                         base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})
    trial = _write_trial(sweep, "aaa_00000")  # no resolved config written: still "running"

    resp = client.post("/api/tuning/sweeps/hpo_cancel01/cancel", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cancel_requested"] is True
    assert (sweep / SWEEP_CANCEL_SENTINEL).exists()
    assert (trial / CANCEL_SENTINEL).exists()

    listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in listing if s["sweep_id"] == "hpo_cancel01")
    assert entry["cancel_requested"] is True


def test_cancel_reaches_a_relaunch_before_run_hyperparameter_search_writes_its_own_manifest(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """The relaunch route marks the new sweep id as launching, on the request thread, before
    starting the worker: a cancel that arrives once the route has answered but before run_hyperparameter_search's
    own worker call has written a manifest still reaches the sweep, rather than the 404
    ``cancel_hyperparameter_search`` used to answer in that window."""
    import threading

    from tcip_web.routes import tuning

    release = threading.Event()

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        release.wait(timeout=5)
        return {"status": "cancelled", "study_name": study_name, "error": "stood down for the test"}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_precancel1",
                base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_precancel1"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]

    cancel_resp = client.post(f"/api/tuning/sweeps/{sweep_id}/cancel", json={})
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancel_requested"] is True

    release.set()
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()


def test_worker_marks_an_error_dict_failed_not_completed(hpo_root, monkeypatch) -> None:
    """A relaunch whose data paths moved must read failed with preflight's own words, not
    completed with no useful result (the defect the relaunch door's own worker fixes)."""
    from tcip_web.routes.tuning import HPOJob, _RelaunchSpec, _worker

    monkeypatch.setattr(
        "tcip_mcp.tools.training_tools.run_hyperparameter_search",
        lambda **kwargs: {"error": "the sweep's base config fails preflight", "issues": ["bad"]},
    )
    job = HPOJob(sweep_id="hpo-worker-err")
    spec = _RelaunchSpec(base_config={}, param_space=None, n_trials=1, search_alg="random",
                         scheduler="asha", grace_period=5, reduction_factor=3, max_concurrent=1,
                         warm_start=False, baseline_params=None, resources_per_trial=None)
    _worker(job, spec, str(hpo_root), "hpo-source-1")
    assert job.status == "failed"
    assert job.error == "the sweep's base config fails preflight"


def test_worker_marks_a_cancelled_result_cancelled_with_its_reason(hpo_root, monkeypatch) -> None:
    """A cancelled sweep's row must carry a reason a breeder can read, the same way a failed
    sweep's row does, not job.error left unset while the disk manifest alone holds it."""
    from tcip_web.routes.tuning import HPOJob, _RelaunchSpec, _worker

    monkeypatch.setattr(
        "tcip_mcp.tools.training_tools.run_hyperparameter_search",
        lambda **kwargs: {"status": "cancelled", "study_name": kwargs.get("study_name"),
                          "error": "the sweep was cancelled by request before it could finish"},
    )
    job = HPOJob(sweep_id="hpo-worker-cxl")
    spec = _RelaunchSpec(base_config={}, param_space=None, n_trials=1, search_alg="random",
                         scheduler="asha", grace_period=5, reduction_factor=3, max_concurrent=1,
                         warm_start=False, baseline_params=None, resources_per_trial=None)
    _worker(job, spec, str(hpo_root), "hpo-source-2")
    assert job.status == "cancelled"
    assert job.error == "the sweep was cancelled by request before it could finish"


def test_worker_discards_the_launch_mark_when_it_fails_before_reaching_run_hyperparameter_search(
    hpo_root, monkeypatch
) -> None:
    """A worker that fails before it ever calls ``run_hyperparameter_search`` must still discard the launch mark
    the route recorded for it, so ``cancel_hyperparameter_search`` does not keep finding a study that will never
    exist."""
    from tcip_mcp.tools.training_tools import _LAUNCHING_SWEEPS, mark_sweep_launching
    from tcip_web.routes.tuning import HPOJob, _RelaunchSpec, _persist, _worker

    calls = {"n": 0}
    real_persist = _persist

    def flaky_persist():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the registry write failed before the worker reached run_hyperparameter_search")
        real_persist()

    monkeypatch.setattr("tcip_web.routes.tuning._persist", flaky_persist)

    job = HPOJob(sweep_id="hpo-worker-markleak1")
    mark_sweep_launching(job.sweep_id, str(hpo_root))
    spec = _RelaunchSpec(base_config={}, param_space=None, n_trials=1, search_alg="random",
                         scheduler="asha", grace_period=5, reduction_factor=3, max_concurrent=1,
                         warm_start=False, baseline_params=None, resources_per_trial=None)

    _worker(job, spec, str(hpo_root), "hpo-source-markleak")

    assert job.sweep_id not in _LAUNCHING_SWEEPS


def test_relaunch_refused_at_preflight_reads_failed_not_interrupted(
    client: TestClient, hpo_root
) -> None:
    """A relaunch run_hyperparameter_search refuses at preflight (an unimportable builder) never mints a
    manifest; the registry's own record is the only account of it, so both the listing row
    and the sweep detail must read failed with the refusal's own words, not the interrupted an
    empty, heartbeat-less manifest would otherwise derive to."""
    _write_sweep(hpo_root, "hpo_refused001",
                base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_refused001"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]

    from tcip_web.routes import tuning

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in listing if s["sweep_id"] == sweep_id)
    assert entry["status"] == "failed"
    assert entry["error"]

    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert body["status"] == "failed"
    assert body["error"]


def test_manifest_fields_agree_between_a_live_and_a_disk_row(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """A sweep this process is running (its row built by ``_summary``) and one only found on
    disk (built by ``_manifest_summary``) report the identical projection for the identical
    manifest shape."""
    from tcip_web.routes import tuning

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        manifest = {"study_name": study_name, "status": "completed", "n_trials": 3,
                    "search_alg": "random", "scheduler": "asha",
                    "param_space": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
                    "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}}
        tcip_store.replace(sweep_manifest_key(study_name), manifest)
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_agree001", n_trials=3,
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 search_alg="random", scheduler="asha",
                 param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_agree001"})
    assert resp.status_code == 200
    live_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    listing = client.get("/api/tuning/sweeps").json()["sweeps"]
    by_id = {s["sweep_id"]: s for s in listing}
    live_row, disk_row = by_id[live_id], by_id["hpo_agree001"]
    fields = ("n_trials", "search_alg", "scheduler", "param_space_keys", "relaunchable", "reason")
    assert {k: live_row[k] for k in fields} == {k: disk_row[k] for k in fields}


def test_manifest_fields_of_an_absent_manifest_is_not_relaunchable_with_no_reason() -> None:
    """A caller with no manifest at all (the pre-manifest window, or a refused relaunch that
    never minted one) reads as not relaunchable but with no reason: the reason is reserved for
    a manifest that actually exists and actually lacks base_config."""
    from tcip_web.routes.tuning import _manifest_fields

    fields = _manifest_fields({})
    assert fields["relaunchable"] is False
    assert fields["reason"] is None
    assert fields["cancel_requested"] is False


def test_persisted_summary_carries_no_manifest_field(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """The registry's own persisted document (what ``JobRegistry.persist`` writes under
    ``.tcip/state/hpo_sweeps.json``) carries only the pre-existing fields: a frozen record must
    not gain new fields over time, and the manifest projection belongs to a listing row only."""
    from tcip_web import jobstore
    from tcip_web.routes import tuning

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        tcip_store.replace(
            sweep_manifest_key(study_name),
            {"study_name": study_name, "status": "completed", "n_trials": 1,
             **_RELAUNCH_FIELD_DEFAULTS,
             "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}},
        )
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_persist001",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_persist001"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    persisted = jobstore.load(tuning.HPO_REGISTRY)
    entry = next(s for s in persisted if s["sweep_id"] == sweep_id)
    assert set(entry) == {"sweep_id", "status", "error", "has_result", "platform_root"}


def test_get_sweep_live_branch_carries_no_manifest_field(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """No frontend code reads a manifest field on the live ``get_sweep`` response; it must not
    be added to the response body."""
    from tcip_web.routes import tuning

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        tcip_store.replace(
            sweep_manifest_key(study_name),
            {"study_name": study_name, "status": "completed", "n_trials": 1,
             **_RELAUNCH_FIELD_DEFAULTS,
             "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}}},
        )
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_nomanifest1",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_nomanifest1"})
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert "manifest" not in body


def test_get_sweep_live_branch_exposes_relaunched_from_top_level(
    client: TestClient, hpo_root, monkeypatch
) -> None:
    """The live get_sweep branch projects relaunched_from as a top-level field, the same as
    the listing row, without adding a manifest field to the body (see
    test_get_sweep_live_branch_carries_no_manifest_field)."""
    from tcip_web.routes import tuning

    def fake_run_hyperparameter_search(*, study_name, **kwargs):
        import tcip_store
        from tcip_mcp.tools.training_tools import sweep_manifest_key

        tcip_store.replace(
            sweep_manifest_key(study_name),
            {"study_name": study_name, "status": "completed", "n_trials": 1,
             **_RELAUNCH_FIELD_DEFAULTS,
             "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
             "relaunched_from": "hpo_relsrc_top1"},
        )
        return {"study_name": study_name}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.run_hyperparameter_search", fake_run_hyperparameter_search)
    _write_sweep(hpo_root, "hpo_relsrc_top1",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}})

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_relsrc_top1"})
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert body["relaunched_from"] == "hpo_relsrc_top1"
    assert "manifest" not in body


def test_get_sweep_disk_branch_exposes_relaunched_from_top_level(client, hpo_root) -> None:
    """The disk get_sweep branch projects relaunched_from at the top level too, beside the
    raw manifest that already carries it nested."""
    _write_sweep(hpo_root, "hpo_disk_relsrc1",
                 base_config={"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
                 relaunched_from="hpo_disk_source0")

    body = client.get("/api/tuning/sweeps/hpo_disk_relsrc1").json()
    assert body["relaunched_from"] == "hpo_disk_source0"
    assert body["manifest"]["relaunched_from"] == "hpo_disk_source0"
