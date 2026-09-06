"""Backend coverage for side-by-side run comparison: compare_experiments' new columns
(task/subject/status_error/split/registry), a no-longer-fabricated model builder, an honest
same_dataset_fingerprint over an error column, the experiment_ids filter on best-model ranking,
TrainRun's own experiment_id/experiment_error fields, and the /api/training/compare/best route.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


# ── same_dataset_fingerprint: an error column must never be silently dropped ────────────────


def test_same_dataset_fingerprint_is_none_not_true_when_one_id_is_an_error(tmp_path, monkeypatch):
    """Before this change, an error entry was filtered out of the fingerprint judgment entirely:
    two matching, readable fingerprints beside one missing experiment compared as
    same_dataset_fingerprint=True, silently assuming the unreadable third run shared the data.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("e1", {}, dataset_fingerprint="v1:aaaa")
    create_experiment("e2", {}, dataset_fingerprint="v1:aaaa")

    result = compare_experiments(["e1", "e2", "missing"])
    assert any("error" in c for c in result["experiments"])
    assert result["same_dataset_fingerprint"] is None


# ── model: None, never a fabricated "unknown" ────────────────────────────────────────────────


def test_compare_experiments_reports_model_none_not_unknown_for_a_configless_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-no-builder", {"a": 1})

    result = compare_experiments(["exp-no-builder"])
    c = result["experiments"][0]
    assert c["model"] is None
    assert c["model"] != "unknown"


def test_compare_experiments_reports_task_and_subject_from_config(tmp_path, monkeypatch):
    """A rail must admit valid work: task/subject read alongside the builder, from the same
    config read compare_experiments already performs."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-task-subject", {
        "model_source": {"builder": "my_models:chestnut_burr_det", "task": "detection"},
        "data": {"subject": "bud"},
    })

    c = compare_experiments(["exp-task-subject"])["experiments"][0]
    assert c["task"] == "detection"
    assert c["subject"] == "bud"


# ── registry: absent (with a reason), never silently empty, on an unreadable index ──────────


def test_registry_is_absent_with_a_reason_when_the_index_will_not_decode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import tcip_mcp.model_registry as model_registry
    from tcip_mcp.experiments import compare_experiments, create_experiment
    from tcip_mcp.model_registry import RegistryVersionRefused

    create_experiment("exp-registry-corrupt", {"model_source": {"builder": "m:f"}})

    def _boom(project_path):
        raise RegistryVersionRefused("simulated unreadable registry index")

    monkeypatch.setattr(model_registry, "read_registry_index", _boom)

    c = compare_experiments(["exp-registry-corrupt"])["experiments"][0]
    assert "registry" not in c
    assert "registry unreadable" in c["registry_error"]


def test_registry_names_stale_entries_instead_of_matching_nothing(tmp_path, monkeypatch):
    """An entry predating metrics_source is the same condition best_model refuses ranking on:
    the comparison must say so rather than silently reporting an empty registry column."""
    monkeypatch.chdir(tmp_path)
    import tcip_store as ts
    from tcip_mcp.experiments import compare_experiments, create_experiment
    from tcip_mcp.model_registry import ModelRegistry, registry_index_key

    create_experiment("exp-registry-stale", {"model_source": {"builder": "m:f"}})
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    ModelRegistry(str(tmp_path)).register_model(
        "a", str(ckpt), {}, metrics={"val_loss": 0.5}, tags=[], metrics_source="trainer")

    key = registry_index_key(str(tmp_path))
    with ts.transaction(key) as txn:
        document = txn.read(key)
        del document["entries"][0]["metrics_source"]
        txn.write(key, document)

    c = compare_experiments(["exp-registry-stale"])["experiments"][0]
    assert "registry" not in c
    assert "predate" in c["registry_error"]


def test_registry_lists_only_entries_this_experiment_produced(tmp_path, monkeypatch):
    """Admits valid work, through the platform's own producer (register_model_from_experiment):
    a completed run's own registered checkpoint shows up in its own registry column, reduced to
    name/metrics/metrics_source/registration time."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        compare_experiments, complete_run, create_experiment, register_model_from_experiment,
    )

    create_experiment("exp-registered", {"model_source": {"builder": "m:f"}}, data_source="imgs")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights")
    assert "error" not in complete_run("exp-registered", str(ckpt))
    reg = register_model_from_experiment("exp-registered", str(ckpt))
    assert "error" not in reg

    c = compare_experiments(["exp-registered"])["experiments"][0]
    assert [e["name"] for e in c["registry"]] == ["exp-registered"]
    assert c["registry"][0]["metrics_source"] is None  # a checkpoint with no metrics dict
    assert "registered_at" in c["registry"][0]


# ── split: bound and drawn, through persist_split_manifest (the one writer) ────────────────


def test_split_reports_a_bound_manifest_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    from tcip_mcp.experiments import compare_experiments, create_experiment
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    create_experiment("exp-bound-split", {"model_source": {"builder": "m:f"}})
    train_ds = SimpleNamespace(stems=["a", "b"])
    val_ds = SimpleNamespace(stems=["c"])
    data_cfg = {"split": {
        "manifest_binding": {"manifest_dir": "splits/2024-01-01"}, "resolved_seed": 7,
    }}
    persist_split_manifest("exp-bound-split", train_ds, val_ds, data_cfg)

    c = compare_experiments(["exp-bound-split"])["experiments"][0]
    assert c["split"] == {
        "case": "bound", "manifest_dir": "splits/2024-01-01", "seed": 7,
        "redrawn_within_manifest": False,
    }


def test_split_reports_a_redrawn_bound_manifest_distinctly(tmp_path, monkeypatch):
    """A run bound to a manifest and one that redrew inside that same manifest at the same seed
    must never compare as the same data."""
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    from tcip_mcp.experiments import compare_experiments, create_experiment
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    train_ds = SimpleNamespace(stems=["a", "b"])
    val_ds = SimpleNamespace(stems=["c"])

    create_experiment("exp-bound-plain", {"model_source": {"builder": "m:f"}})
    persist_split_manifest("exp-bound-plain", train_ds, val_ds, {"split": {
        "manifest_binding": {"manifest_dir": "splits/2024-01-01"}, "resolved_seed": 7,
    }})

    create_experiment("exp-bound-redrawn", {"model_source": {"builder": "m:f"}})
    persist_split_manifest("exp-bound-redrawn", train_ds, val_ds, {"split": {
        "manifest_binding": {
            "manifest_dir": "splits/2024-01-01",
            "redraw": {
                "seed": 7, "val_ratio": 0.25, "stratify_foreground": True,
            },
        },
        "resolved_seed": 7,
    }})

    c = compare_experiments(["exp-bound-plain", "exp-bound-redrawn"])["experiments"]
    plain, redrawn = c[0]["split"], c[1]["split"]
    assert plain != redrawn
    assert plain.get("redrawn_within_manifest") is False
    assert redrawn.get("redrawn_within_manifest") is True


def test_split_reports_a_drawn_seed_with_no_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    from tcip_mcp.experiments import compare_experiments, create_experiment
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    create_experiment("exp-drawn-split", {"model_source": {"builder": "m:f"}})
    train_ds = SimpleNamespace(stems=["a", "b"])
    val_ds = SimpleNamespace(stems=["c"])
    persist_split_manifest("exp-drawn-split", train_ds, val_ds, {"split": {"seed": 99}})

    c = compare_experiments(["exp-drawn-split"])["experiments"][0]
    assert c["split"] == {"case": "drawn", "seed": 99}


def test_split_reports_no_record_for_a_run_that_never_wrote_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-no-split", {"model_source": {"builder": "m:f"}})

    c = compare_experiments(["exp-no-split"])["experiments"][0]
    assert c["split"] == {"case": "none"}


# ── status_error: a diverged run names its reason, through the divergence fixture ──────────


def test_status_error_names_a_diverged_run_reason(tmp_path, monkeypatch):
    """Admits valid work through the platform's own producer: run_training_envelope over the
    divergence fixture's always-diverged builder writes status.json's own error field, which
    compare_experiments now surfaces as status_error."""
    monkeypatch.chdir(tmp_path)
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import compare_experiments, create_experiment, read_member, status_key, update_status
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    train_ds = ConstantImageDataset([0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4])
    collate = task_collate("regression")
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate)

    config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_always_diverged_model",
                         "task": "regression", "in_chans": 1},
        "device": "cpu", "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 3}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    create_experiment("exp-diverged-cmp", config, data_source="imgs")
    update_status("exp-diverged-cmp", "running")
    run = create_run(config, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=None,
                       task="regression", experiment_id="exp-diverged-cmp")
    run_training_envelope(ctx)

    status = read_member(status_key("exp-diverged-cmp"), {})
    assert "2 consecutive full training passes" in status.get("error", "")

    c = compare_experiments(["exp-diverged-cmp"])["experiments"][0]
    assert "2 consecutive full training passes" in c["status_error"]


def test_status_error_is_none_for_a_run_that_never_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment, update_status

    create_experiment("exp-healthy-cmp", {"model_source": {"builder": "m:f"}})
    update_status("exp-healthy-cmp", "running")

    c = compare_experiments(["exp-healthy-cmp"])["experiments"][0]
    assert c["status_error"] is None


# ── TrainRun.experiment_id/experiment_error: to_dict reads the run's own fields ────────────


def test_to_dict_reports_none_not_the_configs_stale_parent_id_during_the_fork_window(tmp_path):
    """Before this change, to_dict() read config["experiment_id"] directly: a relaunch's own
    config carries the picked parent's id (set by the caller before _ensure_experiment mints a
    fresh one), so a live row briefly reported the parent's id as if it were this run's own
    resolved experiment. Now to_dict() reads the run's own experiment_id field, which
    launch_training sets only once _ensure_experiment has actually resolved it."""
    from tcip_mcp.pipelines.training.run_registry import create_run

    config = {"model_source": {"builder": "m:f"}, "experiment_id": "parent-exp"}
    run = create_run(config, str(tmp_path / "out"))

    assert run.to_dict()["experiment_id"] is None


def test_to_dict_reports_the_resolved_experiment_id_once_set(tmp_path):
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run({"model_source": {"builder": "m:f"}}, str(tmp_path / "out"))
    run.experiment_id = "fresh-exp"

    assert run.to_dict()["experiment_id"] == "fresh-exp"
    assert run.to_dict()["experiment_error"] is None


def test_to_dict_reports_experiment_error_when_tracking_failed(tmp_path):
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run({"model_source": {"builder": "m:f"}}, str(tmp_path / "out"))
    run.experiment_error = "dataset_identity failed: boom"

    d = run.to_dict()
    assert d["experiment_id"] is None
    assert d["experiment_error"] == "dataset_identity failed: boom"


# ── _all_training_runs: the live row's own resolved id over a stale disk overlay ────────────


def test_all_training_runs_keeps_the_live_rows_own_resolved_id_over_the_disk_overlay(
    tmp_path, monkeypatch,
):
    """Before this change, the disk overlay's experiment_id overwrote the live row's own
    unconditionally, whenever a pid-bearing row had any disk record at all keyed by its run_id
    (the overlay lookup keys only by run_id, not by which experiment the live row itself
    resolved). A live row that has already resolved its own experiment_id must keep it."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import _all_training_runs

    run = create_run({"model_source": {"builder": "m:f"}}, str(tmp_path / "out"))
    run.experiment_id = "exp-live"
    run.pid = 4242  # subprocess-delegated, so the merge takes the disk overlay at all

    create_experiment("exp-disk", {"model_source": {"builder": "m:f"}}, data_source="imgs")
    update_status("exp-disk", "running")
    stamp_run_identity("exp-disk", run.run_id, str(tmp_path / "out"))

    rows = _all_training_runs(read_progress=False)
    row = next(r for r in rows if r["run_id"] == run.run_id)
    assert row["experiment_id"] == "exp-live"


def test_all_training_runs_takes_the_disk_overlays_id_when_the_live_row_has_none(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import _all_training_runs

    run = create_run({"model_source": {"builder": "m:f"}}, str(tmp_path / "out"))
    run.pid = 4243  # experiment_id left unresolved (None)

    create_experiment("exp-disk-2", {"model_source": {"builder": "m:f"}}, data_source="imgs")
    update_status("exp-disk-2", "running")
    stamp_run_identity("exp-disk-2", run.run_id, str(tmp_path / "out"))

    rows = _all_training_runs(read_progress=False)
    row = next(r for r in rows if r["run_id"] == run.run_id)
    assert row["experiment_id"] == "exp-disk-2"


# ── experiment_ids filter: rank_registered_models / ModelRegistry.best_model ────────────────────


def _register(tmp_path, experiment_id: str, metric_value: float, *, metric: str = "val_map50"):
    """Admits valid work through the platform's own producer: a completed run's own checkpoint,
    registered via register_model_from_experiment."""
    import torch

    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    create_experiment(experiment_id, {"model_source": {"builder": "m:f"}}, data_source="imgs")
    ckpt = tmp_path / f"{experiment_id}.pt"
    torch.save({"model_state_dict": {}, "metrics": {metric: metric_value}}, ckpt)
    assert "error" not in complete_run(experiment_id, str(ckpt))
    result = register_model_from_experiment(experiment_id, str(ckpt))
    assert "error" not in result
    return result


def test_available_metrics_excludes_an_unmarked_experiments_metric(tmp_path, monkeypatch):
    """Before this change, experiment_ids narrowed nothing: available_metrics (and every other
    derivation) was built from the whole registry regardless of the marked set."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-marked", 0.7, metric="val_map50")
    _register(tmp_path, "exp-other", 0.9, metric="val_loss")

    res = rank_registered_models(str(tmp_path), experiment_ids=["exp-marked"])
    metric_names = {m["metric"] for m in res["available_metrics"]}
    assert metric_names == {"val_map50"}
    assert "val_loss" not in metric_names


def test_rank_registered_models_ranks_only_within_the_marked_set(tmp_path, monkeypatch):
    """Admits valid work: the filter selects within a marked set through entries
    register_model_from_experiment wrote on real completed runs."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-low", 0.5)
    _register(tmp_path, "exp-high", 0.9)

    res = rank_registered_models(str(tmp_path), metric="val_map50", experiment_ids=["exp-low"])
    assert res["name"] == "exp-low"
    assert res["experiment_id"] == "exp-low"


def test_rank_registered_models_names_the_marked_set_when_the_filter_empties_a_non_empty_listing(
    tmp_path, monkeypatch,
):
    """The registry is not empty (exp-other registered a checkpoint); the filter just names no
    marked experiment in it. That is a distinct fact from an empty registry and needs its own
    text: "No models registered" would mislead a breeder into thinking nothing was ever
    trained, when the real gap is which experiments were marked."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-other", 0.7)

    res = rank_registered_models(str(tmp_path), metric="val_map50", experiment_ids=["exp-marked"])
    assert res["error"] == "none of the marked experiments registered a checkpoint"


def test_best_model_experiment_ids_filter_excludes_a_better_unmarked_entry(tmp_path):
    """Admits valid work through the platform's own producer: _register_entry is the one write
    both register_model and register_model_from_experiment route through, and is the only path
    that can bind an entry's experiment_id at all."""
    from tcip_mcp.model_registry import ModelRegistry, _register_entry

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    _register_entry(str(tmp_path), name="a", checkpoint_path=str(ckpt), config={},
                    metrics={"val_map50": 0.5}, tags=[], kind=None, metrics_source="trainer",
                    experiment_id="exp-a")
    _register_entry(str(tmp_path), name="b", checkpoint_path=str(ckpt), config={},
                    metrics={"val_map50": 0.9}, tags=[], kind=None, metrics_source="trainer",
                    experiment_id="exp-b")

    reg = ModelRegistry(str(tmp_path))
    best = reg.best_model("val_map50", higher_is_better=True, experiment_ids=["exp-a"])
    assert best["name"] == "a"


# ── /api/training/compare/best route: 404, 422, 409 mappings, the projected answer ─────────


def test_compare_best_route_404s_with_no_registry(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 404
    assert resp.json()["detail"] == "no model registry in this project"


def test_compare_best_route_404_leaves_the_registry_directory_uncreated(
    client: TestClient, tmp_path, monkeypatch,
):
    """The route reads the index through read_registry_index before rank_registered_models's own
    ModelRegistry construction ever runs, so a project with no registry never gets one just for
    asking whether it has one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 404
    assert not (tmp_path / ".tcip" / "models").exists()


def test_compare_best_route_409s_when_the_index_will_not_decode(
    client: TestClient, tmp_path, monkeypatch,
):
    """A corrupt or version-refused index is not a project with no models: before this fix, the
    route's own except Exception swallowed RegistryVersionRefused/DecodeError into the same 404
    an absent index answers."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.model_registry as model_registry
    from tcip_mcp.model_registry import RegistryVersionRefused

    def _boom(project_path):
        raise RegistryVersionRefused("simulated unreadable registry index")

    # compare_best_route imports read_registry_index locally on each call, so patching the
    # defining module (not the route module, which never binds the name at import time) reaches it.
    monkeypatch.setattr(model_registry, "read_registry_index", _boom)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 409
    assert "registry unreadable" in resp.json()["detail"]


def test_compare_best_route_409s_when_the_index_carries_a_stale_schema_version_two(
    client: TestClient, tmp_path, monkeypatch,
):
    """A dev-era index stamped schema_version 2, predating the version-1 reset, refuses through
    the seam's own ceiling check (SchemaVersionRefused), a sibling of RegistryVersionRefused
    under StoreError rather than a subclass of it: before this fix the route's except tuple
    named only RegistryVersionRefused/DecodeError, so this refusal escaped uncaught."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import os
    import sqlite3

    import tcip_store as ts
    from tcip_mcp.model_registry import registry_index_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    _register(tmp_path, "exp-a", 0.7)
    key = registry_index_key(str(tmp_path))
    body = ts.read(key)
    encoded = ts.get_descriptor(key.store).codec.encode({**body, "schema_version": 2})
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(encoded)
    else:
        from tcip_store.sqlite_backend import database_path, encode_parts

        conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
        try:
            conn.execute(
                "update records set value = ? where store = ? and parts = ?",
                (encoded, key.store, encode_parts(key.parts)),
            )
        finally:
            conn.close()

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 409
    assert "nothing currently strips in place" in resp.json()["detail"]


def test_compare_best_route_422s_on_the_tools_own_error(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _register(tmp_path, "exp-a", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map99",
    })
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "no declared ranking direction" in detail["error"]
    assert detail["needs_direction"] is True


def test_compare_best_route_422s_when_the_marked_set_registered_nothing(
    client: TestClient, tmp_path, monkeypatch,
):
    """rank_registered_models's own distinct text for an experiment_ids filter that empties a
    non-empty listing, not the generic "No models registered" a project with an empty registry
    would carry."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _register(tmp_path, "exp-other", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-marked"], "metric": "val_map50",
    })
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "none of the marked experiments registered a checkpoint"


def test_compare_best_route_409s_on_a_pre_metrics_source_entry(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_store as ts
    from tcip_mcp.model_registry import registry_index_key

    _register(tmp_path, "exp-a", 0.7)
    key = registry_index_key(str(tmp_path))
    with ts.transaction(key) as txn:
        document = txn.read(key)
        del document["entries"][0]["metrics_source"]
        txn.write(key, document)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 409


def test_compare_best_route_projects_the_answer(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _register(tmp_path, "exp-a", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "name": "exp-a", "experiment_id": "exp-a", "metrics": {"val_map50": 0.7},
        "metrics_source": "trainer", "higher_is_better": True, "direction_source": "declared",
        "excluded_unverified": [],
    }


# ── NOT_FINITE_SUFFIX: a drift guard between the two definitions the wire format sits between ──


def test_not_finite_suffix_matches_the_frontends_own_constant():
    """Nothing on the wire enforces this pairing: the frontend's metric helpers read a metric's
    own ``{key}{NOT_FINITE_SUFFIX}`` companion tcip_store.values writes, and the two definitions
    can drift silently since no shared source spans the Python/TypeScript boundary."""
    import re
    from pathlib import Path

    from tcip_store.values import NOT_FINITE_SUFFIX

    ts_source = (
        Path(__file__).resolve().parent.parent
        / "packages" / "tcip-web" / "frontend" / "src" / "tabs" / "trainingMetrics.ts"
    )
    text = ts_source.read_text(encoding="utf-8")
    match = re.search(r'METRIC_STATE_SUFFIX = "([^"]+)"', text)
    assert match is not None, f"METRIC_STATE_SUFFIX declaration not found in {ts_source}"
    assert match.group(1) == NOT_FINITE_SUFFIX


def test_val_metric_prefix_matches_the_frontends_own_constant():
    """The same drift guard as above, for the other constant this file shares with the
    frontend: a run row's best-value label and RunComparison's direction lookup both strip
    this prefix before looking a bare metric name up, and nothing on the wire enforces that
    the two definitions agree."""
    import re
    from pathlib import Path

    from tcip_mcp.pipelines.training.evaluation import VAL_METRIC_PREFIX

    ts_source = (
        Path(__file__).resolve().parent.parent
        / "packages" / "tcip-web" / "frontend" / "src" / "tabs" / "trainingMetrics.ts"
    )
    text = ts_source.read_text(encoding="utf-8")
    match = re.search(r'VAL_METRIC_PREFIX = "([^"]+)"', text)
    assert match is not None, f"VAL_METRIC_PREFIX declaration not found in {ts_source}"
    assert match.group(1) == VAL_METRIC_PREFIX
