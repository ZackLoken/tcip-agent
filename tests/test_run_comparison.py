"""Backend coverage for side-by-side run comparison: compare_experiments' task, subject,
status_error, split and registry columns, an honest same_dataset_fingerprint over an error
column, the experiment_ids filter on best-model ranking, and the /api/training/compare/best
route.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client(opened_project) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _opened(project, experiment_id: str, *, data_dir=None, split: dict | None = None):
    """A detector run, over two frames of its own (or those under ``data_dir``) and stating
    ``split``, resolved and opened by the launcher's own producer and writer."""
    from tests._verified_checkpoint_fixtures import detection_config, fixture_data_dir, opened_run

    config = detection_config(data_dir or fixture_data_dir(project, experiment_id),
                              model_source={"builder": "m:f", "task": "detection"})
    if split is not None:
        config["data"]["split"] = split
    return opened_run(project, config, experiment_id=experiment_id)


def _bound(experiment_id: str, tmp_path, **split):
    """A run bound to a selection ``draw_splits`` drew at seed 7, its ``data.split`` beside the
    binding being ``split``."""
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.test_selection_binding import _draw, _two_subject_two_date_dataset

    selection_dir = tmp_path / "splits" / "2024-01-01"
    if not selection_dir.exists():
        _draw(tmp_path, _two_subject_two_date_dataset(tmp_path / "bound-ds"), selection_dir, seed=7)
    return opened_run(tmp_path, {"model_source": {"builder": "m:f", "task": "detection"},
                             "data": {"split": {"selection_dir": str(selection_dir), **split}}},
                      experiment_id=experiment_id), selection_dir


def _compared(project, *experiment_ids: str) -> list[dict]:
    from tcip_mcp.experiments import compare_experiments

    return compare_experiments(list(experiment_ids), project=project)["experiments"]


def test_same_dataset_fingerprint_is_none_not_true_when_one_id_is_an_error(tmp_path):
    """An error entry is never dropped from the fingerprint judgment: two matching fingerprints
    beside one missing run do not compare as the same data."""
    from tcip_mcp.experiments import compare_experiments
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.test_dataset_identity_recording import _config, _make_dataset

    shared = _config(_make_dataset(tmp_path / "shared"))
    opened_run(tmp_path, shared, experiment_id="e1")
    opened_run(tmp_path, shared, experiment_id="e2")
    assert compare_experiments(["e1", "e2"], project=tmp_path)["same_dataset_fingerprint"] is True

    result = compare_experiments(["e1", "e2", "missing"], project=tmp_path)
    assert any("error" in c for c in result["experiments"])
    assert result["same_dataset_fingerprint"] is None


def test_compare_experiments_reports_task_and_subject_from_the_runs_records(tmp_path):
    """The task is read off the launch record, the subject off the data section the run
    resolved."""
    _opened(tmp_path, "exp-task-subject")

    (c,) = _compared(tmp_path, "exp-task-subject")
    assert (c["task"], c["subject"]) == ("detection", "bud")


def test_registry_lists_the_checkpoint_this_run_completed(tmp_path):
    """Admits valid work, through the platform's own producer: a completed run's own checkpoint
    shows up in its own registry column, reduced to name/metrics/metrics_source/completion
    time; a run with no completed checkpoint lists none."""
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(tmp_path, experiment_id="exp-registered", metrics={"val_map50": 0.5})
    _opened(tmp_path, "exp-unfinished")

    registered, unfinished = _compared(tmp_path, "exp-registered", "exp-unfinished")
    assert [e["name"] for e in registered["registry"]] == ["exp-registered"]
    assert registered["registry"][0]["metrics"] == {"val_map50": 0.5}
    assert registered["registry"][0]["metrics_source"] == "training_source"
    assert "registered_at" in registered["registry"][0]
    assert unfinished["registry"] == []


def test_split_reports_a_bound_selection_directory(tmp_path):
    _run_dir, selection_dir = _bound("exp-bound-split", tmp_path)

    (c,) = _compared(tmp_path, "exp-bound-split")
    assert c["split"] == {
        "case": "bound", "selection_dir": str(selection_dir), "seed": 7, "redraw": False,
    }


def test_split_reports_a_redrawn_bound_selection_distinctly(tmp_path):
    """A run bound to a selection and one that redrew inside that same selection must never
    compare as the same data."""
    _bound("exp-bound-plain", tmp_path)
    _bound("exp-bound-redrawn", tmp_path, redraw_within_selection=True, seed=7)

    plain, redrawn = (c["split"]
                      for c in _compared(tmp_path, "exp-bound-plain", "exp-bound-redrawn"))
    assert plain != redrawn
    assert (plain["redraw"], redrawn["redraw"]) == (False, True)


def test_split_reports_a_drawn_seed_with_no_binding(tmp_path):
    _opened(tmp_path, "exp-drawn-split", split={"seed": 99})

    (c,) = _compared(tmp_path, "exp-drawn-split")
    assert c["split"] == {"case": "drawn", "seed": 99}


def test_status_error_names_a_diverged_run_reason(tmp_path):
    """Admits valid work through the platform's own producer: run_training_envelope over the
    divergence fixture's always-diverged builder writes the final status's own error, which
    compare_experiments surfaces as status_error; a run with no final status carries none."""
    from tcip_mcp.experiments import observe
    from tests.test_lifecycle_wiring import _stock_run

    run_dir = _stock_run(
        tmp_path, {"builder": "tests.tiny_trainer_fixtures:build_always_diverged_model"}, 3,
        "exp-diverged-cmp")
    assert "2 consecutive full training passes" in observe(run_dir).final["error"]
    _opened(tmp_path, "exp-healthy-cmp")

    diverged, healthy = _compared(tmp_path, "exp-diverged-cmp", "exp-healthy-cmp")
    assert "2 consecutive full training passes" in diverged["status_error"]
    assert healthy["status_error"] is None


def _register(project, experiment_id: str, metric_value: float, *,
              metric: str = "val_map50") -> None:
    """A run completed through the platform's own producer, its checkpoint carrying ``metric``."""
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(project, experiment_id=experiment_id, metrics={metric: metric_value})


def test_available_metrics_excludes_an_unmarked_experiments_metric(tmp_path):
    """experiment_ids narrows every derivation, available_metrics included, to the marked set."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-marked", 0.7, metric="val_map50")
    _register(tmp_path, "exp-other", 0.9, metric="val_loss")

    res = rank_registered_models(tmp_path, experiment_ids=["exp-marked"])
    assert {m["metric"] for m in res["available_metrics"]} == {"val_map50"}


def test_rank_registered_models_ranks_only_within_the_marked_set(tmp_path):
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-low", 0.5)
    _register(tmp_path, "exp-high", 0.9)

    res = rank_registered_models(tmp_path, metric="val_map50", include_unverified=True,
                                 experiment_ids=["exp-low"])
    assert (res["name"], res["experiment_id"]) == ("exp-low", "exp-low")


def test_rank_registered_models_names_the_marked_set_when_the_filter_empties_a_non_empty_listing(
    tmp_path,
):
    """The registry is not empty; the filter just names no marked experiment in it. That is a
    distinct fact from an empty registry and needs its own text."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-other", 0.7)

    res = rank_registered_models(tmp_path, metric="val_map50", experiment_ids=["exp-marked"])
    assert res["error"] == "none of the marked experiments registered a checkpoint"


def test_a_ranking_over_a_marked_set_that_registered_nothing_reads_the_registry_once(
    tmp_path, monkeypatch,
):
    import tcip_mcp.model_registry as model_registry
    from tcip_mcp.tools.model_tools import ranked_registered_model

    _register(tmp_path, "exp-other", 0.7)
    reads: list[object] = []
    real = model_registry.read_registry_index

    def counted(project_path):
        reads.append(project_path)
        return real(project_path)

    monkeypatch.setattr(model_registry, "read_registry_index", counted)
    res = ranked_registered_model(tmp_path, "val_map50", higher_is_better=None,
                                  include_unverified=False, experiment_ids=["exp-marked"], tag=None)
    assert res["error"] == "none of the marked experiments registered a checkpoint"
    assert len(reads) == 1


def test_compare_best_route_422s_with_no_registered_checkpoint(client: TestClient, tmp_path):
    """A project with no completed run and no foreign registration answers the ranking's own
    refusal as 422, and the asking writes no registry index."""
    import tcip_store

    from tcip_mcp.model_registry import registry_index_key

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "No models registered"
    assert tcip_store.read(registry_index_key(tmp_path), default=None) is None


def test_compare_best_route_409s_when_the_index_will_not_decode(client: TestClient, monkeypatch):
    """A corrupt index is not a project with no models: the route answers 409, not the 404 an
    absent index answers."""
    import tcip_mcp.model_registry as model_registry
    from tcip_store import DecodeError

    def _boom(project_path):
        raise DecodeError("simulated unreadable registry index")

    monkeypatch.setattr(model_registry, "read_registry_index", _boom)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50",
    })
    assert resp.status_code == 409
    assert "registry unreadable" in resp.json()["detail"]


def test_compare_best_route_422s_on_the_tools_own_error(client: TestClient, tmp_path):
    _register(tmp_path, "exp-a", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map99",
    })
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "no declared ranking direction" in detail["error"]
    assert detail["needs_direction"] is True


def test_compare_best_route_422s_when_the_marked_set_registered_nothing(client: TestClient,
                                                                        tmp_path):
    _register(tmp_path, "exp-other", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-marked"], "metric": "val_map50",
    })
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "none of the marked experiments registered a checkpoint"


def test_an_empty_metric_lists_through_the_listing_route_and_refuses_through_the_ranking_one(
    client: TestClient, tmp_path,
):
    _register(tmp_path, "exp-a", 0.7)

    listed = client.get("/api/results/models/registered")
    assert listed.status_code == 200
    assert [m["experiment_id"] for m in listed.json()["models"]] == ["exp-a"]

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": ""})
    assert resp.status_code == 422
    assert "names the metric it ranks by" in resp.json()["detail"]["error"]


def test_compare_best_route_projects_the_answer(client: TestClient, tmp_path):
    _register(tmp_path, "exp-a", 0.7)

    resp = client.post("/api/training/compare/best", json={
        "experiment_ids": ["exp-a"], "metric": "val_map50", "include_unverified": True,
    })
    assert resp.status_code == 200
    assert resp.json() == {
        "name": "exp-a", "experiment_id": "exp-a", "metrics": {"val_map50": 0.7},
        "metrics_source": "training_source", "higher_is_better": True,
        "direction_source": "declared", "excluded_unverified": [],
    }


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
    frontend."""
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
