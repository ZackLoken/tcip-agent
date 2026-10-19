"""Side-by-side run comparison in the backend.

Covers compare_experiments' task, subject, status_error, split and registry columns,
same_dataset_fingerprint beside an error entry, the experiment_ids filter on best-model ranking,
and the /api/training/compare/best route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from tests._training_values import drawn_split

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tcip_mcp.model_registry import RegistryEntry


def _opened(
    project: Path,
    experiment_id: str,
    *,
    data_dir: Path | None = None,
    split: dict | None = None,
) -> Path:
    """Open an unbuilt detector run, drawing ``split`` when one is given, and return its directory.

    The run is over two detection frames written under ``data_dir`` (by default a fixture
    directory beside the project named for ``experiment_id``), resolved and opened by the
    launcher's own producer and writer.
    """
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        fixture_data_dir,
        opened_run,
        unbuilt_source,
    )

    config = detection_config(
        data_dir or fixture_data_dir(project, experiment_id),
        model_source=unbuilt_source("detection"),
        split=split,
        project=project,
    )
    return opened_run(project, config, experiment_id=experiment_id)


def _bound(
    experiment_id: str, tmp_path: Path, **split: int
) -> tuple[Path, Path]:
    """Open a run bound to a selection drawn at seed 7; return its directory and the selection's.

    The selection is drawn once per ``tmp_path``, at ``splits/2024-01-01``, over the
    two-subject, two-date dataset, and later calls bind to it; the run's ``data.split`` carries
    ``split``'s keys beside the binding.
    """
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import opened_run, unbuilt_source
    from tests.test_selection_binding import (
        _draw,
        _two_subject_two_date_dataset,
    )

    selection_dir = tmp_path / "splits" / "2024-01-01"
    if not selection_dir.exists():
        _draw(
            tmp_path,
            _two_subject_two_date_dataset(tmp_path / "bound-ds"),
            selection_dir,
            seed=7,
        )
    return opened_run(
        tmp_path,
        training_config(
            tmp_path,
            unbuilt_source("detection"),
            {"split": {"selection_dir": str(selection_dir), **split}},
        ),
        experiment_id=experiment_id,
    ), selection_dir


def _compared(project: Path, *experiment_ids: str) -> list[dict]:
    from tcip_mcp.experiments import compare_experiments

    return compare_experiments(list(experiment_ids), project=project)[
        "experiments"
    ]


def test_same_dataset_fingerprint_is_none_not_true_when_one_id_is_an_error(
    tmp_path: Path,
) -> None:
    """The dataset fingerprint comparison is None, not True, when one id is an error.

    Two runs over one dataset compare as True; the same two beside one missing run compare as
    None.
    """
    from tcip_mcp.experiments import compare_experiments
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.test_dataset_identity_recording import _config, _make_dataset

    shared = _config(tmp_path, _make_dataset(tmp_path / "shared"))
    opened_run(tmp_path, shared, experiment_id="e1")
    opened_run(tmp_path, shared, experiment_id="e2")
    assert (
        compare_experiments(["e1", "e2"], project=tmp_path)[
            "same_dataset_fingerprint"
        ]
        is True
    )

    result = compare_experiments(["e1", "e2", "missing"], project=tmp_path)
    assert any("error" in c for c in result["experiments"])
    assert result["same_dataset_fingerprint"] is None


def test_compare_experiments_reports_task_and_subject_from_the_runs_records(
    tmp_path: Path,
) -> None:
    """``compare_experiments`` reports a run's task and subject from its own records.

    The task is read off the launch record, the subject off the data section the run resolved.
    """
    _opened(tmp_path, "exp-task-subject")

    (c,) = _compared(tmp_path, "exp-task-subject")
    assert (c["task"], c["subject"]) == ("detection", "bud")


def test_registry_lists_the_checkpoint_this_run_completed(
    tmp_path: Path,
) -> None:
    """A completed run's own checkpoint shows up in its registry column; an unfinished run's none.

    The run completes through the platform's own producer, and its entry carries its name, its
    metrics, ``metrics_source`` and ``registered_at``.
    """
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(
        tmp_path, experiment_id="exp-registered", metrics={"val_map50": 0.5}
    )
    _opened(tmp_path, "exp-unfinished")

    registered, unfinished = _compared(
        tmp_path, "exp-registered", "exp-unfinished"
    )
    assert [e["name"] for e in registered["registry"]] == ["exp-registered"]
    assert registered["registry"][0]["metrics"] == {"val_map50": 0.5}
    assert registered["registry"][0]["metrics_source"] == "training_source"
    assert "registered_at" in registered["registry"][0]
    assert unfinished["registry"] == []


def test_split_reports_a_bound_selection_directory(tmp_path: Path) -> None:
    _run_dir, selection_dir = _bound("exp-bound-split", tmp_path)

    (c,) = _compared(tmp_path, "exp-bound-split")
    assert c["split"] == {
        "case": "bound",
        "selection_dir": str(selection_dir),
        "seed": 7,
        "redraw": False,
    }


def test_split_reports_a_redrawn_bound_selection_distinctly(
    tmp_path: Path,
) -> None:
    """A bound run and one that redrew inside the same selection report different splits."""
    _bound("exp-bound-plain", tmp_path)
    _bound("exp-bound-redrawn", tmp_path, seed=7)

    plain, redrawn = (
        c["split"]
        for c in _compared(tmp_path, "exp-bound-plain", "exp-bound-redrawn")
    )
    assert plain != redrawn
    assert (plain["redraw"], redrawn["redraw"]) == (False, True)


def test_split_reports_a_drawn_seed_with_no_binding(tmp_path: Path) -> None:
    _opened(
        tmp_path,
        "exp-drawn-split",
        split=drawn_split(seed=99, val_ratio=0.15),
    )

    (c,) = _compared(tmp_path, "exp-drawn-split")
    assert c["split"] == {"case": "drawn", "seed": 99}


def test_status_error_names_a_diverged_run_reason(tmp_path: Path) -> None:
    """``compare_experiments`` surfaces a diverged run's final-status error as status_error.

    run_training_envelope over the divergence fixture's always-diverged builder writes the final
    status's own error, which compare_experiments surfaces as status_error; a run with no final
    status carries none.
    """
    from tcip_mcp.experiments import observe
    from tests.test_lifecycle_wiring import _stock_run
    from tests.tiny_trainer_fixtures import ALWAYS_DIVERGED_MODEL

    run_dir = _stock_run(
        tmp_path, ALWAYS_DIVERGED_MODEL, 3, "exp-diverged-cmp"
    )
    status_error = observe(run_dir).status_error
    assert status_error is not None
    assert "2 consecutive full training passes" in status_error
    _opened(tmp_path, "exp-healthy-cmp")

    diverged, healthy = _compared(
        tmp_path, "exp-diverged-cmp", "exp-healthy-cmp"
    )
    assert "2 consecutive full training passes" in diverged["status_error"]
    assert healthy["status_error"] is None


def _register(
    project: Path,
    experiment_id: str,
    metric_value: float,
    *,
    metric: str = "val_map50",
) -> None:
    """Complete a run through the platform's own producer, its checkpoint carrying ``metric``."""
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(
        project, experiment_id=experiment_id, metrics={metric: metric_value}
    )


def test_available_metrics_excludes_an_unmarked_experiments_metric(
    tmp_path: Path,
) -> None:
    """experiment_ids narrows ``available_metrics`` to the marked experiments' metrics."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-marked", 0.7, metric="val_map50")
    _register(tmp_path, "exp-other", 0.9, metric="val_loss")

    res = rank_registered_models(tmp_path, experiment_ids=["exp-marked"])
    assert {m["metric"] for m in res["available_metrics"]} == {"val_map50"}


def test_rank_registered_models_ranks_only_within_the_marked_set(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-low", 0.5)
    _register(tmp_path, "exp-high", 0.9)

    res = rank_registered_models(
        tmp_path,
        metric="val_map50",
        include_unverified=True,
        experiment_ids=["exp-low"],
    )
    assert (res["name"], res["experiment_id"]) == ("exp-low", "exp-low")


def test_rank_registered_models_names_the_marked_set_when_the_filter_empties_a_non_empty_listing(
    tmp_path: Path,
) -> None:
    """A filter naming no registered experiment is refused with text of its own.

    Over a registry holding another experiment's checkpoint, a filter naming only an
    unregistered experiment refuses with "none of the marked experiments registered a
    checkpoint".
    """
    from tcip_mcp.tools.model_tools import rank_registered_models

    _register(tmp_path, "exp-other", 0.7)

    res = rank_registered_models(
        tmp_path, metric="val_map50", experiment_ids=["exp-marked"]
    )
    assert (
        res["error"]
        == "none of the marked experiments registered a checkpoint"
    )


def test_a_ranking_over_a_marked_set_that_registered_nothing_reads_the_registry_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_mcp.model_registry as model_registry
    from tcip_mcp.tools.model_tools import ranked_registered_model

    _register(tmp_path, "exp-other", 0.7)
    reads: list[object] = []
    real = model_registry.read_registry_index

    def counted(project_path: str | Path) -> list[RegistryEntry]:
        reads.append(project_path)
        return real(project_path)

    monkeypatch.setattr(model_registry, "read_registry_index", counted)
    res = ranked_registered_model(
        tmp_path,
        "val_map50",
        higher_is_better=None,
        include_unverified=False,
        experiment_ids=["exp-marked"],
        tag=None,
    )
    assert (
        res["error"]
        == "none of the marked experiments registered a checkpoint"
    )
    assert len(reads) == 1


def test_compare_best_route_422s_with_no_registered_checkpoint(
    opened_client: TestClient, tmp_path: Path
) -> None:
    """The compare-best route answers 422 for a project with no registered checkpoint.

    A project with no completed run and no foreign registration answers the ranking's own
    refusal as 422, and the asking writes no registry index.
    """
    import tcip_store
    from tcip_mcp.model_registry import registry_index_key

    resp = opened_client.post(
        "/api/training/compare/best",
        json={
            "experiment_ids": ["exp-a"],
            "metric": "val_map50",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["message"] == "No models registered"
    assert tcip_store.read(registry_index_key(tmp_path), default=None) is None


def test_compare_best_route_409s_when_the_index_will_not_decode(
    opened_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compare-best route answers 409 when the registry index will not decode.

    The detail names the registry as unreadable.
    """
    import tcip_mcp.model_registry as model_registry
    from tcip_store import DecodeError

    def _boom(project_path: str | Path) -> list[dict]:
        raise DecodeError("simulated unreadable registry index")

    monkeypatch.setattr(model_registry, "read_registry_index", _boom)

    resp = opened_client.post(
        "/api/training/compare/best",
        json={
            "experiment_ids": ["exp-a"],
            "metric": "val_map50",
        },
    )
    assert resp.status_code == 409
    assert "registry unreadable" in resp.json()["detail"]


def test_compare_best_route_422s_on_the_tools_own_error(
    opened_client: TestClient, tmp_path: Path
) -> None:
    _register(tmp_path, "exp-a", 0.7)

    resp = opened_client.post(
        "/api/training/compare/best",
        json={
            "experiment_ids": ["exp-a"],
            "metric": "val_map99",
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "no declared ranking direction" in detail["message"]
    assert detail["needs_direction"] is True


def test_compare_best_route_422s_when_the_marked_set_registered_nothing(
    opened_client: TestClient, tmp_path: Path
) -> None:
    _register(tmp_path, "exp-other", 0.7)

    resp = opened_client.post(
        "/api/training/compare/best",
        json={
            "experiment_ids": ["exp-marked"],
            "metric": "val_map50",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["message"] == (
        "none of the marked experiments registered a checkpoint"
    )


def test_an_empty_metric_lists_through_the_listing_route_and_refuses_through_the_ranking_one(
    opened_client: TestClient,
    tmp_path: Path,
) -> None:
    _register(tmp_path, "exp-a", 0.7)

    listed = opened_client.get("/api/results/models/registered")
    assert listed.status_code == 200
    assert [m["experiment_id"] for m in listed.json()["models"]] == ["exp-a"]

    resp = opened_client.post(
        "/api/training/compare/best",
        json={"experiment_ids": ["exp-a"], "metric": ""},
    )
    assert resp.status_code == 422
    assert "names the metric it ranks by" in resp.json()["detail"]["message"]


def test_compare_best_route_projects_the_answer(
    opened_client: TestClient, tmp_path: Path
) -> None:
    _register(tmp_path, "exp-a", 0.7)

    resp = opened_client.post(
        "/api/training/compare/best",
        json={
            "experiment_ids": ["exp-a"],
            "metric": "val_map50",
            "include_unverified": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "name": "exp-a",
        "experiment_id": "exp-a",
        "metrics": {"val_map50": 0.7},
        "metrics_source": "training_source",
        "ranking_basis": "val_map50",
        "higher_is_better": True,
        "direction_source": "declared",
        "excluded_unverified": [],
    }
