"""The experiment tools' list, compare and lineage reads over launcher-opened runs.

Each run is a directory the launcher's own writer opened under a scratch project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _opened(
    project: Path,
    experiment_id: str,
    builder: str = "my_models:tree_detector",
    **facts: str | float | dict | None,
) -> Path:
    """Open a run of a detector built by ``builder`` through the launcher's own writer.

    The run belongs to ``project``, its builder declared in a ``my_models.py`` of its own, over
    its own two frames; ``facts`` are ``open_run``'s other keywords.
    """
    from tests._chain_fixtures import FRAME_STATISTICS
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        fixture_data_dir,
        opened_run,
    )

    models = fixture_data_dir(project, experiment_id) / "my_models.py"
    models.parent.mkdir(parents=True, exist_ok=True)
    models.write_text("", encoding="utf-8")
    config = detection_config(
        fixture_data_dir(project, experiment_id),
        project=project,
        model_source={
            "builder": builder,
            "source_files": [str(models)],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
    )
    return opened_run(project, config, experiment_id=experiment_id, **facts)


def test_experiment_list_compare_lineage(tmp_path: Path) -> None:
    from tcip_mcp.experiments import (
        RUN_FILE,
        compare_experiments,
        get_experiment_lineage,
        read_run_record,
        run_rows,
    )
    from tests._verified_checkpoint_fixtures import log_epoch

    e1 = _opened(
        tmp_path, "e1", "my_models:tv_resnet50_det", relaunched_from="e0"
    )
    log_epoch(e1, 1, {"map50": 0.6})
    _opened(tmp_path, "e2", "my_models:fcos_det")

    assert {e.experiment_id for e in run_rows(tmp_path)} == {"e1", "e2"}

    cmp = compare_experiments(
        ["e1", "e2", "missing"], project=tmp_path
    ).model_dump()
    assert len(cmp["experiments"]) == 3
    first = next(
        c
        for c in cmp["experiments"]
        if "run" in c and c["run"]["experiment_id"] == "e1"
    )
    assert first["run"]["builder"] == "my_models:tv_resnet50_det"
    assert first["last_logged_metrics"]["map50"] == 0.6
    assert any(
        "error" in c for c in cmp["experiments"]
    )  # the missing run is reported

    lineage = get_experiment_lineage("e1", project=tmp_path)["lineage"]
    record = read_run_record(e1 / RUN_FILE)
    assert isinstance(record, dict)
    resolved = record["resolved"]
    assert isinstance(resolved, dict)
    assert lineage["data"] == resolved["data"]
    assert lineage["relaunched_from"] == "e0"
    assert (
        lineage["checkpoint"] is None
    )  # no final status names a checkpoint yet
    assert "error" in get_experiment_lineage("nope", project=tmp_path)


def test_a_completed_runs_lineage_names_the_checkpoint_its_final_status_names(
    tmp_path: Path,
) -> None:
    from tcip_mcp.experiments import (
        RunCheckpoint,
        get_experiment_lineage,
        observe,
    )
    from tests._verified_checkpoint_fixtures import finished_run

    checkpoint = observe(
        finished_run(tmp_path, experiment_id="exp-done")
    ).checkpoint
    assert checkpoint is not None

    lineage = get_experiment_lineage("exp-done", project=tmp_path)["lineage"]
    assert RunCheckpoint.model_validate(lineage["checkpoint"]) == checkpoint


def test_get_experiment_tool_pages_epoch_rows_bounded_by_n_epochs(
    tmp_path: Path,
) -> None:
    """``get_experiment`` pages the epoch rows ``n_epochs`` counts under the full view.

    The MCP tool accepts metrics_limit/metrics_offset under view='full', paging the epoch
    rows ``n_epochs`` counts.
    """
    from tcip_mcp.tools.experiment_tools import get_experiment
    from tests._verified_checkpoint_fixtures import log_epoch

    run_dir = _opened(tmp_path, "exp-paged")
    for epoch in range(5):
        log_epoch(run_dir, epoch, {"loss": float(epoch)})

    full = get_experiment(tmp_path, "exp-paged")
    assert full["n_epochs"] == 5

    page = get_experiment(
        tmp_path, "exp-paged", metrics_limit=2, metrics_offset=1
    )
    assert page["n_epochs"] == 5
    assert [r["epoch"] for r in page["metrics"]] == [1, 2]


def test_get_experiment_tool_lineage_view_admits_defaults_refuses_pagination(
    tmp_path: Path,
) -> None:
    """``get_experiment``'s lineage view admits its defaults and refuses pagination.

    The ordinary view='lineage' call (no pagination args) still succeeds; a non-default
    pagination arg under that view is refused as meaningless.
    """
    from tcip_mcp.tools.experiment_tools import get_experiment

    _opened(tmp_path, "exp-lineage")

    assert "error" not in get_experiment(
        tmp_path, "exp-lineage", view="lineage"
    )
    assert "error" in get_experiment(
        tmp_path, "exp-lineage", view="lineage", metrics_limit=3
    )
    assert "error" in get_experiment(
        tmp_path, "exp-lineage", view="lineage", metrics_offset=2
    )


def test_compare_experiments_reports_the_last_row_and_rows_logged_after_the_end(
    tmp_path: Path,
) -> None:
    """``compare_experiments`` reports the last row and counts rows logged after the end.

    Of two rows appended to a completed run, the one stamped now is counted after the end and
    the one stamped in 2000 is not, and the last row appended is the one reported.
    """
    from tcip_mcp.audit import now
    from tcip_mcp.experiments import (
        METRICS_FILE,
        MetricsRow,
        append_row,
        compare_experiments,
    )
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-ended")
    append_row(
        run_dir / METRICS_FILE,
        MetricsRow.model_validate({"epoch": 0, "timestamp": now(), "loss": 0.5}),
    )
    append_row(
        run_dir / METRICS_FILE,
        MetricsRow.model_validate(
            {"epoch": 1, "timestamp": "2000-01-01T00:00:00+00:00", "loss": 0.4}
        ),
    )

    (c,) = compare_experiments(["exp-ended"], project=tmp_path).model_dump()[
        "experiments"
    ]
    assert c["run"]["state"] == "completed"
    assert c["last_logged_metrics"]["loss"] == 0.4
    assert c["rows_after_end"] == 1  # the row stamped now, after the end
    assert c["n_epochs"] == 2


def test_compare_experiments_stale_heartbeat_compares_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run with no final status compares running, then interrupted when its heartbeat is stale.

    While its heartbeat is fresh it reads running with no rows-after-end count.
    """
    from tcip_mcp import experiments
    from tcip_mcp.experiments import compare_experiments

    _opened(tmp_path, "exp-live")

    (live,) = compare_experiments(["exp-live"], project=tmp_path).experiments
    assert isinstance(live, experiments.ComparedRun)
    assert (live.run.state, live.rows_after_end) == ("running", None)

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    (stale,) = compare_experiments(["exp-live"], project=tmp_path).experiments
    assert isinstance(stale, experiments.ComparedRun)
    assert stale.run.state == "interrupted"


def test_compare_experiments_rows_after_end_compares_instants_across_offsets(
    tmp_path: Path,
) -> None:
    """``rows_after_end`` compares row timestamps as instants across UTC offsets.

    Of five rows stamped in three UTC offsets and a bare "Z", the two whose instants fall
    strictly after the final status's ``ended`` are counted. A row stating no timestamp refuses
    the run's comparison, naming the row.
    """
    from datetime import timedelta, timezone

    from tcip_mcp.experiments import (
        METRICS_FILE,
        ComparedRun,
        MetricsRow,
        UncomparedRun,
        append_row,
        compare_experiments,
        observe,
    )
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-instants")
    final = observe(run_dir).final
    assert final is not None
    ended = final.ended

    def at(offset_s: float, zone: timezone) -> str:
        return (
            (ended + timedelta(seconds=offset_s)).astimezone(zone).isoformat()
        )

    rows = [
        {
            "epoch": 1,
            "timestamp": at(1, timezone(timedelta(hours=-6))),
            "loss": 0.1,
        },  # after
        {
            "epoch": 2,
            "timestamp": at(-1, timezone(timedelta(hours=5))),
            "loss": 0.2,
        },  # before
        {
            "epoch": 3,
            "timestamp": at(-1, timezone.utc).replace("+00:00", "Z"),
            "loss": 0.3,
        },
        {
            "epoch": 4,
            "timestamp": at(0, timezone.utc),
            "loss": 0.4,
        },  # same instant: not after
        {"epoch": 5, "timestamp": at(1, timezone.utc), "loss": 0.5},  # after
    ]
    for row in rows:
        append_row(run_dir / METRICS_FILE, MetricsRow.model_validate(row))

    (compared,) = compare_experiments(
        ["exp-instants"], project=tmp_path
    ).experiments
    assert isinstance(compared, ComparedRun)
    assert compared.rows_after_end == 2

    with (run_dir / METRICS_FILE).open("ab") as log:
        log.write(b'{"epoch": 6, "loss": 0.6}\n')
    (refused,) = compare_experiments(
        ["exp-instants"], project=tmp_path
    ).experiments
    assert isinstance(refused, UncomparedRun)
    assert "epoch': 6" in refused.error and "timestamp" in refused.error
