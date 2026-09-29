"""Experiment-tool coverage (list / compare / lineage), each run a directory the launcher's own
writer opened."""

import pytest

@pytest.fixture(autouse=True)
def _platform(tmp_path, monkeypatch):
    """Pin the platform state root at this test's tmp dir, where its runs land."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))


def _opened(experiment_id: str, builder: str = "my_models:tree_detector", **facts):
    """A run of a detector built by ``builder`` over its own two frames, opened by the
    launcher's own writer; ``facts`` are ``open_run``'s other keywords."""
    from tests._verified_checkpoint_fixtures import detection_config, fixture_data_dir, opened_run

    config = detection_config(fixture_data_dir(None, experiment_id),
                              model_source={"builder": builder, "task": "detection"})
    return opened_run(None, config, experiment_id=experiment_id, **facts)


def test_experiment_list_compare_lineage():
    from tcip_mcp.experiments import (
        compare_experiments, get_experiment_lineage, list_experiments, run_resolution,
    )
    from tests._verified_checkpoint_fixtures import log_epoch

    e1 = _opened("e1", "my_models:tv_resnet50_det", parent_experiment="e0")
    log_epoch(e1, 1, {"map50": 0.6})
    _opened("e2", "my_models:fcos_det")

    assert {e["experiment_id"] for e in list_experiments()} == {"e1", "e2"}

    cmp = compare_experiments(["e1", "e2", "missing"])
    assert cmp["count"] == 3
    first = next(c for c in cmp["experiments"] if c["experiment_id"] == "e1")
    assert first["model"] == "my_models:tv_resnet50_det"
    assert first["last_logged_metrics"]["map50"] == 0.6
    assert any("error" in c for c in cmp["experiments"])   # the missing run is reported

    lineage = get_experiment_lineage("e1")["lineage"]
    assert lineage["data"] == run_resolution("e1")["data"]
    assert lineage["parent_experiment"] == "e0"
    assert lineage["checkpoint"] is None  # no final status names a checkpoint yet
    assert "error" in get_experiment_lineage("nope")


def test_a_completed_runs_lineage_names_the_checkpoint_its_final_status_names():
    from tcip_mcp.experiments import get_experiment_lineage, observe
    from tests._verified_checkpoint_fixtures import finished_run

    checkpoint = observe(finished_run(None, experiment_id="exp-done")).checkpoint
    assert checkpoint is not None

    lineage = get_experiment_lineage("exp-done")["lineage"]
    assert lineage["checkpoint"] == checkpoint


def test_get_experiment_tool_pages_metrics_and_exposes_n_rows():
    """The MCP tool accepts metrics_limit/metrics_offset under view='full'; n_rows (the row
    count) is the paging bound, always present alongside n_epochs."""
    from tcip_mcp.tools.experiment_tools import get_experiment
    from tests._verified_checkpoint_fixtures import log_epoch

    run_dir = _opened("exp-paged")
    for epoch in range(5):
        log_epoch(run_dir, epoch, {"loss": float(epoch)})

    full = get_experiment("exp-paged")
    assert full["n_rows"] == 5 and full["n_epochs"] == 5

    page = get_experiment("exp-paged", metrics_limit=2, metrics_offset=1)
    assert page["n_rows"] == 5
    assert [r["epoch"] for r in page["metrics"]] == [1, 2]


def test_get_experiment_tool_lineage_view_admits_defaults_refuses_pagination():
    """A rail must admit valid work: the ordinary view='lineage' call (no pagination args)
    still succeeds; a non-default pagination arg under that view is refused as meaningless."""
    from tcip_mcp.tools.experiment_tools import get_experiment

    _opened("exp-lineage")

    assert "error" not in get_experiment("exp-lineage", view="lineage")
    assert "error" in get_experiment("exp-lineage", view="lineage", metrics_limit=3)
    assert "error" in get_experiment("exp-lineage", view="lineage", metrics_offset=2)


def _calibration():
    """A calibration run opened by its own writer."""
    from tcip_mcp.experiments import open_calibration_run

    return open_calibration_run({
        "document": "operating_point", "checkpoint_sha256": None,
        "reference_identity": {"calibration_dataset_hash": "h"}, "trait": "bud_50per_date",
        "derived_from": "a calibration door"})


def test_list_experiments_tool_carries_has_model_source():
    """A training run's launch record carries a config; a calibration run's does not."""
    from tcip_mcp.tools.experiment_tools import list_experiments

    _opened("exp-run")
    calibration = _calibration()

    listed = {e["experiment_id"]: e for e in list_experiments()["experiments"]}
    assert listed["exp-run"]["has_model_source"] is True
    assert listed[calibration.name]["has_model_source"] is False


def test_list_experiments_launched_only_serves_the_training_runs_view():
    """launched_only=True switches list_experiments to the training runs view: training run
    directories only, calibration runs left out, in the shape _all_training_runs builds."""
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_mcp.tools.training_tools import _all_training_runs

    _opened("exp-launched-view")
    calibration = _calibration()

    default_view = list_experiments()
    assert "experiments" in default_view and "runs" not in default_view

    launched_view = list_experiments(launched_only=True)
    assert launched_view == {"runs": _all_training_runs()}
    by_id = {r["experiment_id"]: r for r in launched_view["runs"]}
    assert "exp-launched-view" in by_id
    assert calibration.name not in by_id


def test_compare_experiments_reports_the_last_row_and_rows_logged_after_the_end():
    """The final status is written once; a row an outside writer appends later with a later
    instant is counted, never hidden, and a completed run's own rows are not."""
    from tcip_mcp.experiments import METRICS_FILE, append_row, compare_experiments, now_iso
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(None, experiment_id="exp-ended")
    append_row(run_dir / METRICS_FILE, {"epoch": 0, "timestamp": now_iso(), "loss": 0.5})
    append_row(run_dir / METRICS_FILE,
               {"epoch": 1, "timestamp": "2000-01-01T00:00:00+00:00", "loss": 0.4})

    (c,) = compare_experiments(["exp-ended"])["experiments"]
    assert c["state"] == "completed"
    assert c["last_logged_metrics"]["loss"] == 0.4
    assert c["rows_after_end"] == 1  # the row stamped now, after the end
    assert c["n_epochs"] == 2 and c["n_rows"] == 2


def test_compare_experiments_stale_heartbeat_compares_interrupted(monkeypatch):
    """A run with no final status reads running while its heartbeat is fresh, with no
    rows-after-end count since it has not ended, and interrupted once it goes stale."""
    from tcip_mcp import experiments
    from tcip_mcp.experiments import compare_experiments

    _opened("exp-live")

    (live,) = compare_experiments(["exp-live"])["experiments"]
    assert (live["state"], live["rows_after_end"]) == ("running", None)

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    (stale,) = compare_experiments(["exp-live"])["experiments"]
    assert stale["state"] == "interrupted"


def test_compare_experiments_rows_after_end_compares_instants_across_offsets():
    """rows_after_end parses each row's timestamp (and the final status's ``ended``) as an
    instant and compares strictly-after, so a row stamped in a different UTC offset, or a bare
    "Z", still compares on the instant it actually names rather than on its ISO text; a row
    whose timestamp isn't a parseable string (a bespoke loop's own integer counter, say) is
    skipped, not raised on."""
    from datetime import datetime, timedelta, timezone

    from tcip_mcp.experiments import METRICS_FILE, append_row, compare_experiments, observe
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(None, experiment_id="exp-instants")
    final = observe(run_dir).final
    assert final is not None
    ended = datetime.fromisoformat(final["ended"])

    def at(offset_s: float, zone: timezone) -> str:
        return (ended + timedelta(seconds=offset_s)).astimezone(zone).isoformat()

    rows = [
        {"epoch": 1, "timestamp": at(1, timezone(timedelta(hours=-6))), "loss": 0.1},  # after
        {"epoch": 2, "timestamp": at(-1, timezone(timedelta(hours=5))), "loss": 0.2},  # before
        {"epoch": 3, "timestamp": at(-1, timezone.utc).replace("+00:00", "Z"), "loss": 0.3},
        {"epoch": 4, "timestamp": at(0, timezone.utc), "loss": 0.4},  # same instant: not after
        {"epoch": 5, "timestamp": at(1, timezone.utc), "loss": 0.5},  # after
        {"epoch": 6, "timestamp": 1704110401, "loss": 0.6},           # not a string: skipped
    ]
    for row in rows:
        append_row(run_dir / METRICS_FILE, row)

    assert compare_experiments(["exp-instants"])["experiments"][0]["rows_after_end"] == 2
