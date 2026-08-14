"""The metrics rows a run writes are the rows the Training route serves.

Both sides here are the real implementations: ``log_metrics`` writes the file and the web
route's own path helper plus its shared reader read it back, so a change to the row shape or to
where the store lives shows up as a disagreement instead of passing against a hand-built file.
"""

from __future__ import annotations


def test_logged_rows_reach_the_training_route_reader_with_their_epoch_and_values(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics
    from tcip_web.routes._metrics_common import read_metrics_file
    from tcip_web.routes.training import _metrics_path

    run_id = "exp-021-hazelnut-catkin-det"
    create_experiment(run_id, {"model_source": {"builder": "my_models:catkin_det"}})
    log_metrics(run_id, 3, {"loss": 0.94, "val_map50": 0.28})
    log_metrics(run_id, 7, {"loss": 0.31, "val_map50": 0.66})

    body = read_metrics_file(_metrics_path(str(tmp_path), run_id))
    assert body["exists"] is True
    rows = body["metrics"]
    assert len(rows) == 2
    assert [r.get("epoch") for r in rows] == [3, 7]
    assert [r.get("val_map50") for r in rows] == [0.28, 0.66]
    assert rows[-1].get("loss") == 0.31
    assert all(r.get("timestamp") for r in rows)


def test_training_route_reads_the_file_the_run_appends_to(tmp_path):
    from tcip_mcp.experiments import create_experiment, experiments_dir, log_metrics
    from tcip_web.routes.training import _metrics_path

    run_id = "exp-022-chestnut-burr-det"
    create_experiment(run_id, {"model_source": {"builder": "my_models:burr_det"}})
    log_metrics(run_id, 1, {"loss": 1.2})

    written = experiments_dir() / run_id / "metrics.jsonl"
    assert written.is_file()
    assert _metrics_path(str(tmp_path), run_id) == written.resolve()
