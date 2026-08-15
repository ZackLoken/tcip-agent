"""The metrics rows a run writes are the rows the Training route serves.

Both sides here are the real implementations: ``log_metrics`` appends to the run's own log and
the web route resolves that same log and reads it back, so a change to the row shape, to where
the log lives, or to which record a run id resolves to shows up as a disagreement instead of
passing against a hand-built file.
"""

from __future__ import annotations


def test_logged_rows_reach_the_training_route_reader_with_their_epoch_and_values(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics
    from tcip_web.routes.training import get_run_metrics

    run_id = "exp-021-hazelnut-catkin-det"
    create_experiment(run_id, {"model_source": {"builder": "my_models:catkin_det"}})
    log_metrics(run_id, 3, {"loss": 0.94, "val_map50": 0.28})
    log_metrics(run_id, 7, {"loss": 0.31, "val_map50": 0.66})

    body = get_run_metrics(str(tmp_path), run_id)
    assert body["exists"] is True
    rows = body["metrics"]
    assert len(rows) == 2
    assert [r.get("epoch") for r in rows] == [3, 7]
    assert [r.get("val_map50") for r in rows] == [0.28, 0.66]
    assert rows[-1].get("loss") == 0.31
    assert all(r.get("timestamp") for r in rows)


def test_training_route_serves_a_relaunched_run_from_the_record_that_claims_it(tmp_path):
    """A relaunch's experiment id is not its run id, so the route resolves the record.

    ``_ensure_experiment`` mints ``<id>_<run_id>`` when an id already has a run, and the rows
    then live under that minted id. Serving by run id alone would read an id nothing writes.
    """
    from tcip_mcp.experiments import create_experiment, log_metrics, stamp_run_identity
    from tcip_web.routes.training import get_run_metrics

    run_id = "run-20260401-abcdef"
    experiment_id = f"exp-022-chestnut-burr-det_{run_id}"
    create_experiment(experiment_id, {"model_source": {"builder": "my_models:burr_det"}})
    stamp_run_identity(experiment_id, run_id, str(tmp_path / "out"))
    log_metrics(experiment_id, 1, {"loss": 1.2})

    body = get_run_metrics(str(tmp_path), run_id)
    assert body["exists"] is True
    assert [r.get("loss") for r in body["metrics"]] == [1.2]


def test_training_route_serves_nothing_for_a_run_no_record_claims(tmp_path):
    from tcip_web.routes.training import get_run_metrics

    body = get_run_metrics(str(tmp_path), "never-launched")
    assert body == {"metrics": [], "exists": False}
