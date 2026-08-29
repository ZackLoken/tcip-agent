"""Every reader of an experiment's own append-only logs reports a version-refused entry as
its own named fact, distinct from a corrupt one: the write side now refuses a document a
newer writer alone could have produced, so a row at this reader's ceiling reaching a log at
all means a source other than today's writer put it there, and that is worth its own log
line rather than a silent gap in the returned rows.
"""

from __future__ import annotations

import logging

import tcip_store as ts
from tcip_mcp.experiments import (
    create_experiment,
    metrics_key,
    read_metrics,
    read_validations,
    validations_key,
)
from tcip_store.file_backend import FileBackend


def _plant_poisoned_line(key: ts.Key, entry: dict) -> None:
    """One already-poisoned log line, written directly to disk: what a newer writer's row
    looks like to this reader, never reachable through this store's own append any more."""
    data = ts.get_descriptor(key.store).codec.encode(entry)
    with open(FileBackend().path_for(key), "ab") as handle:
        handle.write(data + b"\n")


def test_read_metrics_reports_version_refused_entries_separately_from_corrupt(tmp_path, caplog):
    ts.bind(FileBackend())
    try:
        experiment_id = "exp-metrics-version-refused"
        create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
        key = metrics_key(experiment_id)
        ts.append(key, {"epoch": 1, "loss": 0.5})
        _plant_poisoned_line(key, {"epoch": 2, "schema_version": 99})
        ts.append(key, {"epoch": 3, "loss": 0.1})

        with caplog.at_level(logging.WARNING):
            rows = read_metrics(experiment_id)
    finally:
        ts.unbind()
    assert [r["epoch"] for r in rows] == [1, 3]
    assert any("schema_version" in record.message for record in caplog.records)


def test_read_validations_reports_version_refused_entries_separately_from_corrupt(tmp_path, caplog):
    ts.bind(FileBackend())
    try:
        experiment_id = "exp-validations-version-refused"
        create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
        key = validations_key(experiment_id)
        ts.append(key, {"trait": "count", "document": "a", "verdict": "pass"})
        _plant_poisoned_line(key, {"trait": "count", "schema_version": 99})
        ts.append(key, {"trait": "count", "document": "b", "verdict": "pass"})

        with caplog.at_level(logging.WARNING):
            rows = read_validations(experiment_id)
    finally:
        ts.unbind()
    assert [r["document"] for r in rows] == ["a", "b"]
    assert any("schema_version" in record.message for record in caplog.records)
