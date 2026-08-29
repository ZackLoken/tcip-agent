"""A caller's free-form config or metrics dict must not smuggle a schema_version the
experiment stores' own writer does not produce: the seam's write side refuses it before a
byte lands, so a poisoned record can never make every later read of the workspace fail.
"""

from __future__ import annotations

import pytest

from tcip_mcp import experiments
from tcip_store import SchemaVersionRefused


def test_create_experiment_refuses_a_config_carrying_schema_version(tmp_path):
    experiment_id = "exp-config-schema-version"
    with pytest.raises(SchemaVersionRefused) as raised:
        experiments.create_experiment(
            experiment_id, {"schema_version": 2, "model_source": {"builder": "m:f"}}
        )
    message = str(raised.value)
    assert "schema_version" in message
    assert experiment_id in message
    assert not experiments.experiment_exists(experiment_id)


def test_create_experiment_still_creates_a_legitimate_config(tmp_path):
    experiment_id = "exp-config-legit"
    result = experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    assert "error" not in result, result
    assert experiments.experiment_exists(experiment_id)


def test_log_metrics_refuses_a_metric_named_schema_version(tmp_path):
    experiment_id = "exp-metrics-schema-version"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})

    with pytest.raises(SchemaVersionRefused) as raised:
        experiments.log_metrics(experiment_id, 1, {"schema_version": 2, "loss": 0.1})
    assert "schema_version" in str(raised.value)
    assert experiments.read_metrics(experiment_id) == []

    experiments.log_metrics(experiment_id, 1, {"loss": 0.1})
    assert [row["epoch"] for row in experiments.read_metrics(experiment_id)] == [1]
