"""The registry's producer binding: a run's own completion records the digest it produced, and
nothing else can name a producer for weights the run did not record.

Rails from docs/audit/remediation/milestone-s/registry-producer-binding-design.md, numbered as
that note's section 4 numbers them. Rails 1, 2, 4, 5, 7, 10, 12-15, 17, 18 belong to the
registry entry's own experiment_id field and are not built yet.
"""

from __future__ import annotations

import pytest


# Rail 3: update_lineage(experiment_id, model_weights_sha256=...) and
# update_lineage(experiment_id, model_weights=...) raise naming complete_run.

def test_update_lineage_refuses_the_two_completion_fields(tmp_path, monkeypatch):
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_id = "exp-rail3"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, model_weights_sha256="deadbeef")
    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, model_weights="/some/path.pt")

    lineage = ts_read_lineage(tmp_path, exp_id)
    assert lineage["model_weights"] is None
    assert lineage["model_weights_sha256"] is None


def test_update_lineage_refuses_a_completion_field_even_beside_a_legitimate_one(tmp_path, monkeypatch):
    """The whole call refuses before any field lands: a legitimate companion in the same call
    does not land either, unlike the identity fields' pop-and-audit treatment."""
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_id = "exp-rail3b"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, predictions="/preds/should-not-land", model_weights="/w.pt")

    lineage = ts_read_lineage(tmp_path, exp_id)
    assert lineage["predictions"] is None


def ts_read_lineage(root, experiment_id: str) -> dict:
    from tcip_mcp.experiments import lineage_key, read_member

    return read_member(lineage_key(experiment_id, root=root), {})


# Rail 6 (coverage): a checkpoint whose payload stamps a real experiment's id but whose bytes are
# not the ones the run's completion recorded is reported producer-unknown by corroborated_producer.

def test_corroborated_producer_reports_unknown_when_stamped_bytes_disagree_with_completion(
    tmp_path, monkeypatch,
):
    from tcip_mcp.experiments import complete_run, create_experiment, update_status
    from tcip_mcp.model_registry import _sha256_of_bytes
    from tcip_mcp.pipelines.resolution import corroborated_producer

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_id = "exp-rail6"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    weights = tmp_path / "model_best.pt"
    weights.write_bytes(b"the real weights this run produced")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed

    forged_digest = _sha256_of_bytes(b"a different checkpoint entirely, not what this run wrote")
    assert forged_digest != completed["model_weights_sha256"]
    assert corroborated_producer(forged_digest, exp_id) == (None, None)


# Rail 9 (admits, the part built so far): update_lineage still records predictions, data_source
# and review_session.

def test_update_lineage_still_records_predictions_data_source_and_review_session(tmp_path, monkeypatch):
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_id = "exp-rail9"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    res = update_lineage(exp_id, predictions="/preds/rail9", data_source="/data/rail9",
                         review_session="session-1")
    assert res["lineage"]["predictions"] == "/preds/rail9"
    assert res["lineage"]["data_source"] == "/data/rail9"
    assert res["lineage"]["review_session"] == "session-1"


# Rail 11 (admits): a resumed run (the relaunch producing a fresh id) records and binds its own
# weights, and its delivery names the new run.

def test_resumed_run_records_and_binds_its_own_weights(tmp_path, monkeypatch):
    from tcip_mcp.experiments import complete_run, create_experiment, log_metrics, update_status
    from tcip_mcp.pipelines.resolution import corroborated_producer
    from tcip_mcp.tools.training_tools import _ensure_experiment

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    base = "exp-rail11-base"
    create_experiment(base, {"model_source": {"builder": "x:y"}})
    update_status(base, "running")
    log_metrics(base, 1, {"val_map50": 0.5})
    first_weights = tmp_path / "out1" / "model_best.pt"
    first_weights.parent.mkdir(parents=True, exist_ok=True)
    first_weights.write_bytes(b"the first run's own weights")
    first = complete_run(base, str(first_weights))
    assert "error" not in first, first

    resumed_id = _ensure_experiment(
        base, {"model_source": {"builder": "x:y"}}, None, str(first_weights), "run-2",
        output_dir=str(tmp_path / "out2"))
    assert resumed_id != base  # a non-pristine base mints a fresh id, per _ensure_experiment

    second_weights = tmp_path / "out2" / "model_best.pt"
    second_weights.parent.mkdir(parents=True, exist_ok=True)
    second_weights.write_bytes(b"the resumed run's own, different weights")
    second = complete_run(resumed_id, str(second_weights))
    assert "error" not in second, second
    assert second["model_weights_sha256"] != first["model_weights_sha256"]

    assert corroborated_producer(second["model_weights_sha256"], resumed_id) == (
        second["model_weights_sha256"], resumed_id)
    # The base run's own binding stands, unaffected by the resumed run's completion.
    assert corroborated_producer(first["model_weights_sha256"], base) == (
        first["model_weights_sha256"], base)
