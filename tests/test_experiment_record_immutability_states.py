"""A finished run's record is immutable whichever way it finished.

Both terminal outcomes are locked, the run that completed and the run that failed: neither can
be re-opened to a non-terminal state, gain new epochs, have a recorded artifact pointer or a
populated lineage edge rewritten, and every refusal lands on the append-only audit log. The lock
stays additive, so a still-empty artifact name or lineage field takes its first write even after
the run finished. A cancelled run is the separate case the lock deliberately leaves out: its
record stays writable and re-openable.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key

_MEMBER_KEY_OF = {"status.json": exp.status_key, "artifacts.json": exp.artifacts_key,
                  "lineage.json": exp.lineage_key, "config.json": exp.config_key}


def _record(root: Path, experiment_id: str, name: str) -> dict:
    return ts.read(_MEMBER_KEY_OF[name](experiment_id, root=root))


def _metric_rows(root: Path, experiment_id: str) -> list[dict]:
    return list(ts.read_log(exp.metrics_key(experiment_id, root=root)).records)


def _audit_refusals(root: Path) -> list[dict]:
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "experiment_mutation_refused"]


def test_failed_run_cannot_be_reopened_to_a_non_terminal_state(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status

    eid = "exp-014-currant-bud-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:bud_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="out of memory at epoch 4")

    res = update_status(eid, "running")
    assert "error" in res
    assert res["state"] == "failed"

    status = _record(tmp_path, eid, "status.json")
    assert status["state"] == "failed"
    assert status["error"] == "out of memory at epoch 4"


def test_failed_run_metric_history_takes_no_further_epochs(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status

    eid = "exp-015-chestnut-burr-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:burr_det"}})
    update_status(eid, "running")
    log_metrics(eid, 3, {"val_map50": 0.41})
    log_metrics(eid, 4, {"val_map50": 0.47})
    update_status(eid, "failed", error="loss went to nan")

    res = log_metrics(eid, 5, {"val_map50": 0.99})
    assert "error" in res

    rows = _metric_rows(tmp_path, eid)
    assert [r["epoch"] for r in rows] == [3, 4]
    assert rows[-1]["val_map50"] == 0.47


def test_failed_run_artifact_pointer_is_frozen_while_a_new_name_still_records(tmp_path):
    from tcip_mcp.experiments import create_experiment, record_artifact, update_status

    eid = "exp-016-currant-cluster-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    record_artifact(eid, "model_final", "/runs/016/model_final.pt")
    update_status(eid, "failed", error="worker process died")

    added = record_artifact(eid, "failure_log", "/runs/016/stderr.txt")
    assert added["artifact"] == "failure_log"

    refused = record_artifact(eid, "model_final", "/runs/017/model_final.pt")
    assert "error" in refused

    artifacts = _record(tmp_path, eid, "artifacts.json")
    assert artifacts["model_final"]["path"] == "/runs/016/model_final.pt"
    assert artifacts["failure_log"]["path"] == "/runs/016/stderr.txt"


def test_failed_run_populated_lineage_edge_is_frozen_while_an_empty_one_accepts_its_first_write(
    tmp_path,
):
    """The additive lock, driven through the two fields ``update_lineage`` still admits
    (``model_weights``/``model_weights_sha256`` are ``complete_run``'s alone, refused
    unconditionally, so they cannot demonstrate the additive rule any more)."""
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    eid = "exp-017-elderberry-umbel-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:umbel_det"}})
    update_status(eid, "running")
    update_lineage(eid, predictions="/preds/017")
    update_status(eid, "failed", error="dataloader raised")

    update_lineage(eid, data_source="/data/017")
    update_lineage(eid, predictions="/preds/018")

    lineage = _record(tmp_path, eid, "lineage.json")
    assert lineage["predictions"] == "/preds/017"
    assert lineage["data_source"] == "/data/017"


def test_refused_mutations_on_a_failed_run_are_recorded_on_the_audit_log(tmp_path):
    from tcip_mcp.experiments import (
        create_experiment,
        log_metrics,
        record_artifact,
        update_status,
    )

    eid = "exp-018-persimmon-fruit-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:fruit_det"}})
    update_status(eid, "running")
    log_metrics(eid, 2, {"val_map50": 0.33})
    record_artifact(eid, "model_final", "/runs/018/model_final.pt")
    update_status(eid, "failed", error="killed by the wall-clock watcher")

    log_metrics(eid, 3, {"val_map50": 0.90})
    record_artifact(eid, "model_final", "/runs/019/model_final.pt")
    update_status(eid, "running")

    refusals = _audit_refusals(tmp_path)
    assert len(refusals) == 3
    assert {e["arguments"]["op"] for e in refusals} == {
        "log_metrics", "record_artifact", "update_status",
    }
    assert {e["arguments"]["experiment_id"] for e in refusals} == {eid}
    assert {e["status"] for e in refusals} == {"refused"}


def test_completed_run_refuses_a_move_to_failed_and_audits_both_states(tmp_path):
    """The lock refuses a move between the two terminal states, not only a reopen to a
    non-terminal one: a completed record stays completed and the refusal names both states."""
    from tcip_mcp.experiments import create_experiment, update_status

    eid = "exp-030-chestnut-burr-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:burr_det"}})
    update_status(eid, "running")
    update_status(eid, "completed")

    res = update_status(eid, "failed", error="loss diverged")
    assert "error" in res
    assert res["state"] == "completed"

    status = _record(tmp_path, eid, "status.json")
    assert status["state"] == "completed"

    refusals = _audit_refusals(tmp_path)
    assert len(refusals) == 1
    assert refusals[0]["arguments"] == {"experiment_id": eid, "op": "update_status",
                                        "from": "completed", "to": "failed"}


def test_repeat_of_failed_records_a_reason_onto_a_reasonless_record_with_no_restamp(tmp_path, monkeypatch):
    """A repeat of the current terminal state is idempotent: it applies error only when the
    record holds none, restamps nothing else (proved against a clock frozen to a date far past
    the first write, so any restamp would be unmistakable), and audits nothing (not a refusal)."""
    from datetime import datetime as real_datetime

    import tcip_mcp.experiments as exp_mod
    from tcip_mcp.experiments import create_experiment, update_status

    eid = "exp-031-currant-cluster-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    update_status(eid, "failed")  # the child's own reasonless failure

    ended_before = _record(tmp_path, eid, "status.json")["ended"]

    class _FrozenLaterClock:
        @staticmethod
        def now(tz=None):
            return real_datetime(2099, 1, 1, tzinfo=tz)

    monkeypatch.setattr(exp_mod, "datetime", _FrozenLaterClock)

    res = update_status(eid, "failed", error="exceeded max_wall_clock_seconds (5)")
    assert "error" not in res

    status = _record(tmp_path, eid, "status.json")
    assert status["error"] == "exceeded max_wall_clock_seconds (5)"
    assert status["ended"] == ended_before  # not restamped to the frozen future
    assert _audit_refusals(tmp_path) == []


def test_repeat_of_the_current_terminal_state_with_an_existing_reason_changes_nothing(tmp_path):
    """A second repeat, once a reason is already recorded, changes nothing at all: the first
    reason is never overwritten by a later reasonless (or differently reasoned) repeat."""
    from tcip_mcp.experiments import create_experiment, update_status

    eid = "exp-032-elderberry-umbel-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:umbel_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="loss went to nan")

    res = update_status(eid, "failed", error="a different, later reason")
    assert "error" not in res

    status = _record(tmp_path, eid, "status.json")
    assert status["error"] == "loss went to nan"
    assert _audit_refusals(tmp_path) == []


def test_update_status_refusal_raises_when_the_audit_append_itself_fails(tmp_path, monkeypatch):
    """The repo's rule for every other refusal: an audit line that cannot be appended raises
    rather than vanishing, since the record already reflects the refusal's own outcome (it
    stayed terminal) by the time the append is attempted."""
    import pytest
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.experiments import create_experiment, update_status

    eid = "exp-033-persimmon-fruit-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:fruit_det"}})
    update_status(eid, "running")
    update_status(eid, "completed")

    import tcip_mcp.audit as audit_mod

    def _refuse_append(*args, **kwargs):
        raise RuntimeError("the audit log could not be appended to")

    monkeypatch.setattr(audit_mod, "append", _refuse_append)

    with pytest.raises(AuditEntryNotWritten):
        update_status(eid, "failed")


def test_cancelled_run_record_stays_writable_and_reopenable(tmp_path):
    """A cancelled run is not locked the way a completed or failed one is: it stopped on request
    rather than finishing, so its record must still take the epochs and the state a resumed run
    records against it."""
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status

    eid = "exp-019-black_locust-raceme-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:raceme_det"}})
    update_status(eid, "running")
    log_metrics(eid, 6, {"val_map50": 0.52})
    update_status(eid, "cancelled")

    appended = log_metrics(eid, 7, {"val_map50": 0.58})
    assert "error" not in appended
    assert appended.get("logged") is True

    reopened = update_status(eid, "running")
    assert "error" not in reopened
    assert reopened["state"] == "running"

    rows = _metric_rows(tmp_path, eid)
    assert [r["epoch"] for r in rows] == [6, 7]
    assert _record(tmp_path, eid, "status.json")["state"] == "running"
    assert _audit_refusals(tmp_path) == []


def test_pointer_frozen_admits_an_absent_or_running_pointer(tmp_path):
    """A rail must admit valid work: pointer_frozen answers None (write admitted) both for a
    field that has never been written and for a populated field on a still-running record."""
    from tcip_mcp.experiments import create_experiment, record_artifact, update_status, pointer_frozen

    eid = "exp-020-chestnut-burr-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:burr_det"}})
    update_status(eid, "running")

    assert pointer_frozen(eid, "artifacts", "model_final", "/runs/020/model_final.pt") is None

    record_artifact(eid, "model_final", "/runs/020/model_final.pt")
    assert pointer_frozen(eid, "artifacts", "model_final", "/runs/020/model_final.pt") is None


def test_pointer_frozen_names_a_populated_pointer_on_a_terminal_record(tmp_path):
    """The same predicate record_artifact's own transactional refusal uses, read standalone: a
    populated artifact on a terminal record answers with the refusal text, not None."""
    from tcip_mcp.experiments import create_experiment, record_artifact, update_status, pointer_frozen

    eid = "exp-021-currant-cluster-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    record_artifact(eid, "model_final", "/runs/021/model_final.pt")
    update_status(eid, "completed")

    frozen = pointer_frozen(eid, "artifacts", "model_final", "/runs/021/model_final_v2.pt")
    assert frozen is not None and eid in frozen and "model_final" in frozen


def test_complete_run_writes_the_pointer_and_completes_in_one_call(tmp_path):
    import hashlib

    from tcip_mcp.experiments import create_experiment, update_status, complete_run

    eid = "exp-022-elderberry-umbel-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:umbel_det"}})
    update_status(eid, "running")

    weights = tmp_path / "runs" / "022" / "model_best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"022 weights")

    result = complete_run(eid, str(weights))
    assert "error" not in result
    assert result["state"] == "completed"
    assert result["model_weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()

    status = _record(tmp_path, eid, "status.json")
    assert status["state"] == "completed" and status.get("ended")
    artifacts = _record(tmp_path, eid, "artifacts.json")
    assert artifacts["model_weights"]["path"] == str(weights)
    assert artifacts["model_weights"]["sha256"] == result["model_weights_sha256"]
    lineage = _record(tmp_path, eid, "lineage.json")
    assert lineage["model_weights"] == str(weights)
    assert lineage["model_weights_sha256"] == result["model_weights_sha256"]


def test_complete_run_refuses_a_run_already_terminal_naming_the_weights_file(tmp_path):
    """The reachable refusal case: the wall-clock watchdog's failed landed while the child was
    inside its own finalize step. complete_run refuses, names the weights file on disk, and
    writes neither the pointer nor completed onto the already-failed record."""
    from tcip_mcp.experiments import create_experiment, update_status, complete_run

    eid = "exp-023-persimmon-fruit-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:fruit_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="killed by the wall-clock watcher")

    weights = tmp_path / "runs" / "023" / "model_best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"023 weights")

    result = complete_run(eid, str(weights))
    assert "error" in result
    assert repr(str(weights)) in result["error"]

    status = _record(tmp_path, eid, "status.json")
    assert status["state"] == "failed"
    artifacts = _record(tmp_path, eid, "artifacts.json")
    assert "model_weights" not in artifacts
    lineage = _record(tmp_path, eid, "lineage.json")
    assert lineage["model_weights_sha256"] is None


def test_complete_run_transaction_names_artifacts_then_lineage_then_status(tmp_path, monkeypatch):
    """Structural: a probe on the transaction's key order, not a behavior it produces. A
    file-backend transaction applies its writes in named-key order and is not crash-atomic
    across keys, so naming the pointer's key first, then the lineage digest, then status is what
    keeps a crash mid-write detectably stale rather than the reverse."""
    import tcip_mcp.experiments as exp

    eid = "exp-024-black_locust-raceme-det"
    exp.create_experiment(eid, {"model_source": {"builder": "my_models:raceme_det"}})
    exp.update_status(eid, "running")

    weights = tmp_path / "runs" / "024" / "model_best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"024 weights")

    seen: list[tuple] = []
    real_transaction = exp.store.transaction

    def spy(*keys, **kwargs):
        seen.append(keys)
        return real_transaction(*keys, **kwargs)

    monkeypatch.setattr(exp.store, "transaction", spy)
    exp.complete_run(eid, str(weights))

    assert seen, "complete_run never opened a transaction"
    named = seen[0]
    assert named[0].parts[1] == "artifacts"
    assert named[1].parts[1] == "lineage"
    assert named[2].parts[1] == "status"
