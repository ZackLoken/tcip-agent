"""Tests for audit logging and experiment tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts

from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run


# ── Audit logging ──


def _rows(project: Path) -> list[dict]:
    import tcip_mcp.audit as audit_mod

    return list(ts.read_log(audit_mod.audit_log_key(project)).records)


def test_audited_logs_success(tmp_path):
    from tcip_mcp.audit import audited

    @audited
    def my_tool(project: Path, x: int = 0) -> dict:
        return {"result": x + 1}

    assert my_tool(tmp_path, x=5) == {"result": 6}

    (entry,) = _rows(tmp_path)
    assert entry["tool"] == "my_tool"
    assert entry["status"] == "ok"
    assert entry["arguments"] == {"x": 5}
    assert "duration_ms" in entry
    assert "timestamp" in entry


def test_audited_logs_exception(tmp_path):
    from tcip_mcp.audit import audited

    @audited
    def failing_tool(project: Path) -> dict:
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_tool(tmp_path)

    (entry,) = _rows(tmp_path)
    assert entry["status"] == "exception"
    assert "test error" in entry["error"]


def test_redaction():
    from tcip_mcp.audit import _redact

    args = {"name": "test", "api_key": "secret123", "token": "tok123"}
    redacted = _redact(args)
    assert redacted["name"] == "test"
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["token"] == "***REDACTED***"


# -- positional args are bound to their parameter names --------------


def test_audited_binds_positional_args_to_names(tmp_path):
    from tcip_mcp.audit import audited

    @audited
    def my_tool(project: Path, x: int, y: str = "default") -> dict:
        return {"result": x}

    my_tool(tmp_path, 5, "explicit")  # positional, the way the web routes call audited tools

    assert _rows(tmp_path)[0]["arguments"] == {"x": 5, "y": "explicit"}


def test_audited_positional_binding_fills_unstated_defaults(tmp_path):
    from tcip_mcp.audit import audited

    @audited
    def my_tool(project: Path, x: int, y: str = "default") -> dict:
        return {"result": x}

    my_tool(tmp_path, 5)  # positional, y left at its default

    assert _rows(tmp_path)[0]["arguments"] == {"x": 5, "y": "default"}


def test_audited_call_that_does_not_bind_raises_before_the_body_runs(tmp_path):
    """A call-site bug (wrong arity) raises the binding's own TypeError: without a bound
    ``project`` there is no log to record it in, and the body never runs."""
    from tcip_mcp.audit import audited

    ran: list[int] = []

    @audited
    def my_tool(project: Path, x: int) -> dict:
        ran.append(x)
        return {"result": x}

    with pytest.raises(TypeError):
        my_tool(tmp_path, 1, 2, 3)  # type: ignore[call-arg]

    assert ran == []
    assert _rows(tmp_path) == []


# ── Experiment tracking ──


def test_get_experiment_reads_the_run_directory(tmp_path):
    import tcip_mcp.experiments as exp

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "ds", backbone="resnet50"),
                         experiment_id="exp-005")
    log_epoch(run_dir, 0, {"loss": 1.0})

    result = exp.get_experiment("exp-005", project=tmp_path)
    assert result["experiment_id"] == "exp-005"
    assert result["run"]["config"]["backbone"] == "resnet50"
    assert result["final_status"] is None
    assert result["n_epochs"] == 1
    assert len(result["metrics"]) == 1


def test_get_experiment_not_found(tmp_path):
    import tcip_mcp.experiments as exp

    assert "error" in exp.get_experiment("nonexistent", project=tmp_path)


def test_list_experiments(tmp_path):
    import tcip_mcp.experiments as exp

    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-a")
    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-b")

    listing = exp.run_rows(tmp_path)
    assert {e.experiment_id for e in listing} == {"exp-a", "exp-b"}
    assert {e.state for e in listing} == {"running"}


def test_compare_experiments(tmp_path):
    import tcip_mcp.experiments as exp

    x = opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-x")
    y = opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-y")
    log_epoch(x, 0, {"mAP50": 0.6})
    log_epoch(y, 0, {"mAP50": 0.7})

    result = exp.compare_experiments(["exp-x", "exp-y"], project=tmp_path)
    assert result["count"] == 2
    exps = {e["experiment_id"]: e for e in result["experiments"]}
    assert exps["exp-x"]["model"] == "tests.bespoke_models:build_bespoke_detection"
    assert exps["exp-y"]["last_logged_metrics"]["mAP50"] == 0.7


def test_get_experiment_lineage(tmp_path):
    import tcip_mcp.experiments as exp

    config = detection_config(tmp_path / "ds")
    opened_run(tmp_path, config, experiment_id="exp-l", relaunched_from="exp-k")

    lineage = exp.get_experiment_lineage("exp-l", project=tmp_path)["lineage"]
    assert lineage["data"]["images_dir"] == config["data"]["images_dir"]
    assert lineage["data"]["scope"]["subject"] == config["data"]["scope"]["subject"]
    assert lineage["relaunched_from"] == "exp-k"
    assert lineage["checkpoint"] is None


# ── a foreign registration is keyed by its bytes and audited ──


def _entry(project: Path, sha256: str) -> dict:
    """The one foreign entry ``project``'s registry holds for ``sha256``."""
    from tcip_mcp.model_registry import registered_entries

    [entry] = [e for e in registered_entries(project) if e["sha256"] == sha256]
    return entry


def _ckpt(project: Path, name: str, content: bytes) -> str:
    from tests._verified_checkpoint_fixtures import checkpoint_file

    return str(checkpoint_file(project / name, content.decode()))


def test_a_first_registration_and_a_replacement_each_leave_one_row(tmp_path):
    """The same bytes registered again under another name replace their one entry, the row
    naming the name it superseded; other bytes under the first name are an entry of their
    own."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    first_sha = reg.register_model("exp1", _ckpt(tmp_path, "a.pt", b"first"), {})["sha256"]
    (first,) = _rows(tmp_path)
    assert first["tool"] == "model_registered"
    assert first["arguments"] == {"name": "exp1", "new_sha256": first_sha}

    reg.register_model("exp2", _ckpt(tmp_path, "a.pt", b"first"), {})
    _, replaced = _rows(tmp_path)
    assert replaced["arguments"] == {"name": "exp2", "new_sha256": first_sha,
                                     "superseded_name": "exp1", "superseded_tags": []}
    assert _entry(tmp_path, first_sha)["name"] == "exp2"

    second_sha = reg.register_model("exp1", _ckpt(tmp_path, "b.pt", b"second, different"),
                                    {})["sha256"]
    assert second_sha != first_sha
    assert "superseded_name" not in _rows(tmp_path)[-1]["arguments"]
    assert _entry(tmp_path, first_sha)["name"] == "exp2"


def test_the_register_model_door_leaves_only_the_registrys_own_rows(tmp_path):
    """Through the door: one row per registry write that changed content, the registry's,
    and none for the door on top of it or for an idempotent re-registration."""
    from tcip_mcp.tools.model_tools import register_model

    first = _ckpt(tmp_path, "a.pt", b"first")
    for path in (first, _ckpt(tmp_path, "b.pt", b"second, different"),
                 _ckpt(tmp_path, "c.pt", b"first")):
        assert "error" not in register_model(tmp_path, name="door", checkpoint_path=path,
                                             config={})

    rows = _rows(tmp_path)
    assert [r["tool"] for r in rows] == ["model_registered", "model_registered", "model_registered"]
    assert ["superseded_name" in r["arguments"] for r in rows] == [False, False, True]
    assert "error" not in register_model(tmp_path, name="door",
                                         checkpoint_path=str(tmp_path / "c.pt"), config={})
    assert len(_rows(tmp_path)) == 3  # the same entry re-registered: nothing changed, no row


def test_reregistering_an_identical_entry_changes_nothing_and_leaves_no_row(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry, registry_index_key

    reg = ModelRegistry(str(tmp_path))
    ckpt = _ckpt(tmp_path, "a.pt", b"same bytes")
    reg.register_model("exp1", ckpt, {})
    before = ts.read_versioned(registry_index_key(tmp_path))
    reg.register_model("exp1", ckpt, {})

    after = ts.read_versioned(registry_index_key(tmp_path))
    assert (after.value, after.version) == (before.value, before.version)
    assert [e["tool"] for e in _rows(tmp_path)] == ["model_registered"]


def test_the_same_weights_under_new_tags_change_the_entry_and_leave_one_row(tmp_path):
    """A write is decided by the entry it would store, never by the weights' digest alone:
    the same checkpoint re-registered under new tags changes the entry and leaves its line."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = _ckpt(tmp_path, "a.pt", b"same bytes")
    sha = reg.register_model("exp1", ckpt, {})["sha256"]
    reg.register_model("exp1", ckpt, {}, tags=["chestnut"])

    assert _entry(tmp_path, sha)["tags"] == ["chestnut"]
    assert [e["tool"] for e in _rows(tmp_path)] == ["model_registered", "model_registered"]


def test_replace_raises_and_stays_committed_when_its_audit_line_fails(tmp_path, monkeypatch):
    """The transaction has already replaced the entry by the time the audit line is
    attempted, so a failed append must not be swallowed: the caller is told through
    AuditEntryNotWritten, and the registry keeps the replace regardless."""
    import tcip_mcp.audit as audit_mod
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    sha = reg.register_model("exp1", _ckpt(tmp_path, "a.pt", b"first"), {})["sha256"]

    def _refuse(*args, **kwargs):
        raise RuntimeError("the audit log could not be appended to")

    monkeypatch.setattr(audit_mod, "append", _refuse)

    with pytest.raises(audit_mod.AuditEntryNotWritten) as caught:
        reg.register_model("exp2", _ckpt(tmp_path, "a.pt", b"first"), {})

    assert caught.value.tool == "model_registered"
    assert _entry(tmp_path, sha)["name"] == "exp2"
