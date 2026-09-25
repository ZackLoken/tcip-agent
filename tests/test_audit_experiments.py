"""Tests for audit logging and experiment tracking."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import tcip_store as ts


# ── Audit logging ──


class TestAuditLogging:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_audited_logs_success(self):
        from tcip_mcp.audit import audited

        with patch.object(
            __import__("tcip_mcp.audit", fromlist=["AUDIT_ROOT"]),
            "AUDIT_ROOT",
            self.tmpdir,
        ):
            # Re-import to get patched version
            import tcip_mcp.audit as audit_mod
            original = audit_mod.AUDIT_ROOT
            audit_mod.AUDIT_ROOT = self.tmpdir

            @audited
            def my_tool(x: int = 0) -> dict:
                return {"result": x + 1}

            result = my_tool(x=5)
            assert result == {"result": 6}

            # Check audit log, through the seam rather than the file backend's raw jsonl
            page = ts.read_log(audit_mod.audit_log_key())
            assert len(page.records) == 1
            entry = page.records[0]
            assert entry["tool"] == "my_tool"
            assert entry["status"] == "ok"
            assert entry["arguments"] == {"x": 5}
            assert "duration_ms" in entry
            assert "timestamp" in entry

            audit_mod.AUDIT_ROOT = original

    def test_audited_logs_exception(self):
        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        @audited
        def failing_tool() -> dict:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_tool()

        page = ts.read_log(audit_mod.audit_log_key())
        assert len(page.records) == 1
        entry = page.records[0]
        assert entry["status"] == "exception"
        assert "test error" in entry["error"]

        audit_mod.AUDIT_ROOT = original

    def test_redaction(self):
        from tcip_mcp.audit import _redact

        args = {"name": "test", "api_key": "secret123", "token": "tok123"}
        redacted = _redact(args)
        assert redacted["name"] == "test"
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["token"] == "***REDACTED***"

    # -- positional args are bound to their parameter names --------------

    def test_audited_binds_positional_args_to_names(self):
        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        @audited
        def my_tool(x: int, y: str = "default") -> dict:
            return {"result": x}

        my_tool(5, "explicit")  # positional, the way the web routes call audited tools

        entry = ts.read_log(audit_mod.audit_log_key()).records[0]
        assert entry["arguments"] == {"x": 5, "y": "explicit"}

        audit_mod.AUDIT_ROOT = original

    def test_audited_positional_binding_fills_unstated_defaults(self):
        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        @audited
        def my_tool(x: int, y: str = "default") -> dict:
            return {"result": x}

        my_tool(5)  # positional, y left at its default

        entry = ts.read_log(audit_mod.audit_log_key()).records[0]
        assert entry["arguments"] == {"x": 5, "y": "default"}

        audit_mod.AUDIT_ROOT = original

    def test_audited_call_arity_error_still_logs_and_raises(self):
        """A real call-site bug (wrong arity) must still be logged before it propagates: the
        decorator's own exception handling isn't disturbed by the binding step."""
        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        @audited
        def my_tool(x: int) -> dict:
            return {"result": x}

        with pytest.raises(TypeError):
            my_tool(1, 2, 3)  # too many positional args: the real call itself fails, not just binding

        page = ts.read_log(audit_mod.audit_log_key())
        assert len(page.records) == 1
        entry = page.records[0]
        assert entry["status"] == "exception"

        audit_mod.AUDIT_ROOT = original

    def test_audited_binding_failure_falls_back_without_aborting_a_call_that_would_succeed(
        self, monkeypatch,
    ):
        """Isolates the sig.bind() failure from the underlying call: even when parameter binding
        itself raises (simulated here; for every real @audited tool the two happen to fail
        together, since none take *args/**kwargs), the real call must still run and be logged,
        just with a degraded (kwargs-only) argument record instead of aborting or losing the
        entry entirely."""
        import inspect

        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        @audited
        def my_tool(x: int, y: str = "default") -> dict:
            return {"result": x}

        def _boom(self, *a, **k):
            raise TypeError("synthetic binding failure")

        monkeypatch.setattr(inspect.Signature, "bind", _boom)

        result = my_tool(5, y="explicit")  # the real call must still succeed
        assert result == {"result": 5}

        entry = ts.read_log(audit_mod.audit_log_key()).records[0]
        assert entry["status"] == "ok"
        # Degraded fallback: the positional x is lost, y survives via kwargs.
        assert entry["arguments"] == {"y": "explicit"}

        audit_mod.AUDIT_ROOT = original


# ── Experiment tracking ──


class TestExperiments:
    def setup_method(self):
        import tcip_mcp.experiments as exp

        self.tmpdir = Path(tempfile.mkdtemp())
        self._experiments_dir = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

    def teardown_method(self):
        import tcip_mcp.experiments as exp

        # Restored here, not at each test's end, so a failing test cannot leave the next one
        # reading this test's experiments.
        exp.EXPERIMENTS_DIR = self._experiments_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_experiment(self):
        import tcip_mcp.experiments as exp

        result = exp.create_experiment("exp-001", {"model": "resnet50"})
        assert result["experiment_id"] == "exp-001"
        assert result["state"] == "created"

        # Every member document the record is made of exists, through the seam its own
        # readers use (backend-general: creation is a claim about the record, not the layout).
        assert ts.exists(exp.config_key("exp-001"))
        assert ts.exists(exp.status_key("exp-001"))
        assert ts.exists(exp.lineage_key("exp-001"))
        assert ts.exists(exp.artifacts_key("exp-001"))

    def test_create_duplicate_experiment(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-001", {"model": "resnet50"})
        result = exp.create_experiment("exp-001", {"model": "resnet50"})
        assert "error" in result

    def test_log_metrics(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-002", {})
        exp.log_metrics("exp-002", 0, {"loss": 1.5, "mAP50": 0.2})
        exp.log_metrics("exp-002", 1, {"loss": 0.8, "mAP50": 0.5})

        rows = exp.read_metrics("exp-002")
        assert len(rows) == 2
        assert rows[0]["epoch"] == 0
        assert rows[1]["mAP50"] == 0.5

    def test_update_status(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-003", {})
        exp.update_status("exp-003", "running")
        status = ts.read(exp.status_key("exp-003"))
        assert status["state"] == "running"
        assert status["started"] is not None

        exp.update_status("exp-003", "completed")
        status = ts.read(exp.status_key("exp-003"))
        assert status["state"] == "completed"
        assert status["ended"] is not None

    def test_record_artifact(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-004", {})
        exp.record_artifact("exp-004", "model_weights", "/path/to/model.pt")

        artifacts = ts.read(exp.artifacts_key("exp-004"))
        assert "model_weights" in artifacts
        assert artifacts["model_weights"]["path"] == "/path/to/model.pt"

    def test_get_experiment(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-005", {"backbone": "resnet50"})
        exp.log_metrics("exp-005", 0, {"loss": 1.0})

        result = exp.get_experiment("exp-005")
        assert result["experiment_id"] == "exp-005"
        assert result["config"]["backbone"] == "resnet50"
        assert result["n_epochs"] == 1
        assert len(result["metrics"]) == 1

    def test_get_experiment_not_found(self):
        import tcip_mcp.experiments as exp

        result = exp.get_experiment("nonexistent")
        assert "error" in result

    def test_list_experiments(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-a", {})
        exp.create_experiment("exp-b", {})
        exp.update_status("exp-a", "completed")

        listing = exp.list_experiments()
        assert len(listing) == 2
        names = {e["experiment_id"] for e in listing}
        assert names == {"exp-a", "exp-b"}

    def test_compare_experiments(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-x", {"model_source": {"builder": "my_models:resnet50_det",
                                                         "task": "detection"}})
        exp.create_experiment("exp-y", {"model_source": {"builder": "my_models:effb0_cls",
                                                         "task": "classification"}})
        exp.log_metrics("exp-x", 0, {"mAP50": 0.6})
        exp.log_metrics("exp-y", 0, {"mAP50": 0.7})

        result = exp.compare_experiments(["exp-x", "exp-y"], stale_seconds=600.0)
        assert result["count"] == 2
        exps = {e["experiment_id"]: e for e in result["experiments"]}
        assert exps["exp-x"]["model"] == "my_models:resnet50_det"
        assert exps["exp-y"]["last_logged_metrics"]["mAP50"] == 0.7

    # -- overwrite_config_if_pristine --------------------------

    def test_overwrite_config_if_pristine_rewrites_when_pristine(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-006", {"a": 1})
        result = exp.overwrite_config_if_pristine("exp-006", {"a": 2, "seed": 7})
        assert result["overwritten"] is True
        config = ts.read(exp.config_key("exp-006"))
        assert config == {"a": 2, "seed": 7}

    def test_overwrite_config_if_pristine_refuses_once_metrics_exist(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-007", {"a": 1})
        exp.log_metrics("exp-007", 0, {"loss": 1.0})
        result = exp.overwrite_config_if_pristine("exp-007", {"a": 2})
        assert "error" in result
        config = ts.read(exp.config_key("exp-007"))
        assert config == {"a": 1}  # untouched

    def test_overwrite_config_if_pristine_refuses_when_terminal(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-008", {"a": 1})
        exp.update_status("exp-008", "running")
        exp.update_status("exp-008", "completed")
        result = exp.overwrite_config_if_pristine("exp-008", {"a": 2})
        assert "error" in result
        config = ts.read(exp.config_key("exp-008"))
        assert config == {"a": 1}

    def test_log_metrics_stamps_the_status_record_before_its_append(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-009", {"a": 1})
        assert "metrics_logged" not in ts.read(exp.status_key("exp-009"))

        exp.log_metrics("exp-009", 0, {"loss": 1.0})
        assert ts.read(exp.status_key("exp-009"))["metrics_logged"] is True

    def test_overwrite_config_if_pristine_reads_the_marker_not_the_log(self):
        """The predicate now decides pristineness from the status record's own field, not by
        re-scanning the log: an experiment whose marker is set (with no rows at all, a state the
        real log_metrics can never produce alone, manufactured here to isolate what the
        predicate actually reads) still refuses."""
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-010", {"a": 1})
        assert exp.read_metrics("exp-010") == []
        key = exp.status_key("exp-010")
        with ts.transaction(key) as txn:
            status = txn.read(key, default={})
            status["metrics_logged"] = True
            txn.write(key, status)

        result = exp.overwrite_config_if_pristine("exp-010", {"a": 2})
        assert "error" in result
        config = ts.read(exp.config_key("exp-010"))
        assert config == {"a": 1}

    def test_get_experiment_lineage(self):
        import tcip_mcp.experiments as exp

        exp.create_experiment("exp-l", {"data": {"images_dir": "/data/images", "task": "detection"}},
                             data_source="/data/images")
        exp.update_lineage("exp-l", predictions="/preds/best")

        result = exp.get_experiment_lineage("exp-l")
        assert result["lineage"]["data_source"] == "/data/images"
        assert result["lineage"]["predictions"] == "/preds/best"
        assert result["lineage"]["data_config"]["task"] == "detection"


# ── model registry replace-by-name is audited ──


class TestModelRegistryReplaceAudit:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ckpt(self, name: str, content: bytes) -> str:
        p = self.tmpdir / name
        p.write_bytes(content)
        return str(p)

    def _rows(self) -> list[dict]:
        import tcip_mcp.audit as audit_mod

        return list(ts.read_log(audit_mod.audit_log_key()).records)

    def test_a_first_registration_and_a_replacement_each_leave_one_row(self):
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        reg.register_model("exp1", self._ckpt("a.pt", b"first"), {}, metrics_source=None)
        first_sha = reg.get_model("exp1")["sha256"]
        (first,) = self._rows()
        assert first["tool"] == "model_registered"
        assert first["arguments"] == {"name": "exp1", "new_sha256": first_sha, "experiment_id": None}

        reg.register_model("exp1", self._ckpt("b.pt", b"second, different"), {}, metrics_source=None)
        second_sha = reg.get_model("exp1")["sha256"]
        assert first_sha != second_sha
        _, replaced = self._rows()
        assert replaced["tool"] == "model_registered"
        assert replaced["arguments"]["name"] == "exp1"
        assert replaced["arguments"]["superseded_sha256"] == first_sha
        assert replaced["arguments"]["new_sha256"] == second_sha

        audit_mod.AUDIT_ROOT = original

    def test_the_register_model_door_leaves_only_the_registrys_own_rows(self):
        """Through the door: one row per registry write that changed content, the registry's,
        and none for the door on top of it or for an idempotent re-registration."""
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.tools.model_tools import register_model

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        first = self._ckpt("a.pt", b"first")
        for path in (first, self._ckpt("b.pt", b"second, different"), self._ckpt("c.pt", b"first")):
            assert "error" not in register_model(name="door", checkpoint_path=path, config={},
                                                 project_path=str(self.tmpdir))

        rows = self._rows()
        assert [r["tool"] for r in rows] == ["model_registered", "model_registered", "model_registered"]
        assert ["superseded_sha256" in r["arguments"] for r in rows] == [False, True, True]
        assert "error" not in register_model(name="door", checkpoint_path=str(self.tmpdir / "c.pt"),
                                             config={}, project_path=str(self.tmpdir))
        assert len(self._rows()) == 3  # the same entry re-registered: nothing changed, no row

        audit_mod.AUDIT_ROOT = original

    def test_reregistering_an_identical_entry_changes_nothing_and_leaves_no_row(self):
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry, registry_index_key

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        ckpt = self._ckpt("a.pt", b"same bytes")
        reg.register_model("exp1", ckpt, {}, metrics_source=None)
        before = ts.read_versioned(registry_index_key(self.tmpdir))
        reg.register_model("exp1", ckpt, {}, metrics_source=None)

        after = ts.read_versioned(registry_index_key(self.tmpdir))
        assert (after.value, after.version) == (before.value, before.version)
        assert [e["tool"] for e in self._rows()] == ["model_registered"]

        audit_mod.AUDIT_ROOT = original

    def test_the_same_weights_under_new_tags_change_the_entry_and_leave_one_row(self):
        """A write is decided by the entry it would store, never by the weights' digest alone:
        the same checkpoint re-registered under new tags changes the entry and leaves its line."""
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        ckpt = self._ckpt("a.pt", b"same bytes")
        reg.register_model("exp1", ckpt, {}, metrics_source=None)
        reg.register_model("exp1", ckpt, {}, tags=["chestnut"], metrics_source=None)

        assert ModelRegistry(str(self.tmpdir)).get_model("exp1")["tags"] == ["chestnut"]
        assert [e["tool"] for e in self._rows()] == ["model_registered", "model_registered"]

        audit_mod.AUDIT_ROOT = original

    def test_replace_raises_and_stays_committed_when_its_audit_line_fails(self, monkeypatch):
        """The transaction has already replaced the entry by the time the audit line is
        attempted, so a failed append must not be swallowed: the caller is told through
        AuditEntryNotWritten, and the registry keeps the replace regardless."""
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        reg.register_model("exp1", self._ckpt("a.pt", b"first"), {}, metrics_source=None)
        second_ckpt = self._ckpt("b.pt", b"second, different")

        def _refuse(*args, **kwargs):
            raise RuntimeError("the audit log could not be appended to")

        monkeypatch.setattr(audit_mod, "append", _refuse)

        with pytest.raises(audit_mod.AuditEntryNotWritten) as caught:
            reg.register_model("exp1", second_ckpt, {}, metrics_source=None)

        assert caught.value.tool == "model_registered"
        reloaded = ModelRegistry(str(self.tmpdir)).get_model("exp1")
        assert reloaded["file_size_bytes"] == len(b"second, different")

        audit_mod.AUDIT_ROOT = original
