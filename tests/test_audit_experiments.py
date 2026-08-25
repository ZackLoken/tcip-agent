"""Tests for audit logging and experiment tracking."""

from __future__ import annotations

import json
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
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_experiment(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        result = exp.create_experiment("exp-001", {"model": "resnet50"})
        assert result["experiment_id"] == "exp-001"
        assert result["state"] == "created"

        # Every member document the record is made of exists, through the seam its own
        # readers use (backend-general: creation is a claim about the record, not the layout).
        assert ts.exists(exp.config_key("exp-001"))
        assert ts.exists(exp.status_key("exp-001"))
        assert ts.exists(exp.lineage_key("exp-001"))
        assert ts.exists(exp.artifacts_key("exp-001"))

        exp.EXPERIMENTS_DIR = original

    def test_create_duplicate_experiment(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-001", {"model": "resnet50"})
        result = exp.create_experiment("exp-001", {"model": "resnet50"})
        assert "error" in result

        exp.EXPERIMENTS_DIR = original

    def test_log_metrics(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-002", {})
        exp.log_metrics("exp-002", 0, {"loss": 1.5, "mAP50": 0.2})
        exp.log_metrics("exp-002", 1, {"loss": 0.8, "mAP50": 0.5})

        rows = exp.read_metrics("exp-002")
        assert len(rows) == 2
        assert rows[0]["epoch"] == 0
        assert rows[1]["mAP50"] == 0.5

        exp.EXPERIMENTS_DIR = original

    def test_update_status(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-003", {})
        exp.update_status("exp-003", "running")
        status = ts.read(exp.status_key("exp-003"))
        assert status["state"] == "running"
        assert status["started"] is not None

        exp.update_status("exp-003", "completed")
        status = ts.read(exp.status_key("exp-003"))
        assert status["state"] == "completed"
        assert status["ended"] is not None

        exp.EXPERIMENTS_DIR = original

    def test_record_artifact(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-004", {})
        exp.record_artifact("exp-004", "model_weights", "/path/to/model.pt")

        artifacts = ts.read(exp.artifacts_key("exp-004"))
        assert "model_weights" in artifacts
        assert artifacts["model_weights"]["path"] == "/path/to/model.pt"

        exp.EXPERIMENTS_DIR = original

    def test_get_experiment(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-005", {"backbone": "resnet50"})
        exp.log_metrics("exp-005", 0, {"loss": 1.0})

        result = exp.get_experiment("exp-005")
        assert result["experiment_id"] == "exp-005"
        assert result["config"]["backbone"] == "resnet50"
        assert result["n_epochs"] == 1
        assert len(result["metrics"]) == 1

        exp.EXPERIMENTS_DIR = original

    def test_get_experiment_not_found(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        result = exp.get_experiment("nonexistent")
        assert "error" in result

        exp.EXPERIMENTS_DIR = original

    def test_list_experiments(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-a", {})
        exp.create_experiment("exp-b", {})
        exp.update_status("exp-a", "completed")

        listing = exp.list_experiments()
        assert len(listing) == 2
        names = {e["experiment_id"] for e in listing}
        assert names == {"exp-a", "exp-b"}

        exp.EXPERIMENTS_DIR = original

    def test_compare_experiments(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-x", {"model_source": {"builder": "my_models:resnet50_det"}})
        exp.create_experiment("exp-y", {"model_source": {"builder": "my_models:effb0_cls"}})
        exp.log_metrics("exp-x", 0, {"mAP50": 0.6})
        exp.log_metrics("exp-y", 0, {"mAP50": 0.7})

        result = exp.compare_experiments(["exp-x", "exp-y"])
        assert result["count"] == 2
        exps = {e["experiment_id"]: e for e in result["experiments"]}
        assert exps["exp-x"]["model"] == "my_models:resnet50_det"
        assert exps["exp-y"]["final_metrics"]["mAP50"] == 0.7

        exp.EXPERIMENTS_DIR = original

    # -- overwrite_config_if_pristine --------------------------

    def test_overwrite_config_if_pristine_rewrites_when_pristine(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-006", {"a": 1})
        result = exp.overwrite_config_if_pristine("exp-006", {"a": 2, "seed": 7})
        assert result["overwritten"] is True
        config = ts.read(exp.config_key("exp-006"))
        assert config == {"a": 2, "seed": 7}

        exp.EXPERIMENTS_DIR = original

    def test_overwrite_config_if_pristine_refuses_once_metrics_exist(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-007", {"a": 1})
        exp.log_metrics("exp-007", 0, {"loss": 1.0})
        result = exp.overwrite_config_if_pristine("exp-007", {"a": 2})
        assert "error" in result
        config = ts.read(exp.config_key("exp-007"))
        assert config == {"a": 1}  # untouched

        exp.EXPERIMENTS_DIR = original

    def test_overwrite_config_if_pristine_refuses_when_terminal(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-008", {"a": 1})
        exp.update_status("exp-008", "running")
        exp.update_status("exp-008", "completed")
        result = exp.overwrite_config_if_pristine("exp-008", {"a": 2})
        assert "error" in result
        config = ts.read(exp.config_key("exp-008"))
        assert config == {"a": 1}

        exp.EXPERIMENTS_DIR = original

    def test_log_metrics_stamps_the_status_record_before_its_append(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-009", {"a": 1})
        assert "metrics_logged" not in ts.read(exp.status_key("exp-009"))

        exp.log_metrics("exp-009", 0, {"loss": 1.0})
        assert ts.read(exp.status_key("exp-009"))["metrics_logged"] is True

        exp.EXPERIMENTS_DIR = original

    def test_overwrite_config_if_pristine_reads_the_marker_not_the_log(self):
        """The predicate now decides pristineness from the status record's own field, not by
        re-scanning the log: an experiment whose marker is set (with no rows at all, a state the
        real log_metrics can never produce alone, manufactured here to isolate what the
        predicate actually reads) still refuses."""
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

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

        exp.EXPERIMENTS_DIR = original

    def test_get_experiment_lineage(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-l", {"data": {"images_dir": "/data/images", "task": "detection"}},
                             data_source="/data/images")
        exp.update_lineage("exp-l", model_weights="/models/best.pt")

        result = exp.get_experiment_lineage("exp-l")
        assert result["lineage"]["data_source"] == "/data/images"
        assert result["lineage"]["model_weights"] == "/models/best.pt"
        assert result["lineage"]["data_config"]["task"] == "detection"

        exp.EXPERIMENTS_DIR = original


# ── model registry replace-by-name is audited ──


class TestModelRegistryReplaceAudit:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.audit_path = self.tmpdir / ".tcip" / "audit.jsonl"

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ckpt(self, name: str, content: bytes) -> str:
        p = self.tmpdir / name
        p.write_bytes(content)
        return str(p)

    def test_replace_by_name_with_different_content_is_audited(self):
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        reg.register_model("exp1", self._ckpt("a.pt", b"first"), {}, metrics_source=None)
        first_sha = reg.get_model("exp1")["sha256"]
        reg.register_model("exp1", self._ckpt("b.pt", b"second, different"), {}, metrics_source=None)
        second_sha = reg.get_model("exp1")["sha256"]
        assert first_sha != second_sha

        events = ts.read_log(audit_mod.audit_log_key()).records
        replace_events = [e for e in events if e.get("tool") == "model_registry_replace"]
        assert len(replace_events) == 1
        assert replace_events[0]["arguments"]["name"] == "exp1"
        assert replace_events[0]["arguments"]["superseded_sha256"] == first_sha
        assert replace_events[0]["arguments"]["new_sha256"] == second_sha

        audit_mod.AUDIT_ROOT = original

    def test_reregistering_identical_content_is_not_audited_as_a_replace(self):
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        ckpt = self._ckpt("a.pt", b"same bytes")
        reg.register_model("exp1", ckpt, {}, metrics_source=None)
        reg.register_model("exp1", ckpt, {}, metrics_source=None)  # idempotent re-registration, same content

        lines = self.audit_path.read_text().strip().splitlines() if self.audit_path.exists() else []
        events = [json.loads(line) for line in lines]
        assert not [e for e in events if e.get("tool") == "model_registry_replace"]

        audit_mod.AUDIT_ROOT = original

    def test_first_registration_under_a_name_is_not_audited_as_a_replace(self):
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        reg.register_model("brand_new", self._ckpt("a.pt", b"content"), {}, metrics_source=None)

        lines = self.audit_path.read_text().strip().splitlines() if self.audit_path.exists() else []
        events = [json.loads(line) for line in lines]
        assert not [e for e in events if e.get("tool") == "model_registry_replace"]

        audit_mod.AUDIT_ROOT = original
