"""Tests for audit logging and experiment tracking."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Audit logging ──


class TestAuditLogging:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.audit_path = self.tmpdir / "audit.jsonl"

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_audited_logs_success(self):
        from tcip_mcp.audit import audited

        with patch.object(
            __import__("tcip_mcp.audit", fromlist=["AUDIT_PATH"]),
            "AUDIT_PATH",
            self.audit_path,
        ):
            # Re-import to get patched version
            import tcip_mcp.audit as audit_mod
            original = audit_mod.AUDIT_PATH
            audit_mod.AUDIT_PATH = self.audit_path

            @audited
            def my_tool(x: int = 0) -> dict:
                return {"result": x + 1}

            result = my_tool(x=5)
            assert result == {"result": 6}

            # Check audit log
            lines = self.audit_path.read_text().strip().splitlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["tool"] == "my_tool"
            assert entry["status"] == "ok"
            assert entry["arguments"] == {"x": 5}
            assert "duration_ms" in entry
            assert "timestamp" in entry

            audit_mod.AUDIT_PATH = original

    def test_audited_logs_exception(self):
        from tcip_mcp.audit import audited

        import tcip_mcp.audit as audit_mod
        original = audit_mod.AUDIT_PATH
        audit_mod.AUDIT_PATH = self.audit_path

        @audited
        def failing_tool() -> dict:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_tool()

        lines = self.audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["status"] == "exception"
        assert "test error" in entry["error"]

        audit_mod.AUDIT_PATH = original

    def test_redaction(self):
        from tcip_mcp.audit import _redact

        args = {"name": "test", "api_key": "secret123", "token": "tok123"}
        redacted = _redact(args)
        assert redacted["name"] == "test"
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["token"] == "***REDACTED***"


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

        # Directory created with files
        d = self.tmpdir / "experiments" / "exp-001"
        assert d.exists()
        assert (d / "config.json").exists()
        assert (d / "status.json").exists()
        assert (d / "lineage.json").exists()
        assert (d / "artifacts.json").exists()

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

        metrics_path = self.tmpdir / "experiments" / "exp-002" / "metrics.jsonl"
        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 0
        assert json.loads(lines[1])["mAP50"] == 0.5

        exp.EXPERIMENTS_DIR = original

    def test_update_status(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-003", {})
        exp.update_status("exp-003", "running")
        status = json.loads((self.tmpdir / "experiments" / "exp-003" / "status.json").read_text())
        assert status["state"] == "running"
        assert status["started"] is not None

        exp.update_status("exp-003", "completed")
        status = json.loads((self.tmpdir / "experiments" / "exp-003" / "status.json").read_text())
        assert status["state"] == "completed"
        assert status["ended"] is not None

        exp.EXPERIMENTS_DIR = original

    def test_record_artifact(self):
        import tcip_mcp.experiments as exp
        original = exp.EXPERIMENTS_DIR
        exp.EXPERIMENTS_DIR = self.tmpdir / "experiments"

        exp.create_experiment("exp-004", {})
        exp.record_artifact("exp-004", "model_weights", "/path/to/model.pt")

        artifacts = json.loads((self.tmpdir / "experiments" / "exp-004" / "artifacts.json").read_text())
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

        exp.create_experiment("exp-x", {"model_spec": {"backbone": {"name": "resnet50"}}})
        exp.create_experiment("exp-y", {"model_spec": {"backbone": {"name": "efficientnet_b0"}}})
        exp.log_metrics("exp-x", 0, {"mAP50": 0.6})
        exp.log_metrics("exp-y", 0, {"mAP50": 0.7})

        result = exp.compare_experiments(["exp-x", "exp-y"])
        assert result["count"] == 2
        exps = {e["experiment_id"]: e for e in result["experiments"]}
        assert exps["exp-x"]["backbone"] == "resnet50"
        assert exps["exp-y"]["final_metrics"]["mAP50"] == 0.7

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


# ── Orchestrator checkpoint/resume ──


class TestOrchestratorCheckpoint:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_checkpoint_saved_on_phase_completion(self):
        """Verify that a checkpoint file is created after phase execution."""
        from tcip_mcp.pipelines.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(work_dir=str(self.tmpdir / "runs"))

        # Run a minimal pipeline that will fail at inference (no checkpoint)
        spec = {
            "name": "test_ckpt",
            "phases": [
                {
                    "name": "phase_agg",
                    "task": "aggregation",
                    "output": "agg_out",
                },
            ],
        }
        result = orch.run_pipeline(spec)
        assert result.status == "completed"

        # Find the run directory
        runs_dir = self.tmpdir / "runs"
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1

        checkpoint = run_dirs[0] / "checkpoint.json"
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert data["pipeline_name"] == "test_ckpt"
        assert len(data["completed_phases"]) == 1

    def test_retry_on_transient_failure(self):
        """Verify the _is_transient method correctly identifies transient errors."""
        from tcip_mcp.pipelines.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        assert orch._is_transient("CUDA out of memory")
        assert orch._is_transient("ConnectionError: failed to connect")
        assert orch._is_transient("TimeoutError")
        assert not orch._is_transient("ValueError: invalid config")
        assert not orch._is_transient("KeyError: missing key")
