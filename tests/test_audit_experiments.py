"""Tests for audit logging and experiment tracking."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import tcip_store as ts

from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run


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


def test_get_experiment_reads_the_run_directory(tmp_path):
    import tcip_mcp.experiments as exp

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "ds", backbone="resnet50"),
                         experiment_id="exp-005")
    log_epoch(run_dir, 0, {"loss": 1.0})

    result = exp.get_experiment("exp-005")
    assert result["experiment_id"] == "exp-005"
    assert result["run"]["config"]["backbone"] == "resnet50"
    assert result["final_status"] is None
    assert result["n_epochs"] == 1
    assert len(result["metrics"]) == 1


def test_get_experiment_not_found(tmp_path):
    import tcip_mcp.experiments as exp

    assert "error" in exp.get_experiment("nonexistent")


def test_list_experiments(tmp_path):
    import tcip_mcp.experiments as exp

    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-a")
    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-b")

    listing = exp.list_experiments()
    assert {e["experiment_id"] for e in listing} == {"exp-a", "exp-b"}
    assert {e["state"] for e in listing} == {"running"}


def test_compare_experiments(tmp_path):
    import tcip_mcp.experiments as exp

    x = opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-x")
    y = opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-y")
    log_epoch(x, 0, {"mAP50": 0.6})
    log_epoch(y, 0, {"mAP50": 0.7})

    result = exp.compare_experiments(["exp-x", "exp-y"])
    assert result["count"] == 2
    exps = {e["experiment_id"]: e for e in result["experiments"]}
    assert exps["exp-x"]["model"] == "tests.bespoke_models:build_bespoke_detection"
    assert exps["exp-y"]["last_logged_metrics"]["mAP50"] == 0.7


def test_get_experiment_lineage(tmp_path):
    import tcip_mcp.experiments as exp

    config = detection_config(tmp_path / "ds")
    opened_run(tmp_path, config, experiment_id="exp-l", parent_experiment="exp-k")

    lineage = exp.get_experiment_lineage("exp-l")["lineage"]
    assert lineage["data"]["images_dir"] == config["data"]["images_dir"]
    assert lineage["data"]["scope"]["id_map"] == config["data"]["scope"]["id_map"]
    assert lineage["parent_experiment"] == "exp-k"
    assert lineage["checkpoint"] is None


# ── a foreign registration is keyed by its bytes and audited ──


def _entry(project: Path, sha256: str) -> dict:
    """The one foreign entry ``project``'s registry holds for ``sha256``."""
    from tcip_mcp.model_registry import registered_entries

    [entry] = [e for e in registered_entries(project) if e["sha256"] == sha256]
    return entry


class TestModelRegistryReplaceAudit:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ckpt(self, name: str, content: bytes) -> str:
        from tests._verified_checkpoint_fixtures import checkpoint_file

        return str(checkpoint_file(self.tmpdir / name, content.decode()))

    def _rows(self) -> list[dict]:
        import tcip_mcp.audit as audit_mod

        return list(ts.read_log(audit_mod.audit_log_key()).records)

    def test_a_first_registration_and_a_replacement_each_leave_one_row(self):
        """The same bytes registered again under another name replace their one entry, the row
        naming the name it superseded; other bytes under the first name are an entry of their
        own."""
        import tcip_mcp.audit as audit_mod
        from tcip_mcp.model_registry import ModelRegistry

        original = audit_mod.AUDIT_ROOT
        audit_mod.AUDIT_ROOT = self.tmpdir

        reg = ModelRegistry(str(self.tmpdir))
        first_sha = reg.register_model("exp1", self._ckpt("a.pt", b"first"), {})["sha256"]
        (first,) = self._rows()
        assert first["tool"] == "model_registered"
        assert first["arguments"] == {"name": "exp1", "new_sha256": first_sha}

        reg.register_model("exp2", self._ckpt("a.pt", b"first"), {})
        _, replaced = self._rows()
        assert replaced["arguments"] == {"name": "exp2", "new_sha256": first_sha,
                                         "superseded_name": "exp1", "superseded_tags": []}
        assert _entry(self.tmpdir, first_sha)["name"] == "exp2"

        second_sha = reg.register_model("exp1", self._ckpt("b.pt", b"second, different"),
                                        {})["sha256"]
        assert second_sha != first_sha
        assert "superseded_name" not in self._rows()[-1]["arguments"]
        assert _entry(self.tmpdir, first_sha)["name"] == "exp2"

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
        assert ["superseded_name" in r["arguments"] for r in rows] == [False, False, True]
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
        reg.register_model("exp1", ckpt, {})
        before = ts.read_versioned(registry_index_key(self.tmpdir))
        reg.register_model("exp1", ckpt, {})

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
        sha = reg.register_model("exp1", ckpt, {})["sha256"]
        reg.register_model("exp1", ckpt, {}, tags=["chestnut"])

        assert _entry(self.tmpdir, sha)["tags"] == ["chestnut"]
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
        sha = reg.register_model("exp1", self._ckpt("a.pt", b"first"), {})["sha256"]

        def _refuse(*args, **kwargs):
            raise RuntimeError("the audit log could not be appended to")

        monkeypatch.setattr(audit_mod, "append", _refuse)

        with pytest.raises(audit_mod.AuditEntryNotWritten) as caught:
            reg.register_model("exp2", self._ckpt("a.pt", b"first"), {})

        assert caught.value.tool == "model_registered"
        assert _entry(self.tmpdir, sha)["name"] == "exp2"

        audit_mod.AUDIT_ROOT = original
