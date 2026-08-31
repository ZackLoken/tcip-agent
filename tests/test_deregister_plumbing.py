"""Guards for the machine-plumbing de-registrations.

De-registering the ``log_metrics`` / ``record_artifact`` MCP wrappers must not touch the internal
``tcip_mcp.experiments`` functions the trainer and feedback tools call directly. These freeze that
internal contract against current code. The registry-absence standing checks (that the three
de-registered names no longer register) are added after the de-regs land; see the bottom of file.
"""

from __future__ import annotations


def test_internal_log_metrics_still_functions(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp import experiments

    experiments.create_experiment("exp-dereg-log", {"backbone": "x"})
    result = experiments.log_metrics("exp-dereg-log", 0, {"train_loss": 0.5, "map50": 0.7})
    assert result["epoch"] == 0
    exp = experiments.get_experiment("exp-dereg-log")
    assert any(m.get("map50") == 0.7 for m in exp["metrics"])


def test_internal_record_artifact_still_functions(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp import experiments

    experiments.create_experiment("exp-dereg-art", {"backbone": "x"})
    weights = tmp_path / "best.pt"
    weights.write_text("weights")
    result = experiments.record_artifact("exp-dereg-art", "model_weights", str(weights))
    assert result["artifact"] == "model_weights"
    exp = experiments.get_experiment("exp-dereg-art")
    assert "model_weights" in exp["artifacts"]


def test_deregistered_wrappers_absent_from_registry():
    """Standing check that the three machine-plumbing wrappers no longer register."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    for name in ("log_metrics", "record_artifact", "get_training_metrics_path"):
        assert name not in registered, f"{name} should be de-registered"
