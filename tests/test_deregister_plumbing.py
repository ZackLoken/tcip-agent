"""The machine-plumbing wrappers stay out of the MCP registry.

A run's metrics and artifacts are written by its own body through the training envelope
(``TrainContext.log_metrics`` / ``record_artifact``, covered in ``test_envelope_ctx_sinks.py``),
never through a tool an agent could call on a run it does not own.
"""

from __future__ import annotations


def test_deregistered_wrappers_absent_from_registry():
    """Standing check that the three machine-plumbing wrappers no longer register."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    for name in ("log_metrics", "record_artifact", "get_training_metrics_path"):
        assert name not in registered, f"{name} should be de-registered"
