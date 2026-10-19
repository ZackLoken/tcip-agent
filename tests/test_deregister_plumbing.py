"""The machine-plumbing wrappers stay out of the MCP registry.

``log_metrics``, ``record_artifact`` and ``get_training_metrics_path`` are no registered tool.
"""

from __future__ import annotations


def test_deregistered_wrappers_absent_from_registry() -> None:
    """The three machine-plumbing wrappers are absent from the MCP registry."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    for name in (
        "log_metrics",
        "record_artifact",
        "get_training_metrics_path",
    ):
        assert name not in registered, f"{name} should be de-registered"
