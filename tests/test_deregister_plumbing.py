"""The machine-plumbing wrappers stay out of the MCP registry.

``log_metrics``, ``record_artifact`` and ``get_training_metrics_path`` are no registered tool.
"""

from __future__ import annotations


def test_deregistered_wrappers_absent_from_registry() -> None:
    """The three machine-plumbing wrappers are absent from the MCP registry."""
    from tcip_web.cli import tool_listing

    registered = {t.name for t in tool_listing()}
    for name in (
        "log_metrics",
        "record_artifact",
        "get_training_metrics_path",
    ):
        assert name not in registered, f"{name} should be de-registered"
