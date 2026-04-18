"""Shared error types for MCP tools and pipelines."""

from __future__ import annotations


class ToolError(Exception):
    """Raised by MCP tools when a recoverable error occurs.

    Attributes:
        message: Human-readable error description.
        tool_name: Name of the tool that raised the error.
        details: Optional dict of extra context for the agent.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details or {}


class PipelineError(Exception):
    """Raised when a pipeline fails in a non-resumable way."""

    def __init__(self, message: str, *, phase: str = "") -> None:
        super().__init__(message)
        self.phase = phase
