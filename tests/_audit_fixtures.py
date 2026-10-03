"""Audit lines read back through the store seam, the shape a line may take, and a fault injected
into the log's append."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import agent_identity
from tcip_mcp.audit import audit_log_key

AUDIT_ENTRY_KEYS = frozenset({"timestamp", "tool", "arguments", "status",
                              *agent_identity.RECORD_FIELDS})
"""Every key an audit line may carry besides ``actor``."""


def audit_rows(scope: Path, tool: str | None = None) -> list[dict]:
    """Every line in the log ``scope`` names, or only ``tool``'s lines when one is named."""
    return [row for row in ts.read_log(audit_log_key(scope)).records
            if tool is None or row["tool"] == tool]


def refuse_audit_appends(monkeypatch: pytest.MonkeyPatch, *, error: BaseException | None = None,
                         landing: int = 0) -> None:
    """Let the first ``landing`` audit appends land, then raise ``error`` (by default a
    ``RuntimeError`` naming an unwritable log) from every later one."""
    import tcip_mcp.audit as audit_module

    real_append = audit_module.append
    failure = error if error is not None else RuntimeError("the audit log could not be appended to")
    calls = {"n": 0}

    def append(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= landing:
            return real_append(*args, **kwargs)
        raise failure

    monkeypatch.setattr(audit_module, "append", append)
