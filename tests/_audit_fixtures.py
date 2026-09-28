"""A fault injected into the audit log's append."""

from __future__ import annotations

import pytest


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
