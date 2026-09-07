"""The shared shape a GUI route answers with when a mutation it already committed could not be
recorded to the audit log.

Not ``_audit``, which already names two route-local helpers (``review.py``'s and ``results.py``'s
own best-effort writers before this module existed); this is the one place the marker, the
raising call and the 409 body are built, so every route that answers this gap composes it the
same way.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise

# Also coverage.py's own marker for its analogous 500: moved here and imported back, so every
# route names the same literal for the same fact.
AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written"


def record_committed(tool: str, arguments: dict, *, scope: str | None) -> None:
    """Record a mutation a route has already committed, letting a failed append propagate as
    :class:`~tcip_mcp.audit.AuditEntryNotWritten`.

    A thin wrapper over :func:`~tcip_mcp.audit.record_event_or_raise`, imported here at module
    level (unlike the route-local helpers this backs) rather than inside this function, so a test
    can refuse a route's own line by patching this one name without reaching the library's.
    """
    record_event_or_raise(tool, arguments, source="gui", scope=scope)


def audit_gap_409(
    exc: AuditEntryNotWritten, committed: Any, *, message: str | None = None,
) -> HTTPException:
    """The 409 a route answers with when a mutation it already committed could not be recorded.

    ``committed`` is the response body a healthy call would have returned: a dict, a pydantic
    model, or ``None`` for the one case where the route cannot say what committed. A client that
    adopts ``committed`` reaches the state a 200 would have left it in. ``message`` overrides
    ``str(exc)`` for the one route that must say more than the exception itself does
    (``build_plant_mapping``'s supersede branch, where the archived copy landed but the new
    record's own receipt is what failed), composed here rather than by mutating a fresh
    exception's ``args`` after construction.
    """
    return HTTPException(409, {
        "error": AUDIT_ENTRY_NOT_WRITTEN,
        "message": message if message is not None else str(exc),
        "committed": jsonable_encoder(committed) if committed is not None else None,
    })
