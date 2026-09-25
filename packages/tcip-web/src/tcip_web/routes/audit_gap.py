"""The shared shape a GUI route answers with when a mutation it already committed could not be
recorded to the audit log.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise

# Also coverage.py's own marker for its analogous 500, imported from here, so every route
# names the same literal for the same fact.
AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written"


def record_committed(tool: str, arguments: dict, *, scope: str | None) -> None:
    """Record a mutation a route has already committed, letting a failed append propagate as
    :class:`~tcip_mcp.audit.AuditEntryNotWritten`.
    """
    record_event_or_raise(tool, arguments, source="gui", scope=scope)


def audit_gap_409(
    exc: AuditEntryNotWritten, committed: Any, *, message: str | None = None,
) -> HTTPException:
    """The 409 a route answers with when a mutation it already committed could not be recorded.

    ``committed`` is the response body a healthy call would have returned: a dict, a pydantic
    model, or ``None`` when the route cannot say what committed. A client that adopts ``committed``
    reaches the state a 200 would have left it in. ``message`` overrides ``str(exc)`` when the
    route must say more than the exception itself does.
    """
    return HTTPException(409, {
        "error": AUDIT_ENTRY_NOT_WRITTEN,
        "message": message if message is not None else str(exc),
        "committed": jsonable_encoder(committed) if committed is not None else None,
    })
