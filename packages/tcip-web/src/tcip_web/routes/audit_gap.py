"""The shared shape a GUI route answers with when a mutation it already committed could not be
recorded to the audit log.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from tcip_mcp.audit import AuditEntryNotWritten

AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written"


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
