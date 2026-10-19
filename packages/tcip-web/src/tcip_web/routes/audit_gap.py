"""The shared shape a GUI route answers with when a committed mutation went unrecorded.

The mutation committed, but its line could not be recorded to the audit log.
"""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from tcip_mcp.audit import AuditEntryNotWrittenError

AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written"


def audit_gap_409(
    exc: AuditEntryNotWrittenError,
    committed: object,
    *,
    message: str | None = None,
) -> HTTPException:
    """Build the 409 a route answers with when a mutation it already committed went unrecorded.

    ``committed`` is the response body a healthy call would have returned: a dict, a pydantic
    model, or ``None`` when the route cannot say what committed. ``message`` overrides
    ``str(exc)`` when the route must say more than the exception itself does.
    """
    return HTTPException(
        409,
        {
            "error": AUDIT_ENTRY_NOT_WRITTEN,
            "message": message if message is not None else str(exc),
            "committed": jsonable_encoder(committed)
            if committed is not None
            else None,
        },
    )
