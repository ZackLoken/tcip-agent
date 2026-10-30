"""The shared shape a GUI route answers with when a committed mutation went unrecorded.

The mutation committed, but its line could not be recorded to the audit log.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from tcip_mcp.audit import AuditEntryNotWrittenError


class AuditGapRefusal(BaseModel):
    """A mutation that committed and whose audit line could not be written.

    ``committed`` is the response body a healthy call would have returned, a dict or a pydantic
    model, serialized as it is; ``None`` when the route cannot say what committed.
    """

    kind: Literal["audit_entry_not_written"] = "audit_entry_not_written"
    message: str
    committed: Any


def audit_gap_409(
    exc: AuditEntryNotWrittenError,
    committed: object,
    *,
    message: str | None = None,
) -> HTTPException:
    """Build the 409 a route answers with when a mutation it already committed went unrecorded.

    Its detail is the :class:`AuditGapRefusal` of ``committed``, a dict, a pydantic model or
    ``None``. ``message`` overrides ``str(exc)`` when the route must say more than the exception
    itself does.
    """
    return HTTPException(
        409,
        AuditGapRefusal(
            message=message if message is not None else str(exc),
            committed=committed,
        ).model_dump(mode="json"),
    )
