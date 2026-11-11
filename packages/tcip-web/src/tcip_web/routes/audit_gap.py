"""The shared shape a GUI route answers with when a call's audit line could not be written.

The refusal says whether the call's change committed and carries the committed body where the
route holds one.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from tcip_mcp.audit import (
    AuditEntryNotWrittenError,
    MutationCommittedWithoutAuditLineError,
)


class AuditGapRefusal(BaseModel):
    """A call whose audit line could not be written.

    ``committed`` is whether the call's change committed. ``what`` is the response body a healthy
    call would have returned, a dict or a pydantic model, serialized as it is; ``None`` when the
    route holds no such body, which is the case whether or not the change committed.
    """

    kind: Literal["audit_entry_not_written"] = "audit_entry_not_written"
    message: str
    committed: bool
    what: Any


def audit_gap_409(
    exc: AuditEntryNotWrittenError | MutationCommittedWithoutAuditLineError,
    *,
    committed: bool,
    what: object,
    message: str | None = None,
) -> HTTPException:
    """Build the 409 a route answers with when a call's audit line could not be written.

    Its detail is the :class:`AuditGapRefusal` saying whether the call's change ``committed`` and
    carrying ``what``, a dict, a pydantic model or ``None``. ``message`` overrides ``str(exc)``
    when the route must say more than the exception itself does.
    """
    return HTTPException(
        409,
        AuditGapRefusal(
            message=message if message is not None else str(exc),
            committed=committed,
            what=what,
        ).model_dump(mode="json"),
    )
