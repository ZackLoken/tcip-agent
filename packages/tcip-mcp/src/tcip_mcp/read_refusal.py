"""Why a caught error says a record will not read (:func:`read_refusal`)."""

from __future__ import annotations

from pydantic import ValidationError

from tcip_store import Invalid, ReadRefusal, Unreadable


def read_refusal(exc: Exception) -> ReadRefusal:
    """Return why a read raised ``exc``: its model's errors, or its own sentence."""
    if isinstance(exc, ValidationError):
        return Invalid.of(exc)
    return Unreadable(message=str(exc))
