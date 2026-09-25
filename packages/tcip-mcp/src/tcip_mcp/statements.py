"""Comparable-value and content-hash primitives shared by every statement kind.

A statement is a record an agent proposes and a breeder confirms. :func:`canonical` gives one
comparable form for a stored or live value, and :func:`content_hash` a content hash over a
caller-declared set of a record's fields.

A statement's ``stated_by``/``authored_by``-shaped field holds the name of the tool that wrote it,
stamped by the writer itself via :func:`now_iso` and a module-level surface constant, never
accepted from a caller; it is not evidence of who a human author was.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


def canonical(value: Any) -> Any:
    """One comparable form for a stored or live value, so JSON round-tripping is not a difference.

    Sequences become lists recursively and mapping keys are sorted; scalars are left alone.
    """
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [canonical(item) for item in value]
    return value


def content_hash(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    """A content hash over ``fields`` of ``record``, in canonical form."""
    payload = {field: canonical(record.get(field)) for field in fields}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """The current UTC timestamp in ISO-8601, the clock every statement writer stamps from."""
    return datetime.now(timezone.utc).isoformat()
