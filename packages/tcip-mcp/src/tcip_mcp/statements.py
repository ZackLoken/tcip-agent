"""Comparable-value and content-hash primitives shared by every statement kind.

A statement is a record an agent proposes and a breeder confirms: what a trait's delivered
number means, the semantic shape of a trait it authors, and any future kind built the same way.
Two things every statement kind needs are generic, not particular to any one field set, and live
here so a second kind reuses them rather than re-implementing them: :func:`canonical`, one
comparable form for a stored or live value so JSON round-tripping is never mistaken for drift, and
:func:`content_hash`, a content hash over a caller-declared set of a record's fields, the
compare-and-set a breeder's confirmation click carries back so it can never land on text the
breeder never read.

A statement's ``stated_by``/``authored_by``-shaped field holds the name of the tool that wrote it,
stamped by the writer itself via :func:`now_iso` and a module-level surface constant, never
accepted from a caller. It says the statement came in through that surface rather than through a
file edit, and it is not evidence of who a human author was.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


def canonical(value: Any) -> Any:
    """One comparable form for a stored or live value, so JSON round-tripping is not a difference.

    Sequences become lists recursively and mapping keys are sorted; scalars are left alone. A
    tuple field written into a JSON record reads back as a list, so a raw comparison would report
    a field as moved seconds after it was confirmed.
    """
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [canonical(item) for item in value]
    return value


def content_hash(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    """A content hash over ``fields`` of ``record``, in canonical form.

    The confirmation carries this back, so a breeder's click confirms the record they read rather
    than whatever an agent rewrote while the card was open. ``fields`` names every field the
    statement kind owns, so leaving one field alone while changing another would otherwise harvest
    a click for content nobody saw.
    """
    payload = {field: canonical(record.get(field)) for field in fields}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """The current UTC timestamp in ISO-8601, the clock every statement writer stamps from."""
    return datetime.now(timezone.utc).isoformat()
