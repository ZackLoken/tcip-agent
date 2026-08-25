"""The review verdict's action vocabulary, declared once.

``review_engine.record_detection_action`` is the write boundary for a stored verdict; it checks
a caller's action against this vocabulary before storing anything, so nothing outside the four
values reaches the store. Every other reader of the vocabulary, in this package or in
tcip-mcp/tcip-web, imports from here rather than restating the values.

``VerdictAction``'s literal strings are the declaration; ``VERDICT_ACTIONS`` is derived from them
rather than the reverse, since a ``Literal`` built from a tuple does not typecheck.
"""

from __future__ import annotations

from typing import Literal, get_args

VerdictAction = Literal["accepted", "rejected", "edited", "swept"]
VERDICT_ACTIONS = get_args(VerdictAction)
