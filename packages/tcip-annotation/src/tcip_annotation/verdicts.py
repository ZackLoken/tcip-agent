"""The review verdict's action vocabulary, declared once.

``review_engine.record_detection_action`` is the write boundary for a stored verdict; it checks
a caller's action against this vocabulary before storing anything, so nothing outside the four
values reaches the store. ``pipelines/feedback/verdicts.py``'s ``POSITIVE_ACTIONS``/
``REJECTED_ACTION``, ``routes/review.py``'s ``ActionPayload.action`` and
``Detection.reviewed_action`` fields, and the generated browser union all carry
``VerdictAction`` rather than restating its values as a separate list. One site still does:
``routes/review.py``'s own ``_apply_gt_mutation`` branches on the action against spelled-out
literals (``review.py:512,529,534``), and a typo there is caught only where ``strict_equality``
is enabled, which the review route is not.

``VerdictAction``'s literal strings are the declaration; ``VERDICT_ACTIONS`` is derived from them
rather than the reverse, since a ``Literal`` built from a tuple does not typecheck.
"""

from __future__ import annotations

from typing import Literal, get_args

VerdictAction = Literal["accepted", "rejected", "edited", "swept"]
VERDICT_ACTIONS: tuple[VerdictAction, ...] = get_args(VerdictAction)
