"""The platform's recorded-actor convention, in one place.

A stamped fact says who or what produced it: a person is ``user:<name>``, a tool producer stays
bare (``sam``, ``materialize_review_dataset``).
"""

from __future__ import annotations


def user_identity(name: str | None) -> str:
    """A person's recorded identity: ``user:<name>``, idempotent, never bare. Refuses an empty
    name.
    """
    value = (name or "").strip()
    if not value:
        raise ValueError(
            "a confirmation records who gave it, so the confirming name is required; resolve it "
            "from the request or the backend's own fallback identity before calling"
        )
    return value if value.startswith("user:") else f"user:{value}"
