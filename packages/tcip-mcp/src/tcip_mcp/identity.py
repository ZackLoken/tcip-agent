"""The platform's recorded-actor convention, in one place.

A stamped fact says who or what produced it: a person is ``user:<name>``, a tool producer stays
bare (``sam``, ``materialize_review_dataset``), so a reader can tell a human's statement from one a
function wrote without consulting a second store. Every module that stamps an actor onto a record
forms the value here rather than re-spelling the prefix.
"""

from __future__ import annotations


def user_identity(name: str | None) -> str:
    """A person's recorded identity: ``user:<name>``, idempotent, never bare.

    Refuses an empty name rather than stamping a placeholder: a record that says who acted is only
    worth reading when the name is real, so the caller resolves one (the request's own, or the
    backend's fallback identity) before calling.
    """
    value = (name or "").strip()
    if not value:
        raise ValueError(
            "a confirmation records who gave it, so the confirming name is required; resolve it "
            "from the request or the backend's own fallback identity before calling"
        )
    return value if value.startswith("user:") else f"user:{value}"
