"""The platform's recorded-actor convention, in one place.

A stamped fact says who or what produced it: a person is ``user:<name>``, a tool producer stays
bare (``model:<checkpoint>``, an agent's own name).
"""

from __future__ import annotations

from tcip_annotation.json_io import PERSON_IDENTITY_PREFIX


def actor(name: str | None) -> str:
    """A person's recorded identity, ``user:<name>``, idempotent, from the ``name`` a request
    states. Refuses (``ValueError``) when ``name`` is missing or blank."""
    value = (name or "").strip()
    if not value:
        raise ValueError("an act records who made it, and the request names no one; state the "
                         "person's name")
    return value if value.startswith(PERSON_IDENTITY_PREFIX) else PERSON_IDENTITY_PREFIX + value
