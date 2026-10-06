"""The platform's recorded-actor convention, in one place.

A stamped fact says who or what produced it: a person is ``user:<name>``, a tool producer stays
bare (``model:<checkpoint>``, an agent's own name).
"""

from __future__ import annotations

from tcip_annotation.json_io import PERSON_IDENTITY_PREFIX, is_person


class NoActorError(ValueError):
    """An act that records who made it was requested naming no one."""


def actor(name: str | None) -> str:
    """A person's recorded identity, ``user:<name>``, idempotent, from the ``name`` a request
    states, with or without the prefix. Refuses (:class:`NoActorError`) when that identity names no
    person (:func:`~tcip_annotation.json_io.is_person`)."""
    identity = (PERSON_IDENTITY_PREFIX
                + (name or "").strip().removeprefix(PERSON_IDENTITY_PREFIX).strip())
    if not is_person(identity):
        raise NoActorError("an act records who made it, and the request names no one; state the "
                      "person's name")
    return identity
