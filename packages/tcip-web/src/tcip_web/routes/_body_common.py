"""The body model for a state-changing route that carries no fields of its own.

Giving the handler a JSON body model, even an empty one, makes the server refuse the content types
a browser simple request can send (no body, form-urlencoded, or plain text) before the handler
runs; only a call that arrives as application/json reaches it.
"""

from __future__ import annotations

from pydantic import BaseModel


class EmptyBodyPayload(BaseModel):
    """A JSON body with no required fields, for a route that takes only path parameters."""


class PersonPayload(BaseModel):
    """A JSON body naming the person making the act, for a route whose other parameters are its
    path's."""

    user: str
