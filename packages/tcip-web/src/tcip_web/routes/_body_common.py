"""The body model for a state-changing route that carries no fields of its own.

``TrustBoundaryMiddleware`` refuses a state-changing request from another origin before it
reaches any route (``tcip_web.trust_boundary``); this model is the second layer behind that
check. A route with no body parameter at all is reachable as a browser simple request (a
cross-origin HTML form submission, for instance), which skips the CORS preflight the rest of
this API relies on. Giving the handler a JSON body model, even an empty one, makes the server
refuse the content types a simple request can send: no body, form-urlencoded, or plain text
all fail before the handler runs. Only a call that arrives as application/json, which a
browser sends only from a preflighted request, still reaches it. That closes the gap without
touching the route's semantics.
"""

from __future__ import annotations

from pydantic import BaseModel


class EmptyBodyPayload(BaseModel):
    """A JSON body with no required fields, for a route that takes only path parameters."""
