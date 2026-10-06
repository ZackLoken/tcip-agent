"""Every POST, PUT, PATCH and DELETE route declares a JSON body model, so it refuses every content
type a browser simple request can send and only application/json reaches its handler."""

from __future__ import annotations

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tcip_web.app import app
from tests._route_walk import state_changing_routes


class _ProbePayload(BaseModel):
    """A JSON body model this file owns, so the probe app below declares one at module scope.

    FastAPI resolves a handler's annotations against the module's globals, so a model bound to a
    local name inside a test reads as no body model at all.
    """


EMPTY_BODY_ROUTES = (
    "/api/training/runs/does-not-exist/tensorboard",
    "/api/sessions/end",
)
"""The state-changing routes whose only body is ``EmptyBodyPayload``.

Named here rather than derived, because deriving them from the app would only ever restate what
the routes currently declare, and what has to be held is that these particular reachable state
changes refuse the browser simple-request shape.
"""


def declares_json_body(route: APIRoute) -> bool:
    """Whether a route's declared body is JSON and required, which is what closes the
    simple-request shape.

    A form or multipart route declares a body field too, so the field's presence is not the
    question; its media type is. Executed against a throwaway app: a pydantic model parameter
    reports application/x-www-form-urlencoded for Form and multipart/form-data for File, and a
    route with no body parameter has no body field at all. Media type alone is not enough: a
    body field with a default value is not required, and FastAPI only parses and validates a
    request body when one is present, so a route declared with a defaulted body model still
    substitutes the default and reaches the handler on the empty body a browser simple request
    sends. Requiring the field closes that: an empty or non-JSON body then fails validation
    before the handler runs.
    """
    field = route.body_field
    return (
        field is not None
        and getattr(field.field_info, "media_type", None) == "application/json"
        and field.field_info.is_required()
    )


def test_every_state_changing_route_declares_a_json_body_model() -> None:
    """No exemption list: a route reaches this assertion whatever it is, and fails it unless the
    body it declares is JSON. A route taking only path parameters, and a form or multipart route,
    both fail it."""
    routes = state_changing_routes(app)
    assert routes, "no state-changing routes found; the route walk itself is broken"
    undeclared = sorted(
        f"{sorted(r.methods - {'HEAD', 'OPTIONS'})} {r.path}"
        for r in routes
        if not declares_json_body(r)
    )
    assert not undeclared, f"routes with no JSON body model: {undeclared}"


def test_a_declared_body_model_admits_an_empty_json_object(
    client: TestClient, opened_project
) -> None:
    """The rail must admit valid work: each route that carries no fields of its own still
    accepts the ``{}`` its real caller sends, and reaches the handler's own outcome rather than
    a 422 from the body model rejecting the call. A project is open, so each handler reaches
    its own lookup: an unknown run is a 404, and ending a session when none is open is a no-op
    200.
    """
    resp = client.post("/api/training/runs/does-not-exist/tensorboard", json={})
    assert resp.status_code == 404, resp.text

    resp = client.post("/api/sessions/end", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "noop"


def test_a_missing_body_is_refused(client: TestClient) -> None:
    """No body at all fails the body model rather than reaching the handler."""
    reached = {url: client.post(url).status_code for url in EMPTY_BODY_ROUTES}
    assert reached == dict.fromkeys(EMPTY_BODY_ROUTES, 422)


def test_a_form_encoded_body_is_refused(client: TestClient) -> None:
    """A form submission is exactly the browser simple-request shape this rail exists to close;
    it must fail the body model, not fall through to the handler."""
    reached = {
        url: client.post(url, data={"id": "does-not-exist"}).status_code
        for url in EMPTY_BODY_ROUTES
    }
    assert reached == dict.fromkeys(EMPTY_BODY_ROUTES, 422)


def test_a_headerless_json_shaped_body_is_refused(client: TestClient) -> None:
    """Pins a dependency property this rail leans on rather than a version number: a request
    carrying a non-empty, JSON-shaped body with no ``Content-Type`` header at all must not be
    parsed as JSON and handed to the route's body model.

    A cross-origin ``fetch`` whose body is an empty-type ``Blob`` sends no ``Content-Type``
    header and stays a CORS simple request, so it never triggers a preflight; that is exactly
    the shape this rail exists to keep out. If the body were parsed as JSON here the way it is
    when the caller declares ``application/json``, ``b"{}"`` would decode to an empty dict, feed
    through the same body model a real ``json={}`` call satisfies, and reach the handler for its
    own outcome instead of failing
    validation. The 422 asserted below is what distinguishes that reverted behavior from the
    one this rail depends on; a dependency upgrade or pin change that stopped enforcing it would
    fail this assertion rather than passing silently.
    """
    for url in EMPTY_BODY_ROUTES:
        request = client.build_request("POST", url, content=b"{}")
        assert "content-type" not in request.headers, (url, dict(request.headers))
        response = client.send(request)
        assert response.status_code == 422, (url, response.status_code, response.text)


def test_the_guard_rejects_the_shapes_it_exists_to_catch() -> None:
    """The predicate has to discriminate, or the sweep above passes for the wrong reason.

    Four shapes are the ones a future route could open the gap with, and a form route is the
    exact browser simple request the rail is named for. A defaulted JSON body model is the
    subtlest of the four: it declares application/json and a body field, same as a real route,
    but FastAPI substitutes the default on an empty body instead of validating one, so it
    reaches the handler on the same empty body a form route sends. All four declare something
    FastAPI is willing to route; only the required JSON model closes the gap.
    """
    from typing import Annotated

    from fastapi import FastAPI, File, Form, UploadFile

    probe = FastAPI()

    @probe.post("/json")
    def json_route(payload: _ProbePayload) -> dict:
        return {}

    @probe.post("/defaulted-json")
    def defaulted_json_route(payload: _ProbePayload = _ProbePayload()) -> dict:
        return {}

    @probe.post("/form")
    def form_route(name: Annotated[str, Form()]) -> dict:
        return {}

    @probe.post("/upload")
    def upload_route(upload: Annotated[UploadFile, File()]) -> dict:
        return {}

    @probe.post("/nothing")
    def no_body_route() -> dict:
        return {}

    verdicts = {r.path: declares_json_body(r) for r in state_changing_routes(probe)}
    assert verdicts == {
        "/json": True,
        "/defaulted-json": False,
        "/form": False,
        "/upload": False,
        "/nothing": False,
    }
