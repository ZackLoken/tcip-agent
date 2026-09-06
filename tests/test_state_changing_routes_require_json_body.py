"""A state-changing route must declare a JSON body model.

CORS is disabled on this backend; nothing here checks origin, which is covered separately
(``test_state_changing_routes_require_same_origin.py``) by the check
``TrustBoundaryMiddleware`` applies ahead of every route. What this guards is narrower: a route
with no body parameter at all is reachable as a browser simple request (a cross-origin HTML
form submission, for instance), which never triggers a CORS preflight. Requiring a JSON body,
even an empty one, makes the server refuse every content type a simple request can send, on
every POST/PUT/PATCH/DELETE route with no exemption; only application/json, which a browser
sends only from a preflighted request, still reaches the handler.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tcip_web.app import app
from tcip_web.trust_boundary import STATE_CHANGING_METHODS

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "generate_frontend_routes.py"


class _ProbePayload(BaseModel):
    """A JSON body model this file owns, so the probe app below declares one at module scope.

    FastAPI resolves a handler's annotations against the module's globals, so a model bound to a
    local name inside a test reads as no body model at all.
    """


EMPTY_BODY_ROUTES = (
    "/api/inference/jobs/does-not-exist/cancel",
    "/api/training/runs/does-not-exist/tensorboard",
    "/api/training/runs/does-not-exist/cancel",
    "/api/tuning/sweeps/does-not-exist/tensorboard",
    "/api/tuning/sweeps/does-not-exist/trials/does-not-exist/tensorboard",
    "/api/tuning/sweeps/does-not-exist/trials/does-not-exist/tensorboard/stop",
)
"""The state-changing routes whose only body is ``EmptyBodyPayload``, each hit with an id
nothing resolves.

Named here rather than derived, because deriving them from the app would only ever restate what
the routes currently declare, and what has to be held is that these particular reachable state
changes refuse the browser simple-request shape.
"""


def _route_generator():
    """The route-walking module, loaded the same way ``test_frontend_route_paths.py`` loads it,
    so this test enumerates routes through the one real walk of the app's router tree rather
    than a second copy of it."""
    spec = importlib.util.spec_from_file_location("tcip_frontend_route_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_changing_routes(target=app) -> list[APIRoute]:
    gen = _route_generator()
    return [
        route
        for route in gen.iter_api_routes(target)
        if isinstance(route, APIRoute)
        and (route.methods - {"HEAD", "OPTIONS"}) & STATE_CHANGING_METHODS
    ]


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


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def test_every_state_changing_route_declares_a_json_body_model() -> None:
    """No exemption list: a route reaches this assertion whatever it is, and fails it unless the
    body it declares is JSON. A route taking only path parameters, and a form or multipart route,
    both fail it."""
    routes = _state_changing_routes()
    assert routes, "no state-changing routes found; the route walk itself is broken"
    undeclared = sorted(
        f"{sorted(r.methods - {'HEAD', 'OPTIONS'})} {r.path}"
        for r in routes
        if not declares_json_body(r)
    )
    assert not undeclared, f"routes with no JSON body model: {undeclared}"


def test_a_declared_body_model_admits_an_empty_json_object(client: TestClient) -> None:
    """The rail must admit valid work: each route that carries no fields of its own still
    accepts the ``{}`` its real caller sends, and reaches the handler's own outcome rather than
    a 422 from the body model rejecting the call. An unknown id is a 404 everywhere the handler
    resolves an id before acting; the trial-tensorboard stop route acts on a computed process
    key with no id lookup of its own, so an unknown trial is a no-op 200, not a 404.
    """
    unknown_id_routes = tuple(
        url for url in EMPTY_BODY_ROUTES if not url.endswith("/tensorboard/stop")
    )
    for url in unknown_id_routes:
        resp = client.post(url, json={})
        assert resp.status_code != 422, (url, resp.text)
        assert resp.status_code == 404, (url, resp.status_code, resp.text)

    resp = client.post(
        "/api/tuning/sweeps/does-not-exist/trials/does-not-exist/tensorboard/stop", json={}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_running"


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
    own outcome (404, for an unknown id, on every route this test walks) instead of failing
    validation. The 422 asserted below is what distinguishes that reverted behaviour from the
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

    verdicts = {r.path: declares_json_body(r) for r in _state_changing_routes(probe)}
    assert verdicts == {
        "/json": True,
        "/defaulted-json": False,
        "/form": False,
        "/upload": False,
        "/nothing": False,
    }
