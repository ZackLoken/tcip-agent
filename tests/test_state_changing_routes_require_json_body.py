"""Every POST, PUT, PATCH and DELETE route declares a JSON body model, so it refuses every content
type a browser simple request can send and only application/json reaches its handler."""

from __future__ import annotations

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tcip_web.app import app
from tests._route_walk import state_changing_routes


class _ProbePayload(BaseModel):
    """A JSON body model at module scope, where FastAPI resolves a handler's annotations."""


EMPTY_BODY_ROUTES = (
    "/api/training/runs/does-not-exist/tensorboard",
)
"""The state-changing routes whose only body is ``EmptyBodyPayload``."""


def declares_json_body(route: APIRoute) -> bool:
    """Whether a route's declared body is JSON and required: a form or multipart route declares a
    body field of another media type, and a defaulted JSON body model is substituted on an empty
    body instead of validated, so only a required JSON model refuses the browser simple-request
    shape before the handler runs."""
    field = route.body_field
    return (
        field is not None
        and getattr(field.field_info, "media_type", None) == "application/json"
        and field.field_info.is_required()
    )


def test_every_state_changing_route_declares_a_json_body_model() -> None:
    """Every state-changing route declares a required JSON body; one taking only path
    parameters, and a form or multipart route, fail this."""
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
    """A route that carries no fields of its own accepts the ``{}`` its real caller sends and
    reaches the handler's own outcome: with a project open, an unknown run is a 404."""
    resp = client.post("/api/training/runs/does-not-exist/tensorboard", json={})
    assert resp.status_code == 404, resp.text


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
    """A JSON-shaped body with no ``Content-Type`` header (what a cross-origin ``fetch`` of an
    empty-type ``Blob`` sends, a CORS simple request with no preflight) is not parsed as JSON
    and fails the body model rather than reaching the handler."""
    for url in EMPTY_BODY_ROUTES:
        request = client.build_request("POST", url, content=b"{}")
        assert "content-type" not in request.headers, (url, dict(request.headers))
        response = client.send(request)
        assert response.status_code == 422, (url, response.status_code, response.text)


def test_the_guard_rejects_the_shapes_it_exists_to_catch() -> None:
    """The predicate refuses a route with no body, a form route, a multipart route and a
    defaulted JSON body model, and admits a required JSON body model."""
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
