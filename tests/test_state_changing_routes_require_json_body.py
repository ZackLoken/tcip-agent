"""Every POST, PUT, PATCH and DELETE route declares a JSON body model, but one exempt by name.

A missing body, a form-encoded body and a headerless JSON-shaped body each answer 422, and an
empty JSON object reaches the handler. The exempt route forwards TensorBoard's own
form-encoded posts to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tcip_web.app import app
from tcip_web.routes.training import forward_to_board
from tests._route_walk import state_changing_routes

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


class _ProbePayload(BaseModel):
    """A JSON body model at module scope, where FastAPI resolves a handler's annotations."""


EMPTY_BODY_ROUTES = ("/api/training/runs/does-not-exist/tensorboard",)
"""The state-changing routes whose only body is ``EmptyBodyPayload``."""

FORM_POSTED_ROUTES = frozenset({forward_to_board.__name__})
"""The state-changing routes exempt by name, whose posts are another program's forms."""


def declares_json_body(route: APIRoute) -> bool:
    """Return whether a route's declared body is JSON and required.

    It answers whether ``route.body_field`` exists, its media type is ``application/json``, and
    it is required.
    """
    field = route.body_field
    return (
        field is not None
        and getattr(field.field_info, "media_type", None) == "application/json"
        and field.field_info.is_required()
    )


def undeclared_routes(target: FastAPI) -> list[str]:
    """Return each state-changing route of ``target`` with no JSON body, but those exempt by name.

    Each is named by its methods and its path.
    """
    return sorted(
        f"{sorted(r.methods - {'HEAD', 'OPTIONS'})} {r.path}"
        for r in state_changing_routes(target)
        if r.name not in FORM_POSTED_ROUTES and not declares_json_body(r)
    )


def test_every_state_changing_route_declares_a_json_body_model() -> None:
    """Every state-changing route declares a required JSON body, but those exempt by name."""
    assert state_changing_routes(app), (
        "no state-changing routes found; the route walk itself is broken"
    )
    undeclared = undeclared_routes(app)
    assert not undeclared, f"routes with no JSON body model: {undeclared}"


def test_the_exemption_admits_the_named_route_alone() -> None:
    """A form route sharing the exempt route's path and the start of its name is still refused."""
    from typing import Annotated

    from fastapi import FastAPI, Form

    from tcip_mcp.pipelines.training.tensorboard_manager import BOARD_PATH

    probe = FastAPI()

    @probe.post(BOARD_PATH + "/{path:path}", name=forward_to_board.__name__)
    def exempt_route(name: Annotated[str, Form()]) -> dict:
        return {}

    @probe.post(BOARD_PATH + "/form", name=f"{forward_to_board.__name__}_form")
    def form_route(name: Annotated[str, Form()]) -> dict:
        return {}

    assert undeclared_routes(probe) == [f"['POST'] {BOARD_PATH}/form"]


def test_a_declared_body_model_admits_an_empty_json_object(
    client: TestClient, opened_project: Path
) -> None:
    """A body model with no fields admits an empty JSON object and reaches the handler.

    A route that carries no fields of its own accepts ``{}`` and reaches the handler's own
    outcome: with a project open, an unknown run is a 404.
    """
    resp = client.post(
        "/api/training/runs/does-not-exist/tensorboard", json={}
    )
    assert resp.status_code == 404, resp.text


def test_a_missing_body_is_refused(client: TestClient) -> None:
    """A request with no body fails the body model with 422."""
    reached = {url: client.post(url).status_code for url in EMPTY_BODY_ROUTES}
    assert reached == dict.fromkeys(EMPTY_BODY_ROUTES, 422)


def test_a_form_encoded_body_is_refused(client: TestClient) -> None:
    """A form-encoded body fails the body model with 422."""
    reached = {
        url: client.post(url, data={"id": "does-not-exist"}).status_code
        for url in EMPTY_BODY_ROUTES
    }
    assert reached == dict.fromkeys(EMPTY_BODY_ROUTES, 422)


def test_a_headerless_json_shaped_body_is_refused(client: TestClient) -> None:
    """A JSON-shaped body with no ``Content-Type`` header fails the body model.

    A request built with the body ``{}`` and no ``Content-Type`` header answers 422 on each
    empty-body route.
    """
    for url in EMPTY_BODY_ROUTES:
        request = client.build_request("POST", url, content=b"{}")
        assert "content-type" not in request.headers, (
            url,
            dict(request.headers),
        )
        response = client.send(request)
        assert response.status_code == 422, (
            url,
            response.status_code,
            response.text,
        )


def test_the_guard_rejects_the_shapes_it_exists_to_catch() -> None:
    """The JSON-body predicate refuses bodiless, form, multipart and defaulted-JSON routes.

    The predicate refuses a route with no body, a form route, a multipart route and a
    defaulted JSON body model, and admits a required JSON body model.
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

    verdicts = {
        r.path: declares_json_body(r) for r in state_changing_routes(probe)
    }
    assert verdicts == {
        "/json": True,
        "/defaulted-json": False,
        "/form": False,
        "/upload": False,
        "/nothing": False,
    }
