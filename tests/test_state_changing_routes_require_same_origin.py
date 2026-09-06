"""Every state-changing route must refuse a cross-site Origin before it reaches a handler.

The check runs once, in ``TrustBoundaryMiddleware``, ahead of routing: it never sees whether the
enumerated path resolves to a real record, so the walk below is surface enumeration rather than a
per-handler check. The WebSocket surface is covered in test_trust_boundary_routes.py and
test_tcip_web_security.py; the JSON-body guard that stands behind this one on the state-changing
surface is test_state_changing_routes_require_json_body.py.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tcip_web.app import ActiveTabPayload, app
from tcip_web.trust_boundary import EXPOSURE_REFUSAL

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "generate_frontend_routes.py"

FOREIGN_ORIGIN = "http://evil.example"
_PARAM_RE = re.compile(r"\{[^}]*\}")
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
"""The methods this walk exercises, named independently of trust_boundary.STATE_CHANGING_METHODS
so a baseline that predates that constant still collects and runs this walk."""


def _route_generator():
    """The route-walking module, loaded the same way test_frontend_route_paths.py loads it,
    so this test enumerates routes through the one real walk of the app's router tree."""
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
        and (route.methods - {"HEAD", "OPTIONS"}) & _STATE_CHANGING_METHODS
    ]


def _filled_path(route: APIRoute) -> str:
    """``route.path`` with each ``{param}`` segment replaced by a placeholder no record
    resolves; the check runs before routing, so what fills the placeholder never matters."""
    return _PARAM_RE.sub("does-not-exist", route.path)


def _state_changing_calls() -> list[tuple[str, str]]:
    """(method, filled path) for every state-changing route, one entry per declared method."""
    calls = []
    for route in _state_changing_routes():
        path = _filled_path(route)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            calls.append((method, path))
    return calls


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def test_the_walk_itself_is_non_empty() -> None:
    assert _state_changing_calls(), "no state-changing routes found; the route walk itself is broken"


def test_every_state_changing_route_refuses_a_foreign_origin(client: TestClient) -> None:
    """A foreign Origin never reaches a handler: the check runs before routing, so this fails
    at the first enumerated path whether or not that path resolves to a real record."""
    for method, path in _state_changing_calls():
        resp = client.request(method, path, json={}, headers={"origin": FOREIGN_ORIGIN})
        assert resp.status_code == 403, (method, path, resp.status_code, resp.text)
        assert resp.status_code != 422
        assert "origin not allowed" in resp.text
        assert EXPOSURE_REFUSAL not in resp.text


def test_a_null_or_duplicated_origin_is_refused_the_same_way(client: TestClient) -> None:
    method, path = _state_changing_calls()[0]
    duplicated = [("origin", "http://127.0.0.1"), ("origin", "http://localhost:5173")]
    for headers in ({"origin": "null"}, duplicated):
        resp = client.request(method, path, json={}, headers=headers)
        assert resp.status_code == 403, (headers, resp.status_code, resp.text)
        assert "origin not allowed" in resp.text


def test_a_foreign_origin_post_to_a_path_no_route_serves_still_refuses(client: TestClient) -> None:
    resp = client.post("/api/does/not/exist", json={}, headers={"origin": FOREIGN_ORIGIN})
    assert resp.status_code == 403
    assert "origin not allowed" in resp.text


def test_a_permitted_origin_still_reaches_the_handler(client: TestClient) -> None:
    """The rail must admit valid work: a request from the backend's own origin, from the Vite
    dev server's origin, and with no Origin header reach the handler's own outcome. The
    backend's own origin matches the request's own authority; the Vite origin takes the
    loopback arm; a missing Origin returns before either arm runs."""
    body = ActiveTabPayload(active_tab="annotate").model_dump()
    for origin in ("http://127.0.0.1", "http://localhost:5173", None):
        headers = {"origin": origin} if origin else {}
        resp = client.post("/api/state/tab", json=body, headers=headers)
        assert resp.status_code == 200, (origin, resp.status_code, resp.text)

        resp = client.post(
            "/api/tuning/sweeps/does-not-exist/trials/does-not-exist/tensorboard/stop",
            json={}, headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_running"


def test_an_empty_origin_refuses_on_the_tab_route(client: TestClient) -> None:
    """A present but empty Origin header is checked like any other, not read as absent."""
    body = ActiveTabPayload(active_tab="annotate").model_dump()
    resp = client.post("/api/state/tab", json=body, headers={"origin": ""})
    assert resp.status_code == 403
    assert "origin not allowed" in resp.text


def test_an_exposed_arrival_admits_its_own_origin_only_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local arrival admits the backend's own origin and a loopback origin alike, so the test
    above cannot isolate the request's-own-origin arm from the loopback arm; this exercises the
    request's-own-origin arm on an exposed arrival, where no loopback arm helps."""
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    monkeypatch.delenv("TCIP_WEB_ADVERTISED_HOSTS", raising=False)
    lan = TestClient(app, base_url="http://192.168.1.23:8765")
    body = ActiveTabPayload(active_tab="annotate").model_dump()

    resp = lan.post("/api/state/tab", json=body,
                     headers={"origin": "http://192.168.1.23:8765"})
    assert resp.status_code == 200

    resp = lan.post("/api/state/tab", json=body)
    assert resp.status_code == 200

    resp = lan.post("/api/state/tab", json=body,
                     headers={"origin": "http://gui.example"})
    assert resp.status_code == 403

    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "gui.example:80")
    resp = lan.post("/api/state/tab", json=body,
                     headers={"origin": "http://gui.example"})
    assert resp.status_code == 200


def test_a_reverse_proxy_forwarding_its_own_name_is_admitted_once_advertised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-machine reverse proxy that rewrites Host to a public name is admitted by both
    the Host check and the Origin check once that name is advertised with its port written
    out: the combined case, rather than Host and Origin proven admitted separately. Passes at
    the baseline too, since no HTTP origin check existed there: coverage for the check now in
    the middleware, not a guard for this change."""
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "gui.example:80")
    lan = TestClient(app, base_url="http://192.168.1.23:8765")
    body = ActiveTabPayload(active_tab="annotate").model_dump()
    resp = lan.post("/api/state/tab", json=body,
                     headers={"host": "gui.example", "origin": "http://gui.example"})
    assert resp.status_code == 200


def test_the_local_method_set_matches_the_canonical_one() -> None:
    """The literal above exists only so this module still collects at a baseline that
    predates trust_boundary.STATE_CHANGING_METHODS; this pins it to that constant going
    forward, imported here rather than at module scope so that baseline collection stays
    unaffected by this one test."""
    from tcip_web.trust_boundary import STATE_CHANGING_METHODS

    assert _STATE_CHANGING_METHODS == STATE_CHANGING_METHODS


def test_a_get_with_a_foreign_origin_answers_as_it_does_without_one(client: TestClient) -> None:
    with_origin = client.get("/health", headers={"origin": FOREIGN_ORIGIN})
    without_origin = client.get("/health")
    assert with_origin.status_code == without_origin.status_code == 200
