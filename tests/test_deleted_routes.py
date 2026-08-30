"""Callerless HTTP routes stay deleted.

Each parametrized case names a route that a consumer sweep found no production or test caller
for; the route was deleted along with any test that only existed to exercise it, and the
assertions those tests carried were re-homed onto the surviving route or mechanism that already
serves the same information. Re-adding one of these routes requires a real consumer to justify it,
not a caller resurrecting the deleted path unnoticed.
"""

from __future__ import annotations

import pytest

from scripts.generate_frontend_routes import iter_api_routes
from tcip_web.app import app

DELETED_ROUTES = [
    ("GET", "/api/events/{panel}/recent"),
    ("GET", "/api/terminal/sessions"),
    ("GET", "/api/inference/jobs/{job_id}/preview"),
]


@pytest.mark.parametrize("method,path", DELETED_ROUTES)
def test_deleted_route_is_not_registered(method: str, path: str) -> None:
    from fastapi.routing import APIRoute

    for route in iter_api_routes(app):
        if isinstance(route, APIRoute) and route.path == path:
            assert method not in route.methods, f"{method} {path} is still registered"
