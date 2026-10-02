"""Callerless HTTP routes stay deleted.

Each parametrized case names a route no production or test caller reaches, and asserts the app
does not register it. The information each one served is available from a surviving route or
mechanism, so re-adding one takes a real consumer rather than a caller reviving the path
unnoticed.
"""

from __future__ import annotations

import pytest

from tools.generate_frontend_routes import iter_api_routes
from tcip_web.app import app

DELETED_ROUTES = [
    ("GET", "/api/events/{panel}/recent"),
    ("GET", "/api/terminal/sessions"),
    ("GET", "/api/inference/jobs/{job_id}/preview"),
    ("GET", "/api/images/dimensions"),
    ("GET", "/api/review/image_status"),
    ("POST", "/api/review/save_gt"),
    # One editor: a proposal is accepted or rejected, and an image marked complete, through
    # POST /api/annotate/labels; proposals are served by GET /api/annotate/proposals.
    ("POST", "/api/review/action"),
    ("POST", "/api/review/matches"),
    ("GET", "/api/review/generation_conf"),
    ("POST", "/api/subjects/image_status"),
    ("GET", "/api/coverage/grid"),
    ("GET", "/api/dataset/images"),
    ("POST", "/api/annotate/open"),
    # The two phenology doors merged into one, POST /api/results/phenology_measurement,
    # returning both projections from one _measure_phenology run.
    ("POST", "/api/results/per_plant_curves"),
    ("POST", "/api/results/onset_dates"),
    # A trait revision is read through GET /api/results/traits and confirmed through
    # POST /api/results/traits/confirm, the one confirmation door.
    ("GET", "/api/results/operationalization"),
    ("GET", "/api/results/operationalizations"),
    ("POST", "/api/results/operationalization/confirm"),
    ("GET", "/api/results/trait-spec-statement"),
    ("GET", "/api/results/trait-spec-statements"),
    ("POST", "/api/results/trait-spec-statement/confirm"),
]


@pytest.mark.parametrize("method,path", DELETED_ROUTES)
def test_deleted_route_is_not_registered(method: str, path: str) -> None:
    from fastapi.routing import APIRoute

    for route in iter_api_routes(app):
        if isinstance(route, APIRoute) and route.path == path:
            assert method not in route.methods, f"{method} {path} is still registered"


def test_the_walk_still_finds_a_live_route() -> None:
    """A positive control: an empty ``DELETED_ROUTES`` sweep above would pass every case
    vacuously, so this asserts the same walk still finds a route nobody deleted."""
    from fastapi.routing import APIRoute

    found = any(
        isinstance(route, APIRoute) and route.path == "/api/dataset/tree" and "GET" in route.methods
        for route in iter_api_routes(app)
    )
    assert found, "GET /api/dataset/tree is missing; iter_api_routes(app) may be walking nothing"
