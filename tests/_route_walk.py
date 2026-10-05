"""The app's routes walked through the frontend route generator's own walk of its router tree,
and the state-changing ones among them."""

from __future__ import annotations

import importlib.util

from fastapi.routing import APIRoute

from tcip_web.trust_boundary import STATE_CHANGING_METHODS
from tests import REPO_ROOT

GENERATOR = REPO_ROOT / "tools" / "generate_frontend_routes.py"


def route_generator():
    """``tools/generate_frontend_routes.py`` loaded as a module, the walk CI regenerates with."""
    spec = importlib.util.spec_from_file_location("tcip_frontend_route_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_changing_routes(target) -> list[APIRoute]:
    """Every route of the app ``target`` declaring a method in
    ``trust_boundary.STATE_CHANGING_METHODS``."""
    return [route for route in route_generator().iter_api_routes(target)
            if isinstance(route, APIRoute) and route.methods & STATE_CHANGING_METHODS]
