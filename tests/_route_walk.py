"""The app's routes, and the state-changing ones among them.

The routes are walked through the frontend route generator's own walk of its router tree.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from fastapi.routing import APIRoute

from tcip_web.trust_boundary import STATE_CHANGING_METHODS
from tests import REPO_ROOT

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi import FastAPI

GENERATOR = REPO_ROOT / "tools" / "generate_frontend_routes.py"


def route_generator() -> ModuleType:
    """Load ``tools/generate_frontend_routes.py`` as a module and return it."""
    spec = importlib.util.spec_from_file_location(
        "tcip_frontend_route_generator", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_changing_routes(target: FastAPI) -> list[APIRoute]:
    """Return every route of the app ``target`` declaring a state-changing method.

    A state-changing method is one in ``trust_boundary.STATE_CHANGING_METHODS``.
    """
    return [
        route
        for route in route_generator().iter_api_routes(target)
        if isinstance(route, APIRoute)
        and route.methods & STATE_CHANGING_METHODS
    ]
