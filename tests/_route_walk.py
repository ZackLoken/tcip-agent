"""The app's routes, and the state-changing ones among them.

The routes are walked through the frontend route generator's own walk of its router tree.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from fastapi.routing import APIRoute

from tcip_web.trust_boundary import STATE_CHANGING_METHODS

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi import FastAPI


def route_generator() -> ModuleType:
    """Import and return ``tools.generate_frontend_routes``."""
    return importlib.import_module("tools.generate_frontend_routes")


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
