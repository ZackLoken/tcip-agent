"""The app's routes, and the state-changing ones among them.

The routes are the frontend route generator's own FastAPI route contexts.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from fastapi.routing import APIRoute

from tcip_web.trust_boundary import STATE_CHANGING_METHODS

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi import FastAPI
    from fastapi.routing import RouteContext


def route_generator() -> ModuleType:
    """Import and return ``tools.generate_frontend_routes``."""
    return importlib.import_module("tools.generate_frontend_routes")


def state_changing_routes(target: FastAPI) -> list[RouteContext]:
    """Return the context of every route of the app ``target`` declaring a state-changing method.

    A state-changing method is one in ``trust_boundary.STATE_CHANGING_METHODS``.
    """
    return [
        context
        for context in route_generator().api_route_contexts(target)
        if isinstance(context.original_route, APIRoute)
        and context.methods & STATE_CHANGING_METHODS
    ]
