"""Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives.

The records are enumerated, ordered, decoded and paged by the module that owns their stores.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import PositiveInt

from tcip_mcp.tools.meta_tools import (
    FrictionReport,
    MemoryPage,
    Retrospective,
    memory_entries,
    memory_page,
)
from tcip_web.state import store

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/reports")
def get_reports(limit: PositiveInt = 50) -> MemoryPage[FrictionReport]:
    """Return a page of the open project's friction reports, newest stated timestamp first."""
    return memory_page(
        memory_entries(FrictionReport, str(store.held().root)), limit
    )


@router.get("/retrospectives")
def get_retrospectives(
    limit: PositiveInt = 20,
) -> MemoryPage[Retrospective]:
    """Return a page of the open project's retrospectives, newest stated timestamp first."""
    return memory_page(
        memory_entries(Retrospective, str(store.held().root)), limit
    )
