"""Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives.

The records are enumerated, ordered, decoded and paged by the module that owns their stores.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import PositiveInt

from tcip_mcp.tools.meta_tools import (
    MemoryKind,
    MemoryPage,
    MemoryRecord,
    memory_entries,
    memory_page,
)
from tcip_web.state import store

router = APIRouter(prefix="/api/meta", tags=["meta"])


def _route(kind: MemoryKind, default_limit: int) -> None:
    """Register ``GET /<kind>``: a page of the open project's ``kind`` corpus, newest first.

    The page holds ``default_limit`` entries unless the request states its own ``limit``.
    """

    def page(limit: PositiveInt = default_limit) -> MemoryPage[MemoryRecord]:
        return memory_page(
            memory_entries(kind.record, str(store.held().root)), limit
        )

    router.add_api_route(
        f"/{kind}",
        page,
        methods=["GET"],
        response_model=MemoryPage[kind.record],  # type: ignore[name-defined]
    )


_route(MemoryKind.reports, 50)
_route(MemoryKind.retrospectives, 20)
