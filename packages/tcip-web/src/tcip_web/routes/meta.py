"""Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives,
enumerated, ordered and decoded by the module that owns their stores.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/reports")
def get_reports(limit: int = 50) -> dict[str, Any]:
    """Return the open project's recent friction reports, newest stated timestamp first."""
    root = str(store.open_root())

    from tcip_mcp.tools.meta_tools import report_document_name, report_documents

    documents = report_documents(root)
    reports: list[dict[str, Any]] = [
        {
            "file": report_document_name(document.name),
            "timestamp": document.timestamp,
            "category": document.value.get("category", ""),
            "detail": document.value.get("detail", ""),
            "context": document.value.get("context", {}),
        }
        for document in documents[:limit]
    ]

    return {"reports": reports, "count": len(reports), "total_available": len(documents)}


@router.get("/retrospectives")
def get_retrospectives(limit: int = 20) -> dict[str, Any]:
    """Return the open project's recent retrospectives (markdown), latest stated section first."""
    root = str(store.open_root())

    from tcip_mcp.tools.meta_tools import retrospective_documents

    documents = retrospective_documents(root)
    retrospectives: list[dict[str, Any]] = [
        {
            "project_id": document.name,
            "timestamp": document.timestamp,
            "content": document.value,
        }
        for document in documents[:limit]
    ]

    return {
        "retrospectives": retrospectives,
        "count": len(retrospectives),
        "total_available": len(documents),
    }
