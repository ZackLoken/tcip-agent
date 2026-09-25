"""Meta-loop routes: read-only, on-demand views over the friction reports and retrospectives,
enumerated, ordered and decoded by the module that owns their stores.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from tcip_web.paths import assert_project_root_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meta", tags=["meta"])


def _guard(project_root: str) -> str:
    """Confine a client-supplied project_root and hand back the resolved spelling the reads use."""
    try:
        return str(assert_project_root_allowed(project_root))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.get("/reports")
def get_reports(project_root: str, limit: int = 50) -> dict[str, Any]:
    """Return recent friction reports, newest stated timestamp first."""
    root = _guard(project_root)

    from tcip_mcp.tools.meta_tools import report_document_name, report_documents

    documents = report_documents(root)
    reports: list[dict[str, Any]] = [
        {
            "file": report_document_name(document.name),
            "timestamp": document.value.get("timestamp"),
            "category": document.value.get("category", ""),
            "detail": document.value.get("detail", ""),
            "context": document.value.get("context", {}),
        }
        for document in documents[:limit]
    ]

    return {"reports": reports, "count": len(reports), "total_available": len(documents)}


@router.get("/retrospectives")
def get_retrospectives(project_root: str, limit: int = 20) -> dict[str, Any]:
    """Return recent retrospectives (markdown), latest stated section first."""
    root = _guard(project_root)

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
