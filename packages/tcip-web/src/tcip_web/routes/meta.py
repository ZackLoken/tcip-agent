"""Meta-loop routes: surface Claude's friction reports and retrospectives.

Read-only views over ``<project_root>/.tcip/reports/*.json`` (written by the
``claude_reports`` MCP tool) and ``<project_root>/.tcip/retrospectives/*.md``
(written by ``project_retrospective``). These close the loop on the meta-tools:
the agent writes friction/retrospectives, the human can read them in the GUI.

Endpoints are intentionally on-demand reads (not part of the live GUI state /
WebSocket broadcast): this data is occasional and long-form.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from tcip_web.paths import assert_project_root_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meta", tags=["meta"])


def _guard(project_root: str) -> None:
    """Confine a client-supplied project_root (no-op unless TCIP_IMAGE_ROOTS is set)."""
    try:
        assert_project_root_allowed(project_root)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.get("/reports")
def get_reports(project_root: str, limit: int = 50) -> dict[str, Any]:
    """Return recent friction reports, most recent first."""
    _guard(project_root)

    from tcip_mcp.tools.meta_tools import reports_dir

    if not reports_dir(project_root).exists():
        return {"reports": [], "count": 0, "total_available": 0}

    from tcip_mcp.tools.meta_tools import report_documents

    files = report_documents(project_root)

    from tcip_store import DecodeError

    from tcip_mcp.tools.meta_tools import read_report

    reports: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            entry = read_report(project_root, path.stem)
        except DecodeError:
            # Show the breeder what is in an unreadable report rather than dropping the row.
            entry = {"detail": path.read_text(encoding="utf-8").strip(),
                     "category": "", "malformed": True}
        reports.append({
            "file": path.name,
            "timestamp": entry.get("timestamp"),
            "category": entry.get("category", ""),
            "detail": entry.get("detail", ""),
            "context": entry.get("context", {}),
        })

    return {"reports": reports, "count": len(reports), "total_available": len(files)}


@router.get("/retrospectives")
def get_retrospectives(project_root: str, limit: int = 20) -> dict[str, Any]:
    """Return recent retrospectives (markdown), most recent first."""
    _guard(project_root)

    from tcip_mcp.tools.meta_tools import retrospectives_dir

    retros_dir = retrospectives_dir(project_root)
    if not retros_dir.exists():
        return {"retrospectives": [], "count": 0, "total_available": 0}

    files = sorted(
        retros_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    from tcip_mcp.tools.meta_tools import read_retrospective

    retrospectives: list[dict[str, Any]] = []
    for path in files[:limit]:
        retrospectives.append({
            "project_id": path.stem,
            "modified": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "content": read_retrospective(project_root, path.stem),
        })

    return {
        "retrospectives": retrospectives,
        "count": len(retrospectives),
        "total_available": len(files),
    }
