"""Meta-loop tools for self-improvement.

Tools that let Claude sessions leave the system smarter than they started:
- claude_reports: structured friction logging when Claude hits a problem
- load_reports: read recent friction reports at session start (closes the loop)
- project_retrospective: end-of-project reflection written to markdown
- load_retrospectives: read recent retrospectives at session start

See docs/vision.md §6 for the design rationale.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


REPORT_CATEGORIES = {
    "missing_tool",
    "ambiguous_data",
    "cant_find_file",
    "confused_about_domain",
    "failed_repeatedly",
    "needs_human_judgment",
    "unexpected_behavior",
}


def _project_dir(project_path: str) -> Path:
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reports_dir(project_path: str) -> Path:
    d = _project_dir(project_path) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _retrospectives_dir(project_path: str) -> Path:
    d = _project_dir(project_path) / "retrospectives"
    d.mkdir(parents=True, exist_ok=True)
    return d


@mcp.tool()
@audited
def claude_reports(
    project_path: str,
    category: str,
    detail: str,
    context: dict | None = None,
) -> dict:
    """Log structured friction when you get stuck, confused, or surprised.

    Call this whenever you hit a problem you'd otherwise push through silently.
    Surfacing friction is how the system gets smarter. Do not wait until the
    end of the session; report while the context is fresh.

    The free-text `detail` is more load-bearing than the `category` tag —
    categorical labels are easy to get wrong, but a clear written description
    of what went wrong survives mis-labeling.

    Args:
        project_path: Root directory of the project.
        category: One of: missing_tool, ambiguous_data, cant_find_file,
            confused_about_domain, failed_repeatedly, needs_human_judgment,
            unexpected_behavior.
        detail: Free-text description of what went wrong. Be specific:
            what you tried, what you expected, what happened, what you need.
        context: Optional structured context (file paths, tool names, trait,
            crop, session_id, error messages). Preserves raw signal for later
            review.
    """
    if category not in REPORT_CATEGORIES:
        return {
            "error": f"unknown category '{category}'",
            "valid_categories": sorted(REPORT_CATEGORIES),
        }

    now = datetime.now(timezone.utc)
    timestamp_compact = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(2)
    filename = f"{timestamp_compact}_{category}_{suffix}.jsonl"

    reports = _reports_dir(project_path)
    report_path = reports / filename

    entry = {
        "timestamp": now.isoformat(),
        "category": category,
        "detail": detail,
        "context": context or {},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "report_path": str(report_path),
        "category": category,
        "timestamp": entry["timestamp"],
    }


@mcp.tool()
@audited
def load_reports(
    project_path: str = "",
    limit: int = 5,
    category: str = "",
    filter_substring: str = "",
) -> dict:
    """Read recent friction reports so open problems aren't lost between sessions.

    The counterpart to ``claude_reports``: that tool writes friction to
    ``.tcip/reports/``; this one reads it back. Call it early in a session
    (alongside ``load_retrospectives``) to pick up problems a previous session
    surfaced but did not resolve. Returns the most recently written reports first.

    Args:
        project_path: Root directory of the project. Empty defaults to the active
            project (matching ``get_project_status``) so the CLAUDE.md session-start
            flow — load_reports + load_retrospectives + get_project_status — needs no path.
        limit: Maximum number of reports to return (default 5).
        category: Optional exact category filter (e.g. 'missing_tool'). One of the
            ``claude_reports`` categories; empty means all categories.
        filter_substring: Optional case-insensitive substring matched against the
            report's filename or its detail/context text.
    """
    from tcip_mcp import workspace

    project_path = workspace.resolve_project_path(project_path)
    reports_dir = Path(project_path) / ".tcip" / "reports"
    if not reports_dir.exists():
        return {
            "reports": [],
            "count": 0,
            "note": f"{reports_dir} does not exist yet — no friction reports.",
        }

    files = sorted(
        reports_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    cat = category.strip()
    needle = filter_substring.lower().strip()
    results: list[dict] = []
    for path in files:
        raw = path.read_text(encoding="utf-8").strip()
        try:
            entry = json.loads(raw.splitlines()[0]) if raw else {}
        except (json.JSONDecodeError, IndexError):
            entry = {"detail": raw, "category": "", "malformed": True}

        if cat and entry.get("category") != cat:
            continue
        if needle:
            haystack = (path.name + " " + json.dumps(entry)).lower()
            if needle not in haystack:
                continue

        results.append({
            "file": path.name,
            "path": str(path),
            "timestamp": entry.get("timestamp"),
            "category": entry.get("category", ""),
            "detail": entry.get("detail", ""),
            "context": entry.get("context", {}),
        })
        if len(results) >= limit:
            break

    return {
        "reports": results,
        "count": len(results),
        "total_available": len(files),
    }


@mcp.tool()
@audited
def project_retrospective(
    project_path: str,
    project_id: str,
    task: str,
    worked: str,
    did_not_work: str,
    assumptions_wrong: str = "",
    knowledge_for_future: str = "",
    missing_or_hard_tools: str = "",
    would_do_differently: str = "",
) -> dict:
    """Write an end-of-project retrospective to markdown.

    Call this when you finish a substantial piece of work, even if incomplete.
    The retrospective is how future sessions learn from this one. Be honest
    about what did not work — that is the most valuable part.

    Writes to .tcip/retrospectives/<project_id>.md. If the file exists, a new
    dated section is appended rather than overwriting.

    Args:
        project_path: Root directory of the project.
        project_id: Short identifier for the project (e.g. 'chestnut-bur-phase0').
            Becomes the filename.
        task: What you were trying to accomplish.
        worked: What went well. Approaches, tools, decisions that paid off.
        did_not_work: What went badly. Dead ends, failures, confusion.
        assumptions_wrong: Things you assumed that turned out to be false.
        knowledge_for_future: Breeder / trait / domain knowledge a future
            session would benefit from having. Candidates for new skill files.
        missing_or_hard_tools: Tools that did not exist, or existed but were
            unusable. Candidates for tool changes.
        would_do_differently: With hindsight, what would you change about
            your approach?
    """
    now = datetime.now(timezone.utc)
    retros = _retrospectives_dir(project_path)
    retro_path = retros / f"{project_id}.md"

    section_header = f"## Retrospective — {now.isoformat()}"
    body = f"""{section_header}

**Task**

{task.strip()}

**What worked**

{worked.strip()}

**What did not work**

{did_not_work.strip()}

**Assumptions that turned out to be wrong**

{assumptions_wrong.strip() or "_(none noted)_"}

**Knowledge for future sessions**

{knowledge_for_future.strip() or "_(none noted)_"}

**Missing or hard-to-use tools**

{missing_or_hard_tools.strip() or "_(none noted)_"}

**What I would do differently**

{would_do_differently.strip() or "_(none noted)_"}

---
"""

    if retro_path.exists():
        existing = retro_path.read_text(encoding="utf-8")
        content = existing.rstrip() + "\n\n" + body
        appended = True
    else:
        content = f"# {project_id}\n\n{body}"
        appended = False

    retro_path.write_text(content, encoding="utf-8")

    return {
        "retrospective_path": str(retro_path),
        "project_id": project_id,
        "timestamp": now.isoformat(),
        "appended_to_existing": appended,
    }


@mcp.tool()
@audited
def load_retrospectives(
    project_path: str = "",
    limit: int = 5,
    filter_substring: str = "",
) -> dict:
    """Read recent retrospectives so you start the session with context.

    Call this early in a session to learn what past sessions did, what worked,
    what failed, and what knowledge they captured. Returns the most recently
    modified retrospectives first.

    Args:
        project_path: Root directory of the project. Empty defaults to the active
            project (matching ``get_project_status``) so the session-start flow needs no path.
        limit: Maximum number of retrospectives to return (default 5).
        filter_substring: Optional case-insensitive substring. Only
            retrospectives whose filename OR content contains this string
            will be returned.
    """
    from tcip_mcp import workspace

    project_path = workspace.resolve_project_path(project_path)
    retros_dir = Path(project_path) / ".tcip" / "retrospectives"
    if not retros_dir.exists():
        return {
            "retrospectives": [],
            "count": 0,
            "note": f"{retros_dir} does not exist yet — no prior retrospectives.",
        }

    files = sorted(
        retros_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    needle = filter_substring.lower().strip()
    results: list[dict] = []
    for path in files:
        content = path.read_text(encoding="utf-8")
        if needle and needle not in path.name.lower() and needle not in content.lower():
            continue
        results.append({
            "project_id": path.stem,
            "path": str(path),
            "modified": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "content": content,
        })
        if len(results) >= limit:
            break

    return {
        "retrospectives": results,
        "count": len(results),
        "total_available": len(files),
    }
