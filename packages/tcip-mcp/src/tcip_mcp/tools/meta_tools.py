"""Meta-loop tools for self-improvement.

Tools that let Claude sessions leave the system smarter than they started:
- claude_reports: structured friction logging when Claude hits a problem
- project_retrospective: end-of-project reflection written to markdown
- load_project_memory: read recent reports or retrospectives at session start (closes the loop)

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
    user_disagreement: bool = False,
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
        user_disagreement: True when this report is capturing the user pushing
            back on or disagreeing with your approach, independent of category —
            lets a later distill pass pull every disagreement out of the pile on
            its own, rather than mixed into general friction.
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
        "user_disagreement": user_disagreement,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    from tcip_mcp.project_status import record_report

    record_report(project_path)

    return {
        "report_path": str(report_path),
        "category": category,
        "timestamp": entry["timestamp"],
        "user_disagreement": user_disagreement,
    }


@mcp.tool()
@audited
def load_project_memory(
    kind: str,
    project_path: str = "",
    limit: int = 5,
    category: str = "",
    filter_substring: str = "",
) -> dict:
    """Read one project-memory corpus into context so context isn't lost between sessions.

    The read side of the session-start ritual. ``kind`` selects a single corpus (a
    selector, not an aggregator — one honest read of the chosen store):
    ``'reports'`` reads ``.tcip/reports/`` (the counterpart to ``claude_reports``);
    ``'retrospectives'`` reads ``.tcip/retrospectives/`` (the counterpart to
    ``project_retrospective``). Call it early — once per kind — to pick up problems and
    context a previous session surfaced but did not resolve. Returns the most recently
    written entries first.

    Args:
        kind: Which corpus to read — 'reports' or 'retrospectives'.
        project_path: Root directory of the project. Empty defaults to the active
            project (matching ``inspect_project``) so the CLAUDE.md session-start
            flow — load_project_memory + inspect_project — needs no path.
        limit: Maximum number of entries to return (default 5).
        category: Reports only — optional exact category filter (e.g. 'missing_tool'),
            one of the ``claude_reports`` categories; empty means all. Ignored for
            retrospectives.
        filter_substring: Optional case-insensitive substring matched against each
            entry's filename or its text.
    """
    from tcip_mcp import workspace

    project_path = workspace.resolve_project_path(project_path)

    if kind == "reports":
        return _load_reports(project_path, limit, category, filter_substring)
    if kind == "retrospectives":
        return _load_retrospectives(project_path, limit, filter_substring)
    return {"error": f"unknown kind '{kind}'", "valid_kinds": ["reports", "retrospectives"]}


def _load_reports(
    project_path: str, limit: int, category: str, filter_substring: str
) -> dict:
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
            "user_disagreement": entry.get("user_disagreement", False),
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

    from tcip_mcp.project_status import record_retrospective

    record_retrospective(project_path, project_id, retro_path)

    return {
        "retrospective_path": str(retro_path),
        "project_id": project_id,
        "timestamp": now.isoformat(),
        "appended_to_existing": appended,
    }


@mcp.tool()
@audited
def record_distillation_pass(project_path: str) -> dict:
    """Record that you reviewed this project's friction/retrospectives (e.g. via
    ``scripts/distill_learnings.py``) — resets its distillation-backlog counters.

    Call this after actually reading a distillation worksheet, not before. It only records that a
    review happened; it never applies, promotes, or writes anything from the worksheet itself —
    turning a recurring theme into a skill line, a CLAUDE.md rule, or a tool change stays your own,
    separate, explicit edit. ``distill_learnings.py`` itself stays read-only; this is the one
    audited write in the loop, kept out of the script on purpose.

    Args:
        project_path: Root directory of the project (or workspace project) reviewed.
    """
    from tcip_mcp.project_status import record_distillation

    record_distillation(project_path)
    return {"project_path": project_path, "status": "recorded"}


def _load_retrospectives(
    project_path: str, limit: int, filter_substring: str
) -> dict:
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
