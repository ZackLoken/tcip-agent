"""Meta-loop tools: ``report_friction``, ``write_retrospective`` and ``load_project_memory``."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import tcip_store
from tcip_store import DecodeError, Key, Version, VersionConflict

from tcip_mcp.server import tool
from tcip_mcp.audit import audited, now_iso


REPORT_CATEGORIES = {
    "missing_tool",
    "ambiguous_data",
    "cant_find_file",
    "confused_about_domain",
    "failed_repeatedly",
    "needs_human_judgment",
    "unexpected_behavior",
}


FRICTION_REPORT_STORE = "friction_reports"
RETROSPECTIVE_STORE = "retrospectives"


def friction_report_key(project_path: str, report_id: str) -> Key:
    """One friction report."""
    return Key(FRICTION_REPORT_STORE, str(project_path), (report_id,))


def retrospective_key(project_path: str, project_id: str) -> Key:
    """One project's retrospective document, its markdown text."""
    return Key(RETROSPECTIVE_STORE, str(project_path), (project_id,))


def read_report(project_path: str, report_id: str) -> dict:
    """One friction report's decoded document. Raises ``NotFound`` when nothing is recorded under
    that id and ``DecodeError`` for a report that will not read as a JSON object.
    """
    key = friction_report_key(project_path, report_id)
    entry = tcip_store.read(key)
    if not isinstance(entry, dict):
        raise DecodeError(f"{key.store}{list(key.parts)} under {key.root} is not a JSON object")
    return entry


def read_retrospective(project_path: str, project_id: str) -> str:
    """One project's retrospective text, or ``""`` when it has none."""
    return tcip_store.read(retrospective_key(project_path, project_id), default="")


@dataclass(frozen=True)
class MemoryDocument:
    """One project-memory document: the key part naming it, its value, and the time it states.
    ``timestamp`` is the document's own, read out of what it holds, and empty for a document that
    states none.
    """

    name: str
    value: Any
    timestamp: str


def _newest_first(documents: list[MemoryDocument]) -> list[MemoryDocument]:
    """Documents ordered by the timestamp each states, newest first, undated ones last by name, in
    one deterministic order.
    """
    documents.sort(key=lambda document: document.name)
    documents.sort(key=lambda document: document.timestamp, reverse=True)
    return documents


def report_documents(project_path: str) -> list[MemoryDocument]:
    """Every friction report under a project, newest stated timestamp first. A report that will not
    decode is carried with the ``DecodeError`` its reader raised as its value, stating no time.
    """
    documents: list[MemoryDocument] = []
    for key in tcip_store.keys(FRICTION_REPORT_STORE, str(project_path)):
        name = key.parts[0]
        try:
            entry = read_report(project_path, name)
        except DecodeError as exc:
            documents.append(MemoryDocument(name, exc, ""))
        else:
            documents.append(MemoryDocument(name, entry, entry["timestamp"]))
    return _newest_first(documents)


def report_row(document: MemoryDocument) -> dict:
    """One friction report as a listing shows it: its id, its time and every field its writer
    records, or its id and the reason it will not decode (``malformed``) for a report that will
    not. A decoded report lacking a field its writer records raises ``KeyError``."""
    entry = document.value
    if isinstance(entry, DecodeError):
        return {"report_id": document.name, "malformed": str(entry)}
    return {"report_id": document.name, "timestamp": entry["timestamp"],
            "category": entry["category"], "detail": entry["detail"],
            "context": entry["context"], "user_disagreement": entry["user_disagreement"]}


_RETROSPECTIVE_SECTION = re.compile(r"^## Retrospective: (.+)$", re.MULTILINE)
"""The section header :func:`write_retrospective` writes, which is where a retrospective's
own dates are recorded."""


def _latest_section(content: str) -> str:
    """The most recent time a retrospective's own section headers state, or ``""`` for none.

    The headers carry the writer's UTC ``isoformat``, one fixed-width spelling, so the greatest
    string is the latest moment. A document whose headers do not parse states no time and is
    never given one.
    """
    stated = [match.group(1).strip() for match in _RETROSPECTIVE_SECTION.finditer(content)]
    return max(stated) if stated else ""


def retrospective_documents(project_path: str) -> list[MemoryDocument]:
    """Every retrospective under a project, latest stated section first."""
    documents: list[MemoryDocument] = []
    for key in tcip_store.keys(RETROSPECTIVE_STORE, str(project_path)):
        name = key.parts[0]
        content = read_retrospective(project_path, name)
        documents.append(MemoryDocument(name, content, _latest_section(content)))
    return _newest_first(documents)


@tool()
@audited
def report_friction(
    project: Path,
    category: str,
    detail: str,
    context: dict | None = None,
    user_disagreement: bool = False,
) -> dict:
    """Record one friction report in the project: a category, what happened, and its context.

    Args:
        category: One of: missing_tool, ambiguous_data, cant_find_file, confused_about_domain,
            failed_repeatedly, needs_human_judgment, unexpected_behavior.
        detail: Free-text description of what went wrong. Be specific: what you tried, what you
            expected, what happened, what you need.
        context: Optional structured context (file paths, tool names, trait, crop, session_id,
            error messages).
        user_disagreement: True when this report is capturing the user pushing back on or
            disagreeing with your approach, independent of category.
    """
    if category not in REPORT_CATEGORIES:
        return {
            "error": f"unknown category '{category}'",
            "valid_categories": sorted(REPORT_CATEGORIES),
        }

    now = now_iso()
    report_id = f"{datetime.fromisoformat(now):%Y%m%dT%H%M%SZ}_{category}_{secrets.token_hex(2)}"

    entry = {
        "timestamp": now,
        "category": category,
        "detail": detail,
        "context": context or {},
        "user_disagreement": user_disagreement,
    }

    tcip_store.replace(
        friction_report_key(str(project), report_id), entry, expect=Version.ABSENT,
    )

    return {
        "report_id": report_id,
        "category": category,
        "timestamp": entry["timestamp"],
        "user_disagreement": user_disagreement,
    }


@tool()
def load_project_memory(
    project: Path,
    kind: str,
    limit: int = 5,
    category: str = "",
    filter_substring: str = "",
) -> dict:
    """Read one project-memory corpus into context.

    ``kind`` selects a single corpus: ``'reports'`` reads the friction reports (the counterpart to
    ``report_friction``); ``'retrospectives'`` reads the retrospectives (the counterpart to
    ``write_retrospective``). Entries come back newest first, by the timestamp each one states.

    Args:
        kind: Which corpus to read: 'reports' or 'retrospectives'.
        limit: Maximum number of entries to return (default 5).
        category: Reports only, optional exact category filter (e.g. 'missing_tool'), one of the
            ``report_friction`` categories; empty means all. Ignored for retrospectives.
        filter_substring: Optional case-insensitive substring matched against each entry's filename
            or its text.
    """
    if kind == "reports":
        return _load_reports(str(project), limit, category, filter_substring)
    if kind == "retrospectives":
        return _load_retrospectives(str(project), limit, filter_substring)
    return {"error": f"unknown kind '{kind}'", "valid_kinds": ["reports", "retrospectives"]}


def _entry_time(entry: Mapping[str, Any]) -> datetime:
    """An audit entry's own timestamp, as :func:`~tcip_mcp.audit.now_iso` wrote it. Refuses
    (``ValueError``) an entry carrying none, which no producer writes."""
    if "timestamp" not in entry:
        raise ValueError(f"an audit entry carries no timestamp, so no producer wrote it: {entry}")
    return datetime.fromisoformat(entry["timestamp"])


def _parse_audit_bound(label: str, value: str, *, end_of_day: bool = False) -> datetime:
    """Parse a caller-supplied ``since``/``until`` bound (ISO-8601, a trailing ``Z`` accepted, a
    naive value read as UTC), or raise naming which bound and why.

    A date-only bound (no ``T`` separator) names a whole day, not an instant: ``since`` already
    starts at that day's midnight once parsed, and ``until`` with ``end_of_day=True`` is pushed
    to the last microsecond of that day, so ``until="2026-03-02"`` includes every entry from that
    date rather than only one landing on midnight exactly.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} {value!r} is not a parseable ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end_of_day and "T" not in value:
        parsed = parsed + timedelta(days=1) - timedelta(microseconds=1)
    return parsed


@tool()
def read_audit_log(
    project: Path,
    scope: str | None = None,
    *,
    tool: str | None = None,
    since: str | None = None,
    until: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> dict:
    """Read one audit log's own entries: which door touched a dataset or project, when, with what
    status.

    ``scope`` resolves through ``tcip_mcp.audit.dataset_scope_of``: a path under a dataset's
    ``images/`` tree resolves up to its dataset root, and a bare directory counts as a root only when it carries its own ``.tcip``
    directory or a ``subjects.json``, which a project root does too. ``scope=None`` reads the
    project's own log. A ``scope`` that resolves to none of these refuses by name. The whole log is read
    through ``tcip_store.read_log``, filtered in memory on each entry's own ``tool`` name,
    ``status``, and ``timestamp``, then returned newest first by each entry's own stated timestamp.
    ``skipped`` states how many entries this call is not returning, whether filtered out or
    truncated by ``limit``.

    ``since``/``until`` are ISO-8601 strings parsed with ``datetime.fromisoformat`` (a trailing
    ``Z`` is accepted), inclusive on both ends against each entry's own parsed timestamp; a bound
    that will not parse refuses by name. A date-only ``until`` means the end of that whole day, so
    ``until="2026-03-02"`` includes every entry from that date.

    A page carrying undecodable entries is refused.

    Args:
        scope: Dataset root, project root, a path under either, or ``None`` for the project's own
            log.
        tool: Exact tool-name filter, e.g. 'save_annotations'.
        since: Only entries whose own timestamp is at or after this ISO-8601 string.
        until: Only entries whose own timestamp is at or before this ISO-8601 string; a date-only
            string means the end of that day.
        status: Exact status filter, 'ok' or 'exception'.
        limit: Maximum entries to return (default 200), newest first.
    """
    from tcip_mcp.audit import audit_log_key, dataset_scope_of

    if scope is None:
        key = audit_log_key(project)
    else:
        resolved_scope = dataset_scope_of(scope)
        if resolved_scope is None:
            return {
                "error": (
                    f"scope '{scope}' names no dataset root, project root, or path under "
                    "either: omit scope for the project's own log, or pass a dataset root, a "
                    "project root, or a path under one; a project root must carry its own "
                    ".tcip directory or subjects.json for this to resolve it"
                ),
            }
        key = audit_log_key(resolved_scope)

    page = tcip_store.read_log(key)
    if page.corrupt:
        undecodable = len(page.corrupt)
        return {
            "error": (
                f"the audit log at {key.root} carries {undecodable} undecodable "
                f"entr{'y' if undecodable == 1 else 'ies'}; repair the log before trusting a "
                "read of it"
            ),
            "scope_resolved": key.root,
        }

    try:
        since_dt = _parse_audit_bound("since", since) if since is not None else None
        until_dt = (
            _parse_audit_bound("until", until, end_of_day=True) if until is not None else None
        )
        timed = [(_entry_time(entry), entry) for entry in page.records]
    except ValueError as exc:
        return {"error": str(exc), "scope_resolved": key.root}

    def _matches(at: datetime, entry: Mapping[str, Any]) -> bool:
        return ((tool is None or entry.get("tool") == tool)
                and (status is None or entry.get("status") == status)
                and (since_dt is None or at >= since_dt)
                and (until_dt is None or at <= until_dt))

    filtered = [(at, entry) for at, entry in timed if _matches(at, entry)]
    newest_first = [entry for _at, entry in sorted(filtered, key=lambda pair: pair[0],
                                                   reverse=True)]
    truncated = max(0, len(newest_first) - limit)
    entries = newest_first[:limit]
    skipped = (len(page.records) - len(filtered)) + truncated

    return {
        "entries": entries,
        "count": len(entries),
        "skipped": skipped,
        "scope_resolved": key.root,
    }


def _memory_page(corpus: str, noun: str, documents: list[MemoryDocument], limit: int,
                 row: Callable[[MemoryDocument], dict | None]) -> dict:
    """Up to ``limit`` rows ``row`` makes of ``documents`` in order (a ``None`` row is skipped)
    under ``corpus``, with their count and how many documents exist; with no documents, a note
    naming ``noun``."""
    if not documents:
        return {corpus: [], "count": 0, "total_available": 0,
                "note": f"no {noun} recorded under this project yet."}
    rows: list[dict] = []
    for document in documents:
        made = row(document)
        if made is not None:
            rows.append(made)
            if len(rows) >= limit:
                break
    return {corpus: rows, "count": len(rows), "total_available": len(documents)}


def _load_reports(
    project_path: str, limit: int, category: str, filter_substring: str
) -> dict:
    cat = category.strip()
    needle = filter_substring.lower().strip()

    def row(document: MemoryDocument) -> dict | None:
        made = report_row(document)
        if cat and ("malformed" in made or made["category"] != cat):
            return None
        if needle and needle not in json.dumps(made).lower():
            return None
        return made

    return _memory_page("reports", "friction reports", report_documents(project_path), limit, row)


@tool()
@audited
def write_retrospective(
    project: Path,
    project_id: str,
    task: str,
    worked: str,
    did_not_work: str,
    assumptions_wrong: str = "",
    knowledge_for_future: str = "",
    missing_or_hard_tools: str = "",
    would_do_differently: str = "",
) -> dict:
    """Write a retrospective under ``project_id``; when one exists under that name, a new dated
    section is appended rather than overwriting it.

    Args:
        project_id: Short identifier for this retrospective (e.g. '<crop>-<trait>-trial'), not
            the project record's own id.
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
    now = now_iso()
    project_path = str(project)

    section_header = f"## Retrospective: {now}"
    body = f"""{section_header}

### Task

{task.strip()}

### What worked

{worked.strip()}

### What did not work

{did_not_work.strip()}

### Assumptions that turned out to be wrong

{assumptions_wrong.strip() or "_(none noted)_"}

### Knowledge for future sessions

{knowledge_for_future.strip() or "_(none noted)_"}

### Missing or hard-to-use tools

{missing_or_hard_tools.strip() or "_(none noted)_"}

### What I would do differently

{would_do_differently.strip() or "_(none noted)_"}

---
"""

    # Two sessions finishing at once would each append to the text they read and drop a section,
    # so a conflict re-reads and re-appends; the loop ends when this section is the one that lands.
    key = retrospective_key(project_path, project_id)
    while True:
        stored = tcip_store.read_versioned(key, default=None)
        if stored.value is None:
            content = f"# {project_id}\n\n{body}"
            appended = False
        else:
            content = stored.value.rstrip() + "\n\n" + body
            appended = True
        try:
            tcip_store.replace(key, content, expect=stored.version)
        except VersionConflict:
            continue
        break

    return {
        "project_id": project_id,
        "timestamp": now,
        "appended_to_existing": appended,
    }


@tool()
@audited
def record_distillation_pass(project: Path) -> dict:
    """Record, as this door's audit line, that this project's friction reports and retrospectives
    were reviewed; records nothing else."""
    return {"status": "recorded"}


def _load_retrospectives(
    project_path: str, limit: int, filter_substring: str
) -> dict:
    needle = filter_substring.lower().strip()

    def row(document: MemoryDocument) -> dict | None:
        content = document.value
        if needle and needle not in document.name.lower() and needle not in content.lower():
            return None
        return {
            "project_id": document.name,
            "timestamp": document.timestamp,
            "content": content,
        }

    return _memory_page("retrospectives", "retrospectives", retrospective_documents(project_path),
                        limit, row)
