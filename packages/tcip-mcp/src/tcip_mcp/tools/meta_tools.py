"""Meta-loop tools for self-improvement.

Tools that let Claude sessions leave the system smarter than they started:
- report_friction: structured friction logging when Claude hits a problem
- write_retrospective: end-of-project reflection written to markdown
- load_project_memory: read recent reports or retrospectives at session start (closes the loop)

See docs/vision.md §6 for the design rationale.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tcip_store
from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    StoreDescriptor,
    Version,
    VersionConflict,
    register_store,
    text_codec,
)
from tcip_store.file_backend import RootedFileLocator

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


_REPORT_DOC = RootedFileLocator(prefix=(".tcip", "reports"), suffix=".json")
"""One friction report per document, under the project.

A report is one whole JSON document, not a line of a stream, so it carries the extension
that says so and every reader decodes the whole document rather than its first line.
"""

_RETROSPECTIVE_DOC = RootedFileLocator(prefix=(".tcip", "retrospectives"), suffix=".md")
"""One retrospective per project identifier, under the project."""

FRICTION_REPORT_STORE = "friction_reports"
RETROSPECTIVE_STORE = "retrospectives"

_RETROSPECTIVE_TEXT = text_codec()
"""The retrospective's bytes: the markdown itself, with nothing added around it."""

register_store(
    StoreDescriptor(
        name=FRICTION_REPORT_STORE,
        kind="record",
        key_fields=("report",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=_REPORT_DOC,
    )
)

register_store(
    StoreDescriptor(
        name=RETROSPECTIVE_STORE,
        kind="record",
        key_fields=("project",),
        frozen=True,
        cannot_carry_field="retrospective markdown text, parsed by heading and section markers only",
        codec=_RETROSPECTIVE_TEXT,
        concurrency="cas",
        enumerable=True,
        locator=_RETROSPECTIVE_DOC,
    )
)


def friction_report_key(project_path: str, report_id: str) -> Key:
    """One friction report.

    ``last_writer_wins``: the identifier carries a timestamp and a random suffix, so each
    report is written once, whole, by the call that produced it, and a conflict on the
    create-only write means some other call already owns that name.
    """
    return Key(FRICTION_REPORT_STORE, str(project_path), (report_id,))


def retrospective_key(project_path: str, project_id: str) -> Key:
    """One project's retrospective document.

    ``cas``: a retrospective is appended to by reading the stored text and writing the
    concatenation, so :func:`write_retrospective` writes against the version it read and
    re-merges on conflict rather than dropping a section.
    """
    return Key(RETROSPECTIVE_STORE, str(project_path), (project_id,))


def _report_path(project_path: str, report_id: str) -> Path:
    root = Path(project_path)
    return root.joinpath(*_REPORT_DOC.relative_path(str(root), (report_id,)).parts)


def _path_if_written(path: Path) -> str | None:
    """The document's file path when the bound backend wrote one (the file backend), else None:
    under the database backend the record lives in the store and no such file exists, so a path
    answered regardless would name a file a caller cannot open."""
    return str(path) if path.is_file() else None


def _retrospective_path(project_path: str, project_id: str) -> Path:
    root = Path(project_path)
    return root.joinpath(*_RETROSPECTIVE_DOC.relative_path(str(root), (project_id,)).parts)


def report_document_name(report_id: str) -> str:
    """The file name one report's document carries, taken from the store's own locator.

    The extension is stated here and nowhere else, so a reader presenting a report by name
    never restates it.
    """
    return f"{report_id}{_REPORT_DOC.suffix}"


def read_report(project_path: str, report_id: str) -> dict:
    """One friction report's decoded document, or ``{}`` when nothing is recorded under that id.

    The one read of this store, so a consumer never re-spells its decode. Raises
    ``DecodeError`` for a report whose bytes are present but will not read as JSON, which is
    the distinction a panel needs to show the row as malformed rather than drop it.
    """
    entry = tcip_store.read(friction_report_key(project_path, report_id), default=None)
    if entry is None:
        return {}
    return entry if isinstance(entry, dict) else {"detail": str(entry), "category": "",
                                                  "malformed": True}


def read_retrospective(project_path: str, project_id: str) -> str:
    """One project's retrospective text, or ``""`` when it has none."""
    return tcip_store.read(retrospective_key(project_path, project_id), default="")


@dataclass(frozen=True)
class MemoryDocument:
    """One project-memory document: the key part naming it, its value, and the time it states.

    ``timestamp`` is the document's own, read out of what it holds. It is empty for a document
    that states none, and nothing supplies one from the filesystem: a copy, a restore or an
    export rewrites when bytes landed, and ordering a corpus by that would reshuffle a session's
    history every time the state moved.
    """

    name: str
    value: Any
    timestamp: str


def _newest_first(documents: list[MemoryDocument]) -> list[MemoryDocument]:
    """Documents ordered by the timestamp each states, newest first, undated ones last by name.

    Two passes over a stable sort, so documents sharing a timestamp and the undated tail alike
    come back in one deterministic order rather than whichever the enumeration happened to give.
    """
    documents.sort(key=lambda document: document.name)
    documents.sort(key=lambda document: document.timestamp, reverse=True)
    return documents


def report_documents(project_path: str) -> list[MemoryDocument]:
    """Every friction report under a project, newest stated timestamp first.

    The one enumeration of this corpus: the memory tool, the GUI panel and the distillation
    worksheet all read it through here, so they cannot answer with three different orders. A
    report that will not decode is carried as a malformed row rather than dropped, since a
    reader that silently omits it reports a corpus smaller than the one on record.
    """
    documents: list[MemoryDocument] = []
    for key in tcip_store.keys(FRICTION_REPORT_STORE, str(project_path)):
        name = key.parts[0]
        try:
            entry = read_report(project_path, name)
        except DecodeError as exc:
            entry = {"detail": str(exc), "category": "", "malformed": True}
        documents.append(
            MemoryDocument(name, entry, str(entry.get("timestamp") or ""))
        )
    return _newest_first(documents)


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
    """Every retrospective under a project, latest stated section first.

    The counterpart to :func:`report_documents`, and the one ordering of this corpus for the
    same three readers.
    """
    documents: list[MemoryDocument] = []
    for key in tcip_store.keys(RETROSPECTIVE_STORE, str(project_path)):
        name = key.parts[0]
        content = read_retrospective(project_path, name)
        documents.append(MemoryDocument(name, content, _latest_section(content)))
    return _newest_first(documents)


@mcp.tool()
@audited
def report_friction(
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

    The free-text `detail` is more load-bearing than the `category` tag:
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
            back on or disagreeing with your approach, independent of category;
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
    report_id = f"{timestamp_compact}_{category}_{suffix}"

    entry = {
        "timestamp": now.isoformat(),
        "category": category,
        "detail": detail,
        "context": context or {},
        "user_disagreement": user_disagreement,
    }

    tcip_store.replace(
        friction_report_key(project_path, report_id), entry, expect=Version.ABSENT,
    )
    from tcip_mcp.project_status import record_report

    record_report(project_path)

    return {
        "report_id": report_id,
        "report_path": _path_if_written(_report_path(project_path, report_id)),
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
    selector, not an aggregator: one honest read of the chosen store):
    ``'reports'`` reads the friction reports (the counterpart to ``report_friction``);
    ``'retrospectives'`` reads the retrospectives (the counterpart to
    ``write_retrospective``). Call it early, once per kind, to pick up problems and
    context a previous session surfaced but did not resolve. Entries come back newest
    first, by the timestamp each one states rather than by when its bytes landed.

    Args:
        kind: Which corpus to read: 'reports' or 'retrospectives'.
        project_path: Root directory of the project. Empty defaults to the active
            project (matching ``inspect_project``) so the CLAUDE.md session-start
            flow (load_project_memory + inspect_project) needs no path.
        limit: Maximum number of entries to return (default 5).
        category: Reports only, optional exact category filter (e.g. 'missing_tool'),
            one of the ``report_friction`` categories; empty means all. Ignored for
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
    documents = report_documents(project_path)
    if not documents:
        return {
            "reports": [],
            "count": 0,
            "total_available": 0,
            "note": "no friction reports recorded under this project yet.",
        }

    cat = category.strip()
    needle = filter_substring.lower().strip()
    results: list[dict] = []
    for document in documents:
        entry = document.value
        if cat and entry.get("category") != cat:
            continue
        filename = report_document_name(document.name)
        if needle:
            haystack = (filename + " " + json.dumps(entry)).lower()
            if needle not in haystack:
                continue

        results.append({
            "file": filename,
            "report_id": document.name,
            "path": _path_if_written(_report_path(project_path, document.name)),
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
        "total_available": len(documents),
    }


@mcp.tool()
@audited
def write_retrospective(
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
    about what did not work: that is the most valuable part.

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
    retro_path = _retrospective_path(project_path, project_id)

    section_header = f"## Retrospective: {now.isoformat()}"
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

    from tcip_mcp.project_status import record_retrospective

    record_retrospective(project_path, project_id)

    return {
        "retrospective_path": _path_if_written(retro_path),
        "project_id": project_id,
        "timestamp": now.isoformat(),
        "appended_to_existing": appended,
    }


@mcp.tool()
@audited
def record_distillation_pass(project_path: str) -> dict:
    """Record that you reviewed this project's friction/retrospectives (e.g. via
    ``scripts/distill_learnings.py``); resets its distillation-backlog counters.

    Call this after actually reading a distillation worksheet, not before. It only records that a
    review happened; it never applies, promotes, or writes anything from the worksheet itself:
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
    documents = retrospective_documents(project_path)
    if not documents:
        return {
            "retrospectives": [],
            "count": 0,
            "total_available": 0,
            "note": "no retrospectives recorded under this project yet.",
        }

    needle = filter_substring.lower().strip()
    results: list[dict] = []
    for document in documents:
        content = document.value
        if needle and needle not in document.name.lower() and needle not in content.lower():
            continue
        results.append({
            "project_id": document.name,
            "path": _path_if_written(_retrospective_path(project_path, document.name)),
            "timestamp": document.timestamp,
            "content": content,
        })
        if len(results) >= limit:
            break

    return {
        "retrospectives": results,
        "count": len(results),
        "total_available": len(documents),
    }
