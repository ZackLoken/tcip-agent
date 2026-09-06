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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def _parse_audit_timestamp(value: str | None) -> datetime | None:
    """Parse an audit entry's own stated timestamp, or ``None`` when it is absent or will not
    parse. An entry is data the platform itself wrote, never a caller argument to refuse over,
    so a bad or missing timestamp sorts and filters as "unknown" rather than aborting the read.
    A naive result is treated as UTC, matching what every writer through ``audit._entry`` stamps.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_audit_bound(label: str, value: str, *, end_of_day: bool = False) -> datetime:
    """Parse a caller-supplied ``since``/``until`` bound, or raise naming which bound and why.

    A date-only bound (no ``T`` separator) names a whole day, not an instant: ``since`` already
    starts at that day's midnight once parsed, and ``until`` with ``end_of_day=True`` is pushed
    to the last microsecond of that day, so ``until="2026-03-02"`` includes every entry from that
    date rather than only one landing on midnight exactly.
    """
    parsed = _parse_audit_timestamp(value)
    if parsed is None:
        raise ValueError(f"{label} {value!r} is not a parseable ISO-8601 timestamp")
    if end_of_day and "T" not in value:
        parsed = parsed + timedelta(days=1) - timedelta(microseconds=1)
    return parsed


@mcp.tool()
@audited
def read_audit_log(
    scope: str | None = None,
    *,
    tool: str | None = None,
    since: str | None = None,
    until: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> dict:
    """Read one audit log's own entries: which door touched a dataset or project, when, with
    what status.

    ``scope`` resolves through ``tcip_mcp.audit.dataset_scope_of``, the one resolver ``audited``
    itself calls to file a writer's own entry: a path under a dataset's canonical segment
    (``annotations``, ``predictions``, ``images``, ``labels``) resolves up to its dataset root,
    and a bare directory counts as a root only when it carries its own ``.tcip`` directory or a
    registered dataset marker (``classes.json``), which a project root does too. ``scope=None``
    is the platform default. A ``scope`` that resolves to none of these refuses by name, naming
    what was passed, rather than answering from whichever log a typo or an unrecognized inner
    path happened to resolve to (indistinguishable from an empty log otherwise). The whole log is
    read through ``tcip_store.read_log`` (no cursor: this is a bounded lookback, not a stream a
    caller resumes), filtered in memory on each entry's own ``tool`` name, ``status``, and
    ``timestamp``, then returned newest first by each entry's own stated timestamp, the same
    basis ``load_project_memory`` sorts its corpora on, never by append order (a relaunched or
    backfilled entry need not land in the order it was appended). ``skipped`` states how many
    entries this call is not returning, whether filtered out or truncated by ``limit``, so a
    caller can tell "nothing matched" apart from "more exists".

    ``since``/``until`` are ISO-8601 strings parsed with ``datetime.fromisoformat`` (a trailing
    ``Z`` is accepted), inclusive on both ends against each entry's own parsed timestamp; a bound
    that will not parse refuses by name rather than falling back to a lexical string comparison
    that would silently mis-order non-padded or mixed-precision timestamps. A date-only ``until``
    (no time part) means the end of that whole day, not its midnight start, so
    ``until="2026-03-02"`` includes every entry from that date.

    A page carrying undecodable entries, unknown-schema-version entries, or a torn tail (an
    appender's own in-flight, not-yet-newline-terminated fragment) is refused rather than
    answered from what did decode: a provenance read that silently dropped rows would be worse
    than one that says it cannot answer.

    This call's own audit entry is written after this function returns (:func:`audited` appends
    it around the call), so it is never present in this call's own result; a later read of the
    same scope sees it, the same as any earlier call's entry appears in this one.

    This is a read, not a replacement for the two readers that already scan this log for their
    own narrow question: the plant-mapping receipt scan (``pipelines.postprocessing.plant_mapping._scan_receipts``)
    and the refused-experiment-mutation index (``experiments._index_refused_mutations``) each
    keep their own targeted read, since a general-purpose page here would make them re-filter a
    result shaped for something else. Like every MCP tool this call is itself ``@audited``, so a
    read of the record is on the record too, the same as ``load_project_memory``.

    Args:
        scope: Dataset root, project root, a path under either, or ``None`` for the platform log.
        tool: Exact tool-name filter, e.g. 'save_annotations'.
        since: Only entries whose own timestamp is at or after this ISO-8601 string.
        until: Only entries whose own timestamp is at or before this ISO-8601 string; a
            date-only string means the end of that day.
        status: Exact status filter, e.g. 'ok', 'error', 'exception'.
        limit: Maximum entries to return (default 200), newest first.
    """
    from tcip_mcp.audit import audit_log_key, dataset_scope_of

    if scope is None:
        key = audit_log_key(None)
    else:
        resolved_scope = dataset_scope_of(scope)
        if resolved_scope is None:
            return {
                "error": (
                    f"scope '{scope}' names no dataset root, project root, or path under "
                    "either: pass the platform default (omit scope), a dataset root, a "
                    "project root, or a path under one; a project root must carry its own "
                    ".tcip directory or classes.json for this to resolve it"
                ),
            }
        key = audit_log_key(resolved_scope)

    page = tcip_store.read_log(key)
    if page.corrupt or page.version_refused or page.torn_tail:
        undecodable = len(page.corrupt)
        refused = len(page.version_refused)
        parts = [
            f"{undecodable} undecodable entr{'y' if undecodable == 1 else 'ies'}",
            f"{refused} version-refused entr{'y' if refused == 1 else 'ies'}",
        ]
        if page.torn_tail:
            parts.append("a torn tail from an appender still mid-write")
        return {
            "error": (
                f"the audit log at {key.root} carries {', '.join(parts)}; repair the log "
                "before trusting a read of it"
            ),
            "scope_resolved": key.root,
        }

    try:
        since_dt = _parse_audit_bound("since", since) if since is not None else None
        until_dt = (
            _parse_audit_bound("until", until, end_of_day=True) if until is not None else None
        )
    except ValueError as exc:
        return {"error": str(exc), "scope_resolved": key.root}

    def _matches(entry: Mapping[str, Any]) -> bool:
        if tool is not None and entry.get("tool") != tool:
            return False
        if status is not None and entry.get("status") != status:
            return False
        entry_ts = _parse_audit_timestamp(entry.get("timestamp"))
        if since_dt is not None and (entry_ts is None or entry_ts < since_dt):
            return False
        if until_dt is not None and (entry_ts is None or entry_ts > until_dt):
            return False
        return True

    filtered = [entry for entry in page.records if _matches(entry)]

    def _sort_key(entry: Mapping[str, Any]) -> datetime:
        return _parse_audit_timestamp(entry.get("timestamp")) or datetime.min.replace(
            tzinfo=timezone.utc
        )

    newest_first = sorted(filtered, key=_sort_key, reverse=True)
    truncated = max(0, len(newest_first) - limit)
    entries = newest_first[:limit]
    skipped = (len(page.records) - len(filtered)) + truncated

    return {
        "entries": entries,
        "count": len(entries),
        "skipped": skipped,
        "scope_resolved": key.root,
    }


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
    ``tcip distill-learnings``); resets its distillation-backlog counters.

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
