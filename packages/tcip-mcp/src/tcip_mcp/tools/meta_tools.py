"""Meta-loop tools: ``report_friction``, ``write_retrospective`` and ``load_project_memory``.

Each write is one record of its corpus under an id minted for it, and a corpus is paged newest
first by the time each record states.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

import tcip_store
from tcip_mcp.audit import audited, now_iso
from tcip_mcp.pipelines.training.hpo import Count
from tcip_mcp.server import tool
from tcip_store import DecodeError, Key, Version

REPORT_CATEGORIES = {
    "missing_tool",
    "ambiguous_data",
    "cant_find_file",
    "confused_about_domain",
    "failed_repeatedly",
    "needs_human_judgment",
    "unexpected_behavior",
}


class MemoryRecord(BaseModel):
    """One write to a project-memory corpus, stamped with the time it was written.

    ``store`` names the corpus a record of the subclass is stored in.
    """

    model_config = ConfigDict(extra="forbid")

    store: ClassVar[str]
    timestamp: str


class FrictionReport(MemoryRecord):
    """One friction report as :func:`report_friction` records it."""

    store: ClassVar[str] = "friction_reports"
    category: str
    detail: str
    context: dict[str, JsonValue]
    user_disagreement: bool


class Retrospective(MemoryRecord):
    """One retrospective as :func:`write_retrospective` records it, each section as written."""

    store: ClassVar[str] = "retrospectives"
    project_id: str
    task: str
    worked: str
    did_not_work: str
    assumptions_wrong: str
    knowledge_for_future: str
    missing_or_hard_tools: str
    would_do_differently: str


def memory_key(
    record: type[MemoryRecord], project_path: str, record_id: str
) -> Key:
    """Return the key of one record of ``record``'s corpus."""
    return Key(record.store, str(project_path), (record_id,))


def read_memory[M: MemoryRecord](
    record: type[M], project_path: str, record_id: str
) -> M:
    """Read one record of ``record``'s corpus.

    Raises ``NotFoundError`` when nothing is recorded under that id, ``DecodeError`` for a record
    that will not decode, and ``ValidationError`` for one that is not a ``record``.
    """
    return record.model_validate(
        tcip_store.read(memory_key(record, project_path, record_id))
    )


def _write_memory(project: Path, name: str, record: MemoryRecord) -> str:
    """Write ``record`` as a new record of its corpus and return the id minted for it.

    The id is the record's own time to the second, ``name`` and four hex digits; a write whose id
    is already held refuses (``VersionConflictError``) with nothing replaced.
    """
    record_id = (
        f"{datetime.fromisoformat(record.timestamp):%Y%m%dT%H%M%SZ}_{name}_"
        f"{secrets.token_hex(2)}"
    )
    tcip_store.replace(
        memory_key(type(record), str(project), record_id),
        record.model_dump(mode="json"),
        expect=Version.ABSENT,
    )
    return record_id


class Unreadable(BaseModel):
    """A stored record that will not read as its corpus's record, and the reason its reader gave."""

    model_config = ConfigDict(extra="forbid")

    reason: str


class MemoryEntry[M: MemoryRecord](BaseModel):
    """One record of a project-memory corpus under the id it is stored as."""

    id: str
    record: M | Unreadable


class MemoryPage[M: MemoryRecord](BaseModel):
    """The newest entries of a corpus, and how many entries the page was cut from."""

    entries: list[MemoryEntry[M]]
    total_available: int


def memory_entries[M: MemoryRecord](
    record: type[M], project_path: str
) -> list[MemoryEntry[M]]:
    """Return every entry of ``record``'s corpus under a project, newest stated time first.

    A record that will not read (:func:`read_memory`) is carried as :class:`Unreadable` and,
    stating no time, sorts after every one that does; entries stating one time keep id order.
    """
    stated: list[tuple[str, MemoryEntry[M]]] = []
    for key in tcip_store.keys(record.store, str(project_path)):
        name = key.parts[0]
        try:
            read = read_memory(record, project_path, name)
        except (DecodeError, ValidationError) as exc:
            unread: MemoryEntry[M] = MemoryEntry(
                id=name, record=Unreadable(reason=str(exc))
            )
            stated.append(("", unread))
        else:
            stated.append((read.timestamp, MemoryEntry(id=name, record=read)))
    stated.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in stated]


def memory_page[M: MemoryRecord](
    entries: list[MemoryEntry[M]], limit: int
) -> MemoryPage[M]:
    """Return the first ``limit`` of ``entries`` as one page, with how many there are."""
    return MemoryPage(entries=entries[:limit], total_available=len(entries))


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
    report_id = _write_memory(
        project,
        category,
        FrictionReport(
            timestamp=now,
            category=category,
            detail=detail,
            context=context or {},
            user_disagreement=user_disagreement,
        ),
    )

    return {
        "id": report_id,
        "category": category,
        "timestamp": now,
        "user_disagreement": user_disagreement,
    }


@tool()
def load_project_memory(
    project: Path,
    kind: str,
    limit: Count = 5,
    category: str = "",
    filter_substring: str = "",
) -> dict:
    """Read one project-memory corpus into context, one entry per write.

    ``kind`` selects a single corpus: ``'reports'`` reads the friction reports (the counterpart to
    ``report_friction``); ``'retrospectives'`` reads the retrospectives (the counterpart to
    ``write_retrospective``), each the record of one call. Entries come back newest first, by the
    timestamp each one states, each as its id and its record (or ``reason``, for a record that
    will not read), with ``total_available``, how many entries match.

    Args:
        kind: Which corpus to read: 'reports' or 'retrospectives'.
        limit: Maximum number of entries to return, a whole number of at least one (default 5).
        category: Reports only, optional exact category filter (e.g. 'missing_tool'), one of the
            ``report_friction`` categories; empty means all. Ignored for retrospectives.
        filter_substring: Optional case-insensitive substring matched against each entry's id or
            any of its text fields (a report's detail, a retrospective's sections).

    """
    corpora: dict[str, type[MemoryRecord]] = {
        "reports": FrictionReport,
        "retrospectives": Retrospective,
    }
    if kind not in corpora:
        return {
            "error": f"unknown kind '{kind}'",
            "valid_kinds": list(corpora),
        }
    cat = category.strip() if kind == "reports" else ""
    needle = filter_substring.lower().strip()

    def matches(entry: MemoryEntry[Any]) -> bool:
        record = entry.record
        if cat and not (
            isinstance(record, FrictionReport) and record.category == cat
        ):
            return False
        texts = [entry.id, *record.model_dump().values()]
        return any(
            isinstance(text, str) and needle in text.lower() for text in texts
        )

    entries = memory_entries(corpora[kind], str(project))
    return memory_page(
        [entry for entry in entries if matches(entry)], limit
    ).model_dump(mode="json")


def _parse_audit_bound(
    label: str, value: str, *, end_of_day: bool = False
) -> datetime:
    """Parse a caller-supplied ``since``/``until`` bound, or raise naming which bound and why.

    The bound is ISO-8601, a trailing ``Z`` accepted, a naive value read as UTC.

    A date-only bound (no ``T`` separator) names a whole day, not an instant: ``since`` already
    starts at that day's midnight once parsed, and ``until`` with ``end_of_day=True`` is pushed
    to the last microsecond of that day, so ``until="2026-03-02"`` includes every entry from that
    date rather than only one landing on midnight exactly.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"{label} {value!r} is not a parseable ISO-8601 timestamp"
        ) from None
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
    """Read one audit log: which door touched a dataset or project, when, with what status.

    ``scope`` resolves through ``tcip_mcp.audit.dataset_scope_of``: a dataset's image tree, a
    capture directory or a file in either resolves to its dataset root, and a bare directory
    counts as a root only when it carries its own ``.tcip`` directory or a ``subjects.json``,
    which a project root does too. ``scope=None`` reads the
    project's own log. A ``scope`` that resolves to none of these refuses by name. The whole log
    is read
    through ``tcip_mcp.audit.audit_entries``, filtered in memory on each entry's own ``tool`` name,
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
    from tcip_mcp.audit import audit_entries, dataset_scope_of
    from tcip_mcp.registry_paths import located

    root = (
        project if scope is None else dataset_scope_of(located(scope, project))
    )
    if root is None:
        return {
            "error": (
                f"scope '{scope}' names no dataset root, project root, or path under "
                "either: omit scope for the project's own log, or pass a dataset root, a "
                "project root, or a path under one; a project root must carry its own "
                ".tcip directory or subjects.json for this to resolve it"
            ),
        }

    try:
        records, _cursor = audit_entries(root)
        since_dt = (
            _parse_audit_bound("since", since) if since is not None else None
        )
        until_dt = (
            _parse_audit_bound("until", until, end_of_day=True)
            if until is not None
            else None
        )
        timed = [
            (datetime.fromisoformat(entry["timestamp"]), entry)
            for entry in records
        ]
    except ValueError as exc:
        return {"error": str(exc), "scope_resolved": str(root)}

    def _matches(at: datetime, entry: Mapping[str, Any]) -> bool:
        return (
            (tool is None or entry["tool"] == tool)
            and (status is None or entry["status"] == status)
            and (since_dt is None or at >= since_dt)
            and (until_dt is None or at <= until_dt)
        )

    filtered = [(at, entry) for at, entry in timed if _matches(at, entry)]
    newest_first = [
        entry
        for _at, entry in sorted(
            filtered, key=lambda pair: pair[0], reverse=True
        )
    ]
    truncated = max(0, len(newest_first) - limit)
    entries = newest_first[:limit]
    skipped = (len(records) - len(filtered)) + truncated

    return {
        "entries": entries,
        "count": len(entries),
        "skipped": skipped,
        "scope_resolved": str(root),
    }


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
    """Write one retrospective, a new record of each call, under the line of work ``project_id``.

    Answers the id the record was minted, the ``project_id`` and the time it was written.

    Args:
        project_id: Short identifier for the line of work this retrospective belongs to (e.g.
            '<crop>-<trait>-trial'), not the project record's own id.
        task: What you were trying to accomplish.
        worked: What went well. Approaches, tools, decisions that paid off.
        did_not_work: What went badly. Dead ends, failures, confusion.
        assumptions_wrong: Things you assumed that turned out to be false.
        knowledge_for_future: Breeder / trait / domain knowledge a future
            session would benefit from having.
        missing_or_hard_tools: Tools that did not exist, or existed but were
            unusable.
        would_do_differently: With hindsight, what would you change about
            your approach?

    """
    now = now_iso()
    retrospective_id = _write_memory(
        project,
        project_id,
        Retrospective(
            timestamp=now,
            project_id=project_id,
            task=task,
            worked=worked,
            did_not_work=did_not_work,
            assumptions_wrong=assumptions_wrong,
            knowledge_for_future=knowledge_for_future,
            missing_or_hard_tools=missing_or_hard_tools,
            would_do_differently=would_do_differently,
        ),
    )
    return {
        "id": retrospective_id,
        "project_id": project_id,
        "timestamp": now,
    }


@tool()
@audited
def record_distillation_pass(project: Path) -> dict:
    """Record that this project's friction reports and retrospectives were reviewed.

    The record is this door's audit line; it records nothing else.
    """
    return {"status": "recorded"}
