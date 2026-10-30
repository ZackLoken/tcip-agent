"""Meta-loop tools: ``report_friction``, ``write_retrospective`` and ``load_project_memory``.

Each write is one record of its corpus under an id minted for it, and a corpus is paged newest
first by the time each record states.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    JsonValue,
    ValidationError,
)

import tcip_store
from tcip_mcp.audit import audited, now
from tcip_mcp.pipelines.training.hpo import Count
from tcip_mcp.server import tool
from tcip_store import DecodeError, Key, Version

FrictionCategory = Literal[
    "missing_tool",
    "ambiguous_data",
    "cant_find_file",
    "confused_about_domain",
    "failed_repeatedly",
    "needs_human_judgment",
    "unexpected_behavior",
]


class MemoryRecord(BaseModel):
    """One write to a project-memory corpus, stamped with the time it was written."""

    model_config = ConfigDict(extra="forbid")

    timestamp: AwareDatetime


class FrictionReport(MemoryRecord):
    """One friction report as :func:`report_friction` records it."""

    category: FrictionCategory
    detail: str
    context: dict[str, JsonValue]
    user_disagreement: bool


class Retrospective(MemoryRecord):
    """One retrospective as :func:`write_retrospective` records it, each section as written."""

    project_id: str
    task: str
    worked: str
    did_not_work: str
    assumptions_wrong: str
    knowledge_for_future: str
    missing_or_hard_tools: str
    would_do_differently: str


class MemoryKind(StrEnum):
    """A project-memory corpus: the name its records are stored under, and their record class."""

    record: type[MemoryRecord]

    def __new__(cls, name: str, record: type[MemoryRecord]) -> MemoryKind:
        """Make the corpus stored as ``name`` whose records read as ``record``."""
        kind = str.__new__(cls, name)
        kind._value_ = name
        kind.record = record
        return kind

    reports = auto(), FrictionReport
    retrospectives = auto(), Retrospective


def _store(record: type[MemoryRecord]) -> str:
    """Return the name the corpus of ``record`` is stored under."""
    return next(kind.value for kind in MemoryKind if kind.record is record)


def memory_key(
    record: type[MemoryRecord], project_path: str, record_id: str
) -> Key:
    """Return the key of one record of ``record``'s corpus."""
    return Key(_store(record), str(project_path), (record_id,))


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


def _write_memory(project: Path, record: MemoryRecord) -> str:
    """Write ``record`` as a new record of its corpus under a fresh UUID and return that id."""
    record_id = uuid.uuid4().hex
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
    stated: list[tuple[datetime, MemoryEntry[M]]] = []
    unread: list[MemoryEntry[M]] = []
    for key in tcip_store.keys(_store(record), str(project_path)):
        name = key.parts[0]
        try:
            read = read_memory(record, project_path, name)
        except (DecodeError, ValidationError) as exc:
            unread.append(
                MemoryEntry(id=name, record=Unreadable(reason=str(exc)))
            )
        else:
            stated.append((read.timestamp, MemoryEntry(id=name, record=read)))
    stated.sort(key=lambda pair: pair[0], reverse=True)
    return [*(entry for _, entry in stated), *unread]


def memory_page[M: MemoryRecord](
    entries: list[MemoryEntry[M]], limit: int
) -> MemoryPage[M]:
    """Return the first ``limit`` of ``entries`` as one page, with how many there are."""
    return MemoryPage(entries=entries[:limit], total_available=len(entries))


@tool()
@audited
def report_friction(
    project: Path,
    category: FrictionCategory,
    detail: str,
    context: dict | None = None,
    user_disagreement: bool = False,
    *,
    actor: str | None,
) -> dict:
    """Record one friction report in the project: a category, what happened, and its context.

    The report's audit line is ``actor``'s.

    Args:
        category: What the report is about.
        detail: Free-text description of what went wrong. Be specific: what you tried, what you
            expected, what happened, what you need.
        context: Optional structured context (file paths, tool names, trait, crop, session_id,
            error messages).
        user_disagreement: True when this report is capturing the user pushing back on or
            disagreeing with your approach, independent of category.

    """
    written = now()
    report_id = _write_memory(
        project,
        FrictionReport(
            timestamp=written,
            category=category,
            detail=detail,
            context=context or {},
            user_disagreement=user_disagreement,
        ),
    )

    return {
        "id": report_id,
        "category": category,
        "timestamp": written,
        "user_disagreement": user_disagreement,
    }


@tool()
def load_project_memory(
    project: Path,
    kind: MemoryKind,
    limit: Count = 5,
    category: FrictionCategory | None = None,
    filter_substring: str = "",
) -> dict:
    """Read one project-memory corpus into context, one entry per write.

    Entries come back newest first, by the timestamp each one states, each as its id and its
    record (or ``reason``, for a record that will not read), with ``total_available``, how many
    entries match.

    Args:
        kind: Which corpus to read.
        limit: Maximum number of entries to return, a whole number of at least one (default 5).
        category: Only the friction reports of this category; a retrospective has none, so no
            retrospective matches one. Absent, no entry is filtered by category.
        filter_substring: Optional case-insensitive substring matched against each entry's id or
            any of its text fields (a report's detail, a retrospective's sections).

    """
    needle = filter_substring.lower().strip()

    def matches(entry: MemoryEntry[Any]) -> bool:
        record = entry.record
        if category is not None and not (
            isinstance(record, FrictionReport) and record.category == category
        ):
            return False
        texts = [entry.id, *record.model_dump().values()]
        return any(
            isinstance(text, str) and needle in text.lower() for text in texts
        )

    entries = memory_entries(kind.record, str(project))
    return memory_page(
        [entry for entry in entries if matches(entry)], limit
    ).model_dump(mode="json")


def _bound_instant(bound: datetime | date, *, end_of_day: bool) -> datetime:
    """Return the instant an audit-log bound names.

    An instant names itself. A date names its whole UTC day: its first instant, or with
    ``end_of_day`` its last, so ``until`` of a date includes every entry from that day.
    """
    if isinstance(bound, datetime):
        return bound
    start = datetime.combine(bound, time(), tzinfo=timezone.utc)
    return start + timedelta(days=1, microseconds=-1) if end_of_day else start


@tool()
def read_audit_log(
    project: Path,
    scope: str | None = None,
    *,
    tool: str | None = None,
    since: AwareDatetime | date | None = None,
    until: AwareDatetime | date | None = None,
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

    ``since``/``until`` are each an ISO-8601 instant with its offset or a date, inclusive on both
    ends against each entry's own timestamp (:func:`_bound_instant`); a date ``until`` means the
    end of that whole UTC day, so ``until="2026-03-02"`` includes every entry from that date.

    A page carrying undecodable entries is refused.

    Args:
        scope: Dataset root, project root, a path under either, or ``None`` for the project's own
            log.
        tool: Exact tool-name filter, e.g. 'save_annotations'.
        since: Only entries whose own timestamp is at or after this instant, or this date's start.
        until: Only entries whose own timestamp is at or before this instant, or this date's end.
        status: Exact status filter, 'ok' or 'exception'.
        limit: Maximum entries to return (default 200), newest first.

    """
    from tcip_mcp.audit import AuditLine, audit_entries, dataset_scope_of
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
    except ValueError as exc:
        return {"error": str(exc), "scope_resolved": str(root)}
    since_dt = None if since is None else _bound_instant(since, end_of_day=False)
    until_dt = None if until is None else _bound_instant(until, end_of_day=True)

    def _matches(entry: AuditLine) -> bool:
        return (
            (tool is None or entry.tool == tool)
            and (status is None or entry.status == status)
            and (since_dt is None or entry.timestamp >= since_dt)
            and (until_dt is None or entry.timestamp <= until_dt)
        )

    filtered = [entry for entry in records if _matches(entry)]
    newest_first = sorted(filtered, key=lambda e: e.timestamp, reverse=True)
    truncated = max(0, len(newest_first) - limit)
    entries = [
        e.model_dump(mode="json", exclude_unset=True)
        for e in newest_first[:limit]
    ]
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
    *,
    actor: str | None,
) -> dict:
    """Write one retrospective, a new record of each call, under the line of work ``project_id``.

    The write's audit line is ``actor``'s. Answers the id the record was minted, the
    ``project_id`` and the time it was written.

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
    written = now()
    retrospective_id = _write_memory(
        project,
        Retrospective(
            timestamp=written,
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
        "timestamp": written,
    }


@tool()
@audited
def record_distillation_pass(project: Path, *, actor: str | None) -> dict:
    """Record that this project's friction reports and retrospectives were reviewed.

    The record is this door's audit line, by ``actor``; it records nothing else.
    """
    return {"status": "recorded"}
