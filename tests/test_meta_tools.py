"""Tests for meta-loop tools (report_friction, write_retrospective, load_project_memory)."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.tools import meta_tools
from tcip_mcp.tools.meta_tools import (
    FrictionReport,
    MemoryKind,
    Retrospective,
    load_project_memory,
    memory_entries,
    memory_key,
    read_memory,
    record_distillation_pass,
    report_friction,
    write_retrospective,
)
from tcip_mcp.tools.project_tools import _recent_activity
from tcip_mcp.workspace import BoundProject


def _clock(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Stamp the next ``count`` meta-loop writes one second apart, in order."""
    times = iter(
        datetime(2026, 3, 4, 12, 0, second, tzinfo=UTC)
        for second in range(count)
    )
    monkeypatch.setattr(meta_tools, "now", lambda: next(times))


def test_report_friction_writes_one_json_document(tmp_path: Path) -> None:
    result = report_friction(
        tmp_path,
        category="missing_tool",
        detail="I needed a way to fetch trait profiles but no such tool exists.",
        context={"trait": "efb_damage", "crop": "currant"},
        actor=None,
    )

    assert result["category"] == "missing_tool"

    entry = read_memory(FrictionReport, str(tmp_path), result["id"])
    assert entry.category == "missing_tool"
    assert entry.context["trait"] == "efb_damage"
    assert entry.timestamp


def test_a_report_names_the_record_id_it_was_stored_under(
    tmp_path: Path,
) -> None:
    """The record lives in the store, so the tool answers its id, the one a listing names."""
    stored = report_friction(
        tmp_path, category="missing_tool", detail="x", actor=None
    )
    assert "report_path" not in stored
    assert (
        read_memory(FrictionReport, str(tmp_path), stored["id"]).detail == "x"
    )
    listed = load_project_memory(tmp_path, MemoryKind.reports)["entries"][0]
    assert listed["id"] == stored["id"] and "path" not in listed


def _advertised_enum(schema: dict, parameter: str) -> list:
    """Return the enum ``schema`` advertises for ``parameter``, read through a ``$ref`` or null.

    A parameter the schema states no enum for answers an empty list.
    """
    declared = schema["properties"][parameter]
    (choice,) = [
        schema["$defs"][part["$ref"].rsplit("/", 1)[1]]
        if "$ref" in part
        else part
        for part in declared.get("anyOf", [declared])
        if part.get("type") != "null"
    ]
    return choice.get("enum", [])


def test_a_category_or_kind_outside_its_vocabulary_is_refused_at_the_mcp_door(
    project: Path, bound: BoundProject
) -> None:
    """An unknown category or kind is refused by the schema the server advertises, writing nothing.

    Each refused value lies outside the enum the registered tool's input schema advertises for
    its parameter, and the server's answer is its argument validation naming that parameter,
    never the body's own failure. A known category and kind answer, and only the known report is
    recorded and audited.
    """
    import anyio

    from tests._audit_fixtures import audit_rows
    from tests._server_calls import client_server
    from tests.test_agent_identity_records import results_through_handshake

    refusals = [
        ("report_friction", "category", {"category": "nope", "detail": "x"}),
        ("load_project_memory", "kind", {"kind": "nope"}),
        (
            "load_project_memory",
            "category",
            {"kind": "reports", "category": "nope"},
        ),
    ]
    admitted = [
        (
            "report_friction",
            {"category": "missing_tool", "detail": "admitted"},
        ),
        (
            "load_project_memory",
            {"kind": "reports", "category": "missing_tool"},
        ),
    ]
    schemas = {
        listed.name: listed.input_schema
        for listed in anyio.run(client_server(bound).list_tools)
    }
    *refused, written, listed = results_through_handshake(
        [(name, arguments) for name, _, arguments in refusals] + admitted,
        bound=bound,
    )

    for (name, parameter, arguments), result in zip(refusals, refused):
        advertised = _advertised_enum(schemas[name], parameter)
        assert advertised and arguments[parameter] not in advertised, (
            name,
            parameter,
            advertised,
        )
        (answer,) = result.content
        assert result.is_error
        assert f"validation error for {name}Arguments\n{parameter}\n" in (
            answer.text
        ), answer.text
    assert not written.is_error, written
    assert not listed.is_error, listed
    details = [
        entry.record.detail
        for entry in memory_entries(FrictionReport, str(project))
        if isinstance(entry.record, FrictionReport)
    ]
    assert details == ["admitted"]
    assert [
        row["arguments"]["detail"]
        for row in audit_rows(project, "report_friction")
    ] == ["admitted"]


def test_a_stored_report_whose_category_is_outside_the_vocabulary_reads_as_unreadable(
    tmp_path: Path,
) -> None:
    """A written report whose stored category is then changed to an unknown one reads with why."""
    written = report_friction(
        tmp_path, category="missing_tool", detail="x", actor=None
    )
    key = memory_key(FrictionReport, str(tmp_path), written["id"])
    stored = read_memory(FrictionReport, str(tmp_path), written["id"])
    ts.replace(
        key, stored.model_dump(mode="json") | {"category": "not_a_category"}
    )

    (entry,) = load_project_memory(tmp_path, MemoryKind.reports)["entries"]
    assert entry["id"] == written["id"]
    assert "category" in entry["record"]["reason"]


def test_two_reports_in_one_instant_under_one_category_land_under_ids_naming_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reports stamped with one held time under one category are two records.

    Neither id carries the time or the category, which the record's own fields state.
    """
    monkeypatch.setattr(
        meta_tools, "now", lambda: datetime(2026, 3, 4, 12, tzinfo=UTC)
    )
    first, second = (
        report_friction(
            tmp_path, category="missing_tool", detail=detail, actor=None
        )["id"]
        for detail in ("first", "second")
    )

    assert first != second
    for record_id in (first, second):
        assert "20260304" not in record_id and "missing_tool" not in record_id
    assert [
        read_memory(FrictionReport, str(tmp_path), record_id).detail
        for record_id in (first, second)
    ] == ["first", "second"]


def test_report_friction_one_document_per_report(tmp_path: Path) -> None:
    for i in range(3):
        report_friction(
            tmp_path,
            category="unexpected_behavior",
            detail=f"report {i}",
            actor=None,
        )

    assert len(memory_entries(FrictionReport, str(tmp_path))) == 3


def test_a_retrospective_names_the_record_id_a_listing_names(
    tmp_path: Path,
) -> None:
    result = write_retrospective(
        tmp_path,
        project_id="chestnut-bur-phase0",
        task="Bootstrap bur detection from scratch",
        worked="SAM candidates were usable after filtering.",
        did_not_work="Active learning selection stagnated after 50 labels.",
        knowledge_for_future="Burs cluster near canopy edges, not centers.",
        actor=None,
    )

    assert result["project_id"] == "chestnut-bur-phase0"
    (listed,) = load_project_memory(tmp_path, MemoryKind.retrospectives)[
        "entries"
    ]
    assert listed["id"] == result["id"]
    assert listed["record"]["knowledge_for_future"] == (
        "Burs cluster near canopy edges, not centers."
    )


def test_hand_offs_under_one_name_are_each_their_own_entry_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three hand-offs under one name page as three entries; a page of one holds the third alone."""
    _clock(monkeypatch, 3)
    for task in ("first hand-off", "second hand-off", "third hand-off"):
        write_retrospective(
            tmp_path,
            project_id="valley-bloom-trial",
            task=task,
            worked="w",
            did_not_work="d",
            actor=None,
        )

    (newest,) = load_project_memory(
        tmp_path, MemoryKind.retrospectives, limit=1
    )["entries"]
    assert newest["record"]["task"] == "third hand-off"
    assert "first hand-off" not in json.dumps(newest)
    assert "second hand-off" not in json.dumps(newest)

    page = load_project_memory(tmp_path, MemoryKind.retrospectives, limit=3)
    assert [entry["record"]["task"] for entry in page["entries"]] == [
        "third hand-off",
        "second hand-off",
        "first hand-off",
    ]


def test_concurrent_retrospectives_under_one_name_both_survive(
    tmp_path: Path,
) -> None:
    """Two sessions finishing together under one name leave two records, neither overwritten."""
    import threading

    barrier = threading.Barrier(2)

    def _call(index: int) -> None:
        barrier.wait()
        write_retrospective(
            tmp_path,
            project_id="project-under-test",
            task=f"pass {index}",
            worked="a",
            did_not_work="b",
            actor=None,
        )

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    tasks = {
        entry.record.task
        for entry in memory_entries(Retrospective, str(tmp_path))
        if isinstance(entry.record, Retrospective)
    }
    assert tasks == {"pass 0", "pass 1"}


def test_a_retrospective_records_an_unstated_section_as_empty(
    tmp_path: Path,
) -> None:
    result = write_retrospective(
        tmp_path,
        project_id="minimal",
        task="t",
        worked="w",
        did_not_work="d",
        actor=None,
    )
    record = read_memory(Retrospective, str(tmp_path), result["id"])
    assert record.assumptions_wrong == ""


def test_load_retrospectives_returns_empty_when_none_recorded(
    tmp_path: Path,
) -> None:
    result = load_project_memory(tmp_path, MemoryKind.retrospectives)
    assert result == {"entries": [], "total_available": 0}


def test_load_retrospectives_returns_recent_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clock(monkeypatch, 3)
    for project_id in ("first", "second", "third"):
        write_retrospective(
            tmp_path,
            project_id=project_id,
            task="t",
            worked="w",
            did_not_work="d",
            actor=None,
        )

    result = load_project_memory(tmp_path, MemoryKind.retrospectives, limit=10)
    ids = [entry["record"]["project_id"] for entry in result["entries"]]
    assert ids == ["third", "second", "first"]


def test_load_retrospectives_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        write_retrospective(
            tmp_path,
            project_id=f"project-{i}",
            task="t",
            worked="w",
            did_not_work="d",
            actor=None,
        )

    result = load_project_memory(tmp_path, MemoryKind.retrospectives, limit=2)
    assert len(result["entries"]) == 2
    assert result["total_available"] == 5


def test_load_retrospectives_filter_substring(tmp_path: Path) -> None:
    write_retrospective(
        tmp_path,
        project_id="currant-efb",
        task="EFB severity",
        worked="w",
        did_not_work="d",
        actor=None,
    )
    write_retrospective(
        tmp_path,
        project_id="chestnut-bur",
        task="Bur detection",
        worked="w",
        did_not_work="d",
        actor=None,
    )

    (entry,) = load_project_memory(
        tmp_path, MemoryKind.retrospectives, filter_substring="currant"
    )["entries"]
    assert entry["record"]["project_id"] == "currant-efb"

    # Filter also matches on a section's text
    (entry,) = load_project_memory(
        tmp_path, MemoryKind.retrospectives, filter_substring="bur detection"
    )["entries"]
    assert entry["record"]["project_id"] == "chestnut-bur"


def test_load_reports_returns_empty_when_none_recorded(tmp_path: Path) -> None:
    result = load_project_memory(tmp_path, MemoryKind.reports)
    assert result == {"entries": [], "total_available": 0}


def test_load_reports_roundtrips_a_written_report(tmp_path: Path) -> None:
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="needed get_trait_profile",
        context={"crop": "currant"},
        actor=None,
    )
    (entry,) = load_project_memory(tmp_path, MemoryKind.reports)["entries"]
    rep = entry["record"]
    assert rep["category"] == "missing_tool"
    assert rep["detail"] == "needed get_trait_profile"
    assert rep["context"]["crop"] == "currant"
    assert rep["timestamp"]


def test_a_report_missing_a_field_its_writer_records_refuses_rather_than_reading_as_blank(
    tmp_path: Path,
) -> None:
    """A stored report holding its time alone is listed as unreadable, naming ``category``."""
    ts.replace(
        memory_key(FrictionReport, str(tmp_path), "unreadable"),
        {"timestamp": "2026-03-04T12:00:00+00:00"},
    )

    (entry,) = load_project_memory(tmp_path, MemoryKind.reports)["entries"]
    assert "category" in entry["record"]["reason"]


def test_a_report_that_is_not_a_json_object_will_not_decode(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError

    ts.replace(memory_key(FrictionReport, str(tmp_path), "unreadable"), "text")

    with pytest.raises(ValidationError, match="FrictionReport"):
        read_memory(FrictionReport, str(tmp_path), "unreadable")


def test_load_reports_recent_first_and_respects_limit(tmp_path: Path) -> None:
    # Each report is stamped from the clock, whose tick is coarser than these calls, so the writes
    # are spaced far enough apart to state four different times rather than one.
    for i in range(4):
        report_friction(
            tmp_path,
            category="unexpected_behavior",
            detail=f"r{i}",
            actor=None,
        )
        time.sleep(0.05)

    result = load_project_memory(tmp_path, MemoryKind.reports, limit=2)
    assert result["total_available"] == 4
    # Most recent first
    assert [entry["record"]["detail"] for entry in result["entries"]] == [
        "r3",
        "r2",
    ]


def test_a_retrospective_that_will_not_read_sorts_after_every_dated_one_by_id(
    tmp_path: Path,
) -> None:
    """Retrospectives that will not read list after the dated one, ordered by id, with why."""
    dated = write_retrospective(
        tmp_path,
        project_id="dated",
        task="t",
        worked="w",
        did_not_work="d",
        actor=None,
    )
    for record_id in ("zzz-unreadable", "aaa-unreadable"):
        ts.replace(
            memory_key(Retrospective, str(tmp_path), record_id),
            "# a markdown document\n",
            expect=ts.Version.ABSENT,
        )

    entries = load_project_memory(
        tmp_path, MemoryKind.retrospectives, limit=10
    )["entries"]

    assert [entry["id"] for entry in entries] == [
        dated["id"],
        "aaa-unreadable",
        "zzz-unreadable",
    ]
    assert "Retrospective" in entries[1]["record"]["reason"]


def test_load_reports_filters_by_category(tmp_path: Path) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a", actor=None)
    report_friction(
        tmp_path, category="ambiguous_data", detail="b", actor=None
    )

    (entry,) = load_project_memory(
        tmp_path, MemoryKind.reports, category="ambiguous_data"
    )["entries"]
    assert entry["record"]["detail"] == "b"


def test_a_category_filter_over_retrospectives_matches_none(
    tmp_path: Path,
) -> None:
    """A retrospective carries no category, so a category filter over them answers none."""
    write_retrospective(
        tmp_path,
        project_id="p",
        task="t",
        worked="w",
        did_not_work="d",
        actor=None,
    )

    assert load_project_memory(
        tmp_path, MemoryKind.retrospectives, category="missing_tool"
    ) == {"entries": [], "total_available": 0}
    assert (
        load_project_memory(tmp_path, MemoryKind.retrospectives)[
            "total_available"
        ]
        == 1
    )


def test_load_reports_filter_substring(tmp_path: Path) -> None:
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="trait profile lookup",
        actor=None,
    )
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="something else entirely",
        actor=None,
    )

    (entry,) = load_project_memory(
        tmp_path, MemoryKind.reports, filter_substring="trait profile"
    )["entries"]
    assert entry["record"]["detail"] == "trait profile lookup"


def test_the_page_limit_is_refused_at_the_mcp_door_unless_a_whole_number_of_at_least_one(
    bound: BoundProject,
) -> None:
    """Through the server's own registration, a limit of 0, -1 or text refuses and 1 answers."""
    from tests.test_agent_identity_records import results_through_handshake

    calls = [
        ("load_project_memory", {"kind": "reports", "limit": limit})
        for limit in (0, -1, "5", 1)
    ]
    *refused, answered = results_through_handshake(calls, bound=bound)

    assert [result.is_error for result in refused] == [True, True, True]
    assert not answered.is_error, answered


def test_report_friction_defaults_user_disagreement_false(
    tmp_path: Path,
) -> None:
    result = report_friction(
        tmp_path, category="missing_tool", detail="x", actor=None
    )
    assert result["user_disagreement"] is False

    entry = read_memory(FrictionReport, str(tmp_path), result["id"])
    assert entry.user_disagreement is False


def test_report_friction_records_user_disagreement(tmp_path: Path) -> None:
    result = report_friction(
        tmp_path,
        category="needs_human_judgment",
        detail="Zack pushed back on the tiling default.",
        user_disagreement=True,
        actor=None,
    )
    assert result["user_disagreement"] is True

    entry = read_memory(FrictionReport, str(tmp_path), result["id"])
    assert entry.user_disagreement is True


def test_load_reports_roundtrips_user_disagreement(tmp_path: Path) -> None:
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="a",
        user_disagreement=False,
        actor=None,
    )
    report_friction(
        tmp_path,
        category="needs_human_judgment",
        detail="b",
        user_disagreement=True,
        actor=None,
    )

    result = load_project_memory(tmp_path, MemoryKind.reports, limit=10)
    flags = {
        entry["record"]["detail"]: entry["record"]["user_disagreement"]
        for entry in result["entries"]
    }
    assert flags["a"] is False
    assert flags["b"] is True


def test_a_friction_report_counts_toward_the_recent_activity_its_audit_line_records(
    tmp_path: Path,
) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a", actor=None)
    status = _recent_activity(tmp_path)
    assert status["reports_since_last_retrospective"] == 1
    assert status["reports_since_last_distillation"] == 1


def test_a_retrospective_resets_the_report_count_and_is_pointed_at_by_id(
    tmp_path: Path,
) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a", actor=None)
    write_retrospective(
        tmp_path,
        project_id="p",
        task="t",
        worked="w",
        did_not_work="d",
        actor=None,
    )

    status = _recent_activity(tmp_path)
    assert status["reports_since_last_retrospective"] == 0
    assert status["last_retrospective"]["project_id"] == "p"
    assert "worked" not in json.dumps(
        status
    )  # no retrospective text, pointer only


def test_record_distillation_pass_resets_distillation_counters(
    tmp_path: Path,
) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a", actor=None)
    report_friction(tmp_path, category="missing_tool", detail="b", actor=None)

    result = record_distillation_pass(str(tmp_path), actor=None)
    assert result["status"] == "recorded"

    status = _recent_activity(tmp_path)
    assert status["reports_since_last_distillation"] == 0
    assert status["last_distillation_at"]


def test_record_distillation_pass_never_touches_reports_or_retrospectives(
    tmp_path: Path,
) -> None:
    # Bookkeeping only: must never write/modify/delete the underlying records it's counting.
    report_friction(tmp_path, category="missing_tool", detail="a", actor=None)
    before = memory_entries(FrictionReport, str(tmp_path))

    record_distillation_pass(str(tmp_path), actor=None)

    after = memory_entries(FrictionReport, str(tmp_path))
    assert after == before
    report = after[0].record
    assert isinstance(report, FrictionReport) and report.detail == "a"
