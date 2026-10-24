"""Tests for meta-loop tools (report_friction, write_retrospective, load_project_memory)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.tools import meta_tools
from tcip_mcp.tools.meta_tools import (
    FrictionReport,
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
        f"2026-03-04T12:00:{second:02d}+00:00" for second in range(count)
    )
    monkeypatch.setattr(meta_tools, "now_iso", lambda: next(times))


def test_report_friction_writes_one_json_document(tmp_path: Path) -> None:
    result = report_friction(
        tmp_path,
        category="missing_tool",
        detail="I needed a way to fetch trait profiles but no such tool exists.",
        context={"trait": "efb_damage", "crop": "currant"},
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
    stored = report_friction(tmp_path, category="missing_tool", detail="x")
    assert "report_path" not in stored
    assert (
        read_memory(FrictionReport, str(tmp_path), stored["id"]).detail == "x"
    )
    listed = load_project_memory(tmp_path, "reports")["entries"][0]
    assert listed["id"] == stored["id"] and "path" not in listed


def test_report_friction_rejects_invalid_category(tmp_path: Path) -> None:
    result = report_friction(
        tmp_path,
        category="not_a_real_category",
        detail="x",
    )
    assert "error" in result
    assert "valid_categories" in result
    assert "missing_tool" in result["valid_categories"]


def test_report_friction_one_document_per_report(tmp_path: Path) -> None:
    for i in range(3):
        report_friction(
            tmp_path,
            category="unexpected_behavior",
            detail=f"report {i}",
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
    )

    assert result["project_id"] == "chestnut-bur-phase0"
    (listed,) = load_project_memory(tmp_path, "retrospectives")["entries"]
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
        )

    (newest,) = load_project_memory(tmp_path, "retrospectives", limit=1)[
        "entries"
    ]
    assert newest["record"]["task"] == "third hand-off"
    assert "first hand-off" not in json.dumps(newest)
    assert "second hand-off" not in json.dumps(newest)

    page = load_project_memory(tmp_path, "retrospectives", limit=3)
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
    )
    record = read_memory(Retrospective, str(tmp_path), result["id"])
    assert record.assumptions_wrong == ""


def test_load_retrospectives_returns_empty_when_none_recorded(
    tmp_path: Path,
) -> None:
    result = load_project_memory(tmp_path, "retrospectives")
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
        )

    result = load_project_memory(tmp_path, "retrospectives", limit=10)
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
        )

    result = load_project_memory(tmp_path, "retrospectives", limit=2)
    assert len(result["entries"]) == 2
    assert result["total_available"] == 5


def test_load_retrospectives_filter_substring(tmp_path: Path) -> None:
    write_retrospective(
        tmp_path,
        project_id="currant-efb",
        task="EFB severity",
        worked="w",
        did_not_work="d",
    )
    write_retrospective(
        tmp_path,
        project_id="chestnut-bur",
        task="Bur detection",
        worked="w",
        did_not_work="d",
    )

    (entry,) = load_project_memory(
        tmp_path, "retrospectives", filter_substring="currant"
    )["entries"]
    assert entry["record"]["project_id"] == "currant-efb"

    # Filter also matches on a section's text
    (entry,) = load_project_memory(
        tmp_path, "retrospectives", filter_substring="bur detection"
    )["entries"]
    assert entry["record"]["project_id"] == "chestnut-bur"


def test_load_reports_returns_empty_when_none_recorded(tmp_path: Path) -> None:
    result = load_project_memory(tmp_path, "reports")
    assert result == {"entries": [], "total_available": 0}


def test_load_reports_roundtrips_a_written_report(tmp_path: Path) -> None:
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="needed get_trait_profile",
        context={"crop": "currant"},
    )
    (entry,) = load_project_memory(tmp_path, "reports")["entries"]
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
        memory_key(
            FrictionReport, str(tmp_path), "20260304T120000Z_missing_tool_a1b2"
        ),
        {"timestamp": "2026-03-04T12:00:00+00:00"},
    )

    (entry,) = load_project_memory(tmp_path, "reports")["entries"]
    assert "category" in entry["record"]["reason"]


def test_a_report_that_is_not_a_json_object_will_not_decode(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError

    ts.replace(
        memory_key(
            FrictionReport, str(tmp_path), "20260304T120000Z_missing_tool_a1b2"
        ),
        "text",
    )

    with pytest.raises(ValidationError, match="FrictionReport"):
        read_memory(
            FrictionReport, str(tmp_path), "20260304T120000Z_missing_tool_a1b2"
        )


def test_load_reports_recent_first_and_respects_limit(tmp_path: Path) -> None:
    # Each report is stamped from the clock, whose tick is coarser than these calls, so the writes
    # are spaced far enough apart to state four different times rather than one.
    for i in range(4):
        report_friction(
            tmp_path, category="unexpected_behavior", detail=f"r{i}"
        )
        time.sleep(0.05)

    result = load_project_memory(tmp_path, "reports", limit=2)
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
        tmp_path, project_id="dated", task="t", worked="w", did_not_work="d"
    )
    for record_id in ("zzz-unreadable", "aaa-unreadable"):
        ts.replace(
            memory_key(Retrospective, str(tmp_path), record_id),
            "# a markdown document\n",
            expect=ts.Version.ABSENT,
        )

    entries = load_project_memory(tmp_path, "retrospectives", limit=10)[
        "entries"
    ]

    assert [entry["id"] for entry in entries] == [
        dated["id"],
        "aaa-unreadable",
        "zzz-unreadable",
    ]
    assert "Retrospective" in entries[1]["record"]["reason"]


def test_load_reports_filters_by_category(tmp_path: Path) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a")
    report_friction(tmp_path, category="ambiguous_data", detail="b")

    (entry,) = load_project_memory(
        tmp_path, "reports", category="ambiguous_data"
    )["entries"]
    assert entry["record"]["detail"] == "b"


def test_load_reports_filter_substring(tmp_path: Path) -> None:
    report_friction(
        tmp_path, category="missing_tool", detail="trait profile lookup"
    )
    report_friction(
        tmp_path, category="missing_tool", detail="something else entirely"
    )

    (entry,) = load_project_memory(
        tmp_path, "reports", filter_substring="trait profile"
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
    result = report_friction(tmp_path, category="missing_tool", detail="x")
    assert result["user_disagreement"] is False

    entry = read_memory(FrictionReport, str(tmp_path), result["id"])
    assert entry.user_disagreement is False


def test_report_friction_records_user_disagreement(tmp_path: Path) -> None:
    result = report_friction(
        tmp_path,
        category="needs_human_judgment",
        detail="Zack pushed back on the tiling default.",
        user_disagreement=True,
    )
    assert result["user_disagreement"] is True

    entry = read_memory(FrictionReport, str(tmp_path), result["id"])
    assert entry.user_disagreement is True


def test_load_reports_roundtrips_user_disagreement(tmp_path: Path) -> None:
    report_friction(
        tmp_path, category="missing_tool", detail="a", user_disagreement=False
    )
    report_friction(
        tmp_path,
        category="needs_human_judgment",
        detail="b",
        user_disagreement=True,
    )

    result = load_project_memory(tmp_path, "reports", limit=10)
    flags = {
        entry["record"]["detail"]: entry["record"]["user_disagreement"]
        for entry in result["entries"]
    }
    assert flags["a"] is False
    assert flags["b"] is True


def test_a_friction_report_counts_toward_the_recent_activity_its_audit_line_records(
    tmp_path: Path,
) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a")
    status = _recent_activity(tmp_path)
    assert status["reports_since_last_retrospective"] == 1
    assert status["reports_since_last_distillation"] == 1


def test_a_retrospective_resets_the_report_count_and_is_pointed_at_by_id(
    tmp_path: Path,
) -> None:
    report_friction(tmp_path, category="missing_tool", detail="a")
    write_retrospective(
        tmp_path, project_id="p", task="t", worked="w", did_not_work="d"
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
    report_friction(tmp_path, category="missing_tool", detail="a")
    report_friction(tmp_path, category="missing_tool", detail="b")

    result = record_distillation_pass(str(tmp_path))
    assert result["status"] == "recorded"

    status = _recent_activity(tmp_path)
    assert status["reports_since_last_distillation"] == 0
    assert status["last_distillation_at"]


def test_record_distillation_pass_never_touches_reports_or_retrospectives(
    tmp_path: Path,
) -> None:
    # Bookkeeping only: must never write/modify/delete the underlying records it's counting.
    report_friction(tmp_path, category="missing_tool", detail="a")
    before = memory_entries(FrictionReport, str(tmp_path))

    record_distillation_pass(str(tmp_path))

    after = memory_entries(FrictionReport, str(tmp_path))
    assert after == before
    report = after[0].record
    assert isinstance(report, FrictionReport) and report.detail == "a"
