"""Tests for meta-loop tools (report_friction, write_retrospective, load_project_memory)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import tcip_store as ts

from tcip_mcp.tools.project_tools import _recent_activity
from tcip_mcp.tools.meta_tools import (
    report_friction,
    load_project_memory,
    write_retrospective,
    read_report,
    read_retrospective,
    record_distillation_pass,
    report_documents,
    retrospective_key,
)


def test_report_friction_writes_one_json_document(tmp_path: Path):
    result = report_friction(
        tmp_path,
        category="missing_tool",
        detail="I needed a way to fetch trait profiles but no such tool exists.",
        context={"trait": "efb_damage", "crop": "currant"},
    )

    assert result["category"] == "missing_tool"

    entry = read_report(str(tmp_path), result["report_id"])
    assert entry["category"] == "missing_tool"
    assert entry["context"]["trait"] == "efb_damage"
    assert "timestamp" in entry


def test_a_report_names_the_record_id_it_was_stored_under(tmp_path: Path):
    """The record lives in the store, so the tool answers its id, the one a listing names."""
    stored = report_friction(tmp_path, category="missing_tool", detail="x")
    assert "report_path" not in stored
    assert read_report(str(tmp_path), stored["report_id"])["detail"] == "x"
    listed = load_project_memory(tmp_path, "reports")["reports"][0]
    assert listed["report_id"] == stored["report_id"] and "path" not in listed


def test_report_friction_rejects_invalid_category(tmp_path: Path):
    result = report_friction(
        tmp_path,
        category="not_a_real_category",
        detail="x",
    )
    assert "error" in result
    assert "valid_categories" in result
    assert "missing_tool" in result["valid_categories"]


def test_report_friction_one_document_per_report(tmp_path: Path):
    for i in range(3):
        report_friction(
            tmp_path,
            category="unexpected_behavior",
            detail=f"report {i}",
        )

    assert len(report_documents(str(tmp_path))) == 3


def test_write_retrospective_creates_new_file(tmp_path: Path):
    result = write_retrospective(
        tmp_path,
        project_id="chestnut-bur-phase0",
        task="Bootstrap bur detection from scratch",
        worked="SAM candidates were usable after filtering.",
        did_not_work="Active learning selection stagnated after 50 labels.",
        assumptions_wrong="Assumed all burs were similarly sized; ground-truth shows a 5x range.",
        knowledge_for_future="Burs cluster near canopy edges, not centers.",
        missing_or_hard_tools="Needed get_trait_profile, not available.",
        would_do_differently="Start with larger initial label batch.",
    )

    assert result["project_id"] == "chestnut-bur-phase0"
    assert result["appended_to_existing"] is False

    content = read_retrospective(str(tmp_path), "chestnut-bur-phase0")
    assert "# chestnut-bur-phase0" in content
    assert "Bootstrap bur detection from scratch" in content
    assert "Active learning selection stagnated" in content
    assert "Burs cluster near canopy edges" in content


def test_write_retrospective_appends_to_existing(tmp_path: Path):
    write_retrospective(
        tmp_path,
        project_id="chestnut-bur-phase0",
        task="First pass",
        worked="a",
        did_not_work="b",
    )
    write_retrospective(
        tmp_path,
        project_id="chestnut-bur-phase0",
        task="Second pass after three days",
        worked="Different conclusion on reflection",
        did_not_work="c",
    )

    content = read_retrospective(str(tmp_path), "chestnut-bur-phase0")
    # Only one top-level header, two dated sections
    assert content.count("# chestnut-bur-phase0") == 1
    # Count section headers without the em-dash (dash byte can differ by encoding).
    assert content.count("## Retrospective") == 2
    assert "First pass" in content
    assert "Second pass after three days" in content


def test_concurrent_retrospectives_all_survive(tmp_path: Path):
    # Appending is a read, a concatenation and a write. Taken outside one serialized step, two
    # sessions finishing together each append to the text they read and one section is dropped.
    import threading

    n_threads = 16
    barrier = threading.Barrier(n_threads)

    def _call(index: int):
        barrier.wait()  # maximize actual overlap, not just "started around the same time"
        write_retrospective(
            tmp_path,
            project_id="project-under-test",
            task=f"pass {index}",
            worked="a",
            did_not_work="b",
        )

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    content = read_retrospective(str(tmp_path), "project-under-test")
    assert content.count("# project-under-test") == 1
    assert content.count("## Retrospective") == n_threads
    missing = [i for i in range(n_threads) if f"pass {i}" not in content]
    assert not missing, f"sections lost: {missing}"


def test_write_retrospective_handles_empty_optional_fields(tmp_path: Path):
    write_retrospective(
        tmp_path,
        project_id="minimal",
        task="t",
        worked="w",
        did_not_work="d",
    )
    assert "_(none noted)_" in read_retrospective(str(tmp_path), "minimal")


def test_load_retrospectives_returns_empty_when_none_recorded(tmp_path: Path):
    result = load_project_memory(tmp_path, "retrospectives")
    assert result["count"] == 0
    assert result["retrospectives"] == []
    assert "no retrospectives recorded" in result["note"]


def test_load_retrospectives_returns_recent_first(tmp_path: Path):
    # Each section is stamped from the clock, whose tick is coarser than these calls, so the writes
    # are spaced far enough apart to state three different times rather than one.
    write_retrospective(tmp_path, project_id="first", task="t", worked="w", did_not_work="d")
    time.sleep(0.05)
    write_retrospective(tmp_path, project_id="second", task="t", worked="w", did_not_work="d")
    time.sleep(0.05)
    write_retrospective(tmp_path, project_id="third", task="t", worked="w", did_not_work="d")

    result = load_project_memory(tmp_path, "retrospectives", limit=10)
    assert result["count"] == 3
    ids = [r["project_id"] for r in result["retrospectives"]]
    assert ids == ["third", "second", "first"]


def test_load_retrospectives_respects_limit(tmp_path: Path):
    for i in range(5):
        write_retrospective(
            tmp_path,
            project_id=f"project-{i}",
            task="t",
            worked="w",
            did_not_work="d",
        )

    result = load_project_memory(tmp_path, "retrospectives", limit=2)
    assert result["count"] == 2
    assert result["total_available"] == 5


def test_load_retrospectives_filter_substring(tmp_path: Path):
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

    result = load_project_memory(tmp_path, "retrospectives", filter_substring="currant")
    assert result["count"] == 1
    assert result["retrospectives"][0]["project_id"] == "currant-efb"

    # Filter also matches on content
    result = load_project_memory(tmp_path, "retrospectives", filter_substring="bur detection")
    assert result["count"] == 1
    assert result["retrospectives"][0]["project_id"] == "chestnut-bur"


def test_load_reports_returns_empty_when_none_recorded(tmp_path: Path):
    result = load_project_memory(tmp_path, "reports")
    assert result["count"] == 0
    assert result["reports"] == []
    assert "no friction reports recorded" in result["note"]


def test_load_reports_roundtrips_a_written_report(tmp_path: Path):
    report_friction(
        tmp_path,
        category="missing_tool",
        detail="needed get_trait_profile",
        context={"crop": "currant"},
    )
    result = load_project_memory(tmp_path, "reports")
    assert result["count"] == 1
    rep = result["reports"][0]
    assert rep["category"] == "missing_tool"
    assert rep["detail"] == "needed get_trait_profile"
    assert rep["context"]["crop"] == "currant"
    assert rep["timestamp"]


def test_a_report_missing_a_field_its_writer_records_refuses_rather_than_reading_as_blank(
    tmp_path: Path,
):
    """A stored report holding its time alone is not answered with an empty category, detail
    and context and a false disagreement; the listing refuses naming the missing field."""
    import pytest

    from tcip_mcp.tools.meta_tools import friction_report_key

    ts.replace(friction_report_key(str(tmp_path), "20260304T120000Z_missing_tool_a1b2"),
               {"timestamp": "2026-03-04T12:00:00+00:00"})

    with pytest.raises(KeyError, match="category"):
        load_project_memory(tmp_path, "reports")


def test_a_report_that_is_not_a_json_object_will_not_decode(tmp_path: Path):
    import pytest

    from tcip_mcp.tools.meta_tools import friction_report_key

    ts.replace(friction_report_key(str(tmp_path), "20260304T120000Z_missing_tool_a1b2"), "text")

    with pytest.raises(ts.DecodeError, match="not a JSON object"):
        read_report(str(tmp_path), "20260304T120000Z_missing_tool_a1b2")


def test_load_reports_recent_first_and_respects_limit(tmp_path: Path):
    # Each report is stamped from the clock, whose tick is coarser than these calls, so the writes
    # are spaced far enough apart to state four different times rather than one.
    for i in range(4):
        report_friction(tmp_path, category="unexpected_behavior", detail=f"r{i}")
        time.sleep(0.05)

    result = load_project_memory(tmp_path, "reports", limit=2)
    assert result["count"] == 2
    assert result["total_available"] == 4
    # Most recent first
    assert result["reports"][0]["detail"] == "r3"
    assert result["reports"][1]["detail"] == "r2"


def test_a_retrospective_stating_no_section_sorts_after_every_dated_one_by_name(tmp_path: Path):
    """A document that states no time is never given one, and lands in one fixed place by name."""
    write_retrospective(
        tmp_path, project_id="dated", task="t", worked="w", did_not_work="d")
    for project_id in ("zzz-undated", "aaa-undated"):
        ts.replace(
            retrospective_key(str(tmp_path), project_id),
            f"# {project_id}\n",
            expect=ts.Version.ABSENT,
        )

    result = load_project_memory(tmp_path, "retrospectives", limit=10)

    assert [r["project_id"] for r in result["retrospectives"]] == [
        "dated", "aaa-undated", "zzz-undated"]


def test_load_reports_filters_by_category(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="a")
    report_friction(tmp_path, category="ambiguous_data", detail="b")

    result = load_project_memory(tmp_path, "reports", category="ambiguous_data")
    assert result["count"] == 1
    assert result["reports"][0]["detail"] == "b"


def test_load_reports_filter_substring(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="trait profile lookup")
    report_friction(tmp_path, category="missing_tool", detail="something else entirely")

    result = load_project_memory(tmp_path, "reports", filter_substring="trait profile")
    assert result["count"] == 1
    assert result["reports"][0]["detail"] == "trait profile lookup"


def test_report_friction_defaults_user_disagreement_false(tmp_path: Path):
    result = report_friction(tmp_path, category="missing_tool", detail="x")
    assert result["user_disagreement"] is False

    entry = read_report(str(tmp_path), result["report_id"])
    assert entry["user_disagreement"] is False


def test_report_friction_records_user_disagreement(tmp_path: Path):
    result = report_friction(
        tmp_path,
        category="needs_human_judgment",
        detail="Zack pushed back on the tiling default.",
        user_disagreement=True,
    )
    assert result["user_disagreement"] is True

    entry = read_report(str(tmp_path), result["report_id"])
    assert entry["user_disagreement"] is True


def test_load_reports_roundtrips_user_disagreement(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="a", user_disagreement=False)
    report_friction(tmp_path, category="needs_human_judgment", detail="b", user_disagreement=True)

    result = load_project_memory(tmp_path, "reports", limit=10)
    flags = {r["detail"]: r["user_disagreement"] for r in result["reports"]}
    assert flags["a"] is False
    assert flags["b"] is True


def test_a_friction_report_counts_toward_the_recent_activity_its_audit_line_records(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="a")
    status = _recent_activity(tmp_path)
    assert status["reports_since_last_retrospective"] == 1
    assert status["reports_since_last_distillation"] == 1


def test_a_retrospective_resets_the_report_count_and_is_pointed_at_by_id(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="a")
    write_retrospective(tmp_path, project_id="p", task="t", worked="w", did_not_work="d")

    status = _recent_activity(tmp_path)
    assert status["reports_since_last_retrospective"] == 0
    assert status["last_retrospective"]["project_id"] == "p"
    assert "worked" not in json.dumps(status)  # no retrospective text, pointer only


def test_record_distillation_pass_resets_distillation_counters(tmp_path: Path):
    report_friction(tmp_path, category="missing_tool", detail="a")
    report_friction(tmp_path, category="missing_tool", detail="b")

    result = record_distillation_pass(str(tmp_path))
    assert result["status"] == "recorded"

    status = _recent_activity(tmp_path)
    assert status["reports_since_last_distillation"] == 0
    assert status["last_distillation_at"]


def test_record_distillation_pass_never_touches_reports_or_retrospectives(tmp_path: Path):
    # Bookkeeping only: must never write/modify/delete the underlying records it's counting.
    report_friction(tmp_path, category="missing_tool", detail="a")
    before = report_documents(str(tmp_path))

    record_distillation_pass(str(tmp_path))

    after = report_documents(str(tmp_path))
    assert after == before
    assert after[0].value["detail"] == "a"
