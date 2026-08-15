"""Tests for meta-loop tools (claude_reports, project_retrospective, load_project_memory)."""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.project_status import read_project_status
from tcip_mcp.tools.meta_tools import (
    claude_reports,
    load_project_memory,
    project_retrospective,
    record_distillation_pass,
)


def test_claude_reports_writes_jsonl_file(tmp_path: Path):
    result = claude_reports(
        str(tmp_path),
        category="missing_tool",
        detail="I needed a way to fetch trait profiles but no such tool exists.",
        context={"trait": "efb_damage", "crop": "hazelnut"},
    )

    report_path = Path(result["report_path"])
    assert report_path.exists()
    assert report_path.suffix == ".jsonl"
    assert report_path.parent == tmp_path / ".tcip" / "reports"
    assert result["category"] == "missing_tool"

    entry = json.loads(report_path.read_text())
    assert entry["category"] == "missing_tool"
    assert entry["context"]["trait"] == "efb_damage"
    assert "timestamp" in entry


def test_claude_reports_rejects_invalid_category(tmp_path: Path):
    result = claude_reports(
        str(tmp_path),
        category="not_a_real_category",
        detail="x",
    )
    assert "error" in result
    assert "valid_categories" in result
    assert "missing_tool" in result["valid_categories"]


def test_claude_reports_one_file_per_report(tmp_path: Path):
    for i in range(3):
        claude_reports(
            str(tmp_path),
            category="unexpected_behavior",
            detail=f"report {i}",
        )

    files = list((tmp_path / ".tcip" / "reports").glob("*.jsonl"))
    assert len(files) == 3


def test_project_retrospective_creates_new_file(tmp_path: Path):
    result = project_retrospective(
        str(tmp_path),
        project_id="chestnut-bur-phase0",
        task="Bootstrap bur detection from scratch",
        worked="SAM candidates were usable after filtering.",
        did_not_work="Active learning selection stagnated after 50 labels.",
        assumptions_wrong="Assumed all burs were similarly sized; ground-truth shows a 5x range.",
        knowledge_for_future="Burs cluster near canopy edges, not centers.",
        missing_or_hard_tools="Needed get_trait_profile, not available.",
        would_do_differently="Start with larger initial label batch.",
    )

    retro_path = Path(result["retrospective_path"])
    assert retro_path.exists()
    assert retro_path.name == "chestnut-bur-phase0.md"
    assert result["appended_to_existing"] is False

    content = retro_path.read_text()
    assert "# chestnut-bur-phase0" in content
    assert "Bootstrap bur detection from scratch" in content
    assert "Active learning selection stagnated" in content
    assert "Burs cluster near canopy edges" in content


def test_project_retrospective_appends_to_existing(tmp_path: Path):
    project_retrospective(
        str(tmp_path),
        project_id="chestnut-bur-phase0",
        task="First pass",
        worked="a",
        did_not_work="b",
    )
    project_retrospective(
        str(tmp_path),
        project_id="chestnut-bur-phase0",
        task="Second pass after three days",
        worked="Different conclusion on reflection",
        did_not_work="c",
    )

    retro_path = tmp_path / ".tcip" / "retrospectives" / "chestnut-bur-phase0.md"
    content = retro_path.read_text(encoding="utf-8")
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
        project_retrospective(
            str(tmp_path),
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

    content = (tmp_path / ".tcip" / "retrospectives" / "project-under-test.md").read_text(
        encoding="utf-8")
    assert content.count("# project-under-test") == 1
    assert content.count("## Retrospective") == n_threads
    missing = [i for i in range(n_threads) if f"pass {i}" not in content]
    assert not missing, f"sections lost: {missing}"


def test_project_retrospective_handles_empty_optional_fields(tmp_path: Path):
    result = project_retrospective(
        str(tmp_path),
        project_id="minimal",
        task="t",
        worked="w",
        did_not_work="d",
    )
    content = Path(result["retrospective_path"]).read_text()
    assert "_(none noted)_" in content


def test_load_retrospectives_returns_empty_when_dir_missing(tmp_path: Path):
    result = load_project_memory("retrospectives", str(tmp_path))
    assert result["count"] == 0
    assert result["retrospectives"] == []
    assert "does not exist yet" in result["note"]


def test_load_retrospectives_returns_recent_first(tmp_path: Path):
    import time

    project_retrospective(str(tmp_path), project_id="first", task="t", worked="w", did_not_work="d")
    time.sleep(0.05)
    project_retrospective(str(tmp_path), project_id="second", task="t", worked="w", did_not_work="d")
    time.sleep(0.05)
    project_retrospective(str(tmp_path), project_id="third", task="t", worked="w", did_not_work="d")

    result = load_project_memory("retrospectives", str(tmp_path), limit=10)
    assert result["count"] == 3
    ids = [r["project_id"] for r in result["retrospectives"]]
    assert ids == ["third", "second", "first"]


def test_load_retrospectives_respects_limit(tmp_path: Path):
    for i in range(5):
        project_retrospective(
            str(tmp_path),
            project_id=f"project-{i}",
            task="t",
            worked="w",
            did_not_work="d",
        )

    result = load_project_memory("retrospectives", str(tmp_path), limit=2)
    assert result["count"] == 2
    assert result["total_available"] == 5


def test_load_retrospectives_filter_substring(tmp_path: Path):
    project_retrospective(
        str(tmp_path),
        project_id="hazelnut-efb",
        task="EFB severity",
        worked="w",
        did_not_work="d",
    )
    project_retrospective(
        str(tmp_path),
        project_id="chestnut-bur",
        task="Bur detection",
        worked="w",
        did_not_work="d",
    )

    result = load_project_memory("retrospectives", str(tmp_path), filter_substring="hazelnut")
    assert result["count"] == 1
    assert result["retrospectives"][0]["project_id"] == "hazelnut-efb"

    # Filter also matches on content
    result = load_project_memory("retrospectives", str(tmp_path), filter_substring="bur detection")
    assert result["count"] == 1
    assert result["retrospectives"][0]["project_id"] == "chestnut-bur"


def test_load_reports_returns_empty_when_dir_missing(tmp_path: Path):
    result = load_project_memory("reports", str(tmp_path))
    assert result["count"] == 0
    assert result["reports"] == []
    assert "does not exist yet" in result["note"]


def test_load_reports_roundtrips_a_written_report(tmp_path: Path):
    claude_reports(
        str(tmp_path),
        category="missing_tool",
        detail="needed get_trait_profile",
        context={"crop": "hazelnut"},
    )
    result = load_project_memory("reports", str(tmp_path))
    assert result["count"] == 1
    rep = result["reports"][0]
    assert rep["category"] == "missing_tool"
    assert rep["detail"] == "needed get_trait_profile"
    assert rep["context"]["crop"] == "hazelnut"
    assert rep["timestamp"]


def test_load_reports_recent_first_and_respects_limit(tmp_path: Path):
    import time

    for i in range(4):
        claude_reports(str(tmp_path), category="unexpected_behavior", detail=f"r{i}")
        time.sleep(0.02)

    result = load_project_memory("reports", str(tmp_path), limit=2)
    assert result["count"] == 2
    assert result["total_available"] == 4
    # Most recent first
    assert result["reports"][0]["detail"] == "r3"
    assert result["reports"][1]["detail"] == "r2"


def test_load_reports_filters_by_category(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    claude_reports(str(tmp_path), category="ambiguous_data", detail="b")

    result = load_project_memory("reports", str(tmp_path), category="ambiguous_data")
    assert result["count"] == 1
    assert result["reports"][0]["detail"] == "b"


def test_load_reports_filter_substring(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="trait profile lookup")
    claude_reports(str(tmp_path), category="missing_tool", detail="something else entirely")

    result = load_project_memory("reports", str(tmp_path), filter_substring="trait profile")
    assert result["count"] == 1
    assert result["reports"][0]["detail"] == "trait profile lookup"


def test_claude_reports_defaults_user_disagreement_false(tmp_path: Path):
    result = claude_reports(str(tmp_path), category="missing_tool", detail="x")
    assert result["user_disagreement"] is False

    entry = json.loads(Path(result["report_path"]).read_text())
    assert entry["user_disagreement"] is False


def test_claude_reports_records_user_disagreement(tmp_path: Path):
    result = claude_reports(
        str(tmp_path),
        category="needs_human_judgment",
        detail="Zack pushed back on the tiling default.",
        user_disagreement=True,
    )
    assert result["user_disagreement"] is True

    entry = json.loads(Path(result["report_path"]).read_text())
    assert entry["user_disagreement"] is True


def test_load_reports_roundtrips_user_disagreement(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="a", user_disagreement=False)
    claude_reports(str(tmp_path), category="needs_human_judgment", detail="b", user_disagreement=True)

    result = load_project_memory("reports", str(tmp_path), limit=10)
    flags = {r["detail"]: r["user_disagreement"] for r in result["reports"]}
    assert flags["a"] is False
    assert flags["b"] is True


def test_claude_reports_updates_project_status(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 1
    assert status["reports_since_last_distillation"] == 1


def test_project_retrospective_updates_project_status(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    project_retrospective(str(tmp_path), project_id="p", task="t", worked="w", did_not_work="d")

    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 0
    assert status["last_retrospective"]["project_id"] == "p"
    assert "worked" not in json.dumps(status)  # no retrospective text cached, pointer only


def test_record_distillation_pass_resets_distillation_counters(tmp_path: Path):
    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    claude_reports(str(tmp_path), category="missing_tool", detail="b")

    result = record_distillation_pass(str(tmp_path))
    assert result["status"] == "recorded"

    status = read_project_status(tmp_path)
    assert status["reports_since_last_distillation"] == 0
    assert status["last_distillation_at"]


def test_record_distillation_pass_never_touches_reports_or_retrospectives(tmp_path: Path):
    # Bookkeeping only: must never write/modify/delete the underlying records it's counting.
    claude_reports(str(tmp_path), category="missing_tool", detail="a")
    before = list((tmp_path / ".tcip" / "reports").glob("*.jsonl"))

    record_distillation_pass(str(tmp_path))

    after = list((tmp_path / ".tcip" / "reports").glob("*.jsonl"))
    assert before == after
    assert json.loads(before[0].read_text())["detail"] == "a"
