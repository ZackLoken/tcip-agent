"""Governance Part 2 — self-learning capture + distill.

The SessionEnd capture hook files a machine-local backstop record; the distill script gathers the
journal + reports + retros + captures into one worksheet; the fence materialization absolutizes
the SessionEnd hook the same way it does the guards. None of it applies anything (governance
stays human).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_distill():
    spec = importlib.util.spec_from_file_location(
        "distill_learnings", REPO / "scripts" / "distill_learnings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_distill_worksheet_gathers_reports_captures_and_themes(tmp_path):
    distill = _load_distill()
    reports = tmp_path / ".tcip" / "reports"
    reports.mkdir(parents=True)
    (reports / "r.jsonl").write_text(
        json.dumps({"category": "needs_human_judgment", "detail": "the EXIF orientation thing"}) + "\n",
        encoding="utf-8")
    (tmp_path / ".tcip" / "learning_capture.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "session_id": "s1"}) + "\n", encoding="utf-8")

    ws = distill.build_worksheet(tmp_path)
    assert "Friction reports (1)" in ws
    assert "Session captures (1)" in ws
    assert "exif" in ws.lower()          # recurring theme detected
    assert "Nothing here is applied" in ws  # gathering only — governance stays human


def test_distill_worksheet_surfaces_disagreements(tmp_path):
    distill = _load_distill()
    reports = tmp_path / ".tcip" / "reports"
    reports.mkdir(parents=True)
    (reports / "r1.jsonl").write_text(
        json.dumps({"category": "needs_human_judgment", "detail": "kept the old default",
                    "user_disagreement": False}) + "\n", encoding="utf-8")
    (reports / "r2.jsonl").write_text(
        json.dumps({"category": "needs_human_judgment", "detail": "pushed back on the tiling default",
                    "user_disagreement": True}) + "\n", encoding="utf-8")

    ws = distill.build_worksheet(tmp_path)
    assert "Disagreements (1)" in ws
    assert "pushed back on the tiling default" in ws
    disagreements_section = ws.split("## Disagreements", 1)[1].split("##", 1)[0]
    assert "kept the old default" not in disagreements_section


def test_themes_generic_frequency_and_recurrence_floor():
    distill = _load_distill()
    text = ("the wobblesync module keeps desyncing. wobblesync desyncing again. "
            "a one-off mention of zzyzx.")
    themes = dict(distill._themes(text))
    # "wobblesync"/"desyncing" are not, and never will be, in any hardcoded theme vocabulary —
    # they surface purely because they recurred in this project's own text.
    assert themes.get("wobblesync") == 2
    assert themes.get("desyncing") == 2
    assert "zzyzx" not in themes  # single mention, below the recurrence floor


def test_themes_picks_up_bigram_phrases():
    distill = _load_distill()
    text = "operating point drifted. operating point drifted again next session."
    themes = dict(distill._themes(text))
    assert themes.get("operating point") == 2


def test_cross_project_themes_require_multiple_distinct_projects():
    distill = _load_distill()
    # "wobblesync" repeats many times within ONE project's own text — a pooled frequency count
    # would clear a >=2 bar on that alone. The per-project SET approach must not let it.
    per_project = {
        "proj_a": distill._project_token_set(
            "wobblesync desyncing. wobblesync desyncing again and again."
        ),
        "proj_b": distill._project_token_set("an unrelated report about tiling."),
    }
    themes = dict(distill._cross_project_themes(per_project))
    assert "wobblesync" not in themes  # only 1 distinct project, no matter the internal repeats


def test_cross_project_themes_surface_when_shared_across_projects():
    distill = _load_distill()
    per_project = {
        "proj_a": distill._project_token_set("GPS accuracy was too coarse for per-plant work."),
        "proj_b": distill._project_token_set("per-plant GPS accuracy issues came up again."),
        "proj_c": distill._project_token_set("unrelated tiling report."),
    }
    themes = dict(distill._cross_project_themes(per_project))
    assert themes.get("gps") == 2
    assert themes.get("accuracy") == 2


def test_build_workspace_worksheet_gathers_across_projects(tmp_path):
    distill = _load_distill()
    for name, detail in [("proj_a", "shared friction theme theme"),
                          ("proj_b", "shared friction theme theme")]:
        reports = tmp_path / name / ".tcip" / "reports"
        reports.mkdir(parents=True)
        (reports / "r.jsonl").write_text(
            json.dumps({"category": "unexpected_behavior", "detail": detail}) + "\n",
            encoding="utf-8")

    ws = distill.build_workspace_worksheet(tmp_path)
    assert "Cross-project recurring themes" in ws
    assert "proj_a" in ws and "proj_b" in ws
    assert "Nothing here is applied" in ws  # same governance framing as the single-project worksheet
    assert "the judgment is yours" in ws
    assert "record_distillation_pass" in ws


def test_build_workspace_worksheet_ignores_non_project_dirs(tmp_path):
    distill = _load_distill()
    (tmp_path / "not_a_project").mkdir()  # no .tcip/ — must not be treated as a project
    ws = distill.build_workspace_worksheet(tmp_path)
    assert "No projects with a `.tcip/` directory" in ws


def test_workspace_mode_never_writes_anything(tmp_path):
    # The read-only invariant this whole governance surface depends on — a --workspace run must
    # leave every project's .tcip/ untouched, same as the single-project mode already does.
    distill = _load_distill()
    reports_dir = tmp_path / "proj_a" / ".tcip" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "r.jsonl").write_text(
        json.dumps({"category": "missing_tool", "detail": "x"}) + "\n", encoding="utf-8")

    before = {
        p: p.read_bytes()
        for p in (tmp_path / "proj_a" / ".tcip").rglob("*") if p.is_file()
    }
    distill.build_workspace_worksheet(tmp_path)
    after = {
        p: p.read_bytes()
        for p in (tmp_path / "proj_a" / ".tcip").rglob("*") if p.is_file()
    }
    assert before == after


def test_capture_hook_appends_and_never_raises(tmp_path, monkeypatch):
    from tcip_web import agent_learning_capture

    monkeypatch.setattr(sys, "stdin",
                        io.StringIO(json.dumps({"session_id": "s1", "cwd": str(tmp_path), "reason": "clear"})))
    agent_learning_capture.main()

    cap = tmp_path / ".tcip" / "learning_capture.jsonl"
    assert cap.is_file()
    assert json.loads(cap.read_text(encoding="utf-8").strip())["session_id"] == "s1"

    # Malformed / empty stdin must not raise — a capture backstop cannot break the session.
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    agent_learning_capture.main()


def test_materialize_absolutizes_sessionend_capture_hook():
    from tcip_web.terminal import _materialize_fence_settings

    dest = _materialize_fence_settings()
    assert dest is not None
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    cmd = cfg["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    assert "agent_learning_capture.py" in cmd
    assert cmd.startswith('"')  # absolutized + quoted, like the guards (no cwd dependency)


def test_materialize_absolutizes_every_agent_hook():
    # The materialization loop is event-agnostic: every agent_*.py hook command, across all event
    # groups, must be absolutized (the guards, the SessionEnd capture, the SessionStart ritual).
    from tcip_web.terminal import _FENCE_SETTINGS, _materialize_fence_settings

    dest = _materialize_fence_settings()
    assert dest is not None
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    guard_dir = _FENCE_SETTINGS.parent.as_posix()
    seen = 0
    for event_groups in cfg["hooks"].values():
        for group in event_groups:
            for hook in group.get("hooks", []):
                cmd = hook.get("command", "")
                if "agent_" in cmd:
                    seen += 1
                    assert cmd.startswith('"'), cmd
                    assert guard_dir in cmd, cmd
    assert seen >= 3  # the two guards + the SessionEnd capture, at minimum
