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
