"""Governance: self-learning capture + distill.

The SessionEnd capture hook files a machine-local backstop record; the distill script gathers the
journal + reports + retros + captures into one worksheet; the fence materialization absolutizes
the SessionEnd hook the same way it does the guards. None of it applies anything (governance
stays human).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path


def _load_distill():
    from tcip_web.cli import distill_learnings

    return distill_learnings


def _seed_report(project_root: Path, report_id: str, entry: dict) -> None:
    """Record one friction report through the seam, the way ``report_friction`` does."""
    import tcip_store as ts

    from tcip_mcp.tools.meta_tools import friction_report_key

    ts.replace(friction_report_key(str(project_root), report_id), entry,
               expect=ts.Version.ABSENT)


def _seed_capture(project_root: Path, session_id: str) -> None:
    """Record one SessionEnd capture through the hook's own writer, the way the hook does."""
    from tcip_web import agent_learning_capture

    stdin = io.StringIO(json.dumps({"session_id": session_id, "cwd": str(project_root)}))
    old_stdin, sys.stdin = sys.stdin, stdin
    try:
        agent_learning_capture.main(["--project", str(project_root)])
    finally:
        sys.stdin = old_stdin


def test_distill_worksheet_gathers_reports_captures_and_themes(tmp_path):
    distill = _load_distill()
    _seed_report(tmp_path, "r", {"category": "needs_human_judgment",
                                 "detail": "the EXIF orientation thing"})
    # This bites on the database leg (nothing lands at the literal .tcip/learning_capture.jsonl
    # path there); the file leg is coverage only, since its own locator resolves to that path.
    _seed_capture(tmp_path, "s1")

    ws = distill.build_worksheet(tmp_path)
    assert "Friction reports (1)" in ws
    assert "Session captures (1)" in ws
    assert "exif" in ws.lower()          # recurring theme detected
    assert "Nothing here is applied" in ws  # gathering only, governance stays human


def test_distill_worksheet_surfaces_disagreements(tmp_path):
    distill = _load_distill()
    _seed_report(tmp_path, "r1", {"category": "needs_human_judgment",
                                  "detail": "kept the old default",
                                  "user_disagreement": False})
    _seed_report(tmp_path, "r2", {"category": "needs_human_judgment",
                                  "detail": "pushed back on the tiling default",
                                  "user_disagreement": True})

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
    # "wobblesync"/"desyncing" are not, and never will be, in any hardcoded theme vocabulary:
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
    # "wobblesync" repeats many times within one project's own text: a pooled frequency count
    # would clear a >=2 bar on that alone. The per-project set approach must not let it.
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
    workspace = tmp_path.parent
    for name, detail in [("proj_a", "shared friction theme theme"),
                          ("proj_b", "shared friction theme theme")]:
        (workspace / name).mkdir()
        _seed_report(workspace / name, "r", {"category": "unexpected_behavior", "detail": detail})

    ws = distill.build_workspace_worksheet(workspace)
    assert "Cross-project recurring themes" in ws
    assert "proj_a" in ws and "proj_b" in ws
    assert "Nothing here is applied" in ws  # same governance framing as the single-project worksheet
    assert "the judgment is yours" in ws
    assert "record_distillation_pass" in ws


def test_build_workspace_worksheet_ignores_non_project_dirs(tmp_path):
    distill = _load_distill()
    (tmp_path.parent / "not_a_project").mkdir()  # no .tcip/, must not be treated as a project
    ws = distill.build_workspace_worksheet(tmp_path.parent)
    assert "No projects with a `.tcip/` directory" in ws


def test_workspace_mode_never_writes_anything(tmp_path):
    """A --workspace run gathers and writes nothing, the invariant this governance surface rests on.

    Bound to the file backend because the claim is about the bytes on disk, which is where a stray
    write would land.
    """
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    distill = _load_distill()
    proj_a = tmp_path.parent / "proj_a"
    proj_a.mkdir()
    _seed_report(proj_a, "r", {"category": "missing_tool", "detail": "x"})

    before = {p: p.read_bytes() for p in (proj_a / ".tcip").rglob("*") if p.is_file()}
    distill.build_workspace_worksheet(tmp_path.parent)
    after = {p: p.read_bytes() for p in (proj_a / ".tcip").rglob("*") if p.is_file()}
    assert before == after


def test_capture_hook_appends_and_never_raises(tmp_path, monkeypatch):
    import tcip_store as ts
    from tcip_mcp.web_client import learning_capture_key
    from tcip_web import agent_learning_capture

    monkeypatch.setattr(sys, "stdin",
                        io.StringIO(json.dumps({"session_id": "s1", "cwd": str(tmp_path), "reason": "clear"})))
    agent_learning_capture.main(["--project", str(tmp_path)])

    key = learning_capture_key(tmp_path)
    records = ts.read_log(key).records
    assert len(records) == 1
    assert records[0]["session_id"] == "s1"

    # Malformed / empty stdin must not raise: a capture backstop cannot break the session.
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json at all"))
    agent_learning_capture.main(["--project", str(tmp_path)])


def test_capture_hook_records_nothing_for_a_session_with_no_project(tmp_path, monkeypatch):
    """A session started for no project has no project log to write to, so the hook records
    nothing rather than choosing one."""
    import tcip_store as ts
    from tcip_mcp.web_client import learning_capture_key
    from tcip_web import agent_learning_capture

    monkeypatch.setattr(sys, "stdin",
                        io.StringIO(json.dumps({"session_id": "s2", "cwd": str(tmp_path)})))
    agent_learning_capture.main([])

    assert ts.read_log(learning_capture_key(tmp_path)).records == []


def test_materialize_absolutizes_sessionend_capture_hook(tmp_path):
    from tcip_web.terminal import _materialize_fence_settings

    dest = _materialize_fence_settings(tmp_path)
    assert dest is not None
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    cmd = cfg["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    assert "agent_learning_capture.py" in cmd
    assert cmd.startswith('"')  # absolutized + quoted, like the guards (no cwd dependency)


def test_materialize_absolutizes_every_agent_hook(tmp_path):
    # The materialization loop is event-agnostic: every agent_*.py hook command, across all event
    # groups, must be absolutized (the guards, the SessionEnd capture, the SessionStart ritual).
    from tcip_web.terminal import _FENCE_SETTINGS, _materialize_fence_settings

    dest = _materialize_fence_settings(tmp_path)
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
