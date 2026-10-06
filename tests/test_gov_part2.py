"""Governance: the distill worksheet gathers a project's friction reports and retrospectives into
one worksheet and applies nothing (governance stays human).
"""

from __future__ import annotations

from pathlib import Path


def _load_distill():
    from tcip_web.cli import distill_learnings

    return distill_learnings


def _term_set(text: str) -> set[str]:
    """The distinct terms the worksheet reads from ``text``."""
    return set(_load_distill()._terms(text))


def _seed_report(project_root: Path, category: str, detail: str,
                 user_disagreement: bool = False) -> None:
    """Record one friction report through ``report_friction`` itself."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(project_root, category, detail, user_disagreement=user_disagreement)


def test_distill_worksheet_gathers_reports_retrospectives_and_themes(tmp_path):
    from tcip_mcp.tools.meta_tools import write_retrospective

    distill = _load_distill()
    _seed_report(tmp_path, "needs_human_judgment", "the EXIF orientation thing")
    write_retrospective(tmp_path, "exif-handling", task="fix the EXIF orientation",
                        worked="nothing yet", did_not_work="the EXIF orientation came up again")

    ws = distill.build_worksheet(tmp_path)
    assert "Friction reports (1)" in ws
    assert "Retrospectives (1)" in ws
    assert "Session captures" not in ws
    assert "exif" in ws.lower()          # recurring theme detected
    assert "Nothing here is applied" in ws  # gathering only, governance stays human


def test_distill_worksheet_surfaces_disagreements(tmp_path):
    distill = _load_distill()
    _seed_report(tmp_path, "needs_human_judgment", "kept the old default")
    _seed_report(tmp_path, "needs_human_judgment", "pushed back on the tiling default",
                 user_disagreement=True)

    ws = distill.build_worksheet(tmp_path)
    assert "Disagreements (1)" in ws
    assert "pushed back on the tiling default" in ws
    disagreements_section = ws.split("## Disagreements", 1)[1].split("##", 1)[0]
    assert "kept the old default" not in disagreements_section


def test_the_worksheet_lists_a_report_that_will_not_decode_as_malformed(tmp_path):
    """A stored report that is not a JSON object is listed as malformed beside the decoded one,
    and the worksheet still builds."""
    import tcip_store as ts
    from tcip_mcp.tools.meta_tools import friction_report_key

    distill = _load_distill()
    _seed_report(tmp_path, "needs_human_judgment", "a decoded report")
    ts.replace(friction_report_key(str(tmp_path), "20260304T120000Z_missing_tool_a1b2"),
               ["not", "a", "report"])

    ws = distill.build_worksheet(tmp_path)
    assert "Friction reports (2)" in ws
    assert "[malformed]" in ws
    assert "a decoded report" in ws


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
        "proj_a": _term_set(
            "wobblesync desyncing. wobblesync desyncing again and again."
        ),
        "proj_b": _term_set("an unrelated report about tiling."),
    }
    themes = dict(distill._cross_project_themes(per_project))
    assert "wobblesync" not in themes  # only 1 distinct project, no matter the internal repeats


def test_cross_project_themes_surface_when_shared_across_projects():
    distill = _load_distill()
    per_project = {
        "proj_a": _term_set("GPS accuracy was too coarse for per-plant work."),
        "proj_b": _term_set("per-plant GPS accuracy issues came up again."),
        "proj_c": _term_set("unrelated tiling report."),
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
        _seed_report(workspace / name, "unexpected_behavior", detail)

    ws = distill.build_workspace_worksheet(workspace)
    assert "Cross-project recurring themes" in ws
    assert "proj_a" in ws and "proj_b" in ws
    # same governance framing as the single-project worksheet
    assert "Nothing here is applied" in ws
    assert "the judgment is yours" in ws
    assert "record_distillation_pass" in ws


def test_build_workspace_worksheet_ignores_non_project_dirs(tmp_path):
    distill = _load_distill()
    (tmp_path.parent / "not_a_project").mkdir()  # no .tcip/, must not be treated as a project
    ws = distill.build_workspace_worksheet(tmp_path.parent)
    assert "No projects with a `.tcip/` directory" in ws


def test_workspace_mode_never_writes_anything(tmp_path):
    """A workspace worksheet leaves every file under a project's ``.tcip`` but the store's own
    bookkeeping unchanged."""
    from tcip_store.file_backend import is_bookkeeping

    distill = _load_distill()
    proj_a = tmp_path.parent / "proj_a"
    proj_a.mkdir()
    _seed_report(proj_a, "missing_tool", "x")

    def files() -> dict:
        return {p: p.read_bytes() for p in (proj_a / ".tcip").rglob("*")
                if p.is_file() and not is_bookkeeping(p.name)}

    before = files()
    distill.build_workspace_worksheet(tmp_path.parent)
    after = files()
    assert before == after
