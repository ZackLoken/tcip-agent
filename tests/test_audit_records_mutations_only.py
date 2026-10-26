"""The audit lines a door's calls leave, counted through the server and the console.

Each case calls a read-only door, a mutating door, a refused call or a raising body and compares
the project's log before and after.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_mcp.audit as audit_module
from tests._audit_fixtures import audit_rows
from tests._training_values import COMBINED_SCORER, RATIONALE

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import BoundProject


def test_a_successful_monitor_result_audits_nothing_and_carries_no_refusal_key(
    tmp_path: Path,
) -> None:
    """A successful ``monitor_training`` poll audits nothing and carries no refusal key.

    The stream's own read: a run whose status carries ``status_error: None`` polled through
    ``monitor_training`` leaves the log's line count as it found it, and neither its answer nor
    its row carries an ``error`` key.
    """
    from tcip_mcp.tools.training_tools import monitor_training
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    opened_run(
        tmp_path,
        detection_config(tmp_path / "ds", project=tmp_path),
        experiment_id="exp-polled",
    )
    before = len(audit_rows(tmp_path))

    result = monitor_training(tmp_path, "exp-polled")
    status = result["run"]

    assert status["state"] == "running"
    assert status["status_error"] is None
    assert "error" not in result and "error" not in status
    assert len(audit_rows(tmp_path)) == before


def test_read_only_doors_leave_no_line(tmp_path: Path) -> None:
    from tcip_mcp.tools.experiment_tools import (
        get_experiment,
        list_experiments,
    )
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
    from tcip_mcp.tools.meta_tools import read_audit_log
    from tcip_mcp.tools.project_tools import view_gui_state
    from tcip_mcp.tools.training_tools import (
        inspect_compute_resources,
        monitor_training,
    )
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    opened_run(
        tmp_path,
        detection_config(tmp_path / "ds", project=tmp_path),
        experiment_id="exp-read",
    )
    before = len(audit_rows(tmp_path))

    get_experiment(tmp_path, "exp-read")
    list_experiments(tmp_path)
    monitor_training(tmp_path, "exp-read")
    monitor_training(tmp_path, "no-such-run")
    inspect_compute_resources(tmp_path)
    view_gui_state(tmp_path)
    serve_domain_knowledge()
    serve_domain_knowledge("no such document")
    read_audit_log(tmp_path)

    assert len(audit_rows(tmp_path)) == before


def _answered(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Run one console command and assert it printed an answer, never a refusal."""
    from tcip_web.cli import main
    from tests.test_console_tools_through_the_server import read

    assert main(argv) == 0, argv
    answer = read(capsys.readouterr().out)
    assert not (isinstance(answer, dict) and "error" in answer), (argv, answer)


def test_the_console_tools_that_read_or_render_leave_no_line(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every console tool but the shapefile conversion answers and leaves the log as it was.

    A census, an identity check, a scoring, an annotation read, the renders (each written under
    the project's ``.tcip/artifacts/viz/``), a headroom report, the sweep vocabularies, a run
    comparison, a preflight, a triage and a plant-grouped draw stated without an output, each run
    as its command, print an answer that is no refusal and add no line.
    """
    from tcip_mcp.cli import hyphenated
    from tests.test_console_tools_through_the_server import console_cases

    cases = [
        case
        for case in console_cases(project, monkeypatch)
        if case.tool.__name__ != "shp_to_plant_csv"
    ]
    before = len(audit_rows(project))

    for case in cases:
        _answered([hyphenated(case.tool.__name__), *case.command], capsys)

    assert len(audit_rows(project)) == before


def test_the_shapefile_conversion_and_a_written_plant_draw_each_leave_one_line(
    project: Path,
    bound: BoundProject,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conversion leaves its own line, and a plant-grouped draw written leaves the draw's.

    Each runs as its command, by the person its ``--user`` states, who is each line's actor;
    outside any MCP session neither line carries an agent's identity, and the draw records
    through ``draw_splits``' own event, never a line beside it. The same two calls through an
    MCP client's handshake leave the same two lines, by no person.
    """
    from tcip_mcp.agent_identity import RECORD_FIELDS
    from tcip_mcp.identity import actor
    from tests.test_agent_identity_records import results_through_handshake
    from tests.test_console_tools_through_the_server import BY, console_cases

    cases = {
        case.tool.__name__: case
        for case in console_cases(project, monkeypatch)
    }
    before = len(audit_rows(project))
    results_through_handshake(
        [
            ("shp_to_plant_csv", cases["shp_to_plant_csv"].arguments),
            (
                "plant_aware_group_splits",
                cases["plant_aware_group_splits"].arguments
                | {"output_path": "splits/through-a-client"},
            ),
        ],
        bound=bound,
    )
    by_client = audit_rows(project)[before:]
    assert [(r["tool"], r["status"]) for r in by_client] == [
        ("shp_to_plant_csv", "ok"),
        ("draw_splits", "ok"),
    ]
    assert [r.get(audit_module.ACTOR_KEY) for r in by_client] == [None] * 2
    before = len(audit_rows(project))

    _answered(["shp-to-plant-csv", *cases["shp_to_plant_csv"].command], capsys)
    _answered(
        [
            "plant-aware-group-splits",
            *cases["plant_aware_group_splits"].command,
            *("--output-path", "splits/plants"),
        ],
        capsys,
    )

    added = audit_rows(project)[before:]
    assert [(r["tool"], r["status"]) for r in added] == [
        ("shp_to_plant_csv", "ok"),
        ("draw_splits", "ok"),
    ]
    assert [r.get(audit_module.ACTOR_KEY) for r in added] == [actor(BY[1])] * 2
    assert not {key for row in added for key in row} & set(RECORD_FIELDS)


def test_a_phenology_look_on_screen_leaves_no_line(
    tmp_path: Path, client: TestClient
) -> None:
    """A phenology measurement shown on screen leaves no audit line.

    The Results tab's measurement door answers milestone rows and leaves the project's log as it
    found it.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path).body()
    before = audit_rows(tmp_path)

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 200, resp.text
    assert resp.json()["milestones"]["rows"]
    assert audit_rows(tmp_path) == before


def test_ranking_a_review_queue_leaves_no_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ranking a review queue leaves no audit line.

    A successful ranking through a registered run answers a queue and leaves the project's log
    as it found it.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue
    from tests.test_feedback_tools import _bound_checkpoint, _stub_scorer
    from tests.test_selection_ground_truth_digests import (
        DATES,
        _dataset,
        _draw,
    )

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(tmp_path, root, manifest_dir)
    _run_dir, ckpt_path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-ranked"
    )
    _stub_scorer(monkeypatch)
    before = audit_rows(tmp_path)

    result = prioritize_review_queue(
        tmp_path,
        checkpoint_path=ckpt_path,
        images_dir=str(image_dir(root, DATES[0])),
        scorer=COMBINED_SCORER,
        budget=50,
        rationale=RATIONALE,
    )

    assert "error" not in result, result
    assert result["queue"]
    assert audit_rows(tmp_path) == before


def test_a_mutating_door_still_leaves_one_line(tmp_path: Path) -> None:
    """A mutation through the platform's own door leaves exactly one audit line, status ok."""
    from tcip_mcp.tools.meta_tools import report_friction

    before = len(audit_rows(tmp_path))
    result = report_friction(
        tmp_path, "unexpected_behavior", "a real mutation", actor=None
    )

    assert "error" not in result
    rows = audit_rows(tmp_path)
    assert len(rows) == before + 1
    assert rows[-1]["tool"] == "report_friction"
    assert rows[-1]["status"] == "ok"


def test_a_return_is_ok_a_raise_is_an_exception_and_a_refusal_is_no_line(
    tmp_path: Path,
) -> None:
    """A return audits as ok, a raise as an exception, and a returned refusal leaves no line.

    A refusal returned as the tools' error dict is no act and leaves no line; a dict whose
    ``error`` is null is an ordinary answer.
    """
    from tcip_mcp.audit import audited

    @audited
    def refuse(project: Path) -> dict:
        return {"error": "refused by name"}

    @audited
    def answer(project: Path) -> dict:
        return {"error": None, "status": "running"}

    @audited
    def explode(project: Path) -> dict:
        raise RuntimeError("boom")

    refuse(tmp_path)
    answer(tmp_path)
    with pytest.raises(RuntimeError):
        explode(tmp_path)

    statuses = [(r["tool"], r["status"]) for r in audit_rows(tmp_path)]
    assert statuses == [("answer", "ok"), ("explode", "exception")]
