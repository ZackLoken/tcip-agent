"""One project per process, named where the process starts: the MCP server acts on the project
it was started for, the web backend on the project it has open, and neither ever reads the
other's choice or leaves one behind in the environment.

The MCP cases run the real ``tcip-pipeline`` server in memory through a handshake
(``tests.test_agent_identity_records.results_through_handshake``); the backend cases go through
the app's own open route. Every project is made through the platform's own creation door under
this test's workspace (``tmp_path.parent``).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import audit, workspace
from tcip_web.state import store
from tests._web_fixtures import bound_to, named_project
from tests.test_agent_identity_records import results_through_handshake


def _friction_rows(project: Path) -> list[dict]:
    return [row for row in ts.read_log(audit.audit_log_key(project)).records
            if row["tool"] == "report_friction"]


_FRICTION = ("report_friction", {"category": "unexpected_behavior", "detail": "which root"})


def test_the_server_acts_on_the_project_it_was_started_for_while_the_backend_opens_another(
    tmp_path,
):
    ws = tmp_path.parent
    started = named_project(ws / "valley_block", "Valley block")
    opened = named_project(ws / "hill_block", "Hill block")
    asyncio.run(store.open_project(opened))

    (result,) = results_through_handshake([_FRICTION], bound=bound_to(started))

    assert not result.is_error, result
    assert [row["arguments"]["detail"] for row in _friction_rows(started.root)] == ["which root"]
    assert _friction_rows(opened.root) == []


def test_a_tool_input_spelled_like_the_bound_project_stays_the_callers(bound):
    """The server binds its project context, never a tool's own input that shares a name with
    it: ``write_retrospective``'s ``project_id`` names the retrospective, is offered to the
    client, and a call through the server writes under the caller's name."""
    from tcip_mcp.server import build_server
    from tcip_mcp.tools.meta_tools import read_retrospective

    tools = {t.name: t for t in build_server(bound)._tool_manager.list_tools()}
    assert "project_id" in tools["write_retrospective"].parameters["properties"]

    (result,) = results_through_handshake([("write_retrospective", {
        "project_id": "valley-bloom-trial", "task": "t", "worked": "w", "did_not_work": "d",
    })], bound=bound)

    assert not result.is_error, result
    assert read_retrospective(str(bound.root), "valley-bloom-trial").startswith(
        "# valley-bloom-trial")


def test_a_server_for_no_project_refuses_project_tools_naming_project_and_serves_knowledge():
    refused, knowledge = results_through_handshake(
        [_FRICTION, ("serve_domain_knowledge", {})], bound=None)

    assert refused.is_error
    assert "--project" in refused.content[0].text
    assert not knowledge.is_error, knowledge


def test_the_server_refuses_to_start_for_a_directory_holding_no_project(tmp_path):
    import tcip_mcp.server as server_module

    bare = tmp_path.parent / "bare"
    bare.mkdir()

    with pytest.raises(SystemExit, match="--project"):
        server_module.main(["--project", str(bare)])


def test_a_server_started_for_no_project_needs_no_workspace_and_serves_knowledge(monkeypatch):
    import tcip_mcp.server as server_module
    from mcp.server import MCPServer

    served: list[MCPServer] = []
    monkeypatch.setattr(MCPServer, "run", lambda self, **kwargs: served.append(self))
    monkeypatch.delenv("TCIP_WORKSPACE")

    server_module.main([])

    (server,) = served
    result = asyncio.run(server.call_tool("serve_domain_knowledge", {}))
    assert not result.is_error, result


def test_a_server_started_for_a_project_refuses_with_no_workspace_by_name(tmp_path, monkeypatch):
    import tcip_mcp.server as server_module
    from mcp.server import MCPServer

    project = named_project(tmp_path.parent / "valley_block", "Valley block")
    served: list[MCPServer] = []
    monkeypatch.setattr(MCPServer, "run", lambda self, **kwargs: served.append(self))
    monkeypatch.delenv("TCIP_WORKSPACE")

    with pytest.raises(SystemExit, match="TCIP_WORKSPACE"):
        server_module.main(["--project", str(project.root)])
    assert served == []


def test_opening_two_projects_leaves_the_environment_as_it_was(tmp_path, client):
    ws = tmp_path.parent
    first = named_project(ws / "valley_block", "Valley block")
    second = named_project(ws / "hill_block", "Hill block")
    before = dict(os.environ)

    for project in (first, second):
        opened = client.post("/api/projects/open", json={"id": project.id, "user": "grower"})
        assert opened.status_code == 200

    assert dict(os.environ) == before
    assert store.opened == second


def test_the_last_opened_pointer_holds_the_id_and_the_list_names_each_project_by_display_name(
    tmp_path, client,
):
    from tcip_web.routes.projects import open_last_opened

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")

    opened = client.post("/api/projects/open", json={"id": valley.id, "user": "grower"})
    assert opened.status_code == 200

    assert workspace.read_last_opened(ws) == valley.id
    listing = client.get("/api/projects").json()
    assert listing["open_id"] == valley.id
    assert {(p["id"], p["display_name"]) for p in listing["projects"]} == {
        (valley.id, "Valley block"), (hill.id, "Hill block")}

    asyncio.run(store.close_project(valley.id))
    asyncio.run(open_last_opened())
    assert store.opened == valley
