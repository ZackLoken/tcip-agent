"""One project per process, named where it starts or by the project a server for none creates.

The MCP server acts on the project it was started for, or the first one it creates when started
for none; the web backend acts on the project it has open; neither ever reads the other's choice
or leaves one behind in the environment.

The MCP cases run the real ``tcip-pipeline`` server in memory through a handshake
(``tests.test_agent_identity_records.results_through_handshake``); the backend cases go through
the app's own open route. Projects are made through the platform's own creation door under the
workspace ``tmp_path.parent``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_web.state import store
from tests._audit_fixtures import audit_rows
from tests._web_fixtures import bound_to, named_project
from tests.test_agent_identity_records import results_through_handshake

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import BoundProject


_FRICTION = (
    "report_friction",
    {"category": "unexpected_behavior", "detail": "which root"},
)


def test_the_server_acts_on_the_project_it_was_started_for_while_the_backend_opens_another(
    tmp_path: Path,
) -> None:
    ws = tmp_path.parent
    started = named_project(ws / "valley_block", "Valley block")
    opened = named_project(ws / "hill_block", "Hill block")
    asyncio.run(store.open_project(opened))

    (result,) = results_through_handshake([_FRICTION], bound=bound_to(started))

    assert not result.is_error, result
    assert [
        row["arguments"]["detail"]
        for row in audit_rows(started.root, "report_friction")
    ] == ["which root"]
    assert audit_rows(opened.root, "report_friction") == []


def test_a_tool_input_spelled_like_the_bound_project_stays_the_callers(
    bound: BoundProject,
) -> None:
    """A tool input spelled like the bound project stays the caller's.

    The server binds its project context, not a tool's own input that shares a name with it:
    ``write_retrospective``'s ``project_id`` names the retrospective's line of work, is offered to
    the client, and a call through the server writes a record carrying the caller's name.
    """
    from tcip_mcp.tools.meta_tools import Retrospective, memory_entries
    from tests._server_calls import client_server

    tools = {
        t.name: t for t in client_server(bound)._tool_manager.list_tools()
    }
    assert (
        "project_id" in tools["write_retrospective"].parameters["properties"]
    )

    (result,) = results_through_handshake(
        [
            (
                "write_retrospective",
                {
                    "project_id": "valley-bloom-trial",
                    "task": "t",
                    "worked": "w",
                    "did_not_work": "d",
                },
            )
        ],
        bound=bound,
    )

    assert not result.is_error, result
    (entry,) = memory_entries(Retrospective, str(bound.root))
    assert isinstance(entry.record, Retrospective)
    assert entry.record.project_id == "valley-bloom-trial"


def test_a_server_for_no_project_refuses_project_tools_naming_project_and_serves_knowledge() -> (
    None
):
    refused, knowledge = results_through_handshake(
        [_FRICTION, ("serve_domain_knowledge", {})], bound=None
    )

    assert refused.is_error
    assert "--project" in refused.content[0].text
    assert not knowledge.is_error, knowledge


def _creation(path: Path, display_name: str) -> tuple[str, dict]:
    from tests._web_fixtures import INVARIANCES

    return (
        "initialize_project",
        {
            "project_path": str(path),
            "display_name": display_name,
            "site": "north orchard",
            "invariances": INVARIANCES.model_dump(),
        },
    )


def test_a_server_for_no_project_acts_on_the_project_it_creates(
    tmp_path: Path,
) -> None:
    import json

    from PIL import Image

    from tcip_mcp import dataset_layout

    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(source / "IMG_0001.jpg")
    target = tmp_path.parent / "ridge_block"

    created, ingested = results_through_handshake(
        [
            _creation(target, "Ridge block"),
            ("ingest_images", {"source": str(source)}),
        ],
        bound=None,
    )

    assert not created.is_error, created
    assert not ingested.is_error, ingested
    answer = json.loads(created.content[0].text)
    assert answer["session_project"] == str(target.resolve())
    copied = dataset_layout.image_root(target).rglob("*.jpg")
    assert [p.name for p in copied] == ["IMG_0001.jpg"]


def test_a_server_for_a_project_stays_on_it_after_creating_another(
    tmp_path: Path,
) -> None:
    import json

    ws = tmp_path.parent
    started = named_project(ws / "valley_block", "Valley block")
    other = ws / "hill_block"

    created, reported = results_through_handshake(
        [_creation(other, "Hill block"), _FRICTION], bound=bound_to(started)
    )

    assert not created.is_error, created
    assert not reported.is_error, reported
    assert json.loads(created.content[0].text)["session_project"] == str(
        started.root
    )
    assert [
        row["arguments"]["detail"]
        for row in audit_rows(started.root, "report_friction")
    ] == ["which root"]
    assert audit_rows(other, "report_friction") == []


def test_the_server_refuses_to_start_for_a_directory_holding_no_project(
    tmp_path: Path,
) -> None:
    import tcip_mcp.server as server_module

    bare = tmp_path.parent / "bare"
    bare.mkdir()

    with pytest.raises(SystemExit, match="--project"):
        server_module.main(["--project", str(bare)])


def test_a_server_started_for_no_project_needs_no_workspace_and_serves_knowledge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.server import MCPServer

    import tcip_mcp.server as server_module

    served: list[MCPServer] = []
    monkeypatch.setattr(
        MCPServer, "run", lambda self, **kwargs: served.append(self)
    )
    monkeypatch.delenv("TCIP_WORKSPACE")

    server_module.main([])

    (server,) = served
    result = asyncio.run(server.call_tool("serve_domain_knowledge", {}))
    assert not result.is_error, result


def test_a_server_started_for_a_project_refuses_with_no_workspace_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.server import MCPServer

    import tcip_mcp.server as server_module

    project = named_project(tmp_path.parent / "valley_block", "Valley block")
    served: list[MCPServer] = []
    monkeypatch.setattr(
        MCPServer, "run", lambda self, **kwargs: served.append(self)
    )
    monkeypatch.delenv("TCIP_WORKSPACE")

    with pytest.raises(SystemExit, match="TCIP_WORKSPACE"):
        server_module.main(["--project", str(project.root)])
    assert served == []


def test_opening_two_projects_leaves_the_environment_as_it_was(
    tmp_path: Path, client: TestClient
) -> None:
    ws = tmp_path.parent
    first = named_project(ws / "valley_block", "Valley block")
    second = named_project(ws / "hill_block", "Hill block")
    before = dict(os.environ)

    for project in (first, second):
        opened = client.post(
            "/api/projects/open", json={"id": project.id, "user": "grower"}
        )
        assert opened.status_code == 200

    assert dict(os.environ) == before
    assert store.opened == second


def test_the_list_names_the_open_project_and_each_project_by_display_name(
    tmp_path: Path,
    client: TestClient,
) -> None:
    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")

    opened = client.post(
        "/api/projects/open", json={"id": valley.id, "user": "grower"}
    )
    assert opened.status_code == 200

    listing = client.get("/api/projects").json()
    assert listing["open_id"] == valley.id
    assert {(p["id"], p["display_name"]) for p in listing["projects"]} == {
        (valley.id, "Valley block"),
        (hill.id, "Hill block"),
    }
