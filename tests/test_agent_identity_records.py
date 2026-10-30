"""What a record says about the harness that wrote it, proven through the real server.

The MCP server learns which harness connected from the initialize handshake and mints a session id
of its own; the audit lines the process then writes carry both, and the tools' one HTTP push
sends them as headers. These cases run the real ``tcip-pipeline`` server in
memory over the SDK's own streams with a client that declares a name and version, call the tools
through that handshake, and read what landed. A call made with no handshake at all writes its
records with no identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import mcp.types as mcp_types
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from tcip_mcp import traits
from tcip_mcp.agent_identity import RECORD_FIELDS as IDENTITY_FIELDS
from tcip_mcp.workspace import BoundProject, workspace_from_environment
from tests import _trait_fixtures as fx
from tests._audit_fixtures import audit_rows
from tests._server_calls import client_server

DECLARED = mcp_types.Implementation(name="reviewing-harness", version="1.2.3")


def _body(result: mcp_types.CallToolResult) -> dict:
    """Return the tool's own return value, as the server serialized it."""
    if (
        isinstance(result.structured_content, dict)
        and "result" not in result.structured_content
    ):
        return result.structured_content
    if isinstance(result.structured_content, dict):
        return result.structured_content["result"]
    return json.loads(result.content[0].text)


def call_through_handshake(
    calls: list[tuple[str, dict]],
    declared: mcp_types.Implementation = DECLARED,
    *,
    bound: BoundProject | None,
) -> list[dict]:
    """Make ``calls`` in order through a handshake as ``declared`` and return each tool's value.

    The real server runs in memory for ``bound``.
    """
    return [
        _body(result)
        for result in results_through_handshake(calls, declared, bound=bound)
    ]


def results_through_handshake(
    calls: list[tuple[str, dict]],
    declared: mcp_types.Implementation = DECLARED,
    *,
    bound: BoundProject | None,
) -> list[Any]:
    """Make ``calls`` in order through a handshake as ``declared`` and return each call's result.

    The real server runs in memory, built for ``bound`` (``None``: for no project). Each result
    is returned as the server sent it, a refused call's error included.
    """

    async def run() -> list[Any]:
        bodies: list[Any] = []
        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            async with anyio.create_task_group() as tg:
                # The run loop; MCPServer.run is stdio-only.
                server = client_server(bound)._lowlevel_server

                async def serve() -> None:
                    await server.run(
                        server_streams[0],
                        server_streams[1],
                        server.create_initialization_options(),
                        raise_exceptions=True,
                    )

                tg.start_soon(serve)
                async with ClientSession(
                    client_streams[0], client_streams[1], client_info=declared
                ) as session:
                    await session.initialize()
                    for name, arguments in calls:
                        bodies.append(await session.call_tool(name, arguments))
                tg.cancel_scope.cancel()
        return bodies

    return anyio.run(run)


def _report_call(detail: str) -> tuple[str, dict]:
    return (
        "report_friction",
        {"category": "unexpected_behavior", "detail": detail},
    )


# ── the audit line ───────────────────────────────────────────────────────────


def test_an_audited_call_through_a_handshake_records_the_declared_harness_and_a_session(
    project: Path,
    bound: BoundProject,
) -> None:
    call_through_handshake(
        [_report_call("first"), _report_call("second")], bound=bound
    )

    rows = audit_rows(project, "report_friction")
    assert [row["arguments"]["detail"] for row in rows] == ["first", "second"]
    for row in rows:
        assert row["agent_client_name"] == "reviewing-harness"
        assert row["agent_client_version"] == "1.2.3"
        assert row["agent_session"].startswith("mcp_")
    assert rows[0]["agent_session"] == rows[1]["agent_session"]
    assert "terminal_session" not in rows[0]


def test_the_terminal_session_rides_along_only_when_the_launcher_declared_one(
    project: Path, bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term_abc123")
    call_through_handshake([_report_call("under a terminal")], bound=bound)

    (row,) = audit_rows(project, "report_friction")
    assert row["terminal_session"] == "term_abc123"


def test_two_handshakes_in_two_runs_mint_two_sessions(
    project: Path, bound: BoundProject
) -> None:
    call_through_handshake([_report_call("run one")], bound=bound)
    call_through_handshake([_report_call("run two")], bound=bound)

    first, second = audit_rows(project, "report_friction")
    assert first["agent_session"] != second["agent_session"]


def test_a_call_with_no_handshake_records_no_identity(project: Path) -> None:
    """A call with no handshake writes an audit line carrying no identity field."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(project, "unexpected_behavior", "no handshake", actor=None)

    (row,) = audit_rows(project, "report_friction")
    assert not set(IDENTITY_FIELDS) & set(row)


# ── the trait proposal ───────────────────────────────────────────────────────


def test_a_trait_proposed_through_a_handshake_is_named_by_its_audit_line(
    project: Path,
    bound: BoundProject,
) -> None:
    """A trait proposed through a handshake is named by its audit line, never by its revision.

    The entry travels as the tool's declared input schema, JSON over the real server; the
    proposal's own audit line names the harness, and the revision carries no copy of it.
    """
    entry = fx.with_operationalization(
        fx.COUNT_SPEC,
        traits.PER_IMAGE_COUNT,
        measured_subject=fx.COUNT_SUBJECT,
    )

    (revision,) = call_through_handshake(
        [
            (
                "propose_trait",
                {
                    "entry": entry.model_dump(mode="json"),
                    "rationale": "the breeder described the count in their own field-scoring terms",
                },
            )
        ],
        bound=bound,
    )

    assert not set(IDENTITY_FIELDS) & set(revision)
    (row,) = audit_rows(project, "propose_trait")
    assert row["arguments"]["revision"] == revision["number"]
    assert row["agent_client_name"] == "reviewing-harness"
    assert row["agent_client_version"] == "1.2.3"
    assert row["agent_session"].startswith("mcp_")
    stored = traits.read_trait(fx.COUNT_TRAIT, project).latest
    assert (
        stored.entry == entry
        and stored.entry_sha256 == traits.entry_sha256(entry)
    )


# ── the HTTP push ────────────────────────────────────────────────────────────


@pytest.fixture
def captured_requests(monkeypatch: pytest.MonkeyPatch, project: Path) -> list:
    """Return the list every request the tools' push makes through ``httpx.post`` is captured into.

    A backend is recorded as serving the workspace, and every request is answered as up.
    """
    import httpx

    from tcip_mcp.web_client import record_backend_port

    record_backend_port(workspace_from_environment(), 8765)
    seen: list[httpx.Request] = []

    def fake_post(
        url: str,
        *,
        json: object,
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        request = httpx.Request("POST", url, json=json, headers=headers)
        seen.append(request)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    return seen


def test_the_push_through_a_handshake_sends_the_identity_as_headers(
    captured_requests: list,
    bound: BoundProject,
) -> None:
    call_through_handshake(
        [
            (
                "push_panel_event",
                {
                    "panel": "meta",
                    "event": {
                        "event_type": "banner",
                        "text": "identity probe",
                    },
                },
            )
        ],
        bound=bound,
    )

    (req,) = captured_requests
    assert json.loads(req.content)["project_id"] == bound.id
    assert req.headers.get("X-tcip-agent-client-name") == "reviewing-harness"
    assert req.headers.get("X-tcip-agent-client-version") == "1.2.3"
    assert req.headers.get("X-tcip-agent-session").startswith("mcp_")
    assert req.headers.get("X-tcip-terminal-session") is None


def test_the_push_with_no_handshake_sends_no_identity_header(
    captured_requests: list,
    bound: BoundProject,
) -> None:
    from tcip_mcp.agent_identity import HEADERS
    from tcip_mcp.web_client import Banner, post_panel_event

    post_panel_event(bound, "meta", Banner(text="identity probe"))

    (req,) = captured_requests
    assert not {name.lower() for name in req.headers} & {
        header.lower() for header in HEADERS.values()
    }
    assert req.headers["content-type"] == "application/json"
