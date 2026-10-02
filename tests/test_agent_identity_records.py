"""What a record says about the harness that wrote it, proven through the real server.

The MCP server learns which harness connected from the initialize handshake and mints a session id
of its own; every audit line the process then writes carries both, and the
tools' one HTTP push sends them as headers. These cases run the real ``tcip-pipeline`` server in
memory over the SDK's own streams with a client that declares a name and version, call the tools
through that handshake, and read what landed. A call made with no handshake at all is the control:
it writes the same records with no identity, rather than a guessed one.
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

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp import traits
from tcip_mcp.server import build_server
from tcip_mcp.workspace import workspace_from_environment
from tests import _trait_fixtures as fx

DECLARED = mcp_types.Implementation(name="reviewing-harness", version="1.2.3")
IDENTITY_FIELDS = ("agent_client_name", "agent_client_version", "agent_session",
                   "terminal_session")


def _body(result: Any) -> dict:
    """The tool's own return value, as the server serialized it."""
    if isinstance(result.structured_content, dict) and "result" not in result.structured_content:
        return result.structured_content
    if isinstance(result.structured_content, dict):
        return result.structured_content["result"]
    return json.loads(result.content[0].text)


def call_through_handshake(
    calls: list[tuple[str, dict]], declared: mcp_types.Implementation = DECLARED, *,
    project: Path | None,
) -> list[dict]:
    """Run the real server in memory for ``project``, complete a handshake as ``declared``, make
    ``calls`` in order, and hand back each tool's return value."""
    return [_body(result)
            for result in results_through_handshake(calls, declared, project=project)]


def results_through_handshake(
    calls: list[tuple[str, dict]], declared: mcp_types.Implementation = DECLARED, *,
    project: Path | None,
) -> list[Any]:
    """Run the real server in memory, built for ``project`` as ``--project`` builds it, complete a
    handshake as ``declared``, make ``calls`` in order, and hand back each call's result as the
    server sent it, a refused call's error included."""

    async def run() -> list[Any]:
        bodies: list[Any] = []
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                # The run loop; MCPServer.run is stdio-only.
                binding = None if project is None else (project, workspace_from_environment())
                server = build_server(binding)._lowlevel_server

                async def serve() -> None:
                    await server.run(
                        server_streams[0], server_streams[1],
                        server.create_initialization_options(), raise_exceptions=True,
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


def _project_rows(project: Path, tool: str) -> list[dict]:
    key = audit_module.audit_log_key(project)
    return [row for row in ts.read_log(key).records if row["tool"] == tool]


def _report_call(detail: str) -> tuple[str, dict]:
    return ("report_friction", {"category": "unexpected_behavior", "detail": detail})


# ── the audit line ───────────────────────────────────────────────────────────


def test_an_audited_call_through_a_handshake_records_the_declared_harness_and_a_session(
    project: Path,
) -> None:
    call_through_handshake([_report_call("first"), _report_call("second")], project=project)

    rows = _project_rows(project, "report_friction")
    assert [row["arguments"]["detail"] for row in rows] == ["first", "second"]
    for row in rows:
        assert row["agent_client_name"] == "reviewing-harness"
        assert row["agent_client_version"] == "1.2.3"
        assert row["agent_session"].startswith("mcp_")
    assert rows[0]["agent_session"] == rows[1]["agent_session"]
    assert "terminal_session" not in rows[0]


def test_the_terminal_session_rides_along_only_when_the_launcher_declared_one(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term_abc123")
    call_through_handshake([_report_call("under a terminal")], project=project)

    (row,) = _project_rows(project, "report_friction")
    assert row["terminal_session"] == "term_abc123"


def test_two_handshakes_in_two_runs_mint_two_sessions(project: Path) -> None:
    call_through_handshake([_report_call("run one")], project=project)
    call_through_handshake([_report_call("run two")], project=project)

    first, second = _project_rows(project, "report_friction")
    assert first["agent_session"] != second["agent_session"]


def test_a_call_with_no_handshake_records_no_identity(project: Path) -> None:
    """The control: the web backend, a script and this test process import the tools without a
    handshake, and their lines keep the shape they always had."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(project, "unexpected_behavior", "no handshake")

    (row,) = _project_rows(project, "report_friction")
    assert not set(IDENTITY_FIELDS) & set(row)


# ── the trait proposal ───────────────────────────────────────────────────────


def test_a_trait_proposed_through_a_handshake_is_named_by_its_audit_line(project: Path) -> None:
    """The entry travels as the tool's declared input schema, JSON over the real server; the
    proposal's own audit line names the harness, and the revision carries no copy of it."""
    entry = fx.with_operationalization(
        fx.COUNT_SPEC, traits.PER_IMAGE_COUNT, measured_subject=fx.COUNT_SUBJECT)

    (revision,) = call_through_handshake([("propose_trait", {
        "entry": entry.model_dump(mode="json"),
        "rationale": "the breeder described the count in their own field-scoring terms",
    })], project=project)

    assert not set(IDENTITY_FIELDS) & set(revision)
    (row,) = _project_rows(project, "propose_trait")
    assert row["arguments"]["revision"] == revision["number"]
    assert row["agent_client_name"] == "reviewing-harness"
    assert row["agent_client_version"] == "1.2.3"
    assert row["agent_session"].startswith("mcp_")
    stored = traits.read_trait(fx.COUNT_TRAIT, project).latest
    assert stored.entry == entry and stored.entry_sha256 == traits.entry_sha256(entry)


# ── the HTTP push ────────────────────────────────────────────────────────────


class _CapturingResponse:
    status = 200

    def __enter__(self) -> "_CapturingResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return b"{}"


@pytest.fixture
def captured_requests(monkeypatch: pytest.MonkeyPatch) -> list:
    """Every ``urllib`` request the tools' push makes, with the backend answered as up."""
    import urllib.request

    seen: list = []

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        seen.append(req)
        return _CapturingResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    return seen


def test_the_push_through_a_handshake_sends_the_identity_as_headers(
    captured_requests: list, project: Path,
) -> None:
    call_through_handshake([("push_panel_event", {
        "panel": "meta", "event_type": "identity_probe", "data": {"n": 1},
    })], project=project)

    (req,) = captured_requests
    assert req.get_header("X-tcip-agent-client-name") == "reviewing-harness"
    assert req.get_header("X-tcip-agent-client-version") == "1.2.3"
    assert req.get_header("X-tcip-agent-session").startswith("mcp_")
    assert req.get_header("X-tcip-terminal-session") is None


def test_the_push_with_no_handshake_sends_only_the_content_type(
    captured_requests: list, project: Path,
) -> None:
    from tcip_mcp.web_client import post_panel_event

    post_panel_event(project, project.parent, "meta", "identity_probe", {"n": 1})

    (req,) = captured_requests
    assert {name.lower() for name in req.headers} == {"content-type"}
