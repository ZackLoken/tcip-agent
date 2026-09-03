"""What a record says about the harness that wrote it, proven through the real server.

The MCP server learns which harness connected from the initialize handshake and mints a session id
of its own; every audit line and statement record the process then writes carries both, and the
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
from tcip_mcp import operationalization as op
from tcip_mcp import traits
from tcip_mcp.server import mcp as tcip_server
from tests import _operationalization_fixtures as fx

DECLARED = mcp_types.Implementation(name="reviewing-harness", version="1.2.3")
IDENTITY_FIELDS = ("agent_client_name", "agent_client_version", "agent_session",
                   "terminal_session", "harness_session", "harness_effort_at_connect")


def _body(result: Any) -> dict:
    """The tool's own return value, as the server serialized it."""
    if isinstance(result.structured_content, dict) and "result" not in result.structured_content:
        return result.structured_content
    if isinstance(result.structured_content, dict):
        return result.structured_content["result"]
    return json.loads(result.content[0].text)


def call_through_handshake(
    calls: list[tuple[str, dict]], declared: mcp_types.Implementation = DECLARED
) -> list[dict]:
    """Run the real server in memory, complete a handshake as ``declared``, make ``calls`` in order,
    and hand back each tool's return value."""

    async def run() -> list[dict]:
        bodies: list[dict] = []
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                server = tcip_server._lowlevel_server  # the run loop; MCPServer.run is stdio-only

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
                        bodies.append(_body(await session.call_tool(name, arguments)))
                tg.cancel_scope.cancel()
        return bodies

    return anyio.run(run)


def _platform_rows(tool: str) -> list[dict]:
    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    return [row for row in ts.read_log(key).records if row["tool"] == tool]


def _report_call(tmp_path: Path, detail: str) -> tuple[str, dict]:
    return ("report_friction", {"project_path": str(tmp_path), "category": "unexpected_behavior",
                               "detail": detail})


# ── the audit line ───────────────────────────────────────────────────────────


def test_an_audited_call_through_a_handshake_records_the_declared_harness_and_a_session(
    tmp_path: Path,
) -> None:
    call_through_handshake([_report_call(tmp_path, "first"), _report_call(tmp_path, "second")])

    rows = _platform_rows("report_friction")
    assert [row["arguments"]["detail"] for row in rows] == ["first", "second"]
    for row in rows:
        assert row["agent_client_name"] == "reviewing-harness"
        assert row["agent_client_version"] == "1.2.3"
        assert row["agent_session"].startswith("mcp_")
    assert rows[0]["agent_session"] == rows[1]["agent_session"]
    assert "terminal_session" not in rows[0]


def test_the_terminal_session_rides_along_only_when_the_launcher_declared_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term_abc123")
    call_through_handshake([_report_call(tmp_path, "under a terminal")])

    (row,) = _platform_rows("report_friction")
    assert row["terminal_session"] == "term_abc123"


def test_two_handshakes_in_two_runs_mint_two_sessions(tmp_path: Path) -> None:
    call_through_handshake([_report_call(tmp_path, "run one")])
    call_through_handshake([_report_call(tmp_path, "run two")])

    first, second = _platform_rows("report_friction")
    assert first["agent_session"] != second["agent_session"]


def test_a_call_with_no_handshake_records_no_identity(tmp_path: Path) -> None:
    """The control: the web backend, a script and this test process import the tools without a
    handshake, and their lines keep the shape they always had."""
    from tcip_mcp.tools.meta_tools import report_friction

    report_friction(str(tmp_path), "unexpected_behavior", "no handshake")

    (row,) = _platform_rows("report_friction")
    assert not set(IDENTITY_FIELDS) & set(row)


def test_what_claude_code_exports_about_itself_rides_on_its_lines_and_nothing_else_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Code hands its MCP servers its own session id and effort; a harness that declares
    another name gets none of them even with the variables in its environment."""
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "0b56e764-5533-408a-bb5d-d5dd17b4e6b9")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")
    claude = mcp_types.Implementation(name="claude-code", version="2.1.238")

    call_through_handshake([_report_call(tmp_path, "from claude code")], declared=claude)
    call_through_handshake([_report_call(tmp_path, "from another harness")])

    claude_row, other_row = _platform_rows("report_friction")
    assert claude_row["harness_session"] == "0b56e764-5533-408a-bb5d-d5dd17b4e6b9"
    assert claude_row["harness_effort_at_connect"] == "high"
    assert "harness_session" not in other_row and "harness_effort_at_connect" not in other_row


# ── the statement records ────────────────────────────────────────────────────


def _operationalization_call(project: Path) -> tuple[str, dict]:
    return ("state_trait_operationalization", {
        "project_root": str(project),
        "trait": fx.CROSSING_TRAIT,
        "delivery_kind": op.STATE_CROSSING_DATES,
        "statement": "the date each plant reached the state the breeder scores in the field",
        "mechanism": "the calibrated state classifier over the isolated flowers of one plant",
        "measured_subject": "flower",
        "delivered_phenotypes": ["bloom_05per_date", "bloom_50per_date"],
    })


def test_an_operationalization_stated_through_a_handshake_names_the_harness(
    tmp_path: Path,
) -> None:
    project = fx.seed_project(tmp_path / "project")

    (record,) = call_through_handshake([_operationalization_call(project)])

    assert record["stated_by"] == op.STATEMENT_SURFACE
    assert record["agent_client_name"] == "reviewing-harness"
    assert record["agent_client_version"] == "1.2.3"
    assert record["agent_session"].startswith("mcp_")
    assert record["terminal_session"] is None
    stored = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)[1].value
    assert stored["agent_session"] == record["agent_session"]


def test_the_operationalization_hash_covers_the_identity_fields(tmp_path: Path) -> None:
    """A confirmation covers the statement and its stated provenance together."""
    project = fx.seed_project(tmp_path / "project")
    (record,) = call_through_handshake([_operationalization_call(project)])

    for field in IDENTITY_FIELDS:
        assert field in op.STATEMENT_FIELDS
        moved = {**record, field: "something else"}
        assert op.record_seen_hash(moved) != op.record_seen_hash(record), field


def test_an_operationalization_stated_with_no_handshake_carries_the_fields_empty(
    tmp_path: Path,
) -> None:
    project = fx.seed_project(tmp_path / "project")

    record = fx.state_crossing(project)

    assert {record[f] for f in IDENTITY_FIELDS} == {None}


def test_a_trait_spec_authored_through_a_handshake_names_the_harness(tmp_path: Path) -> None:
    (statement,) = call_through_handshake([("author_trait_spec", {
        "project_root": str(tmp_path),
        "trait": "bloom_authored",
        "delivers": ["bloom_05per_date"],
        "positive_class_name": "open",
        "milestone_fractions": [0.05],
        "milestone_on": "positive_fraction",
        "rationale": "the breeder described the state directly, in their own field-scoring terms",
    })])

    assert statement["stated_by"] == traits.TRAIT_SPEC_STATEMENT_SURFACE
    assert statement["agent_client_name"] == "reviewing-harness"
    assert statement["agent_client_version"] == "1.2.3"
    assert statement["agent_session"].startswith("mcp_")
    for field in IDENTITY_FIELDS:
        assert field in traits.TRAIT_SPEC_STATEMENT_FIELDS
        moved = {**statement, field: "something else"}
        assert traits.trait_spec_statement_seen_hash(moved) != traits.trait_spec_statement_seen_hash(
            statement
        ), field


# ── the HTTP push ────────────────────────────────────────────────────────────


class _CapturingResponse:
    status = 200

    def __enter__(self) -> "_CapturingResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


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


def test_the_push_through_a_handshake_sends_the_identity_as_headers(captured_requests: list) -> None:
    call_through_handshake([("push_panel_event", {
        "panel": "meta", "event_type": "identity_probe", "data": {"n": 1},
    })])

    (req,) = captured_requests
    assert req.get_header("X-tcip-agent-client-name") == "reviewing-harness"
    assert req.get_header("X-tcip-agent-client-version") == "1.2.3"
    assert req.get_header("X-tcip-agent-session").startswith("mcp_")
    assert req.get_header("X-tcip-terminal-session") is None


def test_the_push_with_no_handshake_sends_only_the_content_type(captured_requests: list) -> None:
    from tcip_mcp.web_client import post_panel_event

    post_panel_event("meta", "identity_probe", {"n": 1})

    (req,) = captured_requests
    assert {name.lower() for name in req.headers} == {"content-type"}
