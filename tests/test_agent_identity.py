"""The identity holder on its own.

What a run establishes, what each projection says, and that a run's end forgets it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from tcip_mcp import agent_identity

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mcp.server.context import ServerRequestContext


@pytest.fixture(autouse=True)
def _forget_between_tests() -> Iterator[None]:
    agent_identity.end()
    yield
    agent_identity.end()


def test_no_identity_until_a_handshake_and_every_projection_says_so() -> None:
    assert agent_identity.current() is None
    assert agent_identity.audit_fields() == {}
    assert agent_identity.http_headers() == {}


def test_a_handshake_mints_a_session_and_carries_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)

    identity = agent_identity.begin("claude-code", "2.1.238")

    assert identity.agent_client_name == "claude-code"
    assert identity.agent_client_version == "2.1.238"
    session = identity.agent_session
    assert session.startswith("mcp_") and len(session) == len("mcp_") + 16
    assert identity.terminal_session is None
    assert agent_identity.audit_fields() == {
        "agent_client_name": "claude-code",
        "agent_client_version": "2.1.238",
        "agent_session": session,
    }
    assert agent_identity.http_headers() == {
        "X-TCIP-Agent-Client-Name": "claude-code",
        "X-TCIP-Agent-Client-Version": "2.1.238",
        "X-TCIP-Agent-Session": session,
    }


def test_the_identity_fields_are_declared_once_and_carry_no_harness_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identity's declared fields, field tuple, header map and projection agree.

    They name the same four fields, and a harness exporting its session and effort to the server
    it spawns lands in neither the audit fields nor the headers.
    """
    import dataclasses

    monkeypatch.setenv(
        "CLAUDE_CODE_SESSION_ID", "0b56e764-5533-408a-bb5d-d5dd17b4e6b9"
    )
    monkeypatch.setenv("CLAUDE_EFFORT", "high")
    identity = agent_identity.begin("claude-code", "2.1.238")

    declared = tuple(
        f.name for f in dataclasses.fields(agent_identity.AgentIdentity)
    )
    assert declared == (
        "agent_client_name",
        "agent_client_version",
        "agent_session",
        "terminal_session",
    )
    assert agent_identity.RECORD_FIELDS == declared
    assert tuple(agent_identity.HEADERS) == declared
    assert tuple(identity.fields()) == declared
    assert not any(
        "harness" in field for field in agent_identity.audit_fields()
    )
    assert not any(
        "Harness" in header for header in agent_identity.http_headers()
    )
    assert not hasattr(agent_identity, "HARNESS_EXPORTS")


def test_the_terminal_session_is_read_from_the_environment_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term_xyz")

    identity = agent_identity.begin("codex-mcp-client", "0.147.0")

    assert identity.terminal_session == "term_xyz"
    assert agent_identity.audit_fields()["terminal_session"] == "term_xyz"
    assert (
        agent_identity.http_headers()["X-TCIP-Terminal-Session"] == "term_xyz"
    )


def test_an_empty_terminal_session_variable_counts_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "")

    assert (
        agent_identity.begin("codex-mcp-client", "0.147.0").terminal_session
        is None
    )


def test_the_first_handshake_of_a_run_holds_for_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = agent_identity.begin("claude-code", "2.1.238")

    with caplog.at_level("WARNING", logger="tcip_mcp.agent_identity"):
        second = agent_identity.begin("codex-mcp-client", "0.147.0")

    assert second is first
    assert agent_identity.current() is first
    assert "codex-mcp-client 0.147.0" in caplog.text


def test_the_fields_a_request_declared_are_read_through_the_same_header_map() -> (
    None
):
    sent = {
        header: f"declared {field}"
        for field, header in agent_identity.HEADERS.items()
    }

    assert agent_identity.fields_from_headers(sent) == {
        field: f"declared {field}" for field in agent_identity.RECORD_FIELDS
    }
    assert agent_identity.fields_from_headers({}) == {
        field: None for field in agent_identity.RECORD_FIELDS
    }
    assert (
        agent_identity.fields_from_headers({"X-TCIP-Agent-Session": ""})[
            "agent_session"
        ]
        is None
    )


def test_a_server_run_forgets_the_identity_on_its_way_out() -> None:
    async def run() -> None:
        async with agent_identity.session_lifespan(object()):
            agent_identity.begin("claude-code", "2.1.238")
            assert agent_identity.current() is not None

    anyio.run(run)

    assert agent_identity.current() is None


def test_a_caller_cannot_hand_an_audit_line_another_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audit line records the handshake's identity over the caller's own identity keys.

    ``record_event_or_raise(..., agent_session=...)`` records the handshake's session and client
    name, not the caller's.
    """
    import tcip_mcp.audit as audit_module
    import tcip_store as ts

    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    identity = agent_identity.begin("claude-code", "2.1.238")

    audit_module.record_event_or_raise(
        "identity_probe",
        {},
        actor=None,
        scope=tmp_path,
        agent_session="forged",
        agent_client_name="x",
    )

    key = audit_module.audit_log_key(tmp_path)
    (row,) = [
        r for r in ts.read_log(key).records if r["tool"] == "identity_probe"
    ]
    assert row["agent_session"] == identity.agent_session
    assert row["agent_client_name"] == "claude-code"


def test_a_declared_name_travels_as_an_ascii_header_and_comes_back_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term with space")
    identity = agent_identity.begin("harnèss/β", "1.0 (dev)")

    headers = agent_identity.http_headers()

    for value in headers.values():
        assert value.isascii() and " " not in value and "\n" not in value
    assert agent_identity.fields_from_headers(headers) == {
        "agent_client_name": "harnèss/β",
        "agent_client_version": "1.0 (dev)",
        "agent_session": identity.agent_session,
        "terminal_session": "term with space",
    }


def test_a_caller_cannot_supply_an_identity_key_the_handshake_left_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity key the handshake left absent is not recorded from a caller.

    With no handshake a forged ``agent_session`` is not recorded, and after a handshake with no
    terminal session in the environment a forged ``terminal_session`` is not recorded either.
    """
    import tcip_mcp.audit as audit_module
    from tests._audit_fixtures import audit_rows

    audit_module.record_event_or_raise(
        "no_handshake", {}, actor=None, scope=tmp_path, agent_session="forged"
    )
    (row,) = audit_rows(tmp_path, "no_handshake")
    assert "agent_session" not in row

    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    agent_identity.begin("codex-mcp-client", "0.147.0")
    audit_module.record_event_or_raise(
        "codex_session",
        {},
        actor=None,
        scope=tmp_path,
        terminal_session="forged",
    )
    (row,) = audit_rows(tmp_path, "codex_session")
    assert row["agent_client_name"] == "codex-mcp-client"
    assert "terminal_session" not in row


def test_the_declaration_is_taken_on_the_first_message_that_carries_it_whatever_its_method() -> (
    None
):
    """A client whose first message is a ``tools/call`` request gets its declaration recorded."""
    import types

    info = types.SimpleNamespace(name="antigravity-like", version="1.1.17")
    ctx = types.SimpleNamespace(
        method="tools/call",
        session=types.SimpleNamespace(
            client_params=types.SimpleNamespace(client_info=info)
        ),
    )

    async def call_next(c: ServerRequestContext) -> dict[str, str]:
        return {"seen": agent_identity.current().agent_client_name}

    result = anyio.run(agent_identity.record_connecting_client, ctx, call_next)

    assert result == {"seen": "antigravity-like"}
    assert agent_identity.current().agent_client_version == "1.1.17"
