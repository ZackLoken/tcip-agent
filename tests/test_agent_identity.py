"""The identity holder on its own: what a run establishes, what each projection says, and that a
run's end forgets it.

The records these values land on are proven in ``test_agent_identity_records.py`` through the real
server; these cases pin the module's own contract, which that proof relies on.
"""

from __future__ import annotations

import anyio
import pytest

from tcip_mcp import agent_identity


@pytest.fixture(autouse=True)
def _forget_between_tests():
    agent_identity.end()
    yield
    agent_identity.end()


def test_no_identity_until_a_handshake_and_every_projection_says_so() -> None:
    assert agent_identity.current() is None
    assert agent_identity.audit_fields() == {}
    assert agent_identity.http_headers() == {}
    assert agent_identity.statement_fields() == {
        "agent_client_name": None, "agent_client_version": None,
        "agent_session": None, "terminal_session": None,
        "harness_session": None, "harness_effort_at_connect": None,
    }


def test_a_handshake_mints_a_session_and_carries_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)

    identity = agent_identity.begin("claude-code", "2.1.238")

    assert identity.client_name == "claude-code"
    assert identity.client_version == "2.1.238"
    assert identity.session.startswith("mcp_") and len(identity.session) == len("mcp_") + 16
    assert identity.terminal_session is None
    assert agent_identity.audit_fields() == {
        "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.session,
    }
    assert agent_identity.http_headers() == {
        "X-TCIP-Agent-Client-Name": "claude-code",
        "X-TCIP-Agent-Client-Version": "2.1.238",
        "X-TCIP-Agent-Session": identity.session,
    }


def test_the_terminal_session_is_read_from_the_environment_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "term_xyz")

    identity = agent_identity.begin("codex-mcp-client", "0.147.0")

    assert identity.terminal_session == "term_xyz"
    assert agent_identity.audit_fields()["terminal_session"] == "term_xyz"
    assert agent_identity.http_headers()["X-TCIP-Terminal-Session"] == "term_xyz"
    assert agent_identity.statement_fields()["terminal_session"] == "term_xyz"


def test_an_empty_terminal_session_variable_counts_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_SESSION", "")

    assert agent_identity.begin("codex-mcp-client", "0.147.0").terminal_session is None


def test_the_first_handshake_of_a_run_holds_for_the_run(caplog: pytest.LogCaptureFixture) -> None:
    first = agent_identity.begin("claude-code", "2.1.238")

    with caplog.at_level("WARNING", logger="tcip_mcp.agent_identity"):
        second = agent_identity.begin("codex-mcp-client", "0.147.0")

    assert second is first
    assert agent_identity.current() is first
    assert "codex-mcp-client 0.147.0" in caplog.text


def test_the_fields_a_request_declared_are_read_through_the_same_header_map() -> None:
    sent = {header: f"declared {field}" for field, header in agent_identity.HEADERS.items()}

    assert agent_identity.fields_from_headers(sent) == {
        field: f"declared {field}" for field in agent_identity.RECORD_FIELDS
    }
    assert agent_identity.fields_from_headers({}) == {
        field: None for field in agent_identity.RECORD_FIELDS
    }
    assert agent_identity.fields_from_headers({"X-TCIP-Agent-Session": ""})["agent_session"] is None


def test_a_server_run_forgets_the_identity_on_its_way_out() -> None:
    async def run() -> None:
        async with agent_identity.session_lifespan(object()):
            agent_identity.begin("claude-code", "2.1.238")
            assert agent_identity.current() is not None

    anyio.run(run)

    assert agent_identity.current() is None


def test_a_caller_cannot_hand_an_audit_line_another_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identity is applied after a caller's extra facts, so ``record_event(...,
    agent_session=...)`` records the handshake's session, not the caller's."""
    import tcip_mcp.audit as audit_module
    import tcip_store as ts

    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    identity = agent_identity.begin("claude-code", "2.1.238")

    audit_module.record_event("identity_probe", {}, agent_session="forged", agent_client_name="x")

    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    (row,) = [r for r in ts.read_log(key).records if r["tool"] == "identity_probe"]
    assert row["agent_session"] == identity.session
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
        "agent_client_name": "harnèss/β", "agent_client_version": "1.0 (dev)",
        "agent_session": identity.session, "terminal_session": "term with space",
        "harness_session": None, "harness_effort_at_connect": None,
    }


def test_what_the_declaring_harness_exported_about_itself_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "0b56e764-5533-408a-bb5d-d5dd17b4e6b9")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    identity = agent_identity.begin("claude-code", "2.1.238")

    assert identity.harness_session == "0b56e764-5533-408a-bb5d-d5dd17b4e6b9"
    assert identity.harness_effort_at_connect == "high"
    assert agent_identity.audit_fields()["harness_effort_at_connect"] == "high"
    assert agent_identity.http_headers()["X-TCIP-Harness-Session"] == identity.harness_session


def test_another_harness_is_not_attributed_an_enclosing_harness_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A harness that declares another name and passes an enclosing Claude Code session's variables
    through to its server is not attributed them: they are read only under the declaring name."""
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "0b56e764-5533-408a-bb5d-d5dd17b4e6b9")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    identity = agent_identity.begin("codex-mcp-client", "0.147.0")

    assert identity.harness_session is None
    assert identity.harness_effort_at_connect is None
    assert "harness_effort_at_connect" not in agent_identity.audit_fields()


def test_a_caller_cannot_supply_an_identity_key_the_handshake_left_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identity keys are reserved even where the handshake set nothing: with no handshake a
    forged session does not land, and under a harness that exports no session id a forged
    harness_session does not land either."""
    import tcip_mcp.audit as audit_module
    import tcip_store as ts

    def rows(tool: str) -> list[dict]:
        key = audit_module.audit_log_key(audit_module.platform_audit_scope())
        return [r for r in ts.read_log(key).records if r["tool"] == tool]

    audit_module.record_event(
        "no_handshake", {}, agent_session="forged", harness_effort_at_connect="max"
    )
    (row,) = rows("no_handshake")
    assert "agent_session" not in row and "harness_effort_at_connect" not in row

    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    agent_identity.begin("codex-mcp-client", "0.147.0")
    audit_module.record_event("codex_session", {}, harness_session="forged")
    (row,) = rows("codex_session")
    assert row["agent_client_name"] == "codex-mcp-client"
    assert "harness_session" not in row
