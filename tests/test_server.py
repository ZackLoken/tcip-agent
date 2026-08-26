"""The MCP server's own predicates, tested without driving ``main()``.

``main()`` binds a real backend and pins the process root, so nothing here calls it; the
predicate it consults is a pure function of an environment mapping, tested on its own.
"""

from __future__ import annotations

from tcip_mcp import agent_identity


def test_binds_from_marker_true_inside_the_agent_terminal() -> None:
    from tcip_mcp.server import binds_from_marker

    assert binds_from_marker({agent_identity.TERMINAL_SESSION_ENV: "sess-1"}) is True


def test_binds_from_marker_false_outside_the_agent_terminal() -> None:
    from tcip_mcp.server import binds_from_marker

    assert binds_from_marker({}) is False


def test_binds_from_marker_false_when_the_terminal_session_is_empty() -> None:
    from tcip_mcp.server import binds_from_marker

    assert binds_from_marker({agent_identity.TERMINAL_SESSION_ENV: ""}) is False
