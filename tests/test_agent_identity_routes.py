"""What the backend records about the harness that pushed a panel event.

The MCP tools' one HTTP push carries the agent identity as headers; the events route copies what
was declared onto the broadcast payload, so a browser and the replay can say which harness steered
the GUI. A push without the headers is answered as it always was, the fields empty rather than
guessed. The header names are one map shared with the client, read here through the backend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app

DECLARED_HEADERS = {
    "X-TCIP-Agent-Client-Name": "reviewing-harness",
    "X-TCIP-Agent-Client-Version": "1.2.3",
    "X-TCIP-Agent-Session": "mcp_0123",
    "X-TCIP-Terminal-Session": "term_abc",
    "X-TCIP-Harness-Session": "0b56e764",
    "X-TCIP-Harness-Effort-At-Connect": "high",
}
IDENTITY_FIELDS = ("agent_client_name", "agent_client_version", "agent_session",
                   "terminal_session", "harness_session", "harness_effort_at_connect")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _last_event(client: TestClient, event_type: str) -> dict:
    recent = client.get("/api/events/meta/recent").json()["events"]
    return [event for event in recent if event["event_type"] == event_type][-1]


def test_a_push_with_the_identity_headers_is_replayed_with_what_it_declared(
    client: TestClient,
) -> None:
    posted = client.post("/api/events/meta", json={"event_type": "identity_declared", "data": {}},
                         headers=DECLARED_HEADERS)
    assert posted.status_code == 200, posted.text

    event = _last_event(client, "identity_declared")
    assert [event[field] for field in IDENTITY_FIELDS] == [
        "reviewing-harness", "1.2.3", "mcp_0123", "term_abc", "0b56e764", "high"
    ]


def test_a_push_without_the_headers_is_replayed_with_the_fields_empty(client: TestClient) -> None:
    posted = client.post("/api/events/meta", json={"event_type": "identity_absent", "data": {}})
    assert posted.status_code == 200, posted.text
    assert posted.json()["status"] == "ok"

    event = _last_event(client, "identity_absent")
    assert [event[field] for field in IDENTITY_FIELDS] == [None] * len(IDENTITY_FIELDS)


def test_a_partial_declaration_records_only_what_was_sent(client: TestClient) -> None:
    client.post("/api/events/meta", json={"event_type": "identity_partial", "data": {}},
                headers={"X-TCIP-Agent-Client-Name": "reviewing-harness"})

    event = _last_event(client, "identity_partial")
    assert event["agent_client_name"] == "reviewing-harness"
    assert event["agent_client_version"] is None
    assert event["agent_session"] is None
    assert event["terminal_session"] is None
