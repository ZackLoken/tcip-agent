"""What the events route broadcasts about the harness that posted a panel event.

A post carrying the four identity headers is broadcast with their values in the identity fields;
one without them is answered ok and broadcast with those fields empty; one carrying only the
client name is broadcast with that field alone set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from tcip_mcp.agent_identity import RECORD_FIELDS as IDENTITY_FIELDS

if TYPE_CHECKING:
    from tcip_web.state import OpenProject

DECLARED_HEADERS = {
    "X-TCIP-Agent-Client-Name": "reviewing-harness",
    "X-TCIP-Agent-Client-Version": "1.2.3",
    "X-TCIP-Agent-Session": "mcp_0123",
    "X-TCIP-Terminal-Session": "term_abc",
}


def _post_and_capture_broadcast(
    client: TestClient,
    project_id: str,
    event_type: str,
    data: dict,
    headers: dict | None = None,
) -> tuple[Any, dict]:
    """Post a panel event for the open project while subscribed to its live broadcast.

    The open project's id is ``project_id``. Returns the HTTP response and what the broadcast
    carried.
    """
    from tcip_web.state import store

    with client.websocket_connect("ws://127.0.0.1/ws/panel/meta") as ws:
        for _ in store.retained_events("meta"):
            ws.receive_json()  # drain what the open project already retained on this panel
        posted = client.post(
            "/api/events/meta",
            json={
                "event_type": event_type,
                "data": data,
                "project_id": project_id,
            },
            headers=headers or {},
        )
        event = ws.receive_json()
    return posted, event


def test_a_push_with_the_identity_headers_is_replayed_with_what_it_declared(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    posted, event = _post_and_capture_broadcast(
        opened_client,
        opened.id,
        "identity_declared",
        {},
        headers=DECLARED_HEADERS,
    )
    assert posted.status_code == 200, posted.text
    assert [event[field] for field in IDENTITY_FIELDS] == [
        "reviewing-harness",
        "1.2.3",
        "mcp_0123",
        "term_abc",
    ]
    assert not any("harness" in field for field in event)


def test_a_push_without_the_headers_is_replayed_with_the_fields_empty(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    posted, event = _post_and_capture_broadcast(
        opened_client, opened.id, "identity_absent", {}
    )
    assert posted.status_code == 200, posted.text
    assert posted.json()["status"] == "ok"
    assert [event[field] for field in IDENTITY_FIELDS] == [None] * len(
        IDENTITY_FIELDS
    )


def test_a_partial_declaration_records_only_what_was_sent(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    _, event = _post_and_capture_broadcast(
        opened_client,
        opened.id,
        "identity_partial",
        {},
        headers={"X-TCIP-Agent-Client-Name": "reviewing-harness"},
    )
    assert event["agent_client_name"] == "reviewing-harness"
    assert event["agent_client_version"] is None
    assert event["agent_session"] is None
    assert event["terminal_session"] is None
