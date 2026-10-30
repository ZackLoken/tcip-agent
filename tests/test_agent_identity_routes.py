"""What the events route broadcasts about the harness that posted a panel event.

A post carrying the four identity headers is broadcast with their values in its identity; one
without them is answered ok and broadcast with the identity's fields empty; one carrying only the
client name is broadcast with that field alone set. The broadcast is read through its model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from tcip_mcp.agent_identity import RecordedIdentity

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
    text: str,
    headers: dict | None = None,
) -> tuple[Any, RecordedIdentity]:
    """Post a banner reading ``text`` for the open project while subscribed to its broadcast.

    The open project's id is ``project_id``. Returns the HTTP response and the identity the
    broadcast carried.
    """
    from tcip_mcp.web_client import Banner, PanelPost
    from tcip_web.state import PanelEnvelope, store

    with client.websocket_connect("ws://127.0.0.1/ws/panel/meta") as ws:
        for _ in store.retained_events("meta"):
            ws.receive_json()  # drain what the open project already retained on this panel
        posted = client.post(
            "/api/events/meta",
            json=PanelPost(
                event=Banner(text=text), project_id=project_id
            ).model_dump(mode="json"),
            headers=headers or {},
        )
        event = PanelEnvelope.model_validate(ws.receive_json())
    return posted, event.identity


def test_a_push_with_the_identity_headers_is_replayed_with_what_it_declared(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    posted, identity = _post_and_capture_broadcast(
        opened_client,
        opened.id,
        "identity declared",
        headers=DECLARED_HEADERS,
    )
    assert posted.status_code == 200, posted.text
    assert identity == RecordedIdentity(
        agent_client_name="reviewing-harness",
        agent_client_version="1.2.3",
        agent_session="mcp_0123",
        terminal_session="term_abc",
    )


def test_a_push_without_the_headers_is_replayed_with_the_fields_empty(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    posted, identity = _post_and_capture_broadcast(
        opened_client, opened.id, "identity absent"
    )
    assert posted.status_code == 200, posted.text
    assert posted.json() is None
    assert identity == RecordedIdentity()


def test_a_partial_declaration_records_only_what_was_sent(
    opened_client: TestClient,
    opened: OpenProject,
) -> None:
    _, identity = _post_and_capture_broadcast(
        opened_client,
        opened.id,
        "identity partial",
        headers={"X-TCIP-Agent-Client-Name": "reviewing-harness"},
    )
    assert identity == RecordedIdentity(agent_client_name="reviewing-harness")
