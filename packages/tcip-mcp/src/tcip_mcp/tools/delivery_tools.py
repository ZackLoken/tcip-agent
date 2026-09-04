"""Delivery-general tools: doors over the delivery-events record that no one trait or delivery
kind owns.

``supersede_delivery`` is the withdrawal-or-replacement statement for an already-shipped file: a
delivery event records what shipped, and a supersession records that the number it named is
withdrawn or superseded by a fresh delivery, never a deletion or a rewrite of either.
"""

from __future__ import annotations

from datetime import datetime, timezone

import tcip_store
from tcip_mcp.audit import audited
from tcip_mcp.server import mcp

_SUPERSEDED_BY = "supersede_delivery"
"""The actor stamped on every supersession: this door is MCP-only (no HTTP request carries a
breeder identity to it), so the stamp names the door itself, bare, the way a tool producer's
identity is stamped everywhere else (``identity.py``), never a person's ``user:`` prefix."""


@mcp.tool()
@audited
def supersede_delivery(
    event_id: str, reason: str, *, replacement_event_id: str | None = None,
) -> dict:
    """Record that a delivered file's number is withdrawn or replaced, without touching the file
    or the event it names.

    Writes one frozen, enumerable ``delivery_supersessions`` record keyed by ``event_id`` (an
    event can carry at most one), naming the superseded event's own ``output_sha256`` (copied from
    its stored record, so the withdrawal cites exactly the bytes it withdraws), the replacement
    event when one is given, the reason and who. The Results tab's delivery panel and
    ``read_audit_log`` both render a superseded event through the same join
    (``delivery_events_schema.with_supersessions``), never a second reconstruction of it.

    Args:
        event_id: The delivery event this supersedes (``delivery_events``' own ``event_id``,
            as ``list_delivery_events``/the delivery panel names it).
        reason: Why the delivery is withdrawn or replaced. Required, non-empty: a supersession
            with no reason states nothing a reader can act on.
        replacement_event_id: A fresh delivery event that replaces this one, when one already
            exists. Refuses naming the id when it does not resolve to a stored event.

    Refuses (``{"error": ...}``) when ``event_id`` names no stored delivery event, when
    ``reason`` is empty, when ``replacement_event_id`` is given but does not resolve, and when
    ``event_id`` already carries a supersession (a withdrawal is not rewritten; a correction
    is a fresh statement naming this one's own remedy: none exists yet, so the caller removes the
    stored record by hand and asks for one, on the same conservative footing the rewritten-CSV
    refusal takes elsewhere).
    """
    from tcip_mcp.pipelines.delivery_events_schema import DeliverySupersessionRecord
    from tcip_mcp.pipelines.resolution import (
        delivery_event_key,
        delivery_events_scope,
        delivery_supersession_key,
    )
    from tcip_mcp.project_paths import platform_state_root

    if not reason or not reason.strip():
        return {"error": "reason is required and must be non-empty"}

    platform_root = platform_state_root()
    scope = delivery_events_scope(platform_root)

    event = tcip_store.read(delivery_event_key(scope, event_id), default=None)
    if event is None:
        return {"error": f"delivery event {event_id!r} not found under {scope}"}

    if replacement_event_id is not None:
        replacement = tcip_store.read(delivery_event_key(scope, replacement_event_id), default=None)
        if replacement is None:
            return {"error": f"replacement_event_id {replacement_event_id!r} not found under {scope}"}

    body = {
        "superseded_event_id": event_id,
        "output_sha256": event.get("output_sha256"),
        "replacement_event_id": replacement_event_id,
        "reason": reason,
        "superseded_by": _SUPERSEDED_BY,
        "superseded_at": datetime.now(timezone.utc).isoformat(),
    }
    DeliverySupersessionRecord.model_validate(body)

    key = delivery_supersession_key(scope, event_id)
    try:
        tcip_store.replace(key, body, expect=tcip_store.Version.ABSENT)
    except tcip_store.VersionConflict:
        return {"error": f"delivery event {event_id!r} already carries a supersession"}

    return {
        "superseded_event_id": event_id,
        "output_sha256": body["output_sha256"],
        "replacement_event_id": replacement_event_id,
        "reason": reason,
    }
