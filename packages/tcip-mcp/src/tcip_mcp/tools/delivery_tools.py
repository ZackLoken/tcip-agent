"""Delivery-general tools: doors over the delivery record and its writer that no one trait or
delivery kind owns.

``deliver_per_plant_csv`` is the MCP door over ``export_aggregated_csv``, the general per-plant
delivery writer: every specialist per-plant door (``deliver_orthomosaic_plant_counts``,
``deliver_phenology_milestones``) composes its own buckets and mapping into that writer's own
``results`` shape and calls it directly, and until this door existed the general case (a caller
that has already produced ``aggregate_per_plant``'s own output some other way) had no tool surface
at all. ``supersede_delivery`` is the withdrawal-or-replacement statement for an already-shipped
file: a delivery event records what shipped, and a supersession records that the number it named is
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
def deliver_per_plant_csv(
    results: list[dict],
    output_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    plant_mapping: str = "",
    *,
    pred_dirs: list[str] | None = None,
    images_dir: str | None = None,
    scale_capture_id: str | None = None,
) -> dict:
    """The general per-plant CSV door: ``aggregate_per_plant``'s own output plus the existing
    writer, nothing else.

    Every other tool that ships a per-plant CSV (``deliver_orthomosaic_plant_counts``,
    ``deliver_phenology_milestones``) reads its own buckets and its own plant mapping and composes
    them into ``export_aggregated_csv``'s ``results`` shape before calling it; this door is for a
    caller that has already produced that shape by some other composition (a bespoke aggregation
    script, a mapping built by ``build_plant_mapping`` joined against prediction buckets by hand)
    and has no tool surface to reach the writer from. It reads no bucket, resolves no mapping and
    recomputes nothing: ``results`` is taken exactly as given, and every check a delivered number
    has to pass (the meaning door, the evidence gate) is the one ``export_aggregated_csv`` already
    runs, not a second implementation of either kept in step by hand. The delivered schema is that
    writer's own ``fieldnames`` list and nothing else; this docstring does not repeat it.

    Args:
        results: ``aggregate_per_plant``'s own output: one dict per plant, each carrying at least
            ``plant_id``, ``value``, ``value_key``, ``measurement_document`` and
            ``plant_attribution`` (``aggregate_per_plant``'s own required statement fields).
            Produced by calling ``aggregate_per_plant`` over per-image records that already carry
            plant identity, never guessed here.
        output_path: Where to write the delivered CSV. A relative path resolves against the
            platform state root, never the server process's cwd.
        delivered_phenotype: The crop-vocabulary delivered phenotype this CSV ships under, resolved
            to the registered trait whose spec delivers it and whose confirmed operationalization
            this delivery rests on.
        crop: Crop species name, written into every row's own column.
        pipeline_version: Pipeline identifier, written into every row's own column.
        plant_mapping: The name of the plant mapping (``build_plant_mapping``'s own persisted
            record) whose assignments gave ``results`` their ``plant_id`` values. Disclosed in this
            call's own audit line and echoed back in the response so a delivered CSV's plant
            identities can be traced to the mapping that assigned them; this door does not read,
            verify or resolve the named mapping itself, since ``results`` is already the aggregated
            output of it, not a bucket or path this door composes.
        pred_dirs: The prediction buckets ``results`` came from, so the writer can reconcile each
            dimension's validity from their own on-disk sidecars rather than floor to unvalidated
            unconditionally. Omitted or empty has no on-disk validity producer at all, and this door
            takes no acknowledgement, so an unvalidated delivery always refuses.
        images_dir: The buckets' own images directory, required when a result states
            ``scale_document`` alongside ``pred_dirs`` (see ``export_aggregated_csv``).
        scale_capture_id: The capture this delivery's physical scale must match, when the scale is
            capture-scoped.

    Returns:
        A dict carrying ``csv_path``, ``n_plants``, ``plant_mapping`` (echoed back), and the
        writer's own delivered tail (``operating_point_validated``, ``unvalidated_dimensions``,
        ``checkpoint_sha256``, ``producing_experiment_id``, ``validation_record``); or
        ``{"error": ...}`` naming the writer's own refusal, unchanged, when the meaning door or the
        evidence gate refused.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, DeliveryRefused
    from tcip_mcp.project_paths import resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    resolved_output_path = str(resolve_output_path(output_path))
    try:
        csv_path, tail = export_aggregated_csv(
            results, resolved_output_path, delivered_phenotype=delivered_phenotype, crop=crop,
            pipeline_version=pipeline_version, pred_dirs=pred_dirs, images_dir=images_dir,
            scale_capture_id=scale_capture_id, door="deliver_per_plant_csv",
        )
    except DeliveryRefused as exc:
        refusal = {
            "error": str(exc),
            "operating_point_validated": exc.gate.stamp.get("operating_point", VALIDATED_FALSE),
            "unvalidated_dimensions": exc.gate.unvalidated_cell(),
        }
        for dimension in ("tile_size", "scale", "claim_scope"):
            if dimension in exc.gate.stamp:
                refusal[f"{dimension}_validated"] = exc.gate.stamp[dimension]
        return refusal
    except (TraitUnknownError, ValueError) as exc:
        return {"error": str(exc)}

    return {
        "csv_path": csv_path,
        "n_plants": len(results),
        "plant_mapping": plant_mapping,
        "operating_point_validated": tail["operating_point_validated"],
        "unvalidated_dimensions": tail["unvalidated_dimensions"],
        "checkpoint_sha256": tail["producer_model_sha256"],
        "producing_experiment_id": tail["producing_experiment_id"],
        "validation_record": tail["validation_record"],
    }


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
