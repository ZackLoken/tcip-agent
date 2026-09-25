"""Delivery tools not owned by one trait or delivery kind: the general per-plant CSV door and
delivery supersession.
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
def deliver_per_plant_csv(
    results: list[dict],
    output_path: str,
    delivered_phenotype: str,
    crop: str = "",
    pipeline_version: str = "",
    plant_mapping: str = "",
    *,
    predictions_by_date: dict[str, str] | None = None,
    images_dir: str | None = None,
    scale_capture_id: str | None = None,
) -> dict:
    """The general per-plant CSV door: ``aggregate_per_plant``'s own output plus
    ``export_aggregated_csv``, over a plant mapping this door resolves and verifies by name.

    ``results`` is taken exactly as given; every check a delivered number has to pass (the meaning
    door, the evidence gate) is ``export_aggregated_csv``'s, and so is the delivered schema.

    ``plant_mapping``, when named, is a claim about where ``results``' plant identities came from.
    Naming a mapping without ``predictions_by_date`` refuses, stating the argument; naming one that
    is not on record, one that does not cover a named date, one built over a different dataset than
    the buckets belong to, or one whose recorded inputs no longer verify against what is on disk
    now, refuses by name through ``plant_mapping.resolve_delivery_mapping``. Every delivered
    ``plant_id`` in ``results`` must appear among a plot the mapping assigned on the delivered
    dates, refusing by name otherwise.

    Args:
        results: ``aggregate_per_plant``'s own output: one dict per plant, each carrying at least
            ``plant_id``, ``value``, ``value_key``, ``measurement_document`` and
            ``plant_attribution``.
        output_path: Where to write the delivered CSV. A relative path resolves against the
            platform state root, never the server process's cwd.
        delivered_phenotype: The crop-vocabulary delivered phenotype this CSV ships under, resolved
            to the registered trait whose spec delivers it and whose confirmed operationalization
            this delivery rests on.
        crop: Crop species name, written into every row's own column.
        pipeline_version: Pipeline identifier, written into every row's own column.
        plant_mapping: The name of the plant mapping (``build_plant_mapping``'s own persisted
            record) whose assignments gave ``results`` their ``plant_id`` values. Empty (the
            default) states no claim; a non-empty name requires ``predictions_by_date`` and is
            resolved through ``plant_mapping.resolve_delivery_mapping``.
        predictions_by_date: ``{date: predictions_dir}`` for the buckets this delivery reads, keyed
            by capture date. Required whenever ``plant_mapping`` is named. Its values are also the
            writer's ``pred_dirs``: omitted, there is no on-disk validity producer, and this door
            takes no acknowledgment, so an unvalidated delivery always refuses.
        images_dir: The buckets' own images directory, required when a result states
            ``scale_document`` alongside ``predictions_by_date`` (see ``export_aggregated_csv``).
        scale_capture_id: The capture this delivery's physical scale must match, when the scale is
            capture-scoped.

    Returns:
        A dict carrying ``csv_path``, ``n_plants``, ``plant_mapping`` and
        ``plant_mapping_record_sha256`` (both blank together when no mapping was named, both
        carrying the name and its freshly-verified digest together when one was), and the writer's
        own delivered tail (``operating_point_validated``, ``unvalidated_dimensions``,
        ``checkpoint_sha256``, ``producing_experiment_id``, ``validation_record``); or ``{"error":
        ...}`` naming the refusal.
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping as plant_mapping_pipeline
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, DeliveryRefused
    from tcip_mcp.project_paths import platform_state_root, resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    mapping_build = None
    mapping_disclosure = None
    if plant_mapping:
        if predictions_by_date is None:
            return {"error": (
                f"plant_mapping {plant_mapping!r} is named without predictions_by_date: a "
                "delivery either fully verifies the mapping it names or names none, never a name "
                "resolved with nothing checked against it. Pass predictions_by_date, or name no "
                "mapping at all.")}
        try:
            mapping_build, verified = plant_mapping_pipeline.resolve_delivery_mapping(
                platform_state_root(), plant_mapping, predictions_by_date)
        except plant_mapping_pipeline.MappingDeliveryRefusal as exc:
            return {"error": str(exc)}
        mapping_disclosure = mapping_build.delivery_disclosure(verified, predictions_by_date)

        assigned_plots = {
            assignment.plot_name
            for date in predictions_by_date
            for assignment in mapping_build.assignments.get(date, [])
            if plant_mapping_pipeline.assignment_is_attributed(assignment)
        }
        unmapped_plants = sorted(
            {r.get("plant_id") for r in results if r.get("plant_id") not in assigned_plots},
            key=str,
        )
        if unmapped_plants:
            return {"error": (
                f"plant(s) {unmapped_plants} in results carry a plant_id plant_mapping "
                f"{plant_mapping!r} assigned to no plot on the delivered dates "
                f"({sorted(predictions_by_date)}): results' plant identities must come from this "
                "mapping's own assignments, or name no mapping at all")}

    pred_dirs = list(predictions_by_date.values()) if predictions_by_date else None

    resolved_output_path = str(resolve_output_path(output_path))
    try:
        csv_path, tail, _event_recorded = export_aggregated_csv(
            results, resolved_output_path, delivered_phenotype=delivered_phenotype, crop=crop,
            pipeline_version=pipeline_version, pred_dirs=pred_dirs, images_dir=images_dir,
            scale_capture_id=scale_capture_id, door="deliver_per_plant_csv",
            plant_mapping=mapping_disclosure,
        )
    except DeliveryRefused as exc:
        refusal = {
            "error": (
                f"{exc} This kind has no acknowledged route: a Results door only ever composes "
                "the rows it delivers, never a caller-composed table, and this delivery's "
                "results are exactly that. Validate the dimension named above, or, for a "
                "per-image bucket, promote it to validated through the review validation route "
                "(validate_reference) and re-deliver."
            ),
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
        "plant_mapping_record_sha256": mapping_build.record_sha256 if mapping_build else "",
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
    """Record that a delivered file's number is withdrawn or replaced, without touching the file or
    the event it names.

    Writes one frozen, enumerable ``delivery_supersessions`` record keyed by ``event_id`` (an event
    can carry at most one), naming the superseded event's own ``output_sha256`` (copied from its
    stored record), the replacement event when one is given, the reason and who. Readers pair it
    with its event through ``delivery_events_schema.with_supersessions``.

    Args:
        event_id: The delivery event this supersedes (``delivery_events``' own ``event_id``).
        reason: Why the delivery is withdrawn or replaced. Required, non-empty.
        replacement_event_id: A fresh delivery event that replaces this one, when one already
            exists. Refuses naming the id when it does not resolve to a stored event.

    Refuses (``{"error": ...}``) when ``event_id`` names no stored delivery event, when either the
    superseded or the replacement event is stored but does not validate against
    ``DeliveryEventRecord``, when ``reason`` is empty, when ``replacement_event_id`` is given but
    does not resolve, and when ``event_id`` already carries a supersession.
    """
    from tcip_mcp.pipelines.delivery_events_schema import DeliverySupersessionRecord
    from tcip_mcp.pipelines.resolution import (
        DeliveryEventShapeError,
        delivery_events_scope,
        delivery_supersession_key,
        read_one_delivery_event,
    )
    from tcip_mcp.project_paths import platform_state_root

    if not reason or not reason.strip():
        return {"error": "reason is required and must be non-empty"}

    platform_root = platform_state_root()
    scope = delivery_events_scope(platform_root)

    try:
        event = read_one_delivery_event(platform_root, event_id)
    except DeliveryEventShapeError as exc:
        return {"error": str(exc)}
    if event is None:
        return {"error": f"delivery event {event_id!r} not found under {scope}"}

    if replacement_event_id is not None:
        try:
            replacement = read_one_delivery_event(platform_root, replacement_event_id)
        except DeliveryEventShapeError as exc:
            return {"error": str(exc)}
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
