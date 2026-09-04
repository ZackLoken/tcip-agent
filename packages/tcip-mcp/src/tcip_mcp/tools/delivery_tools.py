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
    dataset_root: str | None = None,
    predictions_by_date: dict[str, str] | None = None,
    images_dir: str | None = None,
    scale_capture_id: str | None = None,
) -> dict:
    """The general per-plant CSV door: ``aggregate_per_plant``'s own output plus the existing
    writer, over a plant mapping this door resolves and verifies by name rather than trusting.

    Every other tool that ships a per-plant CSV (``deliver_orthomosaic_plant_counts``,
    ``deliver_phenology_milestones``) reads its own buckets and its own plant mapping and composes
    them into ``export_aggregated_csv``'s ``results`` shape before calling it; this door is for a
    caller that has already produced that shape by some other composition (a bespoke aggregation
    script, a mapping built by ``build_plant_mapping`` joined against prediction buckets by hand)
    and has no tool surface to reach the writer from. ``results`` is taken exactly as given: this
    door never reads a bucket or a mapping to build it, and every check a delivered number has to
    pass (the meaning door, the evidence gate) is the one ``export_aggregated_csv`` already runs,
    not a second implementation of either kept in step by hand. The delivered schema is that
    writer's own ``fieldnames`` list and nothing else; this docstring does not repeat it.

    ``plant_mapping``, when named, is a claim about where ``results``' plant identities came from,
    and a claim the data must positively carry: a delivery either fully verifies the mapping it
    names or names none at all, never a half state where a name is resolved but nothing about it is
    checked. Naming a mapping without both ``dataset_root`` and ``predictions_by_date`` refuses,
    stating the two arguments; naming one that is not on record refuses by name, the way
    ``deliver_phenology_milestones`` refuses one; naming one that resolves, with both arguments
    given, is freshly verified against them (``plant_mapping.verify_mapping_inputs``, the same
    check the phenology doors run), and the resulting disclosure is recorded onto the delivery
    event, naming the mapping's own digest rather than a bare string a reader cannot check.

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
            record) whose assignments gave ``results`` their ``plant_id`` values. Empty (the
            default) states no claim and neither ``dataset_root`` nor ``predictions_by_date`` is
            required; a non-empty name requires both and is resolved through
            ``plant_mapping.load_mapping``, refusing by name when no such record exists.
        pred_dirs: The prediction buckets whose on-disk validity the writer reconciles into this
            delivery's evidence gate (the same buckets ``results``' own values came from). Omitted
            or empty has no on-disk validity producer at all, and this door takes no
            acknowledgement, so an unvalidated delivery always refuses. Derived from
            ``predictions_by_date``'s own values when omitted and that is given.
        dataset_root: The dataset ``predictions_by_date`` belongs to. Required whenever
            ``plant_mapping`` is named (to verify its own recorded inputs); refuses naming both
            arguments when ``plant_mapping`` is named and this is omitted. Has no effect when
            ``plant_mapping`` is empty.
        predictions_by_date: ``{date: predictions_dir}`` for the buckets this delivery reads,
            keyed by capture date. Required whenever ``plant_mapping`` is named, so its own
            recorded inputs are freshly verified against them rather than trusted unresolved;
            refuses naming both arguments when ``plant_mapping`` is named and this is omitted.
        images_dir: The buckets' own images directory, required when a result states
            ``scale_document`` alongside ``pred_dirs`` (see ``export_aggregated_csv``).
        scale_capture_id: The capture this delivery's physical scale must match, when the scale is
            capture-scoped.

    Returns:
        A dict carrying ``csv_path``, ``n_plants``, ``plant_mapping`` and
        ``plant_mapping_record_sha256`` (both blank together when no mapping was named, both
        carrying the name and its freshly-verified digest together when one was, never one without
        the other), and the writer's own delivered tail (``operating_point_validated``,
        ``unvalidated_dimensions``, ``checkpoint_sha256``, ``producing_experiment_id``,
        ``validation_record``); or ``{"error": ...}`` naming the refusal, unchanged from
        ``export_aggregated_csv`` or from the mapping's own resolution/verification.
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping as plant_mapping_pipeline
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, DeliveryRefused
    from tcip_mcp.project_paths import platform_state_root, resolve_output_path
    from tcip_mcp.traits import TraitUnknownError

    mapping_build = None
    mapping_disclosure = None
    if plant_mapping:
        if dataset_root is None or predictions_by_date is None:
            return {"error": (
                f"plant_mapping {plant_mapping!r} is named without both dataset_root and "
                "predictions_by_date: a delivery either fully verifies the mapping it names or "
                "names none, never a name resolved with nothing checked against it. Pass both, "
                "or name no mapping at all.")}
        mapping_build = plant_mapping_pipeline.load_mapping(platform_state_root(), plant_mapping)
        if mapping_build is None:
            return {"error": (
                f"mapping not found: {plant_mapping!r}; build one with build_plant_mapping "
                "before delivering a CSV whose plant identities it produced")}
        verified = plant_mapping_pipeline.verify_mapping_inputs(
            mapping_build, dataset_root, predictions_by_date)
        if "refusal" in verified:
            return {"error": verified["refusal"]}
        mapping_disclosure = mapping_build.delivery_disclosure(verified, predictions_by_date)

    if pred_dirs is None and predictions_by_date is not None:
        pred_dirs = list(predictions_by_date.values())

    resolved_output_path = str(resolve_output_path(output_path))
    try:
        csv_path, tail = export_aggregated_csv(
            results, resolved_output_path, delivered_phenotype=delivered_phenotype, crop=crop,
            pipeline_version=pipeline_version, pred_dirs=pred_dirs, images_dir=images_dir,
            scale_capture_id=scale_capture_id, door="deliver_per_plant_csv",
            plant_mapping=mapping_disclosure,
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
