"""The agent-facing surface for authoring and field-editing a trait spec: two doors of one
surface, deliberately no more.

``author_trait_spec`` registers a trait that does not yet exist, recording the agent's own
account of why, in the breeder's terms; the breeder confirms it from the web GUI, through a route
no MCP tool can reach. A single tool doing both authoring and confirmation would put the
confirmation inside the agent's own tool surface and make honest attribution depend on the agent
choosing not to fill a field. ``update_trait_spec_fields`` is the other door: it edits fields on a
spec already on record, never creates one, so the two doors never overlap.
"""

from __future__ import annotations

from tcip_mcp import traits
from tcip_mcp.audit import audited
from tcip_mcp.server import mcp


@mcp.tool()
@audited(scope_arg="project_root")
def author_trait_spec(
    project_root: str,
    trait: str,
    delivers: list[str],
    rationale: str,
    positive_class_name: str = "",
    milestone_fractions: list[float] | None = None,
    milestone_on: str = "",
    majority_milestone: str = "",
    majority_provisional: bool = False,
    phenology_prefix: str = "",
    majority_label: str = "",
    count_objective: str = "",
    count_bias_tolerance_frac: float | None = None,
    count_error_tolerance: float | None = None,
    classifier_agreement_floor: float | None = None,
    ordinal_agreement_floor: float | None = None,
    regression_skill_floor: float | None = None,
    scale_tolerance_frac: float | None = None,
    holdout_match_quality_floor: float | None = None,
    notes: str = "",
    relayed_note: str = "",
) -> dict:
    """Register a trait that does not yet exist, and record why, in the breeder's terms.

    Cross-checked against the crops.yml controlled vocabulary: `delivers` must name at least one
    real phenotype, or this refuses rather than registering a fabricated trait. Refuses when a
    spec and its authoring statement are both already on record for this trait; change an
    already-registered spec's fields with `update_trait_spec_fields` instead.

    Ask the breeder what the trait's measurement is, in their own terms, and record their answer
    here. Propose the semantics the breeder actually stated, never a plausible-sounding guess: a
    fabricated definition becomes the definition, and the measurement is theirs to define. Writing
    this does not confirm anything on its own; the breeder confirms the statement in the GUI before
    it can back a delivery.

    Args:
        project_root: The project whose registry holds the trait.
        trait: The name to register this trait's spec under.
        delivers: The crop-vocabulary phenotype name(s) this trait's spec claims to deliver.
            Required, and every entry must be in crops.yml.
        rationale: The agent's account of why it chose these values, from the breeder's own
            words. Prose, read by a breeder, not parsed.
        positive_class_name: The `classes.json` class the positive call resolves to, if any.
        milestone_fractions: Crossing fractions for a milestone-delivering trait.
        milestone_on: The quantity the milestones cross, e.g. `positive_fraction`.
        majority_milestone: The crops.yml majority-date crossing key this trait's milestones map
            to, e.g. `95per`.
        majority_provisional: Whether that mapping is provisional, pending breeder confirmation.
        phenology_prefix: The phenology CSV column prefix this trait's milestones use.
        majority_label: The label the majority-alias column carries.
        count_objective: What the delivered number needs to be reliable for
            (`count_unbiased`/`detection_f1`/`presence`), a consequence judgment only the breeder
            can make. Left empty, the platform defaults to `count_unbiased` until stated.
        count_bias_tolerance_frac: Max acceptable relative per-image count bias on the held-out
            split, a breeder-authored measurement decision with no platform-derived value.
        count_error_tolerance: Max acceptable p90 per-image count error on the held-out split.
        classifier_agreement_floor: Min acceptable Cohen's kappa for the classifier operating
            point to count as validated.
        ordinal_agreement_floor: Min acceptable ordinal agreement criterion value.
        regression_skill_floor: Min acceptable regression skill/agreement criterion value.
        scale_tolerance_frac: Max acceptable relative disagreement a physical-scale calibration's
            held-out reference half may show, a breeder-authored measurement decision with no
            platform-derived value.
        holdout_match_quality_floor: Min acceptable held-out precision and recall of the detection
            gate's governing localization criterion, both required, a breeder-authored measurement
            decision with no platform-derived value.
        notes: Free-text notes on the trait's measurement.
        relayed_note: What the breeder said away from the GUI, recorded as a relay attributed to
            the agent. It is surfaced in a delivery refusal and never clears it.

    Returns the unconfirmed statement as written, plus `record_seen`, the content hash the
    confirming surface compares against so a click cannot confirm text nobody displayed.
    """
    try:
        statement = traits.author_trait_spec(
            project_root,
            trait,
            delivers=delivers,
            positive_class_name=positive_class_name,
            milestone_fractions=milestone_fractions or (),
            milestone_on=milestone_on,
            majority_milestone=majority_milestone,
            majority_provisional=majority_provisional,
            phenology_prefix=phenology_prefix,
            majority_label=majority_label,
            count_objective=count_objective,
            count_bias_tolerance_frac=count_bias_tolerance_frac,
            count_error_tolerance=count_error_tolerance,
            classifier_agreement_floor=classifier_agreement_floor,
            ordinal_agreement_floor=ordinal_agreement_floor,
            regression_skill_floor=regression_skill_floor,
            scale_tolerance_frac=scale_tolerance_frac,
            holdout_match_quality_floor=holdout_match_quality_floor,
            notes=notes,
            rationale=rationale,
            relayed_note=relayed_note,
        )
    except ValueError as e:
        return {"error": str(e)}
    return {**statement, "record_seen": traits.trait_spec_statement_seen_hash(statement)}


@mcp.tool()
@audited
def update_trait_spec_fields(
    project_root: str, trait_name: str, fields: dict
) -> dict:
    """Update one or more fields on an already-registered trait's spec.

    Hand-editing a trait spec's YAML directly bypasses the audit record and skips re-validation.
    This refuses if the trait has no existing spec file (creating a new
    trait is a separate, still-manual authoring step) or if the merged result would fail the same
    crops.yml cross-check every config-authored spec already goes through. Returns the updated
    spec.

    This is what a real localization-kind derivation (from actual GT box geometry) or a real
    breeder-answered count objective gets recorded through, never a silent default and never
    copied from another trait's values, both durable, audited facts instead of living only in a
    session's memory.

    An operationalization the breeder confirmed covers the field values it was confirmed against,
    so a field this call moves can leave one superseded. That is reported in `superseded`, naming
    the delivery kind and both values, as a convenience so the agent learns here rather than at the
    next delivery refusal. It is not the enforcement point: the delivery precondition re-reads the
    spec and refuses on its own, which also catches a spec edited by hand.

    Args:
        project_root: The project whose spec registry to update. Required: the platform root this
            process is pinned to can be a different project entirely, and a spec written to the
            wrong registry is a measurement decision recorded where nothing reads it.
        trait_name: Name of the already-registered trait whose spec file to update.
        fields: `TraitSpec` field names to new values, merged into the existing spec (unknown
            fields, off-vocab `delivers` entries, or an invalid value refuse the whole write).
    """
    from tcip_mcp import operationalization

    spec = traits.write_trait_spec_fields(trait_name, fields, project_root=project_root)
    updated = traits._encode_spec(spec)
    updated["superseded"] = operationalization.superseded_confirmations(
        project_root, trait_name, spec=spec
    )
    return updated
