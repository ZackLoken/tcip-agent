"""The agent-facing doors for authoring (``author_trait_spec``) and field-editing
(``revise_trait_spec``) a trait spec; the breeder confirms from the web GUI.
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
    positive_value: str = "",
    milestone_fractions: list[float] | None = None,
    milestone_on: str = "",
    majority_milestone: str = "",
    crossing_unconfirmed: bool = False,
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
    real phenotype, or this refuses. Refuses when a spec and its authoring statement are both
    already on record for this trait; change an already-registered spec's fields with
    `revise_trait_spec` instead.

    Writing this does not confirm anything on its own; the breeder confirms the statement in the
    GUI before it can back a delivery.

    Args:
        project_root: The project whose registry holds the trait.
        trait: The name to register this trait's spec under.
        delivers: The crop-vocabulary phenotype name(s) this trait's spec claims to deliver.
            Required, and every entry must be in crops.yml.
        rationale: The agent's account of why it chose these values, from the breeder's own words.
            Prose, read by a breeder, not parsed.
        positive_value: The `subjects.json` subject the positive call resolves to, if any.
        milestone_fractions: Crossing fractions for a milestone-delivering trait.
        milestone_on: The quantity the milestones cross, e.g. `positive_fraction`.
        majority_milestone: The crops.yml majority-date crossing key this trait's milestones map
            to, e.g. `95per`.
        crossing_unconfirmed: Whether that mapping is not yet breeder-confirmed.
        phenology_prefix: The phenology CSV column prefix this trait's milestones use.
        majority_label: The label the majority-alias column carries.
        count_objective: What the delivered number needs to be reliable for
            (`count_unbiased`/`detection_f1`/`presence`), a consequence judgment only the breeder
            can make. Left empty, the platform defaults to `count_unbiased` until stated.
        count_bias_tolerance_frac: Max acceptable relative per-image count bias on the held-out
            split, a breeder-authored measurement decision with no platform-derived value.
        count_error_tolerance: Max acceptable p90 per-image count error on the held-out split.
        classifier_agreement_floor: Min acceptable Cohen's kappa for the classifier operating point
            to count as validated.
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
    confirming surface compares against.
    """
    try:
        statement = traits.author_trait_spec(
            project_root,
            trait,
            delivers=delivers,
            positive_value=positive_value,
            milestone_fractions=milestone_fractions or (),
            milestone_on=milestone_on,
            majority_milestone=majority_milestone,
            crossing_unconfirmed=crossing_unconfirmed,
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
def revise_trait_spec(
    project_root: str, trait_name: str, fields: dict, rationale: str, relayed_note: str = "",
) -> dict:
    """Update one or more fields on an already-registered trait's spec, and record why.

    Refuses if the trait has no existing spec file or if the merged result would fail the crops.yml
    cross-check every config-authored spec goes through. Returns the updated spec.

    Every call carries a rationale, so after its write the trait's own trait-spec statement is
    stated fresh whenever it was absent or stale, or whenever this call moves a field the
    statement's authored fields cover (a moved authored field, `fields={}` over a spec with no
    statement or a stale one, a carried-forward edit over a stale statement all restate), and is
    left alone whenever it is current and this call moved no authored field. A fresh statement is
    unconfirmed; the breeder confirms it in the Results tab, and `statement_restated` and
    `statement_note` say what happened here. `rationale` is the breeder's own account of the spec's
    authored values whenever this call restates, a carried-forward edit included; it is not parsed.

    An operationalization the breeder confirmed covers the field values it was confirmed against,
    so a field this call moves can leave one superseded. That is reported in `superseded`, naming
    the delivery kind and both values; the delivery precondition re-reads the spec and refuses on
    its own.

    Args:
        project_root: The project whose spec registry to update. Required.
        trait_name: Name of the already-registered trait whose spec file to update.
        fields: `TraitSpec` field names to new values, merged into the existing spec (unknown
            fields, off-vocab `delivers` entries, or an invalid value refuse the whole write).
        rationale: The agent's account of why it chose these values, from the breeder's own words.
            Prose, read by a breeder, not parsed. Required, and must say something.
        relayed_note: What the breeder said away from the GUI, recorded as a relay attributed to
            the agent, on the trait-spec statement this call writes; the breeder reads it in the
            Results tab's statement panel. Nothing surfaces it in a refusal.

    Returns the updated spec as stored, `schema_version` included (the current ceiling, stamped on
    every write), `superseded`, `statement_restated` (true when this call stated or restated the
    trait's trait-spec statement) and `statement_note`; when it did, also `record_seen`, the
    content hash the confirming surface compares against.
    """
    from tcip_mcp import operationalization

    try:
        revision = traits.revise_trait_spec_fields(
            trait_name, fields, project_root=project_root,
            rationale=rationale, relayed_note=relayed_note,
        )
    except ValueError as e:
        return {"error": str(e)}
    updated = traits._encode_spec(revision.spec)
    updated["superseded"] = operationalization.superseded_confirmations(
        project_root, trait_name, spec=revision.spec
    )
    updated["statement_restated"] = revision.statement is not None
    updated["statement_note"] = revision.statement_note
    if revision.statement is not None:
        updated["record_seen"] = traits.trait_spec_statement_seen_hash(revision.statement)
    return updated
