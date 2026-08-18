"""The agent-facing surface for authoring a trait spec that does not yet exist.

One tool, and deliberately only one. The agent authors the spec and its own account of why, in the
breeder's terms; the breeder confirms it from the web GUI, through a route no MCP tool can reach. A
single tool doing both would put the confirmation inside the agent's own tool surface and make
honest attribution depend on the agent choosing not to fill a field.
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
            notes=notes,
            rationale=rationale,
            relayed_note=relayed_note,
        )
    except ValueError as e:
        return {"error": str(e)}
    return {**statement, "record_seen": traits.trait_spec_statement_seen_hash(statement)}
