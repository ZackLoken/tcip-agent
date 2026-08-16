"""The agent-facing surface for recording what a trait's delivered number means.

One tool, and deliberately only one. The agent states the operationalization it worked out with
the breeder; the breeder confirms it from the web GUI, through a route no MCP tool can reach. A
single tool doing both would put the confirmation inside the agent's own tool surface and make
honest attribution depend on the agent choosing not to fill a field.
"""

from __future__ import annotations

from tcip_mcp import operationalization
from tcip_mcp.audit import audited
from tcip_mcp.server import mcp
from tcip_mcp.traits import TraitUnknownError


@mcp.tool()
@audited(scope_arg="project_root")
def state_trait_operationalization(
    project_root: str,
    trait: str,
    delivery_kind: str,
    statement: str,
    mechanism: str,
    measured_subject: str,
    delivered_phenotypes: list[str],
    delivered_value_keys: list[str] | None = None,
    relayed_note: str = "",
) -> dict:
    """Record what this trait's delivered number means, in the breeder's terms, for one delivery.

    Every delivery door reads this record before it writes anything. A trait with none, or one
    nobody confirmed, delivers nothing, and the refusal names which half is missing.

    Ask the breeder what the number should mean in their own terms and record their answer.
    Propose the mechanism that would realize it, never the meaning: a suggested meaning becomes the
    meaning, and the measurement is theirs to define. Writing this does not clear the refusal on its
    own; the breeder confirms it in the Results tab, and the same delivery call then proceeds
    unchanged. Restating clears any confirmation, because a changed definition is unconfirmed.

    Args:
        project_root: The project whose registry holds the trait and whose state holds the record.
        trait: A trait registered in that project's own spec registry.
        delivery_kind: Which delivered artifact this record covers, one of
            `state_crossing_dates`, `per_image_count`, `per_plant_count_aggregate`,
            `per_plant_ordinal_aggregate`, `per_plant_regression_aggregate`. The door derives its
            own kind, and a refusal names the one that would clear it.
        statement: What the delivered number means, in the breeder's own words.
        mechanism: What produces the call or the number: which subject, which attribute, which
            model decides the state. Prose, read by a breeder, not parsed.
        measured_subject: The `classes.json` subject the number is about.
        delivered_phenotypes: Which of the trait's own `delivers` entries this record covers.
            Required empty for `per_image_count`, whose CSV names no phenotype in any column.
        delivered_value_keys: The value keys delivered rows may carry. Required for the three
            per-plant aggregate kinds, whose rows each carry one; empty for the other two, whose
            row schema is fixed by their writer.
        relayed_note: What the breeder said away from the GUI, recorded as a relay attributed to
            the agent. It is surfaced in the refusal and never clears it.

    Returns the record as written, plus `record_seen`, the content hash the confirming surface
    compares against so a click cannot confirm text nobody displayed.
    """
    try:
        record = operationalization.state_operationalization(
            project_root,
            trait,
            delivery_kind,
            statement=statement,
            mechanism=mechanism,
            measured_subject=measured_subject,
            delivered_phenotypes=delivered_phenotypes or (),
            delivered_value_keys=delivered_value_keys or (),
            relayed_note=relayed_note,
        )
    except (TraitUnknownError, ValueError) as e:
        return {"error": str(e)}
    return {**record, "record_seen": operationalization.record_seen_hash(record)}
