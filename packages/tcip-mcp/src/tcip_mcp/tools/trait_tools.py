"""The agent-facing door for proposing a trait's entry; the breeder confirms it in the Setup tab."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp import traits
from tcip_mcp.server import tool


@tool()
def propose_trait(
    project: Path,
    entry: traits.TraitEntry,
    rationale: str,
    relayed_note: str = "",
    dataset_root: str = "",
) -> dict:
    """Propose a trait's complete entry, its spec fields and what its delivered number means per
    delivery kind, as a new unconfirmed revision of that trait in this project. The whole entry is
    proposed every time, carrying forward whatever is unchanged.

    Args:
        entry: The complete entry; its schema states every field and which kinds require what.
        rationale: Why this entry, from the breeder's own words. Prose, read by the breeder.
        relayed_note: What the breeder said away from the GUI, relayed by you; never a confirmation.
        dataset_root: The dataset whose subject registry a ``state_crossing_dates``
            operationalization's positive state is checked against. Empty resolves the project's
            own registry when the project has one dataset; otherwise this refuses naming them.

    Returns the revision as written (its ``number`` and ``entry_sha256`` among it), or
    ``{"error": ...}`` naming the refusal.
    """
    try:
        revision = traits.propose_trait(
            project, entry, rationale=rationale, relayed_note=relayed_note,
            dataset_root=dataset_root,
        )
    except ValueError as e:
        return {"error": str(e)}
    return revision.model_dump(mode="json")
