"""The delivery population a test hands ``deliver_phenology_milestones``.

A phenology delivery takes its population as an explicit list of plant ids; nothing in the
platform derives one from a mapping. A test that built a mapping of two plants and wants both
delivered states that here, by reading the plant ids back out of the persisted mapping it
wrote, so the population it passes is exactly the plants its own fixture named. A test whose
door refuses before the mapping is ever read (a missing or forged mapping) gets an empty list,
which that earlier refusal answers first.
"""

from __future__ import annotations

from pathlib import Path


def mapped_plants(project: Path, mapping_name: str) -> list[str]:
    """Every plant id the mapping ``mapping_name`` under ``project`` names, in first-seen order;
    ``[]`` when no such mapping loads."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import assignment_is_attributed, load_mapping

    try:
        build = load_mapping(project, mapping_name)
    except Exception:
        return []
    if build is None:
        return []
    plants: dict[str, None] = {}
    for assignments in build.assignments.values():
        for a in assignments:
            if assignment_is_attributed(a):
                plants.setdefault(a.plot_name, None)
    return list(plants)
