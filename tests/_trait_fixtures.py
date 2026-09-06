"""Shared test-only trait fixture.

``BUD_OPENING`` used to be a hardcoded builtin in ``tcip_mcp.traits`` that every trait-consuming
test imported directly. There are no built-ins anymore;
every trait is authored as a per-project ``.tcip/state/trait_specs/*.yml`` file, so it is
registered only where that file actually exists. This module is that same value, reconstructed
locally with neutral names (a mechanism fixture, not a crop's own trait) so the existing test suite
keeps exercising trait-consuming code paths without depending on any specific project's config
being present. It is a plain local ``TraitSpec`` literal, not a registered trait: config-loaded
specs are rebuilt fresh on every ``get_trait()`` call, never module-load singletons (see
``traits.py``), so nothing here should ever be compared by identity against one.

The trait's own name, ``bud_opening``, is deliberately not the ``bud`` subject it measures: a
subject is an object class to isolate, a trait is the measurement over it, and this fixture keeps
that distinction rather than reusing one string for both.
"""

from __future__ import annotations

from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec

BUD_OPENING = TraitSpec(
    name="bud_opening",
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    localization_tolerance="half_class_avg_size",
    localization_tolerance_frac=0.5,
    holdout_match_quality_floor=0.5,  # fixture value: loose enough for the synthetic dense references to clear
    positive_class_name="open",
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    majority_milestone="95per",
    majority_provisional=True,
    phenology_prefix="bud",
    majority_label="majority",
    sliver_policy="class_avg_size",
    sliver_frac=0.5,
    delivers=("leaf_out_05per_date", "leaf_out_50per_date"),
    notes="The fraction of a plant's bud objects that are open: a texture call on the object "
          "itself, never a bbox-ratio proxy.",
)
