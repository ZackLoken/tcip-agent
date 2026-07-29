"""Shared test-only trait fixture (K5/K6/round-10).

``CATKIN`` used to be a hardcoded builtin in ``tcip_mcp.traits`` that every trait-consuming test
imported directly. There are no built-ins anymore (2026-07-29) — every trait, including the real
catkin, is authored as a per-project ``.tcip/state/trait_specs/*.yml`` file, so it is registered
only where that file actually exists. This module is that same value, reconstructed locally so the
existing test suite keeps exercising trait-consuming code paths without depending on any specific
project's config being present. It is a plain local ``TraitSpec`` literal, not a registered trait —
config-loaded specs are rebuilt fresh on every ``get_trait()`` call, never module-load singletons
(see ``traits.py``), so nothing here should ever be compared by identity against one.
"""

from __future__ import annotations

from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec

CATKIN = TraitSpec(
    name="catkin",
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    localization_tolerance="half_class_avg_size",
    localization_tolerance_frac=0.5,
    positive_class_name="elongated",
    positive_is_texture=True,
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    majority_milestone="95per",
    majority_provisional=True,
    phenology_prefix="catkin",
    majority_label="elongation",
    sliver_policy="class_avg_size",
    sliver_frac=0.5,
    delivers=("catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"),
    notes="Bloom = fraction of a plant's catkins that are elongated. Elongated is a texture call "
          "(frilled/salt-and-peppery), never a bbox-ratio proxy.",
)
