"""Trait knowledge — the human-defined *semantics* of each measurable trait (Tier C).

These are the things the domain expert defines once per trait and the agent *reads* — never derives,
never re-asks per dataset (CLAUDE.md: the human defines a trait's intent/semantics; the agent derives
the operating points that realize it). Keeping them in one place, versioned with the code, stops a
measurement definition from living only in a session's memory.

A ``TraitSpec`` says: what the phenotype *is* (count objective), what "finding one" *means*
(localization), how the elongated/dormant call is defined (texture, not geometry), the milestone
convention, and the tile-seam sliver policy. Operating-point *values* (conf, IoU, tolerances) are
deliberately absent — those are derived per dataset at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

# Count objectives — what the resolved operating point optimizes.
COUNT_UNBIASED = "count_unbiased"  # minimize signed per-image count bias E[FP-FN]; the phenotype is a count
DETECTION_F1 = "detection_f1"      # optimize matching quality; the phenotype is presence/localization
PRESENCE = "presence"             # only whether the object is present

# Localization — what counts as "finding" an object.
CENTER_MATCH = "center_match"  # predicted center within a derived tolerance of a GT center
IOU_MATCH = "iou_match"        # IoU >= a derived/def threshold


@dataclass(frozen=True)
class TraitSpec:
    """The semantics of one trait — read, never derived."""

    name: str
    task: str  # "detection" | "classification" | ...
    # What the delivered phenotype is, and hence what the operating point optimizes.
    count_objective: str = COUNT_UNBIASED
    # What "a hit" means when validating counts. For small objects, center-match (IoU is noise).
    localization: str = CENTER_MATCH
    # How the localization tolerance is DERIVED (semantic names the recipe; the value is per-dataset).
    localization_tolerance: str = "half_class_avg_size"
    # The class id that is the phenotype-positive ("elongated"), or None if the trait is single-class.
    positive_class_id: int | None = None
    # Elongation/positive call is a learned TEXTURE classification — geometric proxies are forbidden.
    positive_is_texture: bool = False
    # Milestone crossing fractions and the quantity they cross.
    milestone_fractions: tuple[float, ...] = ()
    milestone_on: str = ""  # e.g. "positive_fraction"
    # How the tile-seam sliver cutoff is derived (partial objects count unless below this).
    sliver_policy: str = "class_avg_size"
    # Max acceptable mean per-image count bias on the held-out split for the operating point to count
    # as validated (a measurement decision — how unbiased the count must be to be trustworthy).
    count_bias_tolerance: float = 1.0
    notes: str = ""


class TraitUnknownError(KeyError):
    """Raised for an unregistered trait — lists the available traits, like validate_model_spec."""


# --- Registry ---------------------------------------------------------------
# Hazelnut catkin bloom (Phase 1). Semantics confirmed with the domain expert 2026-07-10:
#   1. elongated = texture (salt-and-peppery frills/gills), NOT geometry (length:width too variable);
#      the detector's 2-class output (class 1 = elongated) is a learned texture classification.
#   2. "a hit" = center within ~half the class's average size (IoU is noise at ~40px).
#   3. phenotype = count-unbiased on elongated and total counts -> the elongated fraction.
#   4. milestones 0.05/0.50/0.95 on the elongated fraction.
#   5. tile-seam slivers dropped below a class-average-size threshold.
CATKIN = TraitSpec(
    name="catkin",
    task="detection",
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    localization_tolerance="half_class_avg_size",
    positive_class_id=1,
    positive_is_texture=True,
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    sliver_policy="class_avg_size",
    notes="Bloom = fraction of a plant's catkins that are elongated. Elongated is a texture call "
          "(frilled/salt-and-peppery), never a bbox-ratio proxy.",
)

_TRAITS: dict[str, TraitSpec] = {t.name: t for t in (CATKIN,)}


def get_trait(name: str) -> TraitSpec:
    """Return the ``TraitSpec`` for ``name``, or raise ``TraitUnknownError`` listing the registered traits."""
    spec = _TRAITS.get(name)
    if spec is None:
        raise TraitUnknownError(f"Unknown trait {name!r}. Registered traits: {sorted(_TRAITS)}")
    return spec


def registered_traits() -> list[str]:
    return sorted(_TRAITS)
