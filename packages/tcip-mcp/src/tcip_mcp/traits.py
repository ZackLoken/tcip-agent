"""Trait knowledge — the human-defined *semantics* of each measurable trait (Tier C).

These are the things the domain expert defines once per trait and the agent *reads* — never derives,
never re-asks per dataset (CLAUDE.md: the human defines a trait's intent/semantics; the agent derives
the operating points that realize it). Keeping them in one place, versioned with the code, stops a
measurement definition from living only in a session's memory.

A ``TraitSpec`` says: what the phenotype *is* (count objective), what "finding one" *means*
(localization), how the elongated/dormant call is defined (texture, not geometry), the milestone
convention, and the tile-seam sliver policy. Operating-point *values* (conf, IoU, tolerances) and
the CV task / pipeline decomposition (detection vs classification, one model vs detect-then-classify)
are deliberately absent — those the agent derives and validates per dataset at runtime, the same way
the values are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

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
    # What the delivered phenotype is, and hence what the operating point optimizes.
    count_objective: str = COUNT_UNBIASED
    # What "a hit" means when validating counts. For small objects, center-match (IoU is noise).
    localization: str = CENTER_MATCH
    # How the localization tolerance is derived (the recipe string names it; ``localization_tolerance_frac``
    # is the single owner of the multiplier it names, read by the operating-point / eval surfaces).
    localization_tolerance: str = "half_class_avg_size"
    localization_tolerance_frac: float = 0.5  # the "half" in half_class_avg_size — one owner, no scattered 0.5s
    # The class the elongated/positive call resolves to in classes.json, by NAME (the id is a mapping
    # FACT derived from the labels, not a pinned magic number). Empty = the trait has no positive class.
    positive_class_name: str = ""
    # The elongated/positive call is learned from texture (frills/gills), never geometry — proxies forbidden.
    positive_is_texture: bool = False
    # Milestone crossing fractions and the quantity they cross.
    milestone_fractions: tuple[float, ...] = ()
    milestone_on: str = ""  # e.g. "positive_fraction"
    # The "majority" milestone (a crops.yml date such as "most elongated") maps to this crossing key
    # (e.g. "95per"), flagged provisional until the breeders confirm the reading. Read-semantics, not a
    # frozen literal buried in the phenology code.
    majority_milestone: str = ""
    majority_provisional: bool = False
    # How the tile-seam sliver cutoff is derived (the policy string names the basis; ``sliver_frac`` owns
    # the multiplier). Partial objects count unless below ``sliver_frac * class_avg_size``.
    sliver_policy: str = "class_avg_size"
    sliver_frac: float = 0.5
    # Max acceptable mean per-image count bias on the held-out split for the operating point to count
    # as validated (a measurement decision — how unbiased the count must be to be trustworthy).
    # ABSOLUTE count/image, breeder-set per trait (D12) — never scaled to a "typical" count, never derived.
    count_bias_tolerance: float = 1.0
    # crops.yml controlled-vocab trait names this spec is authored to deliver — the anti-fabrication
    # anchor a config-loaded spec is cross-checked against (a spec can't claim a phenotype not in the vocab).
    delivers: tuple[str, ...] = ()
    notes: str = ""


class TraitUnknownError(KeyError):
    """Raised for an unregistered trait — lists the available traits (the honest no-fabrication signal)."""


# --- Built-in specs ---------------------------------------------------------
# Hazelnut catkin bloom (Phase 1). Semantics confirmed with the domain expert 2026-07-10:
#   1. elongated = a learned texture call (salt-and-peppery frills/gills), not geometry
#      (length:width too variable); how the call is produced is the agent's pipeline choice.
#   2. "a hit" = center within ~half the class's average size (IoU is noise at ~40px).
#   3. phenotype = count-unbiased on elongated and total counts -> the elongated fraction.
#   4. milestones 0.05/0.50/0.95 on the elongated fraction.
#   5. tile-seam slivers dropped below a class-average-size threshold.
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
    majority_milestone="95per",       # crops.yml "most catkins elongated" -> the 95% majority crossing
    majority_provisional=True,        # provisional reading, pending breeder confirmation
    sliver_policy="class_avg_size",
    sliver_frac=0.5,
    delivers=("catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"),
    notes="Bloom = fraction of a plant's catkins that are elongated. Elongated is a texture call "
          "(frilled/salt-and-peppery), never a bbox-ratio proxy.",
)

_BUILTIN_TRAITS: dict[str, TraitSpec] = {t.name: t for t in (CATKIN,)}


# --- Config-driven authoring ------------------------------------------------
# The registry is built-ins UNION breeder-authored per-trait spec files, so registering trait #2 is a
# config edit rather than a code edit — but the config path is cross-checked against the crops.yml
# controlled vocabulary, so an agent cannot fabricate a trait definition. Resolution is per-call (not a
# module-load snapshot) so a repin of the project root is picked up.

_TRAIT_SPECS_RELPATH = Path(".tcip") / "state" / "trait_specs"
_SPEC_FIELDS = {f.name for f in fields(TraitSpec)}
_TUPLE_FIELDS = {"milestone_fractions", "delivers"}


def _crops_vocab() -> set[str]:
    """The crops.yml controlled-vocab trait names, or an empty set if it can't be read (fail-closed:
    a config spec that can't be cross-checked is not registered)."""
    from tcip_mcp.project_paths import repo_root_from_here

    crops_yml = repo_root_from_here() / ".github" / "skills" / "crops" / "crops.yml"
    try:
        import yaml

        data = yaml.safe_load(crops_yml.read_text(encoding="utf-8"))
        return {t["name"] for t in data.get("traits", []) if isinstance(t, dict) and "name" in t}
    except (OSError, ValueError, KeyError, ImportError):
        return set()


def _spec_from_config(data: dict, vocab: set[str]) -> TraitSpec | None:
    """Build a ``TraitSpec`` from one breeder-authored config dict, cross-checked against ``vocab``.

    Rejects (returns ``None``) a spec with no ``name``, an unknown field, or a ``delivers`` that is
    empty or names a phenotype absent from crops.yml — so a config file can never introduce a
    fabricated trait definition. Registering a real new trait means its delivered outputs are all in
    the controlled vocabulary.
    """
    name = data.get("name")
    if not isinstance(name, str) or not name:
        logger.warning("trait spec skipped: missing/invalid 'name' (%r)", name)
        return None
    unknown = set(data) - _SPEC_FIELDS
    if unknown:
        logger.warning("trait spec %r skipped: unknown field(s) %s", name, sorted(unknown))
        return None
    delivers = data.get("delivers") or []
    off_vocab = [d for d in delivers if d not in vocab]
    if not delivers or off_vocab:
        logger.warning("trait spec %r skipped: delivers must be non-empty and all in crops.yml "
                       "(off-vocab: %s)", name, off_vocab)
        return None
    kwargs = {k: (tuple(v) if k in _TUPLE_FIELDS else v) for k, v in data.items()}
    try:
        return TraitSpec(**kwargs)
    except TypeError as e:
        logger.warning("trait spec %r skipped: %s", name, e)
        return None


def load_trait_specs(specs_dir: Path | None = None) -> list[TraitSpec]:
    """Breeder-authored per-trait spec files (``<root>/.tcip/state/trait_specs/*.yml``), each
    cross-checked against the crops.yml controlled vocab. A missing directory yields none; an invalid
    or fabricated spec is skipped (so ``get_trait`` later hard-fails honestly rather than serving it)."""
    from tcip_mcp.project_paths import resolve_state

    directory = specs_dir or resolve_state(_TRAIT_SPECS_RELPATH)
    if not directory.is_dir():
        return []
    vocab = _crops_vocab()
    specs: list[TraitSpec] = []
    for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")]):
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError) as e:
            logger.warning("trait spec %s skipped: %s", path.name, e)
            continue
        if not isinstance(data, dict):
            logger.warning("trait spec %s skipped: not a mapping", path.name)
            continue
        spec = _spec_from_config(data, vocab)
        if spec is not None:
            specs.append(spec)
    return specs


def _all_traits() -> dict[str, TraitSpec]:
    """The live registry: config-authored specs, with the trusted built-ins overriding on name
    collision (a config file can never redefine a built-in trait's semantics)."""
    traits = {spec.name: spec for spec in load_trait_specs()}
    traits.update(_BUILTIN_TRAITS)
    return traits


def get_trait(name: str) -> TraitSpec:
    """Return the ``TraitSpec`` for ``name``, or raise ``TraitUnknownError`` listing the registered traits."""
    traits = _all_traits()
    spec = traits.get(name)
    if spec is None:
        raise TraitUnknownError(f"Unknown trait {name!r}. Registered traits: {sorted(traits)}")
    return spec


def registered_traits() -> list[str]:
    return sorted(_all_traits())
