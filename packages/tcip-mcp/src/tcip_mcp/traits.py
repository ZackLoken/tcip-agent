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

# The single source of truth for which objective names exist — lives here (torch-free) rather than
# in ``operating_point.py`` (which imports the torch-heavy ``pipelines.training.evaluation`` at
# module level) so validating a config-authored ``count_objective`` never drags torch into
# ``get_trait``/``registered_traits``. ``operating_point.py``'s picker/label registry
# (``COUNT_OBJECTIVE_PICKERS``) shares these same keys rather than maintaining a second list.
COUNT_OBJECTIVES = {COUNT_UNBIASED, DETECTION_F1, PRESENCE}

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
    # is the fallback multiplier when a caller has no GT to derive one from — the real per-dataset
    # value comes from ``derivations.derive_localization_tolerance_frac`` at runtime).
    localization_tolerance: str = "half_class_avg_size"
    localization_tolerance_frac: float = 0.5  # fallback only — see derive_localization_tolerance_frac
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
    # Phenology CSV column vocabulary — the milestone-column prefix and the label the majority
    # alias/provisional columns carry, so the delivered schema derives its names from this spec rather
    # than from literals in the phenology module (catkin -> catkin_elongation_date / catkin_05per_date).
    phenology_prefix: str = ""
    majority_label: str = ""
    # How the tile-seam sliver cutoff is derived (the policy string names the basis). Partial objects
    # count unless below ``sliver_frac * class_avg_size``. Not read by ``TiledDetectionDataset``
    # directly — it derives its own default from the dataset's own size spread
    # (``derivations.derive_sliver_frac``) unless the caller passes an explicit override; this field
    # is that optional override, not a value the platform applies for you.
    sliver_policy: str = "class_avg_size"
    sliver_frac: float = 0.5
    # Max acceptable mean per-image count bias on the held-out split for the operating point to count
    # as validated (a measurement decision — how unbiased the count must be to be trustworthy).
    # ABSOLUTE count/image, breeder-set per trait (D12) — never scaled to a "typical" count, never derived.
    count_bias_tolerance: float = 1.0
    # Max acceptable p90 |per-image count error| (a TAIL statistic, not a mean — a population mean
    # can hide one badly-off image among many) on the held-out split. NO DEFAULT: an invented number
    # here would be platform-picked measurement semantics masquerading as a domain-expert one. `None`
    # means "not yet authored for this trait" and the dispersion term is skipped, not gated on a
    # guessed value — it needs the domain expert (or a derivation from real dense-imagery detector
    # statistics), not a value picked by the agent. PROVISIONAL until authored.
    count_error_tolerance: float | None = None
    # Min acceptable Cohen's kappa (chance-corrected classifier/GT agreement) on the held-out split
    # for K3's classifier operating point to count as validated — catches a compensating-error
    # classifier (flips k positives to negative and k negatives to positive, net count-bias ~0) a
    # bare count-bias check can't see. How much agreement is "enough" for a trait's own phenotype is
    # measurement semantics, the same shape as `count_error_tolerance` above: `None` means "not yet
    # authored for this trait" (stage-6 review Finding B) — it needs the domain expert, not a value
    # picked by the agent. Unlike `count_error_tolerance`'s dispersion
    # term, an unauthored floor here does NOT skip the check: `operating_point.py`'s
    # `_PROVISIONAL_KAPPA_FLOOR` (0.41, platform-chosen, not domain-authored) applies as the real
    # operative floor until a trait sets its own — the gate is never satisfied by the bare
    # mathematical minimum `kappa > 0` alone once the platform default is in effect.
    classifier_agreement_floor: float | None = None
    # crops.yml controlled-vocab trait names this spec is authored to deliver — the anti-fabrication
    # anchor a config-loaded spec is cross-checked against (a spec can't claim a phenotype not in the vocab).
    delivers: tuple[str, ...] = ()
    notes: str = ""
    # Per-field provenance: who actually asserted each semantic choice above, and how firmly — so a
    # spec's own history is legible instead of living only in a session's memory or, worse, a fabricated
    # comment (2026-07-29: a prior session's "confirmed with the domain expert" claim on this trait was
    # exactly that, and got removed). Each entry is ``"<field>: <kind> — <note>"``; ``<kind>`` is one of
    # domain_expert_confirmed / zack_methodology_correction / claude_proposed_unvalidated /
    # claude_recommended_unconfirmed / vocabulary_derived / data_derived_at_runtime. Not a validation
    # gate — nothing reads this to decide anything; it exists so the next reader (human or agent) does
    # not have to guess, or worse, invent, why a field holds the value it does.
    provenance: tuple[str, ...] = ()


class TraitUnknownError(KeyError):
    """Raised for an unregistered trait — lists the available traits (the honest no-fabrication signal)."""


# --- Config-driven authoring ------------------------------------------------
# There are no built-in traits (2026-07-29: catkin's hardcoded TraitSpec — including a prior
# session's fabricated "confirmed with the domain expert" comment on it — is removed; every trait,
# catkin included, is authored the same way, as a per-project spec file). This is the only
# registration path, so it can never be silently outranked by trusted Python the way a builtin
# used to outrank config on name collision. Cross-checked against the crops.yml controlled
# vocabulary, so an agent cannot fabricate a trait definition. Resolution is per-call (not a
# module-load snapshot) so a repin of the project root is picked up.

_TRAIT_SPECS_RELPATH = Path(".tcip") / "state" / "trait_specs"
_SPEC_FIELDS = {f.name for f in fields(TraitSpec)}
_TUPLE_FIELDS = {"milestone_fractions", "delivers", "provenance"}


def _crops_traits() -> list[dict]:
    """The raw crops.yml trait records, or [] if it can't be read — the one YAML load every
    crops.yml-derived reader (vocab, units) shares, never re-parsed per reader."""
    from tcip_mcp.project_paths import repo_root_from_here

    crops_yml = repo_root_from_here() / ".github" / "skills" / "crops" / "crops.yml"
    try:
        import yaml

        data = yaml.safe_load(crops_yml.read_text(encoding="utf-8"))
        return [t for t in data.get("traits", []) if isinstance(t, dict) and "name" in t]
    except (OSError, ValueError, KeyError, ImportError):
        return []


def _crops_vocab() -> set[str]:
    """The crops.yml controlled-vocab trait names, or an empty set if it can't be read (fail-closed:
    a config spec that can't be cross-checked is not registered)."""
    return {t["name"] for t in _crops_traits()}


def crops_units() -> dict[str, str]:
    """trait name -> crops.yml's declared physical unit (``mm``/``g``/``kg``/``m``/``cm``/…), for
    every trait that declares one. crops.yml is THE trait-unit authority (CLAUDE.md); a count/
    ordinal trait with no physical unit is simply absent from this mapping, never guessed."""
    return {t["name"]: t["units"] for t in _crops_traits() if isinstance(t.get("units"), str)}


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
    if "count_objective" in data:
        # COUNT_OBJECTIVES (this module) is the single source of truth for which objectives exist —
        # operating_point.py's picker/label registry shares these same keys, so validating here never
        # needs to import that torch-heavy module. A value outside this set would otherwise silently
        # fall into operating_point.py's permissive else-branch at resolution time instead of failing
        # here.
        objective = data["count_objective"]
        if objective not in COUNT_OBJECTIVES:
            logger.warning("trait spec %r skipped: count_objective %r is not one of %s",
                           name, objective, sorted(COUNT_OBJECTIVES))
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
    """The live registry: every config-authored spec found under this project's trait_specs dir."""
    return {spec.name: spec for spec in load_trait_specs()}


def get_trait(name: str) -> TraitSpec:
    """Return the ``TraitSpec`` for ``name``, or raise ``TraitUnknownError`` listing the registered traits."""
    traits = _all_traits()
    spec = traits.get(name)
    if spec is None:
        raise TraitUnknownError(f"Unknown trait {name!r}. Registered traits: {sorted(traits)}")
    return spec


def registered_traits() -> list[str]:
    return sorted(_all_traits())
