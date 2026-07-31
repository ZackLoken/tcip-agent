"""Trait knowledge — the human-defined *semantics* of each measurable trait (Tier C).

Most fields here are things the domain expert defines once per trait and the agent *reads* — never
derives, never re-asks per dataset (CLAUDE.md: the human defines a trait's intent/semantics; the
agent derives the operating points that realize it). Keeping them in one place, versioned with the
code, stops a measurement definition from living only in a session's memory.

Two fields are a different shape, by design (K18 B3/B4) — neither is authored blind, and neither
has a default: ``localization`` (what "finding one" means) is derived once from real GT the first
time it's needed and recorded (a genuine geometric fact about the object's scale, computable from
data — see ``pipelines.derivations.derive_localization_kind``); ``count_objective`` (what the
phenotype *is*, hence what the operating point optimizes) is decided once from a real, plain-language
answer the breeder gives about what the delivered number needs to be reliable for, and recorded the
same way. Both get written through ``write_trait_spec_fields``, with a ``provenance`` entry naming
who decided and how, and both are read from the recorded value on every later call — never silently
defaulted, never copied from another trait's values (the exact failure this replaced: an earlier
design let both fields default to catkin's own historical values, silently inherited by any trait
whose config omitted them).

Everything else in ``TraitSpec`` says: which class in ``classes.json`` is the positive/target state,
the milestone convention, and the tile-seam sliver policy — genuinely authored-once breeder facts.
Operating-point *values* (conf, IoU, tolerances) and the CV task / pipeline decomposition (detection
vs classification, one model vs detect-then-classify) are deliberately absent from this whole class —
those the agent derives and validates per dataset at runtime, the same way the values are.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

# Count objectives — what the resolved operating point optimizes. NOT a closed enum (K18 B4):
# these three names are today's real, implemented picker capabilities
# (``operating_point.COUNT_OBJECTIVE_PICKERS``), not the only objectives a trait may ever declare.
# The agent can write and register a new named picker for a trait whose breeder-stated need these
# three don't cover — a capability the platform can grow, not a category TraitSpec closes over.
COUNT_UNBIASED = "count_unbiased"  # minimize signed per-image count bias E[FP-FN]; the phenotype is a count
DETECTION_F1 = "detection_f1"      # optimize matching quality; the phenotype is presence/localization
PRESENCE = "presence"             # only whether the object is present

# The currently-implemented objective names — lives here (torch-free) rather than in
# ``operating_point.py`` (which imports the torch-heavy ``pipelines.training.evaluation`` at module
# level) purely so referencing these three names never drags torch into
# ``get_trait``/``registered_traits``. ``operating_point.py``'s picker/label registry
# (``COUNT_OBJECTIVE_PICKERS``) shares these same keys rather than maintaining a second list. NOT a
# validation whitelist — ``_spec_from_config`` no longer rejects a ``count_objective`` outside this
# set; a trait may name any objective an agent has implemented and registered a picker for.
COUNT_OBJECTIVES = {COUNT_UNBIASED, DETECTION_F1, PRESENCE}

# Localization — what counts as "finding" an object.
CENTER_MATCH = "center_match"  # predicted center within a derived tolerance of a GT center
IOU_MATCH = "iou_match"        # IoU >= a derived/def threshold


@dataclass(frozen=True)
class TraitSpec:
    """The semantics of one trait. Most fields are read, never derived; ``count_objective`` and
    ``localization`` are the two exceptions — see the module docstring."""

    name: str
    # What the delivered phenotype needs to be reliable FOR — hence what the operating point
    # optimizes. NOT authored blind: a consequence judgment only a human stakeholder can make (does
    # this number need every object found correctly, or is it fine if errors cancel out as long as
    # the total is right?), asked in plain domain terms, never CV vocabulary. Empty = not yet
    # decided — never silently defaulted or copied from another trait's value (K18 B4: this field
    # used to default to "count_unbiased" and get silently inherited by any trait whose config
    # omitted it). Record the real answer via ``write_trait_spec_fields``, with a ``provenance``
    # entry naming who decided and why; ``resolve_operating_point`` refuses rather than guessing
    # when this is unset.
    count_objective: str = ""
    # What "a hit" means when validating counts (center_match vs iou_match) — NOT authored: derived
    # once from real GT the first time it's needed and recorded via
    # ``write_trait_spec_fields``/``pipelines.derivations.derive_localization_kind``, then read from
    # here on every later call (K18 B3). Empty = not yet derived — never silently assumed to be
    # either kind; ``resolve_match_criterion`` is what fills this in.
    localization: str = ""
    # How the localization tolerance is derived (the recipe string names it; ``localization_tolerance_frac``
    # is the fallback multiplier when a caller has no GT to derive one from — the real per-dataset
    # value comes from ``derivations.derive_localization_tolerance_frac`` at runtime).
    localization_tolerance: str = "half_class_avg_size"
    localization_tolerance_frac: float = 0.5  # fallback only — see derive_localization_tolerance_frac
    # The class the elongated/positive call resolves to in classes.json, by NAME (the id is a mapping
    # FACT derived from the labels, not a pinned magic number). Empty = the trait has no positive class.
    positive_class_name: str = ""
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
    # count_objective is NOT validated against a closed vocabulary (K18 B4) — a trait may name any
    # objective an agent has implemented and registered a picker for in
    # operating_point.COUNT_OBJECTIVE_PICKERS. resolve_operating_point refuses at resolution time
    # if the name has no registered picker, which is the honest place for that check to live (it
    # needs the picker registry; this module stays torch-free and doesn't import it).
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


def write_trait_spec_fields(
    trait_name: str, fields_: dict, provenance_entries: list[str] | tuple[str, ...],
    specs_dir: Path | None = None,
) -> TraitSpec:
    """Update one or more fields on an ALREADY-REGISTERED trait spec, appending provenance
    entries recording who asserted the change and how firmly (K18 B2.5).

    Refuses (raises ``ValueError``) if the trait has no existing spec file — creating a new trait
    is a separate, still-manual authoring step, out of scope here. Re-validates the merged spec
    through ``_spec_from_config`` — the same crops.yml cross-check and field validation every
    config-authored spec already goes through, reused rather than a second implementation — and
    refuses to write anything that would silently fail to load or fall out of
    ``registered_traits()`` afterward. Writes atomically.

    This is the only write path for a trait spec anywhere in the platform. Before this, a spec
    was authored by hand-writing YAML directly, with no ``@audited`` record — this function is
    what the ``update_trait_spec_fields`` MCP tool calls, and what B3's derived localization kind
    and B4's recorded count-objective decision both use to persist themselves; neither gets its
    own write implementation.
    """
    from tcip_mcp.project_paths import resolve_state
    from tcip_mcp.utils.atomic_io import atomic_write_text

    import yaml

    directory = specs_dir or resolve_state(_TRAIT_SPECS_RELPATH)
    path = next(
        (p for p in (directory / f"{trait_name}.yml", directory / f"{trait_name}.yaml") if p.is_file()),
        None,
    )
    if path is None:
        raise ValueError(
            f"no existing trait spec file for {trait_name!r} under {directory} — "
            "write_trait_spec_fields only updates an already-registered trait; author the "
            "initial spec file first."
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a valid trait spec (not a mapping)")

    merged = dict(data)
    merged.update(fields_)
    merged["provenance"] = tuple(data.get("provenance") or ()) + tuple(provenance_entries)

    spec = _spec_from_config(merged, _crops_vocab())
    if spec is None:
        raise ValueError(
            f"update to trait spec {trait_name!r} would produce an invalid spec (unknown field, "
            "off-vocab delivers, or an invalid value) — refusing to write; see the logged warning "
            "above for which check failed."
        )

    write_data = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(spec).items()
    }
    atomic_write_text(path, yaml.safe_dump(write_data), encoding="utf-8")
    return spec


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
