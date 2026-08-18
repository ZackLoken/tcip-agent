"""Trait knowledge, the human-defined *semantics* of each measurable trait (Tier C).

Most fields here are things the domain expert defines once per trait and the agent *reads*, never
derives, never re-asks per dataset (CLAUDE.md: the human defines a trait's intent/semantics; the
agent derives the operating points that realize it). Keeping them in one place, versioned with the
code, stops a measurement definition from living only in a session's memory.

Two fields are a different shape, by design, neither is authored blind: ``localization`` (what
"finding one" means) has no default at all, it is derived once from real GT the first time it's
needed and recorded (a genuine geometric fact about the object's scale, computable from data, see
``pipelines.derivations.derive_localization_kind``). ``count_objective`` (what the phenotype *is*,
hence what the operating point optimizes) *does* have a platform default (``COUNT_UNBIASED``,
errors canceling is the right tolerance for a fraction/ratio phenotype, the common case), used when
the breeder hasn't decided yet, rather than refusing to calibrate at all: nobody can meaningfully
answer what a delivered number needs to be reliable for before any result exists to judge it
against, so the real confirmation point is the delivered result itself (the review-confirmation
loop), not a blind precondition. Either field, once a real answer is recorded, gets written through
``write_trait_spec_fields`` with a ``provenance`` entry naming who decided and how, and is read from
the recorded value on every later call. A trait whose config omits either field must resolve its own
default or derivation, never silently inherit another trait's value. ``resolve_operating_point``
stamps whether a given run's ``count_objective`` was trait-authored or the platform default, so the
distinction is never silently lost downstream.

Everything else in ``TraitSpec`` says: which class in ``classes.json`` is the positive/target state,
the milestone convention, and the tile-seam sliver policy, genuinely authored-once breeder facts.
Operating-point *values* (conf, IoU, tolerances) and the CV task / pipeline decomposition (detection
vs classification, one model vs detect-then-classify) are deliberately absent from this whole class,
those the agent derives and validates per dataset at runtime, the same way the values are.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import tcip_store as ts
from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    NotFound,
    StoreDescriptor,
    VersionConflict,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

# Count objectives, what the resolved operating point optimizes. Not a closed enum:
# these three names are today's real, implemented picker capabilities
# (``operating_point.COUNT_OBJECTIVE_PICKERS``), not the only objectives a trait may ever declare.
# The agent can write and register a new named picker for a trait whose breeder-stated need these
# three don't cover, a capability the platform can grow, not a category TraitSpec closes over.
COUNT_UNBIASED = "count_unbiased"  # minimize signed per-image count bias E[FP-FN]; the phenotype is a count
DETECTION_F1 = "detection_f1"      # optimize matching quality; the phenotype is presence/localization
PRESENCE = "presence"             # only whether the object is present

# The currently-implemented objective names, lives here (torch-free) rather than in
# ``operating_point.py`` (which imports the torch-heavy ``pipelines.training.evaluation`` at module
# level) purely so referencing these three names never drags torch into
# ``get_trait``/``registered_traits``. ``operating_point.py``'s picker/label registry
# (``COUNT_OBJECTIVE_PICKERS``) shares these same keys rather than maintaining a second list. Not a
# validation whitelist, ``_spec_from_config`` does not reject a ``count_objective`` outside this
# set; a trait may name any objective an agent has implemented and registered a picker for.
COUNT_OBJECTIVES = {COUNT_UNBIASED, DETECTION_F1, PRESENCE}

# Localization, what counts as "finding" an object.
CENTER_MATCH = "center_match"  # predicted center within a derived tolerance of a GT center
IOU_MATCH = "iou_match"        # IoU >= a derived/def threshold


@dataclass(frozen=True)
class TraitSpec:
    """The semantics of one trait. Most fields are read, never derived; ``count_objective`` and
    ``localization`` are the two exceptions, see the module docstring."""

    name: str
    # What the delivered phenotype needs to be reliable for, hence what the operating point
    # optimizes. Not authored blind: a consequence judgment only a human stakeholder can make (does
    # this number need every object found correctly, or is it fine if errors cancel out as long as
    # the total is right?), asked in plain domain terms, never CV vocabulary, and never silently
    # copied from another trait's value. Empty = not yet decided; resolve_operating_point defaults
    # to COUNT_UNBIASED (the common case for a fraction/ratio phenotype) rather than refusing to
    # calibrate, since nobody can meaningfully answer this before a result exists to judge it
    # against, stamping the run's provenance as trait-authored or platform-default so the
    # distinction is never lost downstream. Record a real breeder answer via
    # ``write_trait_spec_fields``, with a ``provenance`` entry naming who decided and why, once one
    # exists.
    count_objective: str = ""
    # What "a hit" means when validating counts (center_match vs iou_match), not authored: derived
    # once from real GT the first time it's needed and recorded via
    # ``write_trait_spec_fields``/``pipelines.derivations.derive_localization_kind``, then read from
    # here on every later call. Empty = not yet derived, never silently assumed to be
    # either kind; ``resolve_match_criterion`` is what fills this in.
    localization: str = ""
    # How the localization tolerance is derived (the recipe string names it; ``localization_tolerance_frac``
    # is the fallback multiplier when a caller has no GT to derive one from, the real per-dataset
    # value comes from ``derivations.derive_localization_tolerance_frac`` at runtime).
    localization_tolerance: str = "half_class_avg_size"
    localization_tolerance_frac: float = 0.5  # fallback only, see derive_localization_tolerance_frac
    # The class the positive call resolves to in classes.json, by name (the id is a mapping
    # fact derived from the labels, not a pinned magic number). Empty = the trait has no positive class.
    positive_class_name: str = ""
    # Milestone crossing fractions and the quantity they cross.
    milestone_fractions: tuple[float, ...] = ()
    milestone_on: str = ""  # e.g. "positive_fraction"
    # The "majority" milestone (a crops.yml date such as "most pistillate flowers have opened") maps
    # to this crossing key (e.g. "95per"), flagged provisional until the breeders confirm the
    # reading. Read-semantics, not a frozen literal buried in the phenology code.
    majority_milestone: str = ""
    majority_provisional: bool = False
    # Phenology CSV column vocabulary, the milestone-column prefix and the label the majority
    # alias/provisional columns carry, so the delivered schema derives its names from this spec rather
    # than from literals in the phenology module (<trait> -> <trait>_<milestone>_date / <trait>_<NN>per_date).
    phenology_prefix: str = ""
    majority_label: str = ""
    # How the tile-seam sliver cutoff is derived (the policy string names the basis). Partial objects
    # count unless below ``sliver_frac * class_avg_size``. Not read by ``TiledDetectionDataset``
    # directly, it derives its own default from the dataset's own size spread
    # (``derivations.derive_sliver_frac``) unless the caller passes an explicit override; this field
    # is that optional override, not a value the platform applies for you.
    sliver_policy: str = "class_avg_size"
    sliver_frac: float = 0.5
    # Max acceptable mean per-image count bias on the held-out split, relative to the class's (or,
    # for the pooled gate, the whole reference's) own typical per-image count, for the operating
    # point to count as validated (a measurement decision, how much relative count error is
    # trustworthy). A fraction (e.g. 0.1 == 10% relative error). How much relative count error is
    # "enough" for a trait's own phenotype is measurement semantics, the same shape as
    # `count_error_tolerance`/`classifier_agreement_floor` above: `None` means "not yet authored for
    # this trait", it needs the domain expert, not a value picked by the agent. Unlike
    # `count_error_tolerance`'s dispersion term, an unauthored fraction here does not skip the check:
    # `operating_point.py`'s `_PROVISIONAL_COUNT_BIAS_TOLERANCE_FRAC` (0.01, platform-chosen, not
    # domain-authored) applies as the real operative fraction until a trait sets its own. Applied
    # identically wherever `operating_point._bias_equivalence_ok` is called, the pooled and
    # per-class detector gates and the classifier path's positive-class gate, one field, one unit,
    # everywhere it is read (deliberately not two different units at two call sites). The platform
    # derives what the fraction is relative to (each scope's own typical per-image count, from the
    # same holdout reference the equivalence test itself measures, never calibration, which would let
    # a caller buy a looser holdout tolerance by padding calibration's own density) at runtime, never
    # invented and never breeder-guessed. A near-zero-typical-count scope is protected by a floor that
    # is itself derived (`1 / n`, the same evidence count the equivalence test's own standard error
    # already uses, see `operating_point._effective_count_bias_tolerance`), not by a second authored
    # or platform-invented number. That floor can raise the effective tolerance above what the
    # fraction term alone would give (it is a `max()`), it is bounded, never a runaway number: at
    # n >= 2 (the reference-sufficiency minimum every scope using this floor is independently gated
    # on, see `insufficient_holdout_images`/`insufficient_holdout_images_per_class`) the floor
    # itself never exceeds 0.5. What a given calibration was actually held to (both terms combined) is
    # read from the per-scope `pooled_count_bias_tolerance`/`per_class_count_bias_tolerance` in a
    # run's own sweep record, rather than inferred from this fraction alone.
    count_bias_tolerance_frac: float | None = None
    # Max acceptable p90 |per-image count error| (a tail statistic, not a mean, a population mean
    # can hide one badly-off image among many) on the held-out split. No default: an invented number
    # here would be platform-picked measurement semantics masquerading as a domain-expert one. `None`
    # means "not yet authored for this trait" and the dispersion term is skipped, not gated on a
    # guessed value, it needs the domain expert (or a derivation from real dense-imagery detector
    # statistics), not a value picked by the agent. Provisional until authored.
    count_error_tolerance: float | None = None
    # Min acceptable Cohen's kappa (chance-corrected classifier/GT agreement) on the held-out split
    # for the classifier operating point to count as validated, catches a compensating-error
    # classifier (flips k positives to negative and k negatives to positive, net count-bias ~0) a
    # bare count-bias check can't see. How much agreement is "enough" for a trait's own phenotype is
    # measurement semantics, the same shape as `count_error_tolerance` above: `None` means "not yet
    # authored for this trait", it needs the domain expert, not a value picked by the agent. Unlike
    # `count_error_tolerance`'s dispersion term, an unauthored floor here does not skip the check:
    # `operating_point.py`'s `_PROVISIONAL_KAPPA_FLOOR` (0.41, platform-chosen, not domain-authored)
    # applies as the real operative floor until a trait sets its own, the gate is never satisfied by
    # the bare mathematical minimum `kappa > 0` alone once the platform default is in effect.
    classifier_agreement_floor: float | None = None
    # Min acceptable value for whichever ordinal compensating-error criterion calibration actually
    # used (today only `quadratic_weighted_kappa` is registered in
    # `operating_point.ORDINAL_CRITERIA`, but the field name does not bake that in, since the
    # toolkit may grow further criteria). Same shape as `classifier_agreement_floor` above: `None`
    # means "not yet authored for this trait", it needs the domain expert, not a value picked by the
    # agent. Unlike `count_error_tolerance`'s dispersion term, an unauthored floor here does not skip
    # the check: `operating_point.py`'s `_PROVISIONAL_ORDINAL_AGREEMENT_FLOOR` applies as the real
    # operative floor until a trait sets its own.
    ordinal_agreement_floor: float | None = None
    # Min acceptable value for whichever regression skill/agreement criterion calibration actually
    # used (`r_squared` or `concordance_correlation_coefficient`, `operating_point.
    # REGRESSION_CRITERIA`), the regression counterpart to `ordinal_agreement_floor` above. `None`
    # means "not yet authored for this trait"; `operating_point.py`'s
    # `_PROVISIONAL_REGRESSION_SKILL_FLOOR` applies until a trait sets its own. Unlike
    # `classifier_agreement_floor`'s single criterion (kappa), this platform offers more than one
    # regression criterion with genuinely different scales/conventions (R² is unbounded below and
    # measures skill relative to a trivial mean baseline; CCC is bounded in [-1, 1] and decomposes
    # precision from bias), so a floor authored here is only meaningful paired with the criterion it
    # was set against, whoever authors this value must decide the floor and the criterion together,
    # an honest, documented limitation of a single scalar field, not a design this platform resolves
    # further here.
    regression_skill_floor: float | None = None
    # crops.yml controlled-vocab trait names this spec is authored to deliver, the anti-fabrication
    # anchor a config-loaded spec is cross-checked against (a spec can't claim a phenotype not in the vocab).
    delivers: tuple[str, ...] = ()
    notes: str = ""
    # Per-field provenance: who actually asserted each semantic choice above, and how firmly, so a
    # spec's own history is legible instead of living only in a session's memory or, worse, a fabricated
    # comment. Each entry is ``"<field>: <kind>, <note>"``; ``<kind>`` is one of
    # domain_expert_confirmed / domain_expert_correction / agent_proposed_unvalidated /
    # agent_recommended_unconfirmed / vocabulary_derived / data_derived_at_runtime. Not a validation
    # gate, nothing reads this to decide anything; it exists so the next reader (human or agent) does
    # not have to guess, or worse, invent, why a field holds the value it does.
    provenance: tuple[str, ...] = ()


class TraitUnknownError(KeyError):
    """Raised for an unregistered trait, lists the available traits (the honest no-fabrication signal)."""


# --- Config-driven authoring ------------------------------------------------
# There are no built-in traits; every trait is authored the same way, as a
# per-project spec file. This is the only registration path. Cross-checked against the crops.yml
# controlled vocabulary, so an agent cannot fabricate a trait definition. Resolution is per-call
# (not a module-load snapshot) so a repin of the project root is picked up.

_TRAIT_SPECS_RELPATH = Path(".tcip") / "state" / "trait_specs"
_SPEC_FIELDS = {f.name for f in fields(TraitSpec)}
_TUPLE_FIELDS = {"milestone_fractions", "delivers", "provenance"}

SPEC_SUFFIX = ".json"
"""The canonical spec-file suffix, and the only one the store's locator addresses."""


def crops_yml_path() -> Path:
    """Where the crops.yml controlled vocabulary lives, stated once for every reader of it."""
    from tcip_mcp.project_paths import repo_root_from_here

    return repo_root_from_here() / ".github" / "skills" / "crops" / "crops.yml"


def _crops_traits() -> list[dict]:
    """The raw crops.yml trait records, or [] if it can't be read, the one YAML load every
    crops.yml-derived reader (vocab, units, the skill guardrail) shares, never re-parsed per
    reader. A caller that cannot act on an empty vocabulary checks for one and refuses."""
    try:
        import yaml

        data = yaml.safe_load(crops_yml_path().read_text(encoding="utf-8"))
        return [t for t in data.get("traits", []) if isinstance(t, dict) and "name" in t]
    except (OSError, ValueError, KeyError, ImportError):
        return []


def _crops_vocab() -> set[str]:
    """The crops.yml controlled-vocab trait names, or an empty set if it can't be read (fail-closed:
    a config spec that can't be cross-checked is not registered)."""
    return {t["name"] for t in _crops_traits()}


def crops_units() -> dict[str, str]:
    """trait name -> crops.yml's declared physical unit (``mm``/``g``/``kg``/``m``/``cm``/…), for
    every trait that declares one. crops.yml is the trait-unit authority; a count/
    ordinal trait with no physical unit is simply absent from this mapping, never guessed."""
    return {t["name"]: t["units"] for t in _crops_traits() if isinstance(t.get("units"), str)}


def crops_definitions() -> dict[str, str]:
    """trait name -> crops.yml's declared definition, for every trait that carries one.

    The breeder's own wording for what a delivered phenotype is. A surface that has to show a
    breeder what a number means quotes this rather than paraphrasing it, so the vocabulary the
    breeder reads is the vocabulary the vocabulary file holds.
    """
    return {t["name"]: t["definition"] for t in _crops_traits() if isinstance(t.get("definition"), str)}


def _spec_from_config(data: dict, vocab: set[str]) -> tuple[TraitSpec | None, str | None]:
    """Build a ``TraitSpec`` from one breeder-authored config dict, cross-checked against ``vocab``.

    Rejects (returns ``(None, reason)``) a spec with no ``name``, an unknown field, or a
    ``delivers`` that is empty or names a phenotype absent from crops.yml, so a config file can
    never introduce a fabricated trait definition. Registering a real new trait means its delivered
    outputs are all in the controlled vocabulary. ``reason`` is the same text logged as a warning,
    the one place that wording is authored, so a caller surfacing it (an API response, a
    write_trait_spec_fields refusal, doctor.py) never re-derives its own explanation.
    """
    name = data.get("name")
    if not isinstance(name, str) or not name:
        reason = f"missing/invalid 'name' ({name!r})"
        logger.warning("trait spec skipped: %s", reason)
        return None, reason
    unknown = set(data) - _SPEC_FIELDS
    if unknown:
        reason = f"unknown field(s) {sorted(unknown)}"
        logger.warning("trait spec %r skipped: %s", name, reason)
        return None, reason
    delivers = data.get("delivers") or []
    off_vocab = [d for d in delivers if d not in vocab]
    if not delivers or off_vocab:
        reason = f"delivers must be non-empty and all in crops.yml (off-vocab: {off_vocab})"
        logger.warning("trait spec %r skipped: %s", name, reason)
        return None, reason
    # count_objective is not validated against a closed vocabulary, a trait may name any
    # objective an agent has implemented and registered a picker for in
    # operating_point.COUNT_OBJECTIVE_PICKERS. resolve_operating_point refuses at resolution time
    # if the name has no registered picker, which is the honest place for that check to live (it
    # needs the picker registry; this module stays torch-free and doesn't import it).
    kwargs = {k: (tuple(v) if k in _TUPLE_FIELDS else v) for k, v in data.items()}
    try:
        return TraitSpec(**kwargs), None
    except TypeError as e:
        logger.warning("trait spec %r skipped: %s", name, e)
        return None, str(e)


# ── the trait-spec store ─────────────────────────────────────────────────────


def trait_specs_dir(project_root: str | Path | None = None) -> Path:
    """Where a project's trait specs live: ``<root>/.tcip/state/trait_specs``.

    The one implementation of that placement. ``project_root`` names the project explicitly, for a
    caller (the web backend, ``scripts/doctor.py``) that serves more than one project per process;
    omitting it resolves against this process's pinned platform root.
    """
    if project_root is not None:
        return Path(project_root) / _TRAIT_SPECS_RELPATH
    from tcip_mcp.project_paths import resolve_state

    return resolve_state(_TRAIT_SPECS_RELPATH)


def _resolve_specs_dir(specs_dir: Path | None, project_root: str | Path | None) -> Path:
    if specs_dir is not None and project_root is not None:
        raise ValueError(
            "name either the project root or the spec directory, not both: they can disagree "
            "and nothing here can tell which one the caller meant"
        )
    return Path(specs_dir) if specs_dir is not None else trait_specs_dir(project_root)


TRAIT_SPECS_STORE = "trait_specs"
_SPEC_FILE = RootedFileLocator(suffix=SPEC_SUFFIX)
register_store(
    StoreDescriptor(
        name=TRAIT_SPECS_STORE,
        kind="record",
        key_fields=("trait",),
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_SPEC_FILE,
    )
)


def trait_spec_key(specs_dir: str | Path, trait_name: str) -> Key:
    """One trait's spec record under a spec directory.

    Enumeration answers over the records themselves. :func:`write_trait_spec_fields` merges
    compare-and-set against the version it read, so a field another writer recorded is never
    dropped.
    """
    return Key(TRAIT_SPECS_STORE, str(specs_dir), (trait_name,))


def _spec_filename(specs_dir: Path, trait_name: str) -> str:
    """The spec's file name, taken from the store's own locator rather than spelled again."""
    return _SPEC_FILE.relative_path(str(specs_dir), (trait_name,)).name


def load_trait_specs_with_errors(
    specs_dir: Path | None = None, *, project_root: str | Path | None = None,
) -> tuple[list[TraitSpec], list[dict]]:
    """Same scan as :func:`load_trait_specs`, plus the file/reason for every spec skipped.

    This is the one place that detail exists; a caller that needs to tell a breeder or the agent
    which spec is broken and why (the Results API, ``scripts/doctor.py``) reads it from here
    rather than re-deriving its own explanation or grepping logs. Either name the project whose
    registry to read (``project_root``) or the directory itself; the placement is resolved here.
    """
    directory = _resolve_specs_dir(specs_dir, project_root)
    if not directory.is_dir():
        return [], []
    errors: list[dict] = []
    vocab = _crops_vocab()
    specs: list[TraitSpec] = []
    for key in ts.keys(TRAIT_SPECS_STORE, str(directory)):
        filename = _spec_filename(directory, key.parts[0])
        try:
            data = ts.read_versioned(key).value
        except NotFound:
            continue
        except DecodeError as e:
            logger.warning("trait spec %s skipped: %s", filename, e)
            errors.append({"file": filename, "reason": str(e)})
            continue
        if not isinstance(data, dict):
            reason = "not a mapping"
            logger.warning("trait spec %s skipped: %s", filename, reason)
            errors.append({"file": filename, "reason": reason})
            continue
        spec, reason = _spec_from_config(data, vocab)
        if spec is not None:
            specs.append(spec)
        else:
            errors.append({"file": filename, "reason": reason})
    return specs, errors


def load_trait_specs(
    specs_dir: Path | None = None, *, project_root: str | Path | None = None
) -> list[TraitSpec]:
    """Breeder-authored per-trait spec records (``<root>/.tcip/state/trait_specs/*.json``), each
    cross-checked against the crops.yml controlled vocab. A missing directory yields none; an invalid
    or fabricated spec is skipped (so ``get_trait`` later hard-fails honestly rather than serving it).
    See :func:`load_trait_specs_with_errors` for why each skipped spec was skipped."""
    specs, _errors = load_trait_specs_with_errors(specs_dir, project_root=project_root)
    return specs


def write_trait_spec_fields(
    trait_name: str, fields_: dict, provenance_entries: list[str] | tuple[str, ...],
    specs_dir: Path | None = None, *, project_root: str | Path | None = None,
) -> TraitSpec:
    """Update one or more fields on an already-registered trait spec, appending provenance
    entries recording who asserted the change and how firmly.

    Refuses (raises ``ValueError``) if the trait has no existing spec file, creating a new trait
    is a separate, still-manual authoring step, out of scope here. Re-validates the merged spec
    through ``_spec_from_config``, the same crops.yml cross-check and field validation every
    config-authored spec already goes through, reused rather than a second implementation, and
    refuses to write anything that would silently fail to load or fall out of
    ``registered_traits()`` afterward. The read, the merge and the write are one compare-and-set
    against the version read, retried on conflict against whatever landed meanwhile, so a second
    writer recording a different field cannot be overwritten by what this caller had read before
    it landed.

    This is the only write path for updating a trait spec anywhere in the platform: creating a
    new trait is a separate, still-manual authoring step, out of scope here, but once a spec
    exists, this function is what the ``update_trait_spec_fields`` MCP tool calls, and what the
    derived localization kind and the recorded count-objective decision both use to persist
    themselves; neither gets its own write implementation.
    """
    directory = _resolve_specs_dir(specs_dir, project_root)
    if not directory.is_dir():
        raise _no_spec_error(trait_name, directory)

    key = trait_spec_key(directory, trait_name)
    # A conflict means another writer committed, so the loop only repeats while the spec is
    # actually changing under it and ends when this merge is the one that lands.
    while True:
        stored = ts.read_versioned(key, default=None)
        if stored.value is None:
            raise _no_spec_error(trait_name, directory)
        data = stored.value
        if not isinstance(data, dict):
            filename = _spec_filename(directory, trait_name)
            raise ValueError(f"{directory / filename} is not a valid trait spec (not a mapping)")

        merged = dict(data)
        merged.update(fields_)
        merged["provenance"] = tuple(data.get("provenance") or ()) + tuple(provenance_entries)

        spec, reason = _spec_from_config(merged, _crops_vocab())
        if spec is None:
            raise ValueError(
                f"update to trait spec {trait_name!r} would produce an invalid spec: {reason}. "
                "Refusing to write."
            )

        written = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(spec).items()
        }
        try:
            ts.replace(key, written, expect=stored.version)
        except VersionConflict:
            continue
        return spec


def _no_spec_error(trait_name: str, directory: Path) -> ValueError:
    return ValueError(
        f"no existing trait spec file for {trait_name!r} under {directory}, "
        "write_trait_spec_fields only updates an already-registered trait; author the "
        "initial spec file first."
    )


def _all_traits() -> dict[str, TraitSpec]:
    """The live registry: every config-authored spec found under this project's trait_specs dir."""
    return {spec.name: spec for spec in load_trait_specs()}


def get_trait(name: str) -> TraitSpec:
    """Return the ``TraitSpec`` for ``name``, or raise ``TraitUnknownError`` listing the registered traits."""
    return get_trait_for(name)


def get_trait_for(name: str, project_root: str | Path | None = None) -> TraitSpec:
    """One trait's spec from an explicit project's registry, or from the pinned one.

    For a caller (the web backend, the operationalization resolver) that serves more than one
    project per process and so cannot rely on ``resolve_state``'s single ``$TCIP_PROJECT_ROOT``
    pin. ``get_trait`` is the same lookup against that pin, so both surfaces refuse an unregistered
    trait in the same words.
    """
    traits = {spec.name: spec for spec in load_trait_specs(project_root=project_root)}
    spec = traits.get(name)
    if spec is None:
        raise TraitUnknownError(f"Unknown trait {name!r}. Registered traits: {sorted(traits)}")
    return spec


def registered_traits() -> list[str]:
    return sorted(_all_traits())


def registered_traits_for(project_root: str | Path) -> list[str]:
    """Registered trait names for an explicit project root.

    For a caller (the web backend) that serves more than one project per process and so cannot
    rely on ``resolve_state``'s single ``$TCIP_PROJECT_ROOT`` pin; ``registered_traits()`` stays
    the MCP-server-side entry point for the one pinned project.
    """
    return sorted(spec.name for spec in load_trait_specs(project_root=project_root))
