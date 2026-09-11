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
loop), not a blind precondition. ``count_objective`` is an authored field: once a real answer is recorded, it moves through
``revise_trait_spec`` with a rationale, restating the trait spec's own authoring statement for
the breeder's re-confirmation. ``localization`` is carried forward, never authored, and stays on
the plain ``write_trait_spec_fields``. Both are read from the recorded value on every later call.
A trait whose config omits either field must resolve its own default or derivation, never
silently inherit another trait's value. ``resolve_operating_point``
stamps whether a given run's ``count_objective`` was trait-authored or the platform default, so the
distinction is never silently lost downstream.

Everything else in ``TraitSpec`` says: which subject in ``subjects.json`` is the positive/target state,
the milestone convention, and the tile-seam sliver policy, genuinely authored-once breeder facts.
Operating-point *values* (conf, IoU, tolerances) and the CV task / pipeline decomposition (detection
vs classification, one model vs detect-then-classify) are deliberately absent from this whole class,
those the agent derives and validates per dataset at runtime, the same way the values are.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import tcip_store as ts
from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    NotFound,
    SchemaVersionRefused,
    StoreDescriptor,
    VersionConflict,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp import agent_identity
from tcip_mcp.identity import user_identity
from tcip_mcp.statements import canonical, content_hash, now_iso

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
    # distinction is never lost downstream. It is authored: record a real breeder answer via
    # revise_trait_spec, with a rationale, which restates the trait spec's own authoring statement.
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
    # The value one of the measured subject's attributes declares in subjects.json, checked
    # against that subject's own attribute values. Empty = the trait has no positive value.
    positive_value: str = ""
    # Milestone crossing fractions and the quantity they cross.
    milestone_fractions: tuple[float, ...] = ()
    milestone_on: str = ""  # e.g. "positive_fraction"
    # The "majority" milestone (a crops.yml date such as "most pistillate flowers have opened")
    # maps to this crossing key (e.g. "95per"); read-semantics, not a frozen literal in phenology.py.
    majority_milestone: str = ""
    # Marks the majority crossing mapping as not yet breeder-confirmed; the delivered column
    # spells the same fact crossing_unconfirmed while this stored key keeps its older word.
    majority_provisional: bool = False
    # Phenology CSV column vocabulary: the milestone-column prefix and the label the majority
    # alias/crossing-unconfirmed columns carry, so the delivered schema derives its own names.
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
    # `operating_point.py`'s `_DEFAULT_COUNT_BIAS_TOLERANCE_FRAC` (0.01, platform-chosen, not
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
    # run's own gate-evidence record, rather than inferred from this fraction alone.
    count_bias_tolerance_frac: float | None = None
    # Max acceptable p90 |per-image count error| (a tail statistic, not a mean, a population mean
    # can hide one badly-off image among many) on the held-out split. No default: an invented number
    # here would be platform-picked measurement semantics masquerading as a domain-expert one. `None`
    # means "not yet authored for this trait" and the dispersion term is skipped, not gated on a
    # guessed value, it needs the domain expert (or a derivation from real dense-imagery detector
    # statistics), not a value picked by the agent. Not yet authored.
    count_error_tolerance: float | None = None
    # Min acceptable Cohen's kappa (chance-corrected classifier/GT agreement) on the held-out split
    # for the classifier operating point to count as validated, catches a compensating-error
    # classifier (flips k positives to negative and k negatives to positive, net count-bias ~0) a
    # bare count-bias check can't see. How much agreement is "enough" for a trait's own phenotype is
    # measurement semantics, the same shape as `count_error_tolerance` above: `None` means "not yet
    # authored for this trait", it needs the domain expert, not a value picked by the agent. Unlike
    # `count_error_tolerance`'s dispersion term, an unauthored floor here does not skip the check:
    # `operating_point.py`'s `_DEFAULT_KAPPA_FLOOR` (0.41, platform-chosen, not domain-authored)
    # applies as the real operative floor until a trait sets its own, the gate is never satisfied by
    # the bare mathematical minimum `kappa > 0` alone once the platform default is in effect.
    classifier_agreement_floor: float | None = None
    # Min acceptable value for whichever ordinal compensating-error criterion calibration actually
    # used (today only `quadratic_weighted_kappa` is registered in
    # `operating_point.ORDINAL_CRITERIA`, but the field name does not bake that in, since the
    # toolkit may grow further criteria). Same shape as `classifier_agreement_floor` above: `None`
    # means "not yet authored for this trait", it needs the domain expert, not a value picked by the
    # agent. Unlike `count_error_tolerance`'s dispersion term, an unauthored floor here does not skip
    # the check: `operating_point.py`'s `_DEFAULT_ORDINAL_AGREEMENT_FLOOR` applies as the real
    # operative floor until a trait sets its own.
    ordinal_agreement_floor: float | None = None
    # Min acceptable value for whichever regression skill/agreement criterion calibration actually
    # used (`r_squared` or `concordance_correlation_coefficient`, `operating_point.
    # REGRESSION_CRITERIA`), the regression counterpart to `ordinal_agreement_floor` above. `None`
    # means "not yet authored for this trait"; `operating_point.py`'s
    # `_DEFAULT_REGRESSION_SKILL_FLOOR` applies until a trait sets its own. Unlike
    # `classifier_agreement_floor`'s single criterion (kappa), this platform offers more than one
    # regression criterion with genuinely different scales/conventions (R² is unbounded below and
    # measures skill relative to a trivial mean baseline; CCC is bounded in [-1, 1] and decomposes
    # precision from bias), so a floor authored here is only meaningful paired with the criterion it
    # was set against, whoever authors this value must decide the floor and the criterion together,
    # an honest, documented limitation of a single scalar field, not a design this platform resolves
    # further here.
    regression_skill_floor: float | None = None
    # Max relative disagreement a physical-scale reference half may show (scale_calibration.
    # resolve_physical_scale); None has no platform default fallback, unlike count_bias_tolerance_frac.
    scale_tolerance_frac: float | None = None
    # Min held-out precision and recall the detection gate's governing localization criterion must
    # both clear at the shipped conf. No fallback: None refuses to validate rather than substitute one.
    holdout_match_quality_floor: float | None = None
    # crops.yml controlled-vocab trait names this spec is authored to deliver, the anti-fabrication
    # anchor a config-loaded spec is cross-checked against (a spec can't claim a phenotype not in the vocab).
    delivers: tuple[str, ...] = ()
    notes: str = ""


class TraitUnknownError(KeyError):
    """Raised for an unregistered trait, lists the available traits (the honest no-fabrication signal)."""


# --- Config-driven authoring ------------------------------------------------
# There are no built-in traits; every trait is authored the same way, as a
# per-project spec file. This is the only registration path. Cross-checked against the crops.yml
# controlled vocabulary, so an agent cannot fabricate a trait definition. Resolution is per-call
# (not a module-load snapshot) so a repin of the platform state root is picked up.

_TRAIT_SPECS_RELPATH = Path(".tcip") / "state" / "trait_specs"
_SPEC_FIELDS = {f.name for f in fields(TraitSpec)}
_TUPLE_FIELDS = {"milestone_fractions", "delivers"}

SPEC_SUFFIX = ".json"
"""The canonical spec-file suffix, and the only one the store's locator addresses."""


def crops_yml_path() -> Path:
    """Where the crops.yml controlled vocabulary lives, stated once for every reader of it."""
    from tcip_mcp.knowledge import crops_yml_path as _knowledge_crops_yml_path

    return _knowledge_crops_yml_path()


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


def registered_crops() -> set[str]:
    """Every crop name crops.yml declares (the union of each trait's own ``crops`` list), or an
    empty set if it can't be read.

    The platform's registered crop vocabulary, not a test's: a fixture or a smoke script that
    needs a real ``crop`` for ``register_dataset`` reads one from here rather than spelling one
    of its own.
    """
    return {c for t in _crops_traits() for c in t.get("crops", []) if isinstance(c, str)}


def crops_units() -> dict[str, str]:
    """trait name -> crops.yml's declared physical unit (``mm``/``g``/``kg``/``m``/``cm``/…), for
    every trait that declares one. crops.yml is the trait-unit authority; a count/
    ordinal trait with no physical unit is simply absent from this mapping, never guessed."""
    return {t["name"]: t["units"] for t in _crops_traits() if isinstance(t.get("units"), str)}


_METRIC_LENGTH_UNITS = {"mm", "cm", "m", "km", "um"}
"""The metric length-unit symbols a physical scale may legitimately be expressed in, a dimensional
fact about these symbols, not a trait-by-trait choice: it never grows or shrinks with which traits
crops.yml happens to declare. ``crops_length_units`` intersects this against crops.yml's actual
declared vocabulary, so the length subset always reflects what the vocabulary really contains."""


def crops_length_units() -> set[str]:
    """The subset of :func:`crops_units`'s own declared units that are linear length units.

    Picked out of whatever crops.yml actually declares by a dimensional rule (a unit symbol that
    names a metric length), never a hand list of which traits use one: a mass unit (``g``/``kg``) or
    a mass-ratio concentration (``ug/g``) is never a length regardless of how a trait spells it, and
    a per-pixel physical scale (a length-per-pixel quantity) is refused in any other unit."""
    return {u for u in crops_units().values() if u in _METRIC_LENGTH_UNITS}


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
    write_trait_spec_fields refusal, ``tcip doctor``) never re-derives its own explanation.

    A ``schema_version`` key is not a ``TraitSpec`` field; the store seam already enforces its
    ceiling on every read (``tcip_store.schema_version.check_schema_version``, run inside
    ``read_versioned`` before this function ever sees the document), so it is stripped here
    rather than checked again.
    """
    name = data.get("name")
    if not isinstance(name, str) or not name:
        reason = f"missing/invalid 'name' ({name!r})"
        logger.warning("trait spec skipped: %s", reason)
        return None, reason
    data = {k: v for k, v in data.items() if k != "schema_version"}
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
    floor = data.get("holdout_match_quality_floor")
    if floor is not None and not (isinstance(floor, (int, float)) and 0 < floor <= 1):
        reason = f"holdout_match_quality_floor must be in (0, 1], got {floor!r}"
        logger.warning("trait spec %r skipped: %s", name, reason)
        return None, reason
    # count_objective is not validated against a closed vocabulary, a trait may name any
    # objective an agent has implemented and registered a picker for in
    # operating_point.COUNT_OBJECTIVE_PICKERS. resolve_operating_point refuses at resolution time
    # if the name has no registered picker, which is the honest place for that check to live (it
    # needs the picker registry; this module stays torch-free and doesn't import it).
    # data is arbitrary breeder-authored config; TraitSpec's constructor is the real gate (raises
    # TypeError on an unknown or missing field), so values are typed Any rather than validated here.
    kwargs: dict[str, Any] = {k: (tuple(v) if k in _TUPLE_FIELDS else v) for k, v in data.items()}
    try:
        return TraitSpec(**kwargs), None
    except TypeError as e:
        logger.warning("trait spec %r skipped: %s", name, e)
        return None, str(e)


# ── the trait-spec store ─────────────────────────────────────────────────────


def trait_specs_dir(project_root: str | Path | None = None) -> Path:
    """Where a project's trait specs live: ``<root>/.tcip/state/trait_specs``.

    The one implementation of that placement. ``project_root`` names the project explicitly, for a
    caller (the web backend, the ``doctor`` command) that serves more than one project per process;
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


def _trait_specs_state_root(specs_dir: Path) -> Path:
    """The shared ``.tcip/state`` root this store's records actually key against.

    Refuses rather than silently answering from the wrong place: the locator's own
    ``trait_specs`` prefix is fixed, so ``specs_dir.parent`` is only the right root when
    ``specs_dir`` is genuinely named ``trait_specs``. A caller that names anything else (a typo,
    a deliberate rename) would otherwise be read and written against the real ``trait_specs``
    directory with no refusal anywhere in the path.
    """
    if specs_dir.name != "trait_specs":
        raise ValueError(
            f"{specs_dir} does not end in 'trait_specs', so it cannot name this store's "
            "directory: trait specs live under a fixed 'trait_specs' segment, never a "
            "caller-chosen name"
        )
    return specs_dir.parent


TRAIT_SPECS_STORE = "trait_specs"
# Every write stamps this ceiling; trait_spec_unconformed refuses a stored record with no stamp
# or 1.
TRAIT_SPEC_SCHEMA_VERSION = 2
_SPEC_FILE = RootedFileLocator(prefix=("trait_specs",), suffix=SPEC_SUFFIX)
register_store(
    StoreDescriptor(
        name=TRAIT_SPECS_STORE,
        kind="record",
        key_fields=("trait",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_SPEC_FILE,
        schema_version=TRAIT_SPEC_SCHEMA_VERSION,
    )
)


def trait_spec_key(specs_dir: str | Path, trait_name: str) -> Key:
    """One trait's spec record under a spec directory.

    The key's root is the shared ``.tcip/state`` directory ``specs_dir`` names (via
    :func:`_trait_specs_state_root`), the same root every sibling project-state store hangs off;
    the locator's own ``trait_specs`` prefix supplies the rest of the on-disk placement.
    Enumeration answers over the records themselves. :func:`write_trait_spec_fields` merges
    compare-and-set against the version it read, so a field another writer recorded is never
    dropped.
    """
    return Key(TRAIT_SPECS_STORE, str(_trait_specs_state_root(Path(specs_dir))), (trait_name,))


def _spec_filename(specs_dir: Path, trait_name: str) -> str:
    """The spec's file name, taken from the store's own locator rather than spelled again."""
    return _SPEC_FILE.relative_path(str(specs_dir), (trait_name,)).name


def trait_spec_unconformed(document: dict) -> str | None:
    """Why a stored ``trait_specs`` record predates the subject-registry rename, else ``None``.

    A mapping with no ``schema_version`` key, or with ``1``, was never written by ``_encode_spec``'s
    unconditional stamp: either a record written before the rename (``positive_class_name`` is now
    ``positive_value``), restated through ``author_trait_spec`` with the caller supplying every
    field the spec carries, or a hand-authored file with no stamp, conformed by hand
    (``"schema_version": 2`` and ``positive_value`` in place of ``positive_class_name``). Distinct
    from the seam's own too-new refusal (``SchemaVersionRefused``, ``kind: "version_refused"``):
    that is a document above this store's declared ceiling; this is a document at or under it whose
    shape the ceiling alone does not describe, since a field renamed rather than the store gaining a
    version.
    """
    version = document.get("schema_version")
    if version is None or version == 1:
        return (
            "this trait spec record predates the subject-registry rename (positive_class_name is "
            "now positive_value) and carries no schema_version: 2 stamp; a stored record is "
            "restated by the agent through its author_trait_spec tool, supplying every field the "
            "spec carries, which leaves a fresh unconfirmed statement for the breeder to confirm "
            "in the Results tab; a hand-authored file is conformed by adding "
            "\"schema_version\": 2 and renaming positive_class_name to positive_value, after "
            "which the trait's statement reads as stale until restated and re-confirmed; either "
            "way a confirmed operationalization reads as stale until re-confirmed"
        )
    return None


def load_trait_specs_with_errors(
    specs_dir: Path | None = None, *, project_root: str | Path | None = None,
) -> tuple[list[TraitSpec], list[dict]]:
    """Same scan as :func:`load_trait_specs`, plus the file/reason for every spec skipped.

    This is the one place that detail exists; a caller that needs to tell a breeder or the agent
    which spec is broken and why (the Results API, the ``doctor`` command) reads it from here
    rather than re-deriving its own explanation or grepping logs. Either name the project whose
    registry to read (``project_root``) or the directory itself; the placement is resolved here.
    """
    directory = _resolve_specs_dir(specs_dir, project_root)
    errors: list[dict] = []
    vocab = _crops_vocab()
    specs: list[TraitSpec] = []
    for key in ts.keys(TRAIT_SPECS_STORE, str(_trait_specs_state_root(directory))):
        filename = _spec_filename(directory, key.parts[0])
        try:
            data = ts.read_versioned(key).value
        except NotFound:
            continue
        except DecodeError as e:
            logger.warning("trait spec %s skipped: %s", filename, e)
            errors.append({"file": filename, "reason": str(e)})
            continue
        except SchemaVersionRefused as e:
            logger.warning("trait spec %s skipped: %s", filename, e)
            errors.append({"file": filename, "reason": str(e), "kind": "version_refused"})
            continue
        if not isinstance(data, dict):
            reason: str | None = "not a mapping"
            logger.warning("trait spec %s skipped: %s", filename, reason)
            errors.append({"file": filename, "reason": reason})
            continue
        unconformed_reason = trait_spec_unconformed(data)
        if unconformed_reason is not None:
            logger.warning("trait spec %s skipped: %s", filename, unconformed_reason)
            errors.append({"file": filename, "reason": unconformed_reason, "kind": "unconformed"})
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
    cross-checked against the crops.yml controlled vocab. A project with no trait spec on record
    yields none; an invalid or fabricated spec is skipped (so ``get_trait`` later hard-fails
    honestly rather than serving it). See :func:`load_trait_specs_with_errors` for why each
    skipped spec was skipped."""
    specs, _errors = load_trait_specs_with_errors(specs_dir, project_root=project_root)
    return specs


class TraitSpecRevision(NamedTuple):
    """One call's full answer to updating a trait spec: the spec as written, the trait-spec
    authoring statement as written when this call stated or restated it (``None`` when it did
    neither), and one sentence saying what happened to the statement."""

    spec: TraitSpec
    statement: dict[str, Any] | None
    statement_note: str


def write_trait_spec_fields(
    trait_name: str, fields_: dict,
    specs_dir: Path | None = None, *, project_root: str | Path | None = None,
    rationale: str | None = None, relayed_note: str = "",
) -> TraitSpec:
    """Update one or more fields on an already-registered trait spec, returning it as written.

    ``rationale`` and ``relayed_note`` feed the trait-spec authoring statement this call states
    or restates. A thin wrapper over :func:`revise_trait_spec_fields`, which holds the
    compare-and-set loop and the trait-spec statement it restates; the ``revise_trait_spec`` MCP
    tool calls that function directly for its fuller return, while the derived localization kind
    and every other field-only caller keep calling this one for its plain ``TraitSpec``.
    """
    return revise_trait_spec_fields(
        trait_name, fields_, specs_dir, project_root=project_root,
        rationale=rationale, relayed_note=relayed_note,
    ).spec


def revise_trait_spec_fields(
    trait_name: str, fields_: dict,
    specs_dir: Path | None = None, *, project_root: str | Path | None = None,
    rationale: str | None = None, relayed_note: str = "",
) -> TraitSpecRevision:
    """Update one or more fields on an already-registered trait spec, and state or restate its
    trait-spec authoring statement when the update calls for it.

    Refuses (raises ``ValueError``) if the trait has no spec record on file: creating one is
    ``author_trait_spec``'s job. Refuses a caller-supplied ``schema_version`` in ``fields_``: it
    is not a ``TraitSpec`` field, no caller sets it directly, and merging it in would let a
    config editor stamp a version the store seam never validated; the encoder stamps the current
    ceiling on every write regardless, so no caller-supplied value could ride through anyway.
    Refuses a stored record :func:`trait_spec_unconformed` answers a reason for, before any merge;
    the restatement door for such a record is :func:`author_trait_spec`, never this one.
    A ``rationale`` that is given must say something, checked before any read.

    The statements scope this reads and writes is :func:`trait_spec_statements_scope` (project_root)
    whenever ``project_root`` is given or neither argument is (the pinned root, where
    ``trait_specs_dir(None)`` and ``trait_spec_statements_scope(None)`` resolve under one state
    directory), and :func:`_trait_specs_state_root` (directory) only for a ``specs_dir`` caller,
    which no production code is. The two coincide except under a monkeypatched
    ``_TRAIT_SPECS_RELPATH``, which ``tests/test_trait_authoring.py``'s revise-tool test exercises.

    The read, the merge, the validation and the spec write are one compare-and-set against the
    version read, retried on conflict against whatever landed meanwhile; the candidate is parsed
    once here to decide whether to refuse and what moved, and a second time inside
    :func:`_validate_and_write_spec` on the write itself, the cost of keeping that one shared
    write path. ``moved`` is whether the merge changes any of ``_AUTHORED_SPEC_FIELDS`` from
    what is on file now, both sides parsed through :func:`_spec_from_config` so a field merely
    restated at its own value never reads as moved; a stored record the parser refuses has no
    parsed values to compare, so ``moved`` reads true for it whatever ``fields_`` names, naming
    the authored keys in ``fields_`` when there are any and otherwise saying the stored record's
    authored values could not be read. ``stale`` is whether the
    trait-spec statement on file, if any, is absent or no longer matches the candidate
    (:func:`trait_spec_statement_stale`). A call that moves an authored field over a trait
    carrying a statement and gives no rationale refuses by name, naming the fields that moved,
    before any write; every other combination writes the spec.

    A rationale-bearing call then states or restates the trait-spec statement whenever no
    statement is on file, the one on file is stale, or this call moved an authored field (a spec
    moved past its confirmed statement by a raw write and then restated back to those same
    confirmed values still needs a fresh statement and the breeder's second look, since ``moved``
    is a condition of its own and is never folded into ``stale``); a current, unmoved statement
    is left alone, and a rationale-less call never touches the statement at all, which is what
    the derived localization kind's and every other carried-forward write's call relies on.
    ``relayed_note`` is taken fresh on every statement written here, never carried forward from
    the one it replaces, since a remark about the old values says nothing about the new ones.
    The statement write is its own compare-and-set: on conflict, the spec on file is re-read and
    compared to what this call wrote by authored snapshot; a different snapshot means a
    concurrent revision to other values landed after this call's own write, and the restatement
    is abandoned, since a concurrent writer that itself restates leaves its own statement current
    and one that does not leaves a stale pair the read-time predicate, the doctor and this
    refusal each catch on their own; an unchanged snapshot retries the statement write against
    whatever is on file now, creating fresh when the statement was deleted meanwhile. No branch
    ever leaves a statement whose snapshot already equals this call's own: such a record can only
    be the old confirmed statement or a confirmation of it landing in the same window, and either
    way this call's own rationale is what belongs on record for it.

    Two keys in two stores are never written atomically: a crash between the spec write and the
    statement write, or a no-rationale authored write racing this call's own statement create in
    either ordering, can leave a spec whose statement is stale with no version conflict on either
    side. Both are read-time facts the delivery refusal, the doctor and this function's own
    no-rationale refusal each catch on their own, never assumed away here.
    """
    if "schema_version" in fields_:
        raise ValueError(
            f"update to trait spec {trait_name!r} cannot carry 'schema_version' in fields: it "
            "is not a TraitSpec field, and no caller writes it directly"
        )
    if rationale is not None:
        rationale = _require_text(rationale, "rationale")

    directory = _resolve_specs_dir(specs_dir, project_root)
    if specs_dir is not None and project_root is None:
        statements_scope: str | Path = _trait_specs_state_root(directory)
    else:
        statements_scope = trait_spec_statements_scope(project_root)
    spec_key = trait_spec_key(directory, trait_name)
    statement_key = trait_spec_statement_key(statements_scope, trait_name)
    vocab = _crops_vocab()

    # A conflict at the spec key means another writer committed, so the loop only repeats while
    # the spec is actually changing under it and ends when this merge is the one that lands.
    while True:
        stored = ts.read_versioned(spec_key, default=None)
        if stored.value is None:
            raise _no_spec_error(trait_name, directory)
        data = stored.value
        if not isinstance(data, dict):
            filename = _spec_filename(directory, trait_name)
            raise ValueError(f"{directory / filename} is not a valid trait spec (not a mapping)")
        unconformed_reason = trait_spec_unconformed(data)
        if unconformed_reason is not None:
            raise ValueError(
                f"update to trait spec {trait_name!r} refused: {unconformed_reason}"
            )

        merged = dict(data)
        merged.update(fields_)
        candidate, reason = _spec_from_config(merged, vocab)
        if candidate is None:
            raise ValueError(
                f"update to trait spec {trait_name!r} would produce an invalid spec: {reason}. "
                "Refusing to write."
            )

        stored_spec, _stored_reason = _spec_from_config(data, vocab)
        if stored_spec is None:
            moved = True
            moved_fields = sorted(set(fields_) & set(_AUTHORED_SPEC_FIELDS))
        else:
            stored_snapshot = _statement_snapshot(stored_spec)
            candidate_snapshot = _statement_snapshot(candidate)
            moved_fields = sorted(
                f for f in _AUTHORED_SPEC_FIELDS if stored_snapshot[f] != candidate_snapshot[f]
            )
            moved = bool(moved_fields)
        moved_description = (
            f"moves {moved_fields}" if moved_fields
            else "repairs a stored record whose authored values could not be read"
        )

        existing_statement = ts.read_versioned(statement_key, default=None)
        statement_on_file = existing_statement.value
        stale = trait_spec_statement_stale(candidate, statement_on_file)

        if moved and statement_on_file and rationale is None:
            raise ValueError(
                f"update to trait spec {trait_name!r} {moved_description}, and its authoring "
                "statement already covers the old values; changing an authored field needs a "
                "rationale so the statement restates for the breeder's re-confirmation. Pass "
                "rationale=... naming why."
            )

        try:
            written_spec, write_reason = _validate_and_write_spec(
                spec_key, merged, expect=stored.version,
            )
        except VersionConflict:
            continue
        if written_spec is None:
            raise ValueError(
                f"update to trait spec {trait_name!r} would produce an invalid spec: "
                f"{write_reason}. Refusing to write."
            )
        break

    if rationale is None:
        return TraitSpecRevision(written_spec, None, "left untouched, no rationale")
    if not (moved or stale or statement_on_file is None):
        return TraitSpecRevision(written_spec, None, "left as it is, current")
    if statement_on_file is None:
        note = "stated, since none was on record"
    elif stale:
        note = "restated, since the one on record was stale"
    elif moved_fields:
        note = f"restated for re-confirmation, since {moved_fields} moved"
    else:
        note = (
            "restated for re-confirmation, since the stored record's authored values could "
            "not be read and are read as moved by the repair"
        )

    statement_expect = existing_statement.version
    while True:
        new_statement = {
            "trait": trait_name,
            "statement_fields": _statement_snapshot(written_spec),
            "rationale": rationale,
            "stated_by": TRAIT_SPEC_REVISION_SURFACE,
            "stated_at": now_iso(),
            "relayed_note": str(relayed_note or ""),
            **agent_identity.statement_fields(),
            **{field: None for field in TRAIT_SPEC_CONFIRMATION_FIELDS},
        }
        try:
            ts.replace(statement_key, new_statement, expect=statement_expect)
        except VersionConflict:
            spec_reread = ts.read_versioned(spec_key, default=None)
            reread_data = spec_reread.value
            # This call's own write just stamped reread_data, so trait_spec_unconformed(reread_data)
            # cannot answer a reason here; called anyway for symmetry with the other stored reads.
            reread_spec = (
                _spec_from_config(reread_data, vocab)[0]
                if isinstance(reread_data, dict) and trait_spec_unconformed(reread_data) is None
                else None
            )
            if (
                reread_spec is None
                or _statement_snapshot(reread_spec) != _statement_snapshot(written_spec)
            ):
                return TraitSpecRevision(
                    written_spec, None,
                    "abandoned: the spec's authored values moved again before this call's "
                    "statement could land",
                )
            statement_expect = ts.read_versioned(statement_key, default=None).version
            continue
        return TraitSpecRevision(written_spec, new_statement, note)


def _encode_spec(spec: TraitSpec) -> dict[str, Any]:
    """An already-valid ``TraitSpec`` as the JSON-safe mapping the store's codec accepts: every
    tuple field becomes a list, plus the current ``schema_version`` stamp. The one encoding every
    trait-spec writer and reader shares.

    ``TraitSpec`` itself carries no ``schema_version`` field: the store seam's ceiling check runs
    on the raw document, not the dataclass. Every write stamps ``TRAIT_SPEC_SCHEMA_VERSION``
    unconditionally, so a record this encoder ever touches always carries the current ceiling;
    :func:`trait_spec_unconformed` is what refuses one that predates this encoder.
    """
    encoded = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(spec).items()
    }
    encoded["schema_version"] = TRAIT_SPEC_SCHEMA_VERSION
    return encoded


def _write_spec_record(key: Key, spec: TraitSpec, *, expect: ts.Version | None) -> None:
    """Encode an already-valid ``TraitSpec`` and write it to ``key`` under compare-and-set at
    ``expect``. Never validates: the caller either built ``spec`` through ``_spec_from_config``
    already or otherwise guarantees it is legal."""
    ts.replace(key, _encode_spec(spec), expect=expect)


def _validate_and_write_spec(
    key: Key, data: dict, *, expect: ts.Version | None,
) -> tuple[TraitSpec | None, str | None]:
    """Validate ``data`` as a trait spec against the crops.yml vocabulary and, if legal, encode
    and write it to ``key`` under compare-and-set at ``expect``.

    Returns ``(spec, None)`` on success or ``(None, reason)`` when ``data`` fails validation, so
    each caller states the refusal in its own words around the same failure rather than this
    function picking one wording for all of them. Raises ``VersionConflict`` if another writer
    landed at ``key`` since ``expect`` was read.

    The one write every trait-spec writer shares: ``revise_trait_spec_fields``'s compare-and-set
    loop, ``author_trait_spec``'s single cas attempt, and the test producer
    ``_operationalization_fixtures.write_spec`` all call this rather than repeating the
    validate-encode-write shape.
    """
    spec, reason = _spec_from_config(data, _crops_vocab())
    if spec is None:
        return None, reason
    _write_spec_record(key, spec, expect=expect)
    return spec, None


def _no_spec_error(trait_name: str, directory: Path) -> ValueError:
    return ValueError(
        f"no trait spec record for {trait_name!r} under {directory}, "
        "revise_trait_spec only updates an already-registered trait; register it first "
        "with author_trait_spec."
    )


# ── the trait-spec authoring statement store ─────────────────────────────────

_AUTHORED_SPEC_FIELDS = (
    "delivers", "positive_value", "milestone_fractions", "milestone_on", "majority_milestone",
    "majority_provisional", "phenology_prefix", "majority_label", "count_objective",
    "count_bias_tolerance_frac", "count_error_tolerance", "classifier_agreement_floor",
    "ordinal_agreement_floor", "regression_skill_floor", "scale_tolerance_frac",
    "holdout_match_quality_floor", "notes",
)
"""Every ``TraitSpec`` field ``author_trait_spec`` accepts; see its own docstring for why the rest
of ``TraitSpec`` is not here."""


def _statement_snapshot(spec: TraitSpec) -> dict[str, Any]:
    """The comparable snapshot a trait-spec statement's own ``statement_fields`` holds: every
    authored field's canonical value. Shared by ``author_trait_spec``, ``revise_trait_spec_fields``
    and :func:`trait_spec_statement_stale`, so the three sites cannot disagree over what a
    statement's own fields snapshot means."""
    return {field: canonical(getattr(spec, field)) for field in _AUTHORED_SPEC_FIELDS}


_RETIRED_SPEC_FIELD_NAMES = {"positive_value": "positive_class_name"}
"""An authored field's name before the subject-registry rename, the one rename
:func:`trait_spec_unconformed` exists for: a pre-rename record's ``positive_value`` reads back
under this retired key. Consulted only by :func:`_spec_replaced_values`, when ``author_trait_spec``
restates a record :func:`trait_spec_unconformed` answers a reason for."""


def _spec_replaced_values(
    old: dict[str, Any], authored: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Every authored field whose value in ``old`` (a stored record predating the
    subject-registry rename) differs from the value ``author_trait_spec`` is about to write,
    each as ``{"recorded": <old>, "authored": <new>}`` in canonical form.

    A pre-rename field is read from its retired name (:data:`_RETIRED_SPEC_FIELD_NAMES`) when the
    current name is absent from ``old``, so the renamed ``positive_class_name``/``positive_value``
    pair compares the same underlying value under its two names rather than reading as an
    always-missing field. A field ``old`` never carried under either name reads as
    ``recorded: None``, the same as one it stored as null. ``authored`` is the statement snapshot
    of the parsed spec, so the ``authored`` side agrees with ``statement_fields`` byte for byte.
    """
    replaced: dict[str, dict[str, Any]] = {}
    for field in _AUTHORED_SPEC_FIELDS:
        if field in old:
            recorded = old[field]
        else:
            retired_name = _RETIRED_SPEC_FIELD_NAMES.get(field)
            recorded = old.get(retired_name) if retired_name else None
        recorded_value = canonical(recorded)
        authored_value = canonical(authored.get(field))
        if recorded_value != authored_value:
            replaced[field] = {"recorded": recorded_value, "authored": authored_value}
    return replaced


_CARRIED_FORWARD_SPEC_FIELDS = (
    "localization", "localization_tolerance", "localization_tolerance_frac",
    "sliver_policy", "sliver_frac",
)
"""Fields ``author_trait_spec`` never authors: copied unchanged from an existing spec on a
restatement, or left at ``TraitSpec``'s own dataclass defaults on first creation."""

TRAIT_SPEC_STATEMENT_FIELDS = (
    "statement_fields", "rationale", "stated_by", "stated_at", "relayed_note",
    *agent_identity.RECORD_FIELDS,
)
"""Every field a trait-spec authoring statement owns, the agent identity fields included. The
confirmation compare-and-set hashes all of them, the same discipline
``operationalization.STATEMENT_FIELDS`` already uses."""

TRAIT_SPEC_CONFIRMATION_FIELDS = ("confirmed_by", "confirmed_at", "identity_from_request", "record_seen")
"""Every field only :func:`confirm_trait_spec` sets. A statement write refuses a payload with one."""

TRAIT_SPEC_STATEMENT_SURFACE = "author_trait_spec"
"""The producing surface stamped into ``stated_by``, never accepted from a caller: it says a
statement came in through the authoring tool rather than through a file edit, nothing more. The
harness and session that made the call are the agent identity fields beside it, declared by the
connecting software and no evidence of who the person was."""

TRAIT_SPEC_REVISION_SURFACE = "revise_trait_spec"
"""The producing surface stamped into a restatement's ``stated_by``: the one platform caller
that reaches ``revise_trait_spec_fields``'s statement branch, the field-editing door rather than
the authoring one."""

TRAIT_SPEC_STATEMENTS_STORE = "trait_spec_statements"
_STATEMENT_FILE = RootedFileLocator(prefix=("trait_spec_statements",), suffix=".json")
register_store(
    StoreDescriptor(
        name=TRAIT_SPEC_STATEMENTS_STORE,
        kind="record",
        key_fields=("trait",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_STATEMENT_FILE,
    )
)

_STATE_RELPATH = Path(".tcip") / "state"


def trait_spec_statements_scope(project_root: str | Path | None = None) -> Path:
    """Where a project's trait-spec authoring statements live: ``<root>/.tcip/state``.

    Mirrors ``operationalization.operationalizations_scope``'s own placement and reasoning: a
    statement is a claim about a trait spec it does not own, kept in the project's general state
    directory beside the spec's own registry rather than folded into it. ``project_root`` names the
    project explicitly, for a caller serving more than one project per process; omitting it
    resolves against this process's pinned platform root.
    """
    if project_root is not None:
        return Path(project_root) / _STATE_RELPATH
    from tcip_mcp.project_paths import resolve_state

    return resolve_state(_STATE_RELPATH)


def trait_spec_statement_key(scope: str | Path, trait_name: str) -> Key:
    """One trait's authoring-statement record.

    ``cas``: ``author_trait_spec`` and ``confirm_trait_spec`` are two processes writing the same
    record, so an unconditional write would drop whichever one landed first.
    """
    return Key(TRAIT_SPEC_STATEMENTS_STORE, str(scope), (trait_name,))


def trait_spec_statement_seen_hash(record: dict[str, Any]) -> str:
    """A content hash over every field a trait-spec statement owns, in canonical form.

    The confirmation carries this back, so a breeder's click confirms the record they read rather
    than whatever an agent rewrote while the card was open.
    """
    return content_hash(record, TRAIT_SPEC_STATEMENT_FIELDS)


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            f"{field} is required and must say something: a trait-spec statement with an empty "
            f"{field} states nothing"
        )
    return text


def _spec_collision_text(trait_name: str) -> str:
    return (
        f"author_trait_spec cannot register trait {trait_name!r}: a trait spec and its authoring "
        "statement are both already on record for it. author_trait_spec only creates a trait that "
        "does not yet exist; change an already-registered spec's fields with "
        "revise_trait_spec instead."
    )


def author_trait_spec(
    project_root: str | Path,
    trait: str,
    *,
    delivers: Sequence[str],
    positive_value: str = "",
    milestone_fractions: Sequence[float] = (),
    milestone_on: str = "",
    majority_milestone: str = "",
    majority_provisional: bool = False,
    phenology_prefix: str = "",
    majority_label: str = "",
    count_objective: str = "",
    count_bias_tolerance_frac: float | None = None,
    count_error_tolerance: float | None = None,
    classifier_agreement_floor: float | None = None,
    ordinal_agreement_floor: float | None = None,
    regression_skill_floor: float | None = None,
    scale_tolerance_frac: float | None = None,
    holdout_match_quality_floor: float | None = None,
    notes: str = "",
    rationale: str,
    relayed_note: str = "",
    **payload: Any,
) -> dict[str, Any]:
    """Author a trait spec that does not yet exist, and record the statement of why, unconfirmed.

    Writes the effective spec into ``trait_specs`` first, cross-checked against crops.yml's
    controlled vocabulary through the same ``_spec_from_config`` every config-authored spec already
    goes through, then writes the unconfirmed statement into ``trait_spec_statements`` second.
    Refuses before writing anything when a spec is already registered for this trait *and* a
    statement record already exists for it: a real collision. When a spec exists with no statement
    behind it (the recovery state after a second write that failed partway), this call proceeds as
    a restatement rather than refusing, since there is nothing to collide with.

    A spec :func:`trait_spec_unconformed` answers a reason for (predates the subject-registry
    rename, no ``schema_version: 2`` stamp) is also not a collision, whatever its statement: such a
    record is not one any reader can load, so its own statement describes it no better whether the
    statement exists or not. This call proceeds as a full restatement, the caller supplying every
    field the spec carries exactly as on first creation; the carried-forward fields below still
    come from the old record, since their names did not move. The response, and only the
    response (the persisted statement carries exactly ``TRAIT_SPEC_STATEMENT_FIELDS`` and the
    confirmation fields), then carries two more keys: ``replaced_values``, naming every authored
    field whose value this call replaced (the retired ``positive_class_name`` read as the prior
    value of the now-renamed ``positive_value``; a field the old record never carried reads as
    ``recorded: null``), each as ``{"recorded": <old>, "authored": <new>}`` in the statement
    snapshot's own canonical form; and ``prior_confirmation``, the ``confirmed_by`` and
    ``confirmed_at`` of the statement this call replaced when the breeder had confirmed it, else
    ``None``, since the fresh statement is unconfirmed and that confirmation no longer stands.
    Every field the tool does not receive is written at its default, so the caller reads
    ``replaced_values`` back and the breeder reads the new values in the Results tab before
    confirming; nothing here can tell an omitted field from one stated at its default.

    ``localization``, ``localization_tolerance``, ``localization_tolerance_frac``,
    ``sliver_policy`` and ``sliver_frac`` are not accepted here: they carry forward unchanged from
    an existing spec on a restatement, or stay at ``TraitSpec``'s own dataclass defaults on first
    creation, and remain ``write_trait_spec_fields``'s job to change. ``rationale`` is the agent's
    account of why it chose these values from the breeder's own words, prose read by a breeder,
    never parsed; it is required.

    Refuses a payload naming one of the four confirmation fields
    (``confirmed_by``/``confirmed_at``/``identity_from_request``/``record_seen``): only the
    breeder's own confirmation, from the web backend, writes those.
    """
    if payload:
        offered = sorted(payload)
        confirmation = [key for key in offered if key in TRAIT_SPEC_CONFIRMATION_FIELDS]
        if confirmation:
            raise ValueError(
                f"author_trait_spec cannot carry the confirmation field(s) {confirmation}: only "
                "the breeder's own confirmation writes those, from the web backend, and a "
                "statement that filled them would be the agent confirming its own trait"
            )
        raise ValueError(f"unknown field(s) {offered} for {TRAIT_SPEC_STATEMENTS_STORE}")

    directory = trait_specs_dir(project_root)
    statements_scope = trait_spec_statements_scope(project_root)
    spec_key = trait_spec_key(directory, trait)
    statement_key = trait_spec_statement_key(statements_scope, trait)

    existing_spec = ts.read_versioned(spec_key, default=None)
    existing_statement = ts.read_versioned(statement_key, default=None)
    if existing_spec.value is not None and not isinstance(existing_spec.value, dict):
        raise ValueError(
            f"author_trait_spec cannot restate trait {trait!r}: the stored trait spec record is "
            "not a mapping; delete or repair the record before authoring the trait again"
        )
    unconformed_reason = (
        trait_spec_unconformed(existing_spec.value) if existing_spec.value is not None else None
    )
    if (
        existing_spec.value is not None
        and existing_statement.value is not None
        and unconformed_reason is None
    ):
        raise ValueError(_spec_collision_text(trait))

    authored: dict[str, Any] = {
        "name": trait,
        "delivers": tuple(delivers),
        "positive_value": positive_value,
        "milestone_fractions": tuple(milestone_fractions),
        "milestone_on": milestone_on,
        "majority_milestone": majority_milestone,
        "majority_provisional": majority_provisional,
        "phenology_prefix": phenology_prefix,
        "majority_label": majority_label,
        "count_objective": count_objective,
        "count_bias_tolerance_frac": count_bias_tolerance_frac,
        "count_error_tolerance": count_error_tolerance,
        "classifier_agreement_floor": classifier_agreement_floor,
        "ordinal_agreement_floor": ordinal_agreement_floor,
        "regression_skill_floor": regression_skill_floor,
        "scale_tolerance_frac": scale_tolerance_frac,
        "holdout_match_quality_floor": holdout_match_quality_floor,
        "notes": notes,
    }
    if existing_spec.value is not None:
        authored.update({
            field: existing_spec.value[field]
            for field in _CARRIED_FORWARD_SPEC_FIELDS
            if field in existing_spec.value
        })

    spec, reason = _validate_and_write_spec(spec_key, authored, expect=existing_spec.version)
    if spec is None:
        raise ValueError(f"author_trait_spec cannot register trait {trait!r}: {reason}")

    snapshot = _statement_snapshot(spec)
    statement = {
        "trait": trait,
        "statement_fields": snapshot,
        "rationale": _require_text(rationale, "rationale"),
        "stated_by": TRAIT_SPEC_STATEMENT_SURFACE,
        "stated_at": now_iso(),
        "relayed_note": str(relayed_note or ""),
        **agent_identity.statement_fields(),
        **{field: None for field in TRAIT_SPEC_CONFIRMATION_FIELDS},
    }
    ts.replace(statement_key, statement, expect=existing_statement.version)
    if unconformed_reason is None:
        return statement
    # The two restatement keys are the response's alone: the persisted statement carries exactly
    # TRAIT_SPEC_STATEMENT_FIELDS and the confirmation fields, the shape the seen hash covers.
    prior = existing_statement.value if isinstance(existing_statement.value, dict) else {}
    prior_confirmation = (
        {"confirmed_by": prior.get("confirmed_by"), "confirmed_at": prior.get("confirmed_at")}
        if prior.get("confirmed_by") is not None
        else None
    )
    old_record: dict[str, Any] = existing_spec.value if isinstance(existing_spec.value, dict) else {}
    return {
        **statement,
        "replaced_values": _spec_replaced_values(old_record, snapshot),
        "prior_confirmation": prior_confirmation,
    }


class TraitSpecStatementNotFound(ValueError):
    """Raised when a confirmation arrives for a trait nothing has been stated for."""


class TraitSpecUnconfirmed(ValueError):
    """Raised when a delivery-meaning statement is attempted for a trait whose own trait-spec
    statement is not both confirmed and current: absent, stale, or current but never confirmed by
    the breeder. Defined here, beside :class:`TraitSpecStatementNotFound`, rather than in
    ``operationalization``, since a trait-spec statement is this module's own record."""


class TraitSpecStatementMoved(ValueError):
    """Raised when the statement record moved since the surface read it, carrying what is on
    file now. Defined here rather than reused from ``operationalization.RecordMoved``, since
    ``operationalization`` already imports from this module and the reverse import would be
    circular.
    """

    def __init__(self, message: str, record: dict[str, Any], record_seen: str) -> None:
        super().__init__(message)
        self.record = dict(record)
        self.record_seen = record_seen


def confirm_trait_spec(
    project_root: str | Path,
    trait: str,
    *,
    user: str,
    record_seen: str,
    identity_from_request: bool,
    confirmed: bool = True,
) -> dict[str, Any]:
    """Record that the breeder confirmed the trait-spec statement on file, or withdraw one.

    Exposed by no MCP tool and called only by the web backend, from a route the breeder's own GUI
    action posts, mirroring ``operationalization.confirm_trait_operationalization``'s own "no MCP
    tool" discipline exactly. Reads and writes only ``trait_spec_statements``, never ``trait_specs``
    directly. ``record_seen`` is :func:`trait_spec_statement_seen_hash` over the record the surface
    rendered, compared against what is on file now, so a click cannot land on text the breeder
    never read; a mismatch raises :class:`TraitSpecStatementMoved` carrying the current record.

    ``confirmed=False`` withdraws, clearing exactly the four confirmation fields and leaving the
    statement intact.
    """
    scope = trait_spec_statements_scope(project_root)
    key = trait_spec_statement_key(scope, trait)
    existing = ts.read_versioned(key, default=None)
    stated = existing.value
    if not stated:
        raise TraitSpecStatementNotFound(
            f"nothing is stated for trait {trait!r}, so there is no trait-spec statement to "
            "confirm. This reads only the statement store, so it cannot tell whether a spec is "
            "on record: if one is, state it with revise_trait_spec(project_root=..., "
            f"trait_name={trait!r}, fields={{}}, rationale=...); if none is, register it with "
            "author_trait_spec first"
        )
    current_seen = trait_spec_statement_seen_hash(stated)
    if record_seen != current_seen:
        raise TraitSpecStatementMoved(
            f"the trait-spec statement for {trait!r} moved since it was read, so confirming now "
            "would confirm text nobody displayed; re-read it and confirm what is on file",
            stated,
            current_seen,
        )

    updated = dict(stated)
    if confirmed:
        updated.update({
            "confirmed_by": user_identity(user),
            "confirmed_at": now_iso(),
            "identity_from_request": bool(identity_from_request),
            "record_seen": current_seen,
        })
    else:
        updated.update({field: None for field in TRAIT_SPEC_CONFIRMATION_FIELDS})

    ts.replace(key, updated, expect=existing.version)
    return updated


def trait_spec_statement_stale(spec: TraitSpec, statement: dict[str, Any] | None) -> bool:
    """Whether ``statement`` is absent, or its authored-field snapshot no longer matches
    ``spec``'s live values.

    The one staleness predicate ``revise_trait_spec_fields`` (against the candidate spec),
    ``state_operationalization``'s trait-spec precondition and the doctor's own check all call,
    so the three sites cannot disagree about what counts as stale.
    """
    if not statement:
        return True
    recorded = statement.get("statement_fields") or {}
    return recorded != _statement_snapshot(spec)


def trait_spec_statement_current(spec: TraitSpec, statement: dict[str, Any] | None) -> bool:
    """Whether ``statement`` is both confirmed by the breeder and not stale against ``spec``'s
    live values: "not stale and confirmed".

    Invalidation here is read-time, not write-time: the write-time restatement inside
    ``revise_trait_spec_fields`` is the platform's own mechanism for keeping a statement from
    going stale in the first place, and this is the backstop for a spec written past it (a raw
    store write, the evaluation's own carried-forward write, a crash between the spec write and
    the statement write), the same way an operationalization's constituting-field drift is caught
    by comparison rather than by a write-time rule. Unstated or unconfirmed is never current.
    """
    if not statement or not statement.get("confirmed_by"):
        return False
    return not trait_spec_statement_stale(spec, statement)


def _all_traits() -> dict[str, TraitSpec]:
    """The live registry: every config-authored spec found under this project's trait_specs dir."""
    return {spec.name: spec for spec in load_trait_specs()}


def get_trait(name: str) -> TraitSpec:
    """Return the ``TraitSpec`` for ``name``, or raise ``TraitUnknownError`` listing the registered traits."""
    return get_trait_for(name)


def get_trait_for(name: str, project_root: str | Path | None = None) -> TraitSpec:
    """One trait's spec from an explicit project's registry, or from the pinned one.

    For a caller (the web backend, the operationalization resolver) that serves more than one
    project per process and so cannot rely on ``resolve_state``'s single ``$TCIP_STATE_ROOT``
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
    rely on ``resolve_state``'s single ``$TCIP_STATE_ROOT`` pin; ``registered_traits()`` stays
    the MCP-server-side entry point for the one pinned project.
    """
    return sorted(spec.name for spec in load_trait_specs(project_root=project_root))
