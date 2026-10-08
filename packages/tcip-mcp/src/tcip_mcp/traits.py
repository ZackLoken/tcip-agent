"""A trait: one entry holding its spec fields and the operationalization text for each delivery kind
it delivers, kept per project as an appended list of revisions: ``propose_trait`` appends one,
``confirm_revision`` confirms or withdraws one by its number and content hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

import tcip_store as ts
from tcip_store import Key

from tcip_mcp.audit import now_iso

if TYPE_CHECKING:
    from tcip_mcp.subject_registry import SubjectRegistry

# Count objectives, each the name of one conf picker.
# COUNT_UNBIASED minimizes signed per-image count bias E[FP-FN]; the phenotype is a count.
COUNT_UNBIASED = "count_unbiased"
# DETECTION_F1 optimizes matching quality; the phenotype is presence/localization.
DETECTION_F1 = "detection_f1"
PRESENCE = "presence"             # only whether the object is present

# Localization, what counts as "finding" an object.
CENTER_MATCH = "center_match"  # predicted center within a derived tolerance of a GT center
IOU_MATCH = "iou_match"        # IoU >= a derived/def threshold

# ── the delivery kinds ───────────────────────────────────────────────────────

STATE_CROSSING_DATES = "state_crossing_dates"
PER_IMAGE_COUNT = "per_image_count"
PER_PLANT_COUNT_AGGREGATE = "per_plant_count_aggregate"
PER_PLANT_ORDINAL_AGGREGATE = "per_plant_ordinal_aggregate"
PER_PLANT_REGRESSION_AGGREGATE = "per_plant_regression_aggregate"

DELIVERY_KINDS = (
    STATE_CROSSING_DATES,
    PER_IMAGE_COUNT,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
)
"""The delivered artifact shapes. Each kind decides which spec fields an operationalization of it
rests on, whether it names delivered phenotypes, and whether it names value keys."""

DETECTOR_KINDS = frozenset({STATE_CROSSING_DATES, PER_IMAGE_COUNT, PER_PLANT_COUNT_AGGREGATE})
"""Kinds measured off a detector's detections of the operationalization's measured subject; the
other kinds are measured off a scalar head (ordinal or regression)."""

if TYPE_CHECKING:
    DeliveryKind = str
    Localization = str
else:
    DeliveryKind = Literal[DELIVERY_KINDS]
    Localization = Literal["", CENTER_MATCH, IOU_MATCH]

_COUNT_FIELDS = ("count_objective", "localization", "count_bias_tolerance_frac",
                 "count_error_tolerance", "holdout_match_quality_floor")
CONSTITUTING_FIELDS: dict[str, tuple[str, ...]] = {
    STATE_CROSSING_DATES: ("positive_state", "milestone_on", "milestone_fractions",
                           *_COUNT_FIELDS, "classifier_agreement_floor"),
    PER_IMAGE_COUNT: _COUNT_FIELDS,
    PER_PLANT_COUNT_AGGREGATE: _COUNT_FIELDS,
    PER_PLANT_ORDINAL_AGGREGATE: ("ordinal_agreement_floor",),
    PER_PLANT_REGRESSION_AGGREGATE: ("regression_criterion", "regression_skill_floor"),
}
"""The spec fields an operationalization of each kind rests on, its assessment's criterion
included, each required to hold a value before the operationalization is proposed."""

LOCALIZATION_FIELDS: dict[str, tuple[str, ...]] = {
    CENTER_MATCH: (), IOU_MATCH: ("iou_jitter_px", "iou_margin"),
}
"""The spec fields each localization's match criterion compares by, required beside
``localization`` wherever a kind rests on it."""


def criterion_fields(entry: TraitEntry, kind: str) -> tuple[str, ...]:
    """The spec fields an operationalization of ``kind`` rests on for ``entry``: the kind's own
    (:data:`CONSTITUTING_FIELDS`) and, where it rests on ``localization``, the fields the
    selected localization compares by (:data:`LOCALIZATION_FIELDS`)."""
    fields = CONSTITUTING_FIELDS[kind]
    if "localization" not in fields:
        return fields
    return (*fields, *LOCALIZATION_FIELDS.get(entry.localization, ()))


PHENOTYPE_NAMING_KINDS = frozenset({
    STATE_CROSSING_DATES,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
})
"""Kinds whose delivered file carries a phenotype name, hence whose operationalization binds one."""

VALUE_KEY_KINDS = frozenset({
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
})
"""Kinds whose rows carry a value key, hence whose operationalization bounds the set."""

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Text that must say something: surrounding whitespace is stripped and an empty value refuses."""


class TraitUnknownError(KeyError):
    """Raised for a trait with no record in the project, listing the traits it has."""


def crops_yml_path() -> Path:
    """Where the crops.yml controlled vocabulary lives, stated once for every reader of it."""
    from tcip_mcp.knowledge import crops_yml_path as _knowledge_crops_yml_path

    return _knowledge_crops_yml_path()


def _crops_traits() -> list[dict]:
    """crops.yml's trait records, read whole; a file that is missing or will not parse raises."""
    import yaml

    return yaml.safe_load(crops_yml_path().read_text(encoding="utf-8"))["traits"]


def registered_crops() -> set[str]:
    """Every crop name crops.yml declares (the union of each trait's own ``crops`` list)."""
    return {c for t in _crops_traits() for c in t.get("crops", [])}


def crops_units() -> dict[str, str]:
    """trait name -> crops.yml's declared physical unit, for every trait that declares one; a
    count or ordinal trait with no physical unit is absent from this mapping."""
    return {t["name"]: t["units"] for t in _crops_traits() if isinstance(t.get("units"), str)}


_METRIC_LENGTH_UNITS = {"mm", "cm", "m", "km", "um"}
"""The metric length-unit symbols a physical scale may be expressed in, a dimensional fact about
the symbols that never grows or shrinks with what crops.yml declares."""


def crops_length_units() -> set[str]:
    """The subset of :func:`crops_units`'s declared units that are linear length units."""
    return {u for u in crops_units().values() if u in _METRIC_LENGTH_UNITS}


def crops_definitions() -> dict[str, str]:
    """trait name -> crops.yml's declared definition, for every trait that carries one."""
    return {
        t["name"]: t["definition"] for t in _crops_traits() if isinstance(t.get("definition"), str)
    }


# ── the entry: the one declared schema ───────────────────────────────────────


class Operationalization(BaseModel):
    """What a trait's delivered number means for one delivery kind, in the breeder's terms."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    statement: Text
    """What the delivered number means, in the breeder's own words."""
    mechanism: Text
    """What produces the call or the number: which subject, attribute and model decide it."""
    measured_subject: Text
    """The ``subjects.json`` subject the number is about."""
    delivered_phenotypes: tuple[str, ...]
    """Which of the trait's ``delivers`` entries this kind ships under; empty for
    ``per_image_count``, whose CSV names no phenotype, and at least one for every other kind."""
    delivered_value_keys: tuple[str, ...]
    """The value keys delivered rows may carry: at least one for the per-plant aggregate kinds,
    empty for the others, whose row schema is fixed by their writer."""


class PositiveState(BaseModel):
    """The state a fraction counts: one attribute of the measured subject, and its value that is
    positive."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    attribute: Text
    """The measured subject's attribute, as ``subjects.json`` declares it."""
    value: Text
    """The value of that attribute that is the positive state."""

    def __str__(self) -> str:
        return f"{self.attribute}={self.value!r}"


class TraitEntry(BaseModel):
    """One trait's complete entry: its spec fields and its operationalization per delivery kind.
    Every field is required; a field not yet decided is stated as empty or null. A proposal is
    checked by :func:`check_proposed_entry`."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    name: Text
    """The trait's name in this project; its record is keyed by it."""
    delivers: tuple[str, ...]
    """The crops.yml phenotype names this trait delivers."""
    positive_state: PositiveState | None
    """The state a fraction counts, or null for a trait that counts none."""
    milestone_fractions: tuple[float, ...]
    """Crossing fractions for a milestone-delivering trait."""
    milestone_on: str
    """The quantity the milestones cross, e.g. ``positive_fraction``."""
    majority_milestone: str
    """The crossing key (e.g. ``95per``) the crops.yml majority-date milestone maps to, or empty."""
    phenology_prefix: str
    """The phenology CSV milestone-column prefix."""
    majority_label: str
    """The label the majority-alias column carries."""
    count_objective: str
    """What the delivered number must be reliable for (``count_unbiased``, ``detection_f1``,
    ``presence`` or another registered picker), a consequence judgment only the breeder makes;
    empty until decided."""
    localization: Localization
    """What finding one object means; empty until decided."""
    localization_tolerance: str
    """How the localization tolerance is derived, by name."""
    localization_tolerance_frac: float
    """The tolerance multiplier used when no ground truth is at hand to derive one from."""
    iou_jitter_px: float | None = Field(ge=0, allow_inf_nan=False)
    """For an ``iou_match`` trait, how far in pixels on the reference grid two careful
    annotations of one object sit apart, the displacement the threshold models (two equal boxes of
    the reference's mean characteristic size moved along one axis,
    ``derivations.derive_iou_match_threshold``); null until the breeder authors it."""
    iou_margin: float | None = Field(ge=0, allow_inf_nan=False)
    """For an ``iou_match`` trait, the absolute IoU the threshold sits below that modeled IoU;
    null until authored."""
    count_bias_tolerance_frac: float | None
    """Max acceptable mean per-image count bias on the held-out split, relative to the scope's
    typical per-image count; null until the breeder authors it."""
    count_error_tolerance: float | None
    """Max acceptable p90 per-image count error on the held-out split; null until authored."""
    classifier_agreement_floor: float | None
    """Min acceptable Cohen's kappa for the classifier operating point; null until authored."""
    ordinal_agreement_floor: float | None
    """Min acceptable ordinal agreement criterion value; null until authored."""
    regression_criterion: str
    """The regression skill statistic a regression delivery is assessed by
    (``operating_point.REGRESSION_CRITERIA``); empty until authored."""
    regression_skill_floor: float | None
    """Min acceptable value of ``regression_criterion``; null until authored."""
    scale_tolerance_frac: float | None
    """Max relative disagreement a physical-scale reference half may show; null until authored."""
    holdout_match_quality_floor: float | None = Field(gt=0, le=1)
    """Min held-out precision and recall the detection gate's localization criterion must clear, in
    (0, 1]; null until authored."""
    notes: str
    """Free-text notes on the trait's measurement."""
    operationalizations: dict[DeliveryKind, Operationalization]
    """What the delivered number means, per delivery kind this trait delivers."""


QUESTIONS: dict[str, str] = {
    "count_objective": (
        "Does every object on an image have to be found, or is it enough that misses and false "
        "finds cancel out in the total?"),
    "count_bias_tolerance_frac": (
        "By what fraction of a typical image's count may the model's average count be off "
        "before the number is no use to you?"),
    "count_error_tolerance": (
        "How many objects off may one image's count be, for nine images in ten, before that "
        "image's number is no use to you?"),
    "holdout_match_quality_floor": (
        "What share of the real objects must the model find, and what share of its finds must "
        "be real, on images it never trained on?"),
    "classifier_agreement_floor": (
        "How far beyond chance must the model's call of the state agree with yours before a "
        "fraction built from it means anything?"),
    "ordinal_agreement_floor": (
        "How far beyond chance must the model's scores agree with yours before a plant's "
        "score from it means anything?"),
    "regression_criterion": (
        "Should the model's values be judged by how much of the spread between plants they "
        "explain, or by how closely they agree with yours value for value?"),
    "regression_skill_floor": (
        "By the judgment you chose for the model's values (the share of the spread between "
        "plants they explain, or how closely they agree with yours value for value), how high "
        "must they score before a plant's value from them means anything?"),
    "localization": (
        "Does finding an object mean the model's mark lands near its center, or that the "
        "model's outline overlaps it?"),
    "iou_jitter_px": (
        "If the same person boxed one object twice, how many pixels apart, on the images the "
        "reference is drawn on, would the two boxes sit? The match threshold models two equal "
        "boxes of the objects' typical size moved apart by that much."),
    "iou_margin": (
        "By how much overlap (0 to 1, as an absolute amount) may the model's box fall short of "
        "the overlap those two repeat boxes reach and still count as finding the object?"),
    "milestone_on": "Which share of a plant's objects do the milestone dates track?",
    "milestone_fractions": "At which shares of that state across a plant do you record a date?",
    "scale_tolerance_frac": (
        "By what fraction may two physical measurements of the same reference object disagree "
        "before the scale behind a length is no use to you?"),
    "positive_state": "Which attribute of the subject, and which of its values, is the state a "
                      "fraction counts?",
}
"""The breeder's question for each spec field a measurement criterion compares against, asked
when the field is still unauthored."""


class UnauthoredFieldError(ValueError):
    """A measurement needs a spec field the trait's confirmed revision leaves unauthored."""


def authored(entry: TraitEntry, names: tuple[str, ...]) -> None:
    """Refuse (:class:`UnauthoredFieldError`) ``entry`` when any of the spec fields ``names`` is
    unauthored, naming each field and asking the breeder its question."""
    missing = [n for n in names if getattr(entry, n) in (None, "", ())]
    if missing:
        questions = " ".join(f"{n}: {QUESTIONS[n]}" for n in missing)
        raise UnauthoredFieldError(
            f"Trait {entry.name!r} leaves {missing} unauthored, so nothing states what this "
            f"measurement is held to. Ask the breeder: {questions} Propose their answer with "
            "propose_trait and have them confirm it in the Setup tab.")


def check_proposed_entry(entry: TraitEntry) -> None:
    """Refuse (``ValueError``) an entry a proposal may not append: ``delivers`` empty or naming
    anything outside crops.yml, or an operationalization that leaves a spec field its kind rests
    on (:func:`criterion_fields`) unauthored (:func:`authored`, asking the breeder), covers a
    phenotype the entry does not deliver, or names phenotypes or value keys where its kind
    carries none (or none where it carries them)."""
    vocab = {t["name"] for t in _crops_traits()}
    off_vocab = [d for d in entry.delivers if d not in vocab]
    if not entry.delivers or off_vocab:
        raise ValueError(
            f"delivers must name at least one crops.yml phenotype and nothing else "
            f"(off-vocabulary: {off_vocab})"
        )
    for kind, stated in entry.operationalizations.items():
        authored(entry, criterion_fields(entry, kind))
        off_spec = [p for p in stated.delivered_phenotypes if p not in entry.delivers]
        if off_spec:
            raise ValueError(
                f"the {kind} operationalization covers {off_spec}, which this trait does not "
                f"deliver ({list(entry.delivers)})"
            )
        if (kind in PHENOTYPE_NAMING_KINDS) != bool(stated.delivered_phenotypes):
            raise ValueError(
                f"a {kind} operationalization's delivered_phenotypes must "
                + ("name at least one delivered phenotype" if kind in PHENOTYPE_NAMING_KINDS
                   else "be empty, since its file names no phenotype")
            )
        if (kind in VALUE_KEY_KINDS) != bool(stated.delivered_value_keys):
            raise ValueError(
                f"a {kind} operationalization's delivered_value_keys must "
                + ("name the value keys its rows carry" if kind in VALUE_KEY_KINDS
                   else "be empty, since its writer fixes the row schema")
            )


def entry_sha256(entry: TraitEntry) -> str:
    """The content hash of ``entry``: SHA-256 over its canonical JSON form."""
    encoded = json.dumps(
        entry.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TraitRevision(BaseModel):
    """One proposed entry, as appended: its number, the entry, the entry's content hash, why it
    was proposed, and the breeder's confirmation or withdrawal once made. Who proposed it is the
    ``propose_trait`` audit line's identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    number: int
    """1 for the first revision, one more for each revision after it."""
    entry: TraitEntry
    entry_sha256: str
    """:func:`entry_sha256` of ``entry``, the hash a confirmation must present."""
    rationale: Text
    """The agent's account of why it proposed this entry, from the breeder's own words."""
    relayed_note: str
    """What the breeder said away from the GUI, relayed by the agent; never a confirmation."""
    proposed_at: str
    confirmed_by: str | None
    confirmed_at: str | None
    withdrawn_by: str | None
    withdrawn_at: str | None

    @property
    def confirmed(self) -> bool:
        """Whether the breeder confirmed this revision and has not withdrawn the confirmation."""
        return self.confirmed_at is not None and self.withdrawn_at is None

    @property
    def ref(self) -> dict[str, Any]:
        """This revision as every record and delivered file names it: the trait, the revision
        number and the entry's content hash."""
        return {"trait": self.entry.name, "trait_revision": self.number,
                "trait_revision_sha256": self.entry_sha256}


class TraitRecord(BaseModel):
    """One trait's stored record: every revision proposed for it, oldest first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revisions: tuple[TraitRevision, ...]

    @property
    def latest(self) -> TraitRevision:
        """The most recently proposed revision, confirmed or not."""
        return self.revisions[-1]

    @property
    def latest_confirmed(self) -> TraitRevision | None:
        """The highest-numbered revision the breeder confirmed and has not withdrawn, or
        ``None``."""
        return next((r for r in reversed(self.revisions) if r.confirmed), None)


# ── the store ────────────────────────────────────────────────────────────────

TRAITS_STORE = "traits"


def trait_key(project: str | Path, trait: str) -> Key:
    """One trait's record in a project's state."""
    from tcip_mcp.project_paths import project_state_dir

    return Key(TRAITS_STORE, str(project_state_dir(project)), (trait,))


def trait_names(project: str | Path) -> list[str]:
    """Every trait with a record in the project, sorted."""
    from tcip_mcp.project_paths import project_state_dir

    return sorted(k.parts[0] for k in ts.keys(TRAITS_STORE, str(project_state_dir(project))))


def _record(trait: str, project: str | Path, value: dict | None) -> TraitRecord:
    """``trait``'s stored ``value`` read through its schema. An absent value raises
    :class:`TraitUnknownError` naming the project's traits; a value the schema refuses raises."""
    if value is None:
        raise TraitUnknownError(
            f"Unknown trait {trait!r}. Traits in this project: {trait_names(project)}")
    return TraitRecord.model_validate(value)


def read_trait(trait: str, project: str | Path) -> TraitRecord:
    """One trait's record, read through its schema; refuses as :func:`_record` does."""
    return _record(trait, project, ts.read(trait_key(project, trait), default=None))


# ── the proposal and the confirmation ────────────────────────────────────────


def resolve_statement_registry(project: str | Path, dataset_root: str) -> SubjectRegistry:
    """The registry a ``state_crossing_dates`` operationalization's positive state is checked
    against.

    ``dataset_root`` given: that dataset's own registry. Empty: the project root's own registry,
    served when ``project`` is unambiguously the one dataset the project uses (its own
    ``subjects.json`` exists, and the project's dataset registry names at most one dataset).
    Otherwise refuses by name, naming the registered datasets and the ``dataset_root`` parameter.
    """
    from tcip_mcp.subject_registry import read_registry
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets

    if dataset_root:
        try:
            return read_registry(dataset_root)
        except FileNotFoundError as exc:
            raise ValueError(
                f"dataset_root {dataset_root!r} carries no subject registry of its own. Write one "
                "(write_subject_registry) before a crossing's state can be checked against it."
            ) from exc

    registered = read_datasets(project)
    roots = [str(dataset_entry_path(project, d)) for d in registered]
    if len(registered) > 1:
        raise ValueError(
            f"project {project!r} registers {len(registered)} datasets {roots}, so which one "
            "this crossing's classes belong to cannot be guessed. Pass dataset_root naming it."
        )
    try:
        return read_registry(project)
    except FileNotFoundError as exc:
        raise ValueError(
            f"project root {project!r} carries no subject registry of its own (registered "
            f"datasets: {roots}). Pass dataset_root naming the dataset this crossing's classes "
            "belong to."
        ) from exc


def propose_trait(
    project: str | Path,
    entry: TraitEntry,
    *,
    rationale: str,
    relayed_note: str,
    dataset_root: str = "",
) -> TraitRevision:
    """Append ``entry`` to its trait's record as a new, unconfirmed revision, creating the record
    for a trait the project does not have yet, then write the proposal's audit line.

    Refuses an entry :func:`check_proposed_entry` refuses, and a ``state_crossing_dates``
    operationalization whose positive state names an attribute the registry
    :func:`resolve_statement_registry` resolves (``dataset_root`` names the dataset, empty the
    project's own) does not declare for the measured subject, or a value that attribute does not
    list. A rationale that says nothing refuses. Returns the revision as written; an audit line
    that cannot be written raises ``AuditEntryNotWrittenError`` with the revision already appended.
    """
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.subject_registry import positive_state_problem

    check_proposed_entry(entry)
    crossing = entry.operationalizations.get(STATE_CROSSING_DATES)
    if crossing is not None:
        registry = resolve_statement_registry(project, dataset_root)
        problem = positive_state_problem(registry, crossing.measured_subject,
                                         cast(PositiveState, entry.positive_state))
        if problem is not None:
            raise ValueError(
                f"the {STATE_CROSSING_DATES} operationalization of {entry.name!r} names positive "
                f"state {entry.positive_state} for subject {crossing.measured_subject!r}, and "
                f"{problem}. Name a state the registry declares, or update the registry first."
            )
    key = trait_key(project, entry.name)
    with ts.transaction(key) as txn:
        stored = txn.read(key, default=None)
        revisions = () if stored is None else _record(entry.name, project, stored).revisions
        revision = TraitRevision(
            number=len(revisions) + 1, entry=entry, entry_sha256=entry_sha256(entry),
            rationale=rationale, relayed_note=relayed_note, proposed_at=now_iso(),
            confirmed_by=None, confirmed_at=None, withdrawn_by=None, withdrawn_at=None,
        )
        txn.write(key, TraitRecord(revisions=(*revisions, revision)).model_dump(mode="json"))
    record_event_or_raise(
        "propose_trait",
        {"trait": entry.name, "revision": revision.number, "entry_sha256": revision.entry_sha256},
        actor=None, scope=project,
    )
    return revision


class RevisionMovedError(ValueError):
    """Raised when a confirmation's hash is not the hash of the revision it names."""


def confirm_revision(
    project: str | Path,
    trait: str,
    number: int,
    entry_sha256: str,
    *,
    actor: str,
    confirmed: bool,
) -> TraitRevision:
    """Record the breeder's confirmation of revision ``number`` of ``trait`` (``confirmed``), or
    the withdrawal of one they gave (not ``confirmed``), by ``actor``
    (:func:`~tcip_mcp.identity.actor`'s spelling). Neither edits the entry.

    ``entry_sha256`` is the hash of the entry the surface showed; one that is not the revision's
    own raises :class:`RevisionMovedError`. Refuses a revision that does not exist, a
    confirmation of a revision already confirmed or withdrawn, and a withdrawal of a revision
    not confirmed. Writes
    the audit line after the record; one that cannot be written raises ``AuditEntryNotWrittenError``
    with the record already written. Returns the revision as written.
    """
    from tcip_mcp.audit import record_event_or_raise

    key = trait_key(project, trait)
    with ts.transaction(key) as txn:
        revisions = list(_record(trait, project, txn.read(key, default=None)).revisions)
        if not 1 <= number <= len(revisions):
            raise ValueError(f"trait {trait!r} has revisions 1 to {len(revisions)}, not {number}")
        revision = revisions[number - 1]
        if entry_sha256 != revision.entry_sha256:
            raise RevisionMovedError(
                f"revision {number} of {trait!r} holds entry {revision.entry_sha256}, not the "
                f"{entry_sha256} that was shown; re-read it and confirm what it holds"
            )
        if confirmed and revision.confirmed_at is not None:
            raise ValueError(f"revision {number} of {trait!r} was already confirmed")
        if not confirmed and not revision.confirmed:
            raise ValueError(f"revision {number} of {trait!r} holds no confirmation to withdraw")
        stamp = ({"confirmed_by": actor, "confirmed_at": now_iso()} if confirmed
                 else {"withdrawn_by": actor, "withdrawn_at": now_iso()})
        revisions[number - 1] = revision.model_copy(update=stamp)
        txn.write(key, TraitRecord(revisions=tuple(revisions)).model_dump(mode="json"))
    record_event_or_raise(
        "confirm_trait_revision",
        {"trait": trait, "revision": number, "entry_sha256": entry_sha256,
         "confirmed": confirmed},
        actor=actor, scope=project,
    )
    return revisions[number - 1]
