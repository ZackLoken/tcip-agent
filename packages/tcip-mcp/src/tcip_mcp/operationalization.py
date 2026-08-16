"""What a trait's delivered number means, recorded per project and confirmed by the breeder.

``crops.yml`` gives a breeder's field criterion for each delivered phenotype, which is not
something a model can realize on its own: which object, what visual call, and what a fraction is a
fraction of are all unstated there. This module holds the per-project record that states those
things, the one check a delivery door runs before it writes, and the confirmation writer the web
backend calls.

The record never copies a ``TraitSpec`` value as an operative one. Every consumer of
``positive_class_name``, ``count_objective`` and the rest keeps reading the spec, unchanged. What
the record stores is a snapshot of the fields the confirmation covered, read for one purpose:
detecting that they moved since.

Two writers, two identities. :func:`state_operationalization` records what the agent and the
breeder worked out and stamps the surface it came through; :func:`confirm_trait_operationalization`
is called only by the web backend, from a route the breeder's own action posts. No MCP tool writes
a confirmation field, and a statement write refuses a payload carrying one.

Nothing here authenticates anybody. ``confirmed_by`` holds a name the request supplied, and the
record is an ordinary file. What this buys is that a delivered number names the definition it rests
on and the person recorded as confirming it, not that either one is verified.

Every refusal text below names the trait, the delivery kind, which half is missing and the
primitive that fixes it, and none of them proposes a positive class, a crossing fraction or a
mechanism: a suggested answer becomes the answer, and the meaning is the breeder's to give.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import tcip_store as ts
from tcip_store import RECORD_JSON, Key, StoreDescriptor, Version, Versioned, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.identity import user_identity
from tcip_mcp.traits import TraitSpec, crops_definitions, get_trait_for, trait_specs_dir

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
"""The delivered artifact shapes, one per door group, each with its own required content.

This is a taxonomy with membership rules, not a bag of constants: which spec fields constitute an
operationalization, whether the record names delivered phenotypes, and whether it names value keys
all follow from the kind. A sixth member is added deliberately, with its own three answers.
"""

_CONSTITUTING_FIELDS: dict[str, tuple[str, ...]] = {
    STATE_CROSSING_DATES: ("positive_class_name", "milestone_on", "milestone_fractions"),
    PER_IMAGE_COUNT: ("count_objective", "localization"),
    PER_PLANT_COUNT_AGGREGATE: ("count_objective",),
    PER_PLANT_ORDINAL_AGGREGATE: ("ordinal_agreement_floor",),
    PER_PLANT_REGRESSION_AGGREGATE: ("regression_skill_floor",),
}

_PHENOTYPE_NAMING_KINDS = frozenset({
    STATE_CROSSING_DATES,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
})
"""Kinds whose delivered file carries a phenotype name, hence whose record binds one.

A per-image count CSV names no phenotype in any column, so a delivered-phenotype field on that
record would be a claim about the file rather than a binding checkable against it.
"""

_VALUE_KEY_KINDS = frozenset({
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
})
"""Kinds whose rows carry a caller-supplied value key, hence whose record bounds the set."""

STATEMENT_FIELDS = (
    "statement",
    "mechanism",
    "measured_subject",
    "delivered_phenotypes",
    "delivered_value_keys",
    "stated_by",
    "stated_at",
    "relayed_note",
)
"""Every field a statement owns. The confirmation compare-and-set hashes all of them."""

CONFIRMATION_FIELDS = ("confirmed_by", "confirmed_at", "identity_from_request", "confirmed_fields")
"""Every field only the confirmation writer sets. A statement write refuses a payload with one."""

STATEMENT_SURFACE = "state_trait_operationalization"
"""The producing surface stamped into ``stated_by``, never accepted from a caller.

It says a statement came in through the statement tool rather than through a file edit, and
nothing more. A stdio MCP server carries no caller identity, so this is not evidence of who wrote
the record, and anyone writing the file directly can write the same value.
"""


def constituting_fields(delivery_kind: str) -> tuple[str, ...]:
    """The ``TraitSpec`` fields a confirmation of ``delivery_kind`` covers.

    A pure function of the record's own key, called by the statement writer, the confirmation
    writer and the precondition alike, so a stater cannot narrow the set and a reader cannot widen
    it.
    """
    fields = _CONSTITUTING_FIELDS.get(delivery_kind)
    if fields is None:
        raise ValueError(
            f"unknown delivery kind {delivery_kind!r}; the kinds are {list(DELIVERY_KINDS)}"
        )
    return fields


def canonical(value: Any) -> Any:
    """One comparable form for a stored or live value, so JSON round-tripping is not a difference.

    Sequences become lists recursively and mapping keys are sorted; scalars are left alone. A
    ``TraitSpec`` tuple field written into a JSON record reads back as a list, so a raw comparison
    would report a field as moved seconds after it was confirmed.
    """
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [canonical(item) for item in value]
    return value


# ── the record store ─────────────────────────────────────────────────────────

OPERATIONALIZATIONS_STORE = "trait_operationalizations"
_RECORD_FILE = RootedFileLocator(prefix=("trait_operationalizations",), suffix=".json")
register_store(
    StoreDescriptor(
        name=OPERATIONALIZATIONS_STORE,
        kind="record",
        key_fields=("trait", "delivery_kind"),
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_RECORD_FILE,
    )
)

_STATE_RELPATH = Path(".tcip") / "state"


def operationalizations_scope(project_root: str | Path | None = None) -> Path:
    """Where a project's operationalization records live: ``<root>/.tcip/state``.

    The one implementation of that placement, beside the trait specs the records reference.
    ``project_root`` names the project explicitly, for a caller serving more than one project per
    process; omitting it resolves against this process's pinned platform root.
    """
    if project_root is not None:
        return Path(project_root) / _STATE_RELPATH
    from tcip_mcp.project_paths import resolve_state

    return resolve_state(_STATE_RELPATH)


def operationalization_key(scope: str | Path, trait: str, delivery_kind: str) -> Key:
    """One trait's record for one delivery kind.

    ``cas``: the statement writer and the confirmation writer are two processes writing the same
    record, so an unconditional write would drop whichever one landed first.
    """
    return Key(OPERATIONALIZATIONS_STORE, str(scope), (trait, delivery_kind))


class ResolvedOperationalization(NamedTuple):
    """A trait's spec, its record for one delivery kind, and the registry both came from."""

    spec: TraitSpec
    record: Versioned
    specs_dir: Path


def resolve_trait_and_record(
    trait: str, delivery_kind: str, *, project_root: str | Path | None = None
) -> ResolvedOperationalization:
    """The one way a door obtains a spec and its operationalization record.

    Both come from one call so they cannot come from two different roots: the backend pins a
    process-wide platform root at startup and repins it on project adoption, so a request naming
    its own project can otherwise write a confirmation to one file and look for it in another.

    Raises ``TraitUnknownError`` when the trait is not registered for this project, and
    ``ValueError`` for an unknown delivery kind. The record's value is ``None`` when nothing is
    stated, and it is read versioned so a door can prove the record it checked is the record it
    delivered against.
    """
    constituting_fields(delivery_kind)
    spec = get_trait_for(trait, project_root)
    key = operationalization_key(operationalizations_scope(project_root), trait, delivery_kind)
    return ResolvedOperationalization(
        spec=spec,
        record=ts.read_versioned(key, default=None),
        specs_dir=trait_specs_dir(project_root),
    )


# ── the precondition ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OperationalizationBasis:
    """What a passing check rested on, so a door can prove it still rests on it at write time.

    A door checks once, does its work, and checks again immediately before its first write, passing
    what the first check returned. The record's version token catches a restatement or a
    withdrawal; the canonical constituting values catch a spec edit. Two keys in two stores cannot
    be read atomically together on the file backend, so this is what closes the window instead.
    """

    record_version: Version
    constituting: dict[str, Any]


@dataclass(frozen=True)
class OperationalizationCheck:
    """The precondition's answer: which failure state applies, its refusal, and its basis."""

    trait: str
    delivery_kind: str
    state: int | None
    message: str = ""
    basis: OperationalizationBasis | None = None
    superseded: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the delivery may proceed. Derived, so it cannot disagree with ``state``."""
        return self.state is None

    def as_detail(self) -> dict[str, Any]:
        """The structured refusal body a web door raises, so no reader dispatches on prose."""
        return {
            "kind": "operationalization",
            "state": self.state,
            "trait": self.trait,
            "delivery_kind": self.delivery_kind,
            "message": self.message,
        }


def _is_empty(value: Any) -> bool:
    """Whether a spec field holds nothing. Zero and false are values, not absences."""
    return value is None or (isinstance(value, (str, bytes, tuple, list, dict, set)) and not value)


def _live_constituting(spec: TraitSpec, delivery_kind: str) -> dict[str, Any]:
    """The canonical values the live spec holds for the fields this kind rests on."""
    return {field: canonical(getattr(spec, field)) for field in constituting_fields(delivery_kind)}


def _moved_fields(
    spec: TraitSpec, record: Mapping[str, Any], delivery_kind: str
) -> tuple[dict[str, Any], ...]:
    """Every constituting field whose live value differs from the one the confirmation covered.

    The one comparison behind both failure state 3 and the edit-time supersession signal, so the
    refusal and the convenience report cannot disagree about what moved.
    """
    confirmed = record.get("confirmed_fields") or {}
    live = _live_constituting(spec, delivery_kind)
    moved = []
    for field, current in live.items():
        was = canonical(confirmed.get(field))
        if was != current:
            moved.append({"field": field, "confirmed_value": was, "current_value": current})
    return tuple(moved)


def check_operationalization(
    spec: TraitSpec,
    record: Versioned,
    delivery_kind: str,
    *,
    delivered_phenotype: str | None = None,
    value_keys: Sequence[Any] | None = None,
    id_maps: Mapping[str, Mapping[str, Any]] | None = None,
    basis: OperationalizationBasis | None = None,
) -> OperationalizationCheck:
    """Whether this trait's delivered number has a confirmed meaning, and what to say if not.

    The six failure states are checked in the order they are numbered and the first one reports
    alone, because a number with no defined meaning has nothing for a later check to test. This
    answers whether there is a defined quantity at all; ``check_delivery_gate`` answers whether the
    quantity's error is characterized against a reference. The two refusals are never blurred.

    ``delivered_phenotype``, ``value_keys`` and ``id_maps`` are what a door knows about the file it
    is about to write, and each is checked only where the delivered artifact carries it. ``basis``
    is what an earlier call returned: pass it on the re-check immediately before the first write,
    and a record or spec that moved in between refuses with state 6 rather than delivering against
    a confirmation nobody gave.

    ``acknowledge_unvalidated`` is deliberately not a parameter. An acknowledged provisional number
    whose meaning is stated is honest and ships stamped false; an acknowledged number whose meaning
    is unstated is not a number.
    """
    trait = spec.name
    stated = record.value

    if not stated:
        return OperationalizationCheck(trait, delivery_kind, 1, _state_1_text(spec, delivery_kind))

    if not stated.get("confirmed_by"):
        return OperationalizationCheck(
            trait, delivery_kind, 2, _state_2_text(spec, delivery_kind, stated)
        )

    moved = _moved_fields(spec, stated, delivery_kind)
    if moved:
        return OperationalizationCheck(
            trait, delivery_kind, 3, _state_3_text(spec, delivery_kind, stated, moved[0]),
            superseded=moved,
        )

    empty = [f for f in constituting_fields(delivery_kind) if _is_empty(getattr(spec, f))]
    if empty:
        return OperationalizationCheck(
            trait, delivery_kind, 4, _state_4_text(spec, delivery_kind, empty[0])
        )

    binding = _check_bindings(
        spec, stated, delivery_kind,
        delivered_phenotype=delivered_phenotype, value_keys=value_keys, id_maps=id_maps,
    )
    if binding is not None:
        return OperationalizationCheck(trait, delivery_kind, 5, binding)

    current = OperationalizationBasis(
        record_version=record.version, constituting=_live_constituting(spec, delivery_kind)
    )
    if basis is not None and basis != current:
        return OperationalizationCheck(
            trait, delivery_kind, 6, _state_6_text(spec, delivery_kind, basis, current)
        )
    return OperationalizationCheck(trait, delivery_kind, None, basis=current)


def _check_bindings(
    spec: TraitSpec,
    stated: Mapping[str, Any],
    delivery_kind: str,
    *,
    delivered_phenotype: str | None,
    value_keys: Sequence[Any] | None,
    id_maps: Mapping[str, Mapping[str, Any]] | None,
) -> str | None:
    """The refusal for whichever delivered binding fails, or ``None`` when they all hold."""
    covered_phenotypes = tuple(stated.get("delivered_phenotypes") or ())
    if (
        delivered_phenotype is not None
        and delivery_kind in _PHENOTYPE_NAMING_KINDS
        and delivered_phenotype not in covered_phenotypes
    ):
        return _state_5_phenotype_text(spec, delivery_kind, delivered_phenotype, covered_phenotypes)

    covered_keys = tuple(stated.get("delivered_value_keys") or ())
    if value_keys is not None and delivery_kind in _VALUE_KEY_KINDS:
        missing = sum(1 for key in value_keys if not key)
        if missing:
            return _state_5_missing_key_text(spec, missing, covered_keys)
        offending = sorted({str(key) for key in value_keys} - set(covered_keys))
        if offending:
            return _state_5_value_key_text(spec, delivery_kind, offending, covered_keys)

    if id_maps is not None and delivery_kind == PER_IMAGE_COUNT:
        subject = str(stated.get("measured_subject") or "")
        if not any(subject in (id_map or {}) for id_map in id_maps.values()):
            return _state_5_subject_text(spec, subject, sorted(id_maps))
    return None


# ── refusal texts ────────────────────────────────────────────────────────────


def _delivered_names(spec: TraitSpec) -> str:
    return ", ".join(spec.delivers)


def _delivered_definitions(spec: TraitSpec) -> str:
    """The crop vocabulary's own wording for what this trait delivers, quoted, never paraphrased."""
    definitions = crops_definitions()
    quoted = [f"{name}: {definitions[name]}" for name in spec.delivers if name in definitions]
    return "; ".join(quoted)


def _statement_call_form(delivery_kind: str) -> str:
    """The statement call with the arguments this kind actually requires."""
    phenotypes = "[...]" if delivery_kind in _PHENOTYPE_NAMING_KINDS else "[]"
    value_keys = ", delivered_value_keys=[...]" if delivery_kind in _VALUE_KEY_KINDS else ""
    return (
        f"state_trait_operationalization(project_root=..., trait=..., "
        f"delivery_kind='{delivery_kind}', statement=..., mechanism=..., measured_subject=..., "
        f"delivered_phenotypes={phenotypes}{value_keys})"
    )


def _state_1_text(spec: TraitSpec, delivery_kind: str) -> str:
    definitions = _delivered_definitions(spec)
    vocabulary = f", defined in the crop vocabulary as {definitions}" if definitions else ""
    return (
        f"Delivery refused for trait {spec.name!r}: no operationalization is recorded for a "
        f"{delivery_kind} delivery. This trait delivers {_delivered_names(spec)}{vocabulary}. That "
        "definition is a field criterion, not something a model can realize on its own: what the "
        "delivered number means, and what decides it in the imagery, have to be stated before any "
        "number can be delivered. Ask the breeder what this number should mean in their terms, "
        "decide with them what produces the call, then record both with "
        f"{_statement_call_form(delivery_kind)}. Do not supply the meaning yourself. The breeder "
        "then confirms it in the Results tab and this delivery proceeds unchanged."
    )


def _state_2_text(spec: TraitSpec, delivery_kind: str, stated: Mapping[str, Any]) -> str:
    text = (
        f"Delivery refused for trait {spec.name!r}: its operationalization for a {delivery_kind} "
        "delivery is stated but not confirmed by the breeder. On record, recorded through "
        f"{stated.get('stated_by')} on {stated.get('stated_at')}: {stated.get('statement')}. "
        f"Decided by: {stated.get('mechanism')}. An operationalization the agent wrote and nobody "
        "confirmed is the agent's own definition, which is not what this delivery may rest on. Ask "
        f"them to open the Results tab and confirm the statement recorded for {spec.name!r} there. "
        "If it no longer matches what they mean, restate it with state_trait_operationalization "
        "first."
    )
    relayed = stated.get("relayed_note")
    if relayed:
        text += (
            f" A relayed confirmation is recorded on this statement: {relayed}. It does not clear "
            "this refusal; the confirming act happens in the Results tab."
        )
    return text


def _state_3_text(
    spec: TraitSpec, delivery_kind: str, stated: Mapping[str, Any], moved: Mapping[str, Any]
) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: its operationalization for a {delivery_kind} "
        f"delivery was confirmed by {stated.get('confirmed_by')} on {stated.get('confirmed_at')}, "
        f"and {moved['field']} has changed since, confirmed as {moved['confirmed_value']!r}, now "
        f"{moved['current_value']!r}. The confirmation covered the old value, so it no longer "
        "covers what would be delivered. Ask the breeder to confirm the current statement in the "
        "Results tab, or restore the confirmed value."
    )


def _state_4_text(spec: TraitSpec, delivery_kind: str, field: str) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: its confirmed operationalization for a "
        f"{delivery_kind} delivery rests on {field}, which this trait's spec leaves empty. Nothing "
        "resolves the call the statement describes, so there is no measurement to deliver. Record "
        "the breeder's value with update_trait_spec_fields(project_root=..., "
        f"trait_name={spec.name!r}, ...), then restate and re-confirm."
    )


def _state_5_value_key_text(
    spec: TraitSpec, delivery_kind: str, offending: Sequence[str], covered: Sequence[str]
) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: its confirmed operationalization for a "
        f"{delivery_kind} delivery covers value keys {list(covered)}, and these rows carry "
        f"{list(offending)}. A quantity the breeder never confirmed cannot ship under this trait's "
        "name. Deliver the confirmed quantity, or state and confirm an operationalization that "
        "covers what is actually being aggregated."
    )


def _state_5_missing_key_text(spec: TraitSpec, missing: int, covered: Sequence[str]) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: {missing} of these rows carry no value key, so "
        f"there is nothing to check against the confirmed operationalization's {list(covered)}. A "
        "row with no stated quantity cannot ship under a trait's name. Give every row the value "
        "key naming what it holds."
    )


def _state_5_phenotype_text(
    spec: TraitSpec, delivery_kind: str, delivered: str, covered: Sequence[str]
) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: this CSV would ship under the delivered "
        f"phenotype {delivered}, and the confirmed operationalization for a {delivery_kind} "
        f"delivery covers {list(covered)}. Deliver under a confirmed phenotype, or state and "
        "confirm an operationalization covering this one."
    )


def _state_5_subject_text(spec: TraitSpec, subject: str, buckets: Sequence[str]) -> str:
    return (
        f"Delivery refused for trait {spec.name!r}: its confirmed operationalization names "
        f"{subject} as the measured subject, and no prediction bucket in this delivery recorded "
        f"that subject in its id map ({list(buckets)}). The counts are of something else. Deliver "
        f"from buckets that detected {subject}, or state and confirm an operationalization for the "
        "subject these buckets actually carry."
    )


def _state_6_text(
    spec: TraitSpec,
    delivery_kind: str,
    basis: OperationalizationBasis,
    current: OperationalizationBasis,
) -> str:
    if basis.record_version != current.record_version:
        what = "the record itself was rewritten, so it is no longer the statement that was checked"
    else:
        changed = sorted(
            field for field, value in current.constituting.items()
            if basis.constituting.get(field) != value
        )
        what = f"the trait's spec moved {changed}"
    return (
        f"Delivery refused for trait {spec.name!r}: the confirmed operationalization for a "
        f"{delivery_kind} delivery changed while this delivery was being produced. What was "
        f"checked at the start no longer matches what is on record now ({what}). Nothing was "
        "written. Re-read the record, and if it still says what the breeder means, run this "
        "delivery again."
    )


# ── the statement writer ─────────────────────────────────────────────────────


def record_seen_hash(record: Mapping[str, Any]) -> str:
    """A content hash over every field a statement owns, in canonical form.

    The confirmation carries this back, so a breeder's click confirms the record they read rather
    than whatever an agent rewrote while the card was open. It covers all eight statement fields:
    leaving the statement text alone while changing the measured subject or the value keys would
    otherwise harvest a click for content nobody saw.
    """
    payload = {field: canonical(record.get(field)) for field in STATEMENT_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            f"{field} is required and must say something: an operationalization with an empty "
            f"{field} states nothing, and a delivery cannot rest on it"
        )
    return text


def state_operationalization(
    project_root: str | Path,
    trait: str,
    delivery_kind: str,
    *,
    statement: str,
    mechanism: str,
    measured_subject: str,
    delivered_phenotypes: Sequence[str] = (),
    delivered_value_keys: Sequence[str] = (),
    relayed_note: str = "",
    **payload: Any,
) -> dict[str, Any]:
    """Record what a trait's delivered number means for one delivery kind, unconfirmed.

    The writer behind the ``state_trait_operationalization`` tool, and the only path that writes a
    statement. It stamps ``stated_by`` and ``stated_at`` itself and refuses any further payload
    key, naming the confirmation fields when one of those is what arrived: a writer that could fill
    a confirmation field would make honest attribution depend on the agent choosing not to.

    Restating clears the confirmation. A changed definition is unconfirmed by construction, and the
    breeder confirms the new one on the surface where the confirming act happens.
    """
    if payload:
        offered = sorted(payload)
        confirmation = [key for key in offered if key in CONFIRMATION_FIELDS]
        if confirmation:
            raise ValueError(
                f"a statement cannot carry the confirmation field(s) {confirmation}: only the "
                "breeder's own confirmation writes those, from the web backend, and a statement "
                "that filled them would be the agent confirming its own definition"
            )
        raise ValueError(f"unknown statement field(s) {offered} for {OPERATIONALIZATIONS_STORE}")

    spec, existing, _specs_dir = resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root
    )
    phenotypes = [str(name) for name in delivered_phenotypes]
    value_keys = [str(key) for key in delivered_value_keys]

    off_spec = [name for name in phenotypes if name not in spec.delivers]
    if off_spec:
        raise ValueError(
            f"delivered_phenotypes {off_spec} are not in trait {trait!r}'s own delivers "
            f"{list(spec.delivers)}; a record cannot cover a phenotype the spec does not deliver"
        )
    if delivery_kind in _PHENOTYPE_NAMING_KINDS and not phenotypes:
        raise ValueError(
            f"a {delivery_kind} delivery ships under a phenotype name, so delivered_phenotypes "
            f"must name at least one of trait {trait!r}'s delivers {list(spec.delivers)}"
        )
    if delivery_kind not in _PHENOTYPE_NAMING_KINDS and phenotypes:
        raise ValueError(
            f"a {delivery_kind} delivery names no phenotype in any column, so "
            "delivered_phenotypes must be empty; a name the file never carries is a claim rather "
            "than a binding"
        )
    if delivery_kind in _VALUE_KEY_KINDS and not value_keys:
        raise ValueError(
            f"a {delivery_kind} delivery's rows each carry a value key, so delivered_value_keys "
            "must name the ones this record covers"
        )
    if delivery_kind not in _VALUE_KEY_KINDS and value_keys:
        raise ValueError(
            f"a {delivery_kind} delivery's row schema is fixed by its writer, so "
            "delivered_value_keys must be empty"
        )

    record = {
        "trait": trait,
        "delivery_kind": delivery_kind,
        "statement": _require_text(statement, "statement"),
        "mechanism": _require_text(mechanism, "mechanism"),
        "measured_subject": _require_text(measured_subject, "measured_subject"),
        "delivered_phenotypes": phenotypes,
        "delivered_value_keys": value_keys,
        "stated_by": STATEMENT_SURFACE,
        "stated_at": _now(),
        "relayed_note": str(relayed_note or ""),
        **{field: None for field in CONFIRMATION_FIELDS},
    }
    key = operationalization_key(operationalizations_scope(project_root), trait, delivery_kind)
    ts.replace(key, record, expect=existing.version)
    return record


# ── the confirmation writer ──────────────────────────────────────────────────


class NothingStated(ValueError):
    """Raised when a confirmation arrives for a trait and kind nothing has been stated for."""


class RecordMoved(ValueError):
    """Raised when the record moved since the surface read it, carrying what is on file now."""

    def __init__(self, message: str, record: Mapping[str, Any], record_seen: str) -> None:
        super().__init__(message)
        self.record = dict(record)
        self.record_seen = record_seen


def confirm_trait_operationalization(
    project_root: str | Path,
    trait: str,
    delivery_kind: str,
    *,
    user: str,
    record_seen: str,
    identity_from_request: bool,
    confirmed: bool = True,
) -> dict[str, Any]:
    """Record that the breeder confirmed what is on file, or withdraw a confirmation.

    Exposed by no MCP tool and called only by the web backend, from a route the breeder's own GUI
    action posts. ``record_seen`` is :func:`record_seen_hash` over the record the surface rendered,
    compared against what is on file now, so a click cannot land on text the breeder never read.
    ``identity_from_request`` says whether the confirming request carried a name or the backend
    fell back to its process environment; the caller passes it because only the caller observed it.

    ``confirmed=False`` withdraws, clearing exactly the four confirmation fields and leaving the
    statement intact.
    """
    spec, existing, _specs_dir = resolve_trait_and_record(
        trait, delivery_kind, project_root=project_root
    )
    stated = existing.value
    if not stated:
        raise NothingStated(
            f"nothing is stated for trait {trait!r} and a {delivery_kind} delivery, so there is no "
            "operationalization to confirm; the agent records one with "
            "state_trait_operationalization first"
        )
    current_seen = record_seen_hash(stated)
    if record_seen != current_seen:
        raise RecordMoved(
            f"the operationalization for trait {trait!r} and a {delivery_kind} delivery moved "
            "since it was read, so confirming now would confirm text nobody displayed; re-read it "
            "and confirm what is on file",
            stated,
            current_seen,
        )

    updated = dict(stated)
    if confirmed:
        updated.update({
            "confirmed_by": user_identity(user),
            "confirmed_at": _now(),
            "identity_from_request": bool(identity_from_request),
            "confirmed_fields": _live_constituting(spec, delivery_kind),
        })
    else:
        updated.update({field: None for field in CONFIRMATION_FIELDS})

    key = operationalization_key(operationalizations_scope(project_root), trait, delivery_kind)
    ts.replace(key, updated, expect=existing.version)
    return updated


# ── the edit-time supersession signal ────────────────────────────────────────


def superseded_confirmations(
    project_root: str | Path, trait: str, *, spec: TraitSpec | None = None
) -> list[dict[str, Any]]:
    """Every confirmed record of ``trait`` a spec edit has just moved out from under, and how.

    A convenience for the spec writer's return value, never enforcement: the precondition's
    read-time comparison is the one place a superseded confirmation stops a delivery. This exists
    so the agent learns at edit time rather than at the next refusal.
    """
    scope = operationalizations_scope(project_root)
    superseded: list[dict[str, Any]] = []
    for key in ts.keys(OPERATIONALIZATIONS_STORE, str(scope), (trait,)):
        delivery_kind = key.parts[1]
        if delivery_kind not in _CONSTITUTING_FIELDS:
            continue
        record = ts.read(key, default=None)
        if not record or not record.get("confirmed_by"):
            continue
        resolved = spec if spec is not None else get_trait_for(trait, project_root)
        for moved in _moved_fields(resolved, record, delivery_kind):
            superseded.append({"delivery_kind": delivery_kind, **moved})
    return superseded
