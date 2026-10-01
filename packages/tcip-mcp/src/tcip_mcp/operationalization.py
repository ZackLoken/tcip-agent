"""A trait's latest confirmed revision, the one every measurement and delivery reads, and the check
of whether its operationalization binds what a delivery door is about to write."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from tcip_mcp.subject_registry import SubjectRegistry, positive_value_problem
from tcip_mcp.traits import (
    PHENOTYPE_NAMING_KINDS,
    STATE_CROSSING_DATES,
    VALUE_KEY_KINDS,
    TraitRevision,
    crops_definitions,
    read_trait,
    trait_names,
)


class OperationalizationRefused(ValueError):
    """The trait has no confirmed revision, or its confirmed revision does not state or bind what
    a delivery is about to write."""

    def as_detail(self) -> dict[str, Any]:
        """The structured refusal body a web door raises."""
        return {"kind": "operationalization", "message": str(self)}


def latest_confirmed(trait: str, project: str | Path) -> TraitRevision:
    """``trait``'s latest confirmed revision. Refuses (:class:`OperationalizationRefused`) naming
    the trait when none of its revisions is confirmed; an unknown trait raises
    ``TraitUnknownError``."""
    record = read_trait(trait, project)
    if record.latest_confirmed is None:
        raise OperationalizationRefused(
            f"Refused for trait {trait!r}: none of its {record.latest.number} revision(s) is "
            "confirmed by the breeder, so nothing states what its numbers mean. An entry the agent "
            "proposed and nobody confirmed is the agent's own definition. Ask the breeder to open "
            "the Setup tab and confirm the revision that says what they mean."
        )
    return record.latest_confirmed


def confirmed_revision(
    delivery_kind: str,
    *,
    project: str | Path,
    trait: str | None = None,
    delivered_phenotype: str | None = None,
    value_keys: Sequence[Any] | None = None,
    registry: SubjectRegistry | None = None,
) -> TraitRevision:
    """The latest confirmed revision a ``delivery_kind`` delivery ships under, bound (:func:`bind`)
    to what the door knows about the file it is about to write.

    ``trait`` names the trait (:func:`latest_confirmed`); ``None`` takes the one trait whose latest
    confirmed revision delivers ``delivered_phenotype``. Refuses
    (:class:`OperationalizationRefused`) when no trait or several qualify, when the revision states
    no operationalization for ``delivery_kind``, and when :func:`bind` refuses.
    """
    revision = (latest_confirmed(trait, project) if trait is not None
                else _revision_delivering(str(delivered_phenotype), project))
    if delivery_kind not in revision.entry.operationalizations:
        raise OperationalizationRefused(_unstated_text(revision, delivery_kind))
    bind(revision, delivery_kind, delivered_phenotype=delivered_phenotype, value_keys=value_keys,
         registry=registry)
    return revision


def bind(
    revision: TraitRevision,
    delivery_kind: str,
    *,
    delivered_phenotype: str | None = None,
    value_keys: Sequence[Any] | None = None,
    buckets: Mapping[str, Collection[str]] | None = None,
    registry: SubjectRegistry | None = None,
) -> None:
    """Refuse (:class:`OperationalizationRefused`) a delivery ``revision``'s ``delivery_kind``
    operationalization does not bind: a delivered phenotype it does not cover, a row value key
    outside its set or missing, a bucket (``buckets``: path to the object classes its detections
    are of) not counting its measured subject, or, with ``registry`` given, a positive class the
    delivered dataset's registry no longer declares."""
    entry = revision.entry
    stated = entry.operationalizations[delivery_kind]

    def refuse(what: str) -> OperationalizationRefused:
        return OperationalizationRefused(
            f"Delivery refused for trait {entry.name!r}: revision {revision.number}'s "
            f"{delivery_kind} operationalization {what}. Deliver what it covers, or propose a "
            "revision covering this delivery with propose_trait and have the breeder confirm it "
            "in the Setup tab."
        )

    if delivery_kind == STATE_CROSSING_DATES and registry is not None:
        problem = positive_value_problem(registry, stated.measured_subject, entry.positive_value)
        if problem is not None:
            raise refuse(
                f"names positive class {entry.positive_value!r} for {stated.measured_subject!r}, "
                f"which the delivered dataset's registry no longer declares: {problem}")
    if (delivered_phenotype is not None and delivery_kind in PHENOTYPE_NAMING_KINDS
            and delivered_phenotype not in stated.delivered_phenotypes):
        raise refuse(f"covers {list(stated.delivered_phenotypes)}, not {delivered_phenotype!r}")
    if value_keys is not None and delivery_kind in VALUE_KEY_KINDS:
        missing = sum(1 for key in value_keys if not key)
        if missing:
            raise refuse(f"covers value keys {list(stated.delivered_value_keys)}, and {missing} "
                         "row(s) carry no value key at all")
        offending = sorted({str(key) for key in value_keys} - set(stated.delivered_value_keys))
        if offending:
            raise refuse(f"covers value keys {list(stated.delivered_value_keys)}, and these "
                         f"rows carry {offending}")
    for bucket, subjects in (buckets or {}).items():
        if stated.measured_subject not in subjects:
            raise refuse(f"measures {stated.measured_subject!r}, and bucket {bucket} counted "
                         f"{sorted(subjects)}")


def _revision_delivering(delivered_phenotype: str, project: str | Path) -> TraitRevision:
    """The latest confirmed revision of the one trait whose latest confirmed revision delivers
    ``delivered_phenotype``; refuses (:class:`OperationalizationRefused`) naming the phenotype,
    the confirmed traits delivering it and the traits proposing it when not exactly one does."""
    confirmed: list[TraitRevision] = []
    proposed: list[str] = []
    for name in trait_names(project):
        record = read_trait(name, project)
        if record.latest_confirmed is not None and (
                delivered_phenotype in record.latest_confirmed.entry.delivers):
            confirmed.append(record.latest_confirmed)
        elif delivered_phenotype in record.latest.entry.delivers:
            proposed.append(name)
    if len(confirmed) == 1:
        return confirmed[0]
    raise OperationalizationRefused(
        f"Delivery refused for phenotype {delivered_phenotype!r}: "
        f"{len(confirmed)} traits with a confirmed revision deliver it "
        f"({[r.entry.name for r in confirmed]}), and {proposed} deliver it only in a revision "
        "nobody confirmed, so no single operationalization answers for this number. Propose one "
        "trait delivering it with propose_trait, have the breeder confirm it in the Setup tab, "
        "and deliver again."
    )


def _unstated_text(revision: TraitRevision, delivery_kind: str) -> str:
    entry = revision.entry
    definitions = crops_definitions()
    quoted = "; ".join(f"{n}: {definitions[n]}" for n in entry.delivers if n in definitions)
    vocabulary = f", defined in the crop vocabulary as {quoted}" if quoted else ""
    return (
        f"Delivery refused for trait {entry.name!r}: its confirmed revision {revision.number} "
        f"states no operationalization for a {delivery_kind} delivery. This trait delivers "
        f"{', '.join(entry.delivers)}{vocabulary}. That is a field criterion, not something a "
        "model realizes on its own: ask the breeder what this number should mean and what decides "
        "it in the imagery, propose the entry with that operationalization through propose_trait, "
        "and have the breeder confirm it in the Setup tab. Do not supply the meaning yourself."
    )
