"""Trait entries a test needs, proposed with ``traits.propose_trait`` and confirmed with
``traits.confirm_revision``: a crossing trait delivering bloom dates, a count trait delivering a
stem count, the vocabulary phenotypes the aggregate deliveries ship under, and ``bud_opening``,
whose name is not the ``bud`` subject it measures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tcip_mcp import subject_registry as cr
from tcip_mcp import traits
from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitEntry, TraitRevision


def entry(name: str, delivers: Sequence[str], **fields: Any) -> TraitEntry:
    """A complete fixture entry: ``fields`` over an entry that authors nothing beyond its name and
    what it delivers."""
    return TraitEntry.model_validate({
        "name": name, "delivers": tuple(delivers), "positive_state": None,
        "milestone_fractions": (), "milestone_on": "", "majority_milestone": "",
        "phenology_prefix": "", "majority_label": "",
        "count_objective": "", "localization": "", "localization_tolerance": "half_class_avg_size",
        "localization_tolerance_frac": 0.5, "count_bias_tolerance_frac": None,
        "count_error_tolerance": None, "classifier_agreement_floor": None,
        "ordinal_agreement_floor": None, "regression_skill_floor": None,
        "regression_criterion": "", "scale_tolerance_frac": None, "holdout_match_quality_floor": None, "notes": "",
        "operationalizations": {}, **fields,
    })


def with_fields(base: TraitEntry, **fields: Any) -> TraitEntry:
    """``base`` with ``fields`` replaced, validated as a new entry."""
    return TraitEntry.model_validate({**base.model_dump(), **fields})


BUD_OPENING = entry(
    "bud_opening", ("leaf_out_05per_date", "leaf_out_50per_date"),
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    holdout_match_quality_floor=0.5,  # fixture value: loose enough for the synthetic dense references to clear
    positive_state={"attribute": "opening", "value": "open"},
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    majority_milestone="95per",
    phenology_prefix="bud",
    majority_label="majority",
    notes="The fraction of a plant's bud objects that are open: a texture call on the object "
          "itself, never a bbox-ratio proxy.",
)

CROSSING_TRAIT = "bloom"
COUNT_TRAIT = "stem"
COUNT_SUBJECT = "stem"
"""What a count operationalization made from :data:`COUNT_SPEC` says the counts are counts of."""

_FLOORS: dict[str, Any] = {
    "count_objective": COUNT_UNBIASED, "localization": CENTER_MATCH,
    "count_bias_tolerance_frac": 0.1, "count_error_tolerance": 1.0,
    "classifier_agreement_floor": 0.6, "ordinal_agreement_floor": 0.6,
    "regression_skill_floor": 0.5, "regression_criterion": "r_squared",
    "holdout_match_quality_floor": 0.5,
    "scale_tolerance_frac": 0.1,
}
"""Every floor filled with a fixture value, so one entry serves whichever delivery kind a test
exercises."""

COUNT_SPEC = entry(COUNT_TRAIT, ("stem_count",), **_FLOORS)


def with_floors(base: TraitEntry) -> TraitEntry:
    """``base`` with each criterion field it leaves unauthored filled from :data:`_FLOORS`, so an
    operationalization proposed on it states every field its kind rests on."""
    return with_fields(base, **{k: v for k, v in _FLOORS.items()
                                if getattr(base, k) in (None, "", ())})


CROSSING_SPEC = with_floors(entry(
    CROSSING_TRAIT, ("bloom_05per_date", "bloom_50per_date"),
    positive_state={"attribute": "state", "value": "open"}, milestone_fractions=(0.05, 0.50),
    milestone_on="positive_fraction",
    phenology_prefix="bloom",
))

DELIVERY_SPECS = (
    COUNT_SPEC,
    *(entry(p, (p,), **_FLOORS)
      for p in ("astringency", "fruit_diameter", "plant_surface_area", "bark_thickness")),
)
"""Every trait the count and aggregate delivery tests ship under; no phenotype is delivered by two
of them, since a phenotype two traits deliver has no one operationalization behind it."""

DELIVERY_TRAIT_BY_PHENOTYPE = {
    phenotype: spec.name for spec in DELIVERY_SPECS for phenotype in spec.delivers
}


def propose(project_root: Path, proposed: TraitEntry, *, dataset_root: str = "") -> TraitRevision:
    """Append ``proposed`` as a new, unconfirmed revision of its trait."""
    return traits.propose_trait(
        project_root, proposed, rationale="fixture proposal", relayed_note="",
        dataset_root=dataset_root)


def confirm(project_root: Path, revision: TraitRevision, *, user: str = "grüne") -> TraitRevision:
    """Confirm ``revision`` the way the Setup tab posts it: by number and the hash it showed."""
    return traits.confirm_revision(
        project_root, revision.entry.name, revision.number, revision.entry_sha256,
        user=user, confirmed=True)


def propose_and_confirm(project_root: Path, proposed: TraitEntry) -> TraitRevision:
    """Propose ``proposed`` and confirm the revision it became."""
    return confirm(project_root, propose(project_root, proposed))


def confirm_bare(project_root: Path, name: str, **fields: Any) -> TraitRevision:
    """Confirm a minimal trait delivering one dimension at ``project_root``, stating nothing
    beyond ``fields``."""
    return propose_and_confirm(project_root, entry(name, ("leaf_length",), **fields))


def latest(trait: str, project_root: Path) -> TraitEntry:
    """The entry of ``trait``'s latest revision, confirmed or not, to build the next one from."""
    return traits.read_trait(trait, project_root).latest.entry


def count_revision(project_root: Path) -> TraitRevision:
    """The count trait's latest confirmed revision, the one a count delivery ships under."""
    from tcip_mcp.operationalization import latest_confirmed

    return latest_confirmed(COUNT_TRAIT, project_root)


def operationalization(**fields: Any) -> dict[str, Any]:
    """One operationalization, its three texts filled with fixture wording unless given."""
    return {
        "statement": "the number the breeder records for one plant",
        "mechanism": "the calibrated model at the derived operating point",
        "measured_subject": COUNT_SUBJECT, "delivered_phenotypes": (),
        "delivered_value_keys": (), **fields,
    }


def with_operationalization(base: TraitEntry, kind: str, **fields: Any) -> TraitEntry:
    """``base`` with an operationalization for ``kind`` added or replaced."""
    stated = {k: v.model_dump() for k, v in base.operationalizations.items()}
    stated[kind] = operationalization(**fields)
    return with_fields(base, operationalizations=stated)


def seed_positive_class(project_root: Path, subject_name: str,
                        state: traits.PositiveState | None) -> cr.SubjectRegistry:
    """Ensure the project's subject registry declares ``state``'s attribute on ``subject_name``,
    listing ``state``'s value, adding the subject, the attribute and the value on first mention and
    leaving an existing declaration alone; returns the registry as stored."""
    from tests._producer_fixtures import registry_over

    registry = cr.registry_for_dataset_root(project_root) or cr.SubjectRegistry()
    subjects = {s.name: s for s in registry.subjects}
    existing = subjects.get(subject_name)
    attrs = {a.name: a for a in (existing.attributes if existing else ())}
    if state is not None:
        attr = attrs.get(state.attribute)
        if attr is None:
            attrs[state.attribute] = cr.Attribute(name=state.attribute, type="categorical",
                                                  values=(state.value,))
        elif state.value not in attr.values:
            attrs[state.attribute] = cr.Attribute(name=attr.name, type=attr.type,
                                                  values=(*attr.values, state.value))
    subjects[subject_name] = cr.Subject(name=subject_name, attributes=tuple(attrs.values()))
    updated = cr.SubjectRegistry(subjects=tuple(subjects.values()))
    registry_over(project_root, updated)
    return updated


def seed_confirmed_crossing(project_root: Path, trait: str, **fields: Any) -> TraitRevision:
    """Confirm a revision of ``trait`` (already proposed at this root) stating a crossing
    operationalization, declaring its positive state for the measured subject in the project's
    own subject registry first. ``fields`` override the operationalization's defaults: the
    measured subject is the trait's own name and the phenotypes are everything it delivers."""
    base = with_floors(latest(trait, project_root))
    stated = {"statement": f"the date each plant reached the state {trait} scores in the field",
              "mechanism": "the calibrated attribute head over one plant's objects",
              "measured_subject": trait, "delivered_phenotypes": base.delivers, **fields}
    seed_positive_class(project_root, stated["measured_subject"], base.positive_state)
    return propose_and_confirm(
        project_root, with_operationalization(base, traits.STATE_CROSSING_DATES, **stated))


def seed_confirmed_count(
    project_root: Path, *, measured_subject: str = COUNT_SUBJECT, **fields: Any
) -> TraitRevision:
    """Confirm the count trait at this root with a per-image-count operationalization measuring
    ``measured_subject``, the subject the delivery's buckets recorded detecting."""
    return propose_and_confirm(project_root, with_operationalization(
        COUNT_SPEC, traits.PER_IMAGE_COUNT, statement="how many stems the model finds in one frame",
        measured_subject=measured_subject, **fields))


def seed_delivery_traits(project_root: Path) -> Path:
    """Propose and confirm every trait the count and aggregate delivery tests deliver under."""
    for spec in DELIVERY_SPECS:
        propose_and_confirm(project_root, spec)
    return Path(project_root)


def seed_confirmed_aggregate(
    project_root: Path,
    delivered_phenotype: str,
    *,
    value_keys: Sequence[str],
    delivery_kind: str = traits.PER_PLANT_COUNT_AGGREGATE,
    **fields: Any,
) -> TraitRevision:
    """Confirm a revision of the trait delivering ``delivered_phenotype`` stating the aggregate
    operationalization ``delivery_kind``, covering that phenotype and ``value_keys``."""
    (trait,) = [name for name in traits.trait_names(project_root)
                if delivered_phenotype in latest(name, project_root).delivers]
    return propose_and_confirm(project_root, with_operationalization(
        with_floors(latest(trait, project_root)), delivery_kind,
        statement=f"the {delivered_phenotype} the breeder records for one plant",
        delivered_phenotypes=(delivered_phenotype,), delivered_value_keys=tuple(value_keys),
        **fields))
