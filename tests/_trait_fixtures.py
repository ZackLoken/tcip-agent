"""Sample trait entries, proposed and confirmed through the platform's trait doors.

They are proposed with ``traits.propose_trait`` and confirmed with ``traits.confirm_revision``:
a crossing trait delivering bloom dates, a count trait delivering a stem count, the vocabulary
phenotypes the aggregate deliveries ship under, and ``bud_opening``, whose name is not the
``bud`` subject it measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tcip_mcp import subject_registry as cr, traits
from tcip_mcp.traits import TraitEntry, TraitRevision


def entry(name: str, delivers: Sequence[str], **fields: object) -> TraitEntry:
    """Return a ``TraitEntry`` validated from ``name``, ``delivers`` and ``fields``.

    Every other field is empty, except a ``half_class_avg_size`` localization tolerance at 0.5;
    ``fields`` are merged over them all.
    """
    return TraitEntry.model_validate(
        {
            "name": name,
            "delivers": tuple(delivers),
            "positive_state": None,
            "milestone_fractions": (),
            "milestone_on": "",
            "crossing_rule": "",
            "majority_milestone": "",
            "phenology_prefix": "",
            "majority_label": "",
            "count_objective": "",
            "localization": "",
            "localization_tolerance": "half_class_avg_size",
            "localization_tolerance_frac": 0.5,
            "iou_jitter_px": None,
            "iou_margin": None,
            "count_bias_tolerance_frac": None,
            "count_error_tolerance": None,
            "classifier_agreement_floor": None,
            "ordinal_agreement_floor": None,
            "regression_skill_floor": None,
            "plant_aggregation": "",
            "regression_criterion": "",
            "scale_tolerance_frac": None,
            "holdout_match_quality_floor": None,
            "notes": "",
            "operationalizations": {},
            **fields,
        }
    )


def with_fields(base: TraitEntry, **fields: object) -> TraitEntry:
    """Return ``base`` with ``fields`` replaced, validated as a new entry."""
    return TraitEntry.model_validate({**base.model_dump(), **fields})


CROSSING_RULE = "linear_whole_day"
"""The crossing rule a milestone-delivering fixture trait authors, a ``phenology.CROSSING_RULES``
name."""
PLANT_AGGREGATION = "median"
"""The per-plant statistic an aggregate-delivering fixture trait authors, an
``aggregation.PLANT_AGGREGATIONS`` name."""


BUD_OPENING = entry(
    "bud_opening",
    ("leaf_out_05per_date", "leaf_out_50per_date"),
    count_objective="count_unbiased",
    localization="center_match",
    # fixture value: loose enough for the synthetic dense references to clear
    holdout_match_quality_floor=0.5,
    positive_state={"attribute": "opening", "value": "open"},
    milestone_fractions=(0.05, 0.50, 0.95),
    milestone_on="positive_fraction",
    crossing_rule=CROSSING_RULE,
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
    "count_objective": "count_unbiased",
    "localization": "center_match",
    "count_bias_tolerance_frac": 0.1,
    "count_error_tolerance": 1.0,
    "classifier_agreement_floor": 0.6,
    "ordinal_agreement_floor": 0.6,
    "regression_skill_floor": 0.5,
    "regression_criterion": "r_squared",
    "holdout_match_quality_floor": 0.5,
    "scale_tolerance_frac": 0.1,
    "crossing_rule": CROSSING_RULE,
    "plant_aggregation": PLANT_AGGREGATION,
}
"""Fixture values for the criterion floors, the crossing rule and the plant aggregation."""

COUNT_SPEC = entry(COUNT_TRAIT, ("stem_count",), **_FLOORS)

IOU_JITTER_PX = 12.0
"""A sample repeat-annotation displacement in pixels an ``iou_match`` trait authors."""
IOU_MARGIN = 0.05
"""A sample absolute IoU margin an ``iou_match`` trait authors."""


def with_floors(base: TraitEntry) -> TraitEntry:
    """Return ``base`` with each criterion field it leaves unauthored taken from :data:`_FLOORS`."""
    return with_fields(
        base,
        **{
            k: v
            for k, v in _FLOORS.items()
            if getattr(base, k) in (None, "", ())
        },
    )


CROSSING_SPEC = with_floors(
    entry(
        CROSSING_TRAIT,
        ("bloom_05per_date", "bloom_50per_date"),
        positive_state={"attribute": "state", "value": "open"},
        milestone_fractions=(0.05, 0.50),
        milestone_on="positive_fraction",
        phenology_prefix="bloom",
    )
)

DELIVERY_SPECS = (
    COUNT_SPEC,
    *(
        entry(p, (p,), **_FLOORS)
        for p in (
            "astringency",
            "fruit_diameter",
            "plant_surface_area",
            "bark_thickness",
        )
    ),
)
"""The count trait and four vocabulary phenotypes' traits, each delivering its own phenotype."""

DELIVERY_TRAIT_BY_PHENOTYPE = {
    phenotype: spec.name
    for spec in DELIVERY_SPECS
    for phenotype in spec.delivers
}


def propose(
    project_root: Path, proposed: TraitEntry, *, dataset_root: str = ""
) -> TraitRevision:
    """Append ``proposed`` as a new, unconfirmed revision of its trait."""
    return traits.propose_trait(
        project_root,
        proposed,
        rationale="fixture proposal",
        relayed_note="",
        dataset_root=dataset_root,
        actor=None,
    )


def confirm(
    project_root: Path, revision: TraitRevision, *, user: str = "grüne"
) -> TraitRevision:
    """Confirm ``revision`` by its number and entry hash, as the person ``user``."""
    from tcip_mcp.identity import actor

    return traits.confirm_revision(
        project_root,
        revision.entry.name,
        revision.number,
        revision.entry_sha256,
        actor=actor(user),
        confirmed=True,
    )


def propose_and_confirm(
    project_root: Path, proposed: TraitEntry
) -> TraitRevision:
    """Propose ``proposed`` and confirm the revision it became."""
    return confirm(project_root, propose(project_root, proposed))


def confirm_bare(
    project_root: Path, name: str, **fields: object
) -> TraitRevision:
    """Propose and confirm ``entry(name, ("leaf_length",), **fields)`` at ``project_root``."""
    return propose_and_confirm(
        project_root, entry(name, ("leaf_length",), **fields)
    )


def latest(trait: str, project_root: Path) -> TraitEntry:
    """Return the entry of ``trait``'s latest revision, confirmed or not."""
    return traits.read_trait(trait, project_root).latest.entry


def count_revision(project_root: Path) -> TraitRevision:
    """Return the count trait's latest confirmed revision."""
    from tcip_mcp.operationalization import latest_confirmed

    return latest_confirmed(COUNT_TRAIT, project_root)


def operationalization(
    **fields: object,
) -> dict[str, Any]:
    """Return an operationalization of fixture texts and :data:`COUNT_SUBJECT`, ``fields`` over."""
    return {
        "statement": "the number the breeder records for one plant",
        "mechanism": "the calibrated model at the derived operating point",
        "measured_subject": COUNT_SUBJECT,
        "delivered_phenotypes": (),
        "delivered_value_keys": (),
        **fields,
    }


def with_operationalization(
    base: TraitEntry, kind: str, **fields: object
) -> TraitEntry:
    """Return ``base`` with an operationalization for ``kind`` added or replaced."""
    stated = {k: v.model_dump() for k, v in base.operationalizations.items()}
    stated[kind] = operationalization(**fields)
    return with_fields(base, operationalizations=stated)


def seed_positive_class(
    project_root: Path, subject_name: str, state: traits.PositiveState | None
) -> cr.SubjectRegistry:
    """Ensure the project's subject registry declares ``state``'s attribute on ``subject_name``.

    The subject is rebuilt from its name and its existing attributes, ``state``'s attribute
    added as categorical over its value or its value appended where missing, and the registry
    replaced through :func:`~tests._producer_fixtures.registry_over`. Returns the new registry.
    """
    from tests._producer_fixtures import registry_over

    registry = cr.registry_for_dataset_root(
        project_root
    ) or cr.SubjectRegistry(subjects=())
    subjects = {s.name: s for s in registry.subjects}
    existing = subjects.get(subject_name)
    attrs = {a.name: a for a in (existing.attributes if existing else ())}
    if state is not None:
        attr = attrs.get(state.attribute)
        if attr is None:
            attrs[state.attribute] = cr.Attribute(
                name=state.attribute, type="categorical", values=(state.value,)
            )
        elif state.value not in attr.values:
            attrs[state.attribute] = cr.Attribute(
                name=attr.name,
                type=attr.type,
                values=(*attr.values, state.value),
            )
    subjects[subject_name] = cr.Subject(
        name=subject_name,
        description=existing.description if existing else "",
        attributes=tuple(attrs.values()),
    )
    updated = cr.SubjectRegistry(subjects=tuple(subjects.values()))
    registry_over(project_root, updated)
    return updated


def seed_confirmed_crossing(
    project_root: Path, trait: str, **fields: object
) -> TraitRevision:
    """Confirm a revision of ``trait`` stating a crossing operationalization.

    ``trait`` must be proposed at this root first. Its positive state is declared for the measured
    subject in the project's own subject registry first. ``fields`` override the
    operationalization's defaults: the measured subject is the trait's own name and the
    phenotypes are everything it delivers.
    """
    base = with_floors(latest(trait, project_root))
    stated = {
        "statement": f"the date each plant reached the state {trait} scores in the field",
        "mechanism": "the calibrated attribute head over one plant's objects",
        "measured_subject": trait,
        "delivered_phenotypes": base.delivers,
        **fields,
    }
    seed_positive_class(
        project_root, stated["measured_subject"], base.positive_state
    )
    return propose_and_confirm(
        project_root,
        with_operationalization(base, traits.STATE_CROSSING_DATES, **stated),
    )


def seed_confirmed_count(
    project_root: Path,
    *,
    measured_subject: str = COUNT_SUBJECT,
    **fields: object,
) -> TraitRevision:
    """Confirm the count trait at this root with a per-image-count operationalization.

    The operationalization measures ``measured_subject``.
    """
    return propose_and_confirm(
        project_root,
        with_operationalization(
            COUNT_SPEC,
            traits.PER_IMAGE_COUNT,
            statement="how many stems the model finds in one frame",
            measured_subject=measured_subject,
            **fields,
        ),
    )


def seed_delivery_traits(project_root: Path) -> Path:
    """Propose and confirm each of :data:`DELIVERY_SPECS`; return the project path."""
    for spec in DELIVERY_SPECS:
        propose_and_confirm(project_root, spec)
    return Path(project_root)


def seed_confirmed_aggregate(
    project_root: Path,
    delivered_phenotype: str,
    *,
    value_keys: Sequence[str],
    delivery_kind: str = traits.PER_PLANT_COUNT_AGGREGATE,
    **fields: object,
) -> TraitRevision:
    """Confirm a revision of the trait delivering ``delivered_phenotype``.

    It adds a ``delivery_kind`` operationalization over the trait's latest entry with its floors
    filled, passing that phenotype, ``value_keys`` and fixture wording, with ``fields`` over them.
    """
    (trait,) = [
        name
        for name in traits.trait_names(project_root)
        if delivered_phenotype in latest(name, project_root).delivers
    ]
    return propose_and_confirm(
        project_root,
        with_operationalization(
            with_floors(latest(trait, project_root)),
            delivery_kind,
            statement=f"the {delivered_phenotype} the breeder records for one plant",
            delivered_phenotypes=(delivered_phenotype,),
            delivered_value_keys=tuple(value_keys),
            **fields,
        ),
    )
