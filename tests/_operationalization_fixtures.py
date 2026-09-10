"""Registered trait specs and stated records a test needs before an operationalization exists.

Traits deliberately not the ones the first pilot uses: a crossing trait delivering bloom dates, a
count trait delivering a stem count, and the handful of vocabulary phenotypes the aggregate
deliveries ship under, all real names in the crop vocabulary. A rail's own tests run on a trait
other than the pilot's, so nothing here generalizes from one trait's vocabulary.

The ``seed_confirmed_*`` half takes whatever a test registered and gives it a confirmed record, for
the many modules whose subject is a delivery rather than the precondition standing in front of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tcip_store import Version, read_versioned

from tcip_mcp import subject_registry as cr
from tcip_mcp import operationalization as op
from tcip_mcp import traits
from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec, trait_spec_key, trait_specs_dir

CROSSING_TRAIT = "bloom"
COUNT_TRAIT = "stem"

CROSSING_SPEC = TraitSpec(
    name=CROSSING_TRAIT,
    positive_value="open",
    milestone_fractions=(0.05, 0.50),
    milestone_on="positive_fraction",
    phenology_prefix="bloom",
    delivers=("bloom_05per_date", "bloom_50per_date"),
)

COUNT_SPEC = TraitSpec(
    name=COUNT_TRAIT,
    count_objective=COUNT_UNBIASED,
    localization=CENTER_MATCH,
    ordinal_agreement_floor=0.6,
    regression_skill_floor=0.5,
    holdout_match_quality_floor=0.5,
    delivers=("stem_count",),
)

COUNT_SUBJECT = "stem"
"""What a count record made from :data:`COUNT_SPEC` says the counts are counts of."""


def _delivering(phenotype: str) -> TraitSpec:
    """A registered trait for one crop-vocabulary phenotype, named for what it delivers.

    Every floor is filled, so one spec serves whichever delivery kind a test exercises: which floor
    a confirmation covers follows from the kind, and a floor left empty would refuse for that rather
    than for the thing under test.
    """
    return TraitSpec(
        name=phenotype,
        count_objective=COUNT_UNBIASED,
        localization=CENTER_MATCH,
        ordinal_agreement_floor=0.6,
        regression_skill_floor=0.5,
        holdout_match_quality_floor=0.5,
        delivers=(phenotype,),
    )


DELIVERY_SPECS = (
    COUNT_SPEC,
    _delivering("astringency"),
    _delivering("fruit_diameter"),
    _delivering("plant_surface_area"),
    _delivering("bark_thickness"),
)
"""Every trait the count and aggregate delivery tests ship under.

``stem_count`` is delivered by :data:`COUNT_SPEC`, whose own name is not a vocabulary name, which
is the case the two namespaces exist for; the rest are named for the phenotype they deliver. No
phenotype is delivered by two of them, because a phenotype two registered traits claim has no one
operationalization behind it and every delivery of it refuses as ambiguous.
"""

DELIVERY_TRAIT_BY_PHENOTYPE = {
    phenotype: spec.name for spec in DELIVERY_SPECS for phenotype in spec.delivers
}
"""Which of the fixture specs above delivers each phenotype, indexed from the specs themselves."""


def write_spec(project_root: Path, spec: TraitSpec) -> None:
    """Write a spec record into a project's own registry, bypassing both authoring doors and
    the trait-spec statement they carry.

    Routes through the platform's shared validate-encode-write entry, so a fixture spec is
    proven to clear the same crops.yml cross-check every authored spec goes through.
    Compare-and-set against whatever is on file, so a test re-registering the same trait to
    simulate a spec that moved since it was read overwrites it rather than being refused.
    """
    data = traits._encode_spec(spec)
    key = trait_spec_key(trait_specs_dir(str(project_root)), spec.name)
    current = read_versioned(key, default=None)
    validated, reason = traits._validate_and_write_spec(key, data, expect=current.version)
    if validated is None:
        raise ValueError(f"fixture spec {spec.name!r} does not clear crops.yml: {reason}")


def confirm_spec_statement(project_root: Path, trait: str) -> dict[str, Any]:
    """Give a registered trait's own trait-spec statement a confirmed, current record, the way
    the breeder's own confirmation would leave it.

    States or restates the spec on file through the revision door (a no-op write when a current
    statement is already on record) and confirms unless it already is, so a direct
    ``state_operationalization`` call the tests below make finds the precondition
    ``state_trait_operationalization`` now checks already cleared. Idempotent: a fixture may call
    this more than once over the same trait.
    """
    traits.write_trait_spec_fields(
        trait, {}, project_root=project_root,
        rationale="fixture restatement so the trait's own trait-spec statement stays current",
    )
    scope = traits.trait_spec_statements_scope(project_root)
    key = traits.trait_spec_statement_key(scope, trait)
    record = read_versioned(key, default=None).value or {}
    if record.get("confirmed_by"):
        return record
    return traits.confirm_trait_spec(
        project_root, trait,
        user="fixture-breeder",
        record_seen=traits.trait_spec_statement_seen_hash(record),
        identity_from_request=False,
    )


def seed_positive_class(project_root: Path, subject_name: str, positive_value: str) -> cr.SubjectRegistry:
    """Ensure the project's subject registry declares ``positive_value`` as a value of
    ``subject_name``'s own attribute, adding both the subject and the value on first mention and
    leaving an existing declaration alone; returns the registry as stored.

    An empty ``positive_value`` (a spec whose field is not yet authored) adds the subject with
    whatever attributes it already carries, never an empty-string value, since the registry
    invariant forbids one and there is nothing yet to declare.
    """
    from tcip_mcp.dataset_layout import subjects_path

    registry = cr.registry_for_dataset_root(project_root) or cr.SubjectRegistry()
    subjects = {s.name: s for s in registry.subjects}
    existing = subjects.get(subject_name)
    attrs = list(existing.attributes) if existing else []
    if positive_value:
        if attrs:
            attr = attrs[0]
            if positive_value not in attr.values:
                attrs[0] = cr.Attribute(name=attr.name, type=attr.type,
                                        values=(*attr.values, positive_value))
        else:
            attrs = [cr.Attribute(name="state", type="categorical", values=(positive_value,))]
    subjects[subject_name] = cr.Subject(name=subject_name, attributes=tuple(attrs))
    updated = cr.SubjectRegistry(subjects=tuple(subjects.values()))
    cr.write_registry(subjects_path(project_root), updated)
    return updated


def seed_project(project_root: Path) -> Path:
    """A project whose registry carries both fixture traits, and whose subject registry declares the
    crossing fixture's positive class for the subject it states its crossing operationalization of."""
    write_spec(project_root, CROSSING_SPEC)
    confirm_spec_statement(project_root, CROSSING_TRAIT)
    write_spec(project_root, COUNT_SPEC)
    confirm_spec_statement(project_root, COUNT_TRAIT)
    seed_positive_class(project_root, "flower", CROSSING_SPEC.positive_value)
    return Path(project_root)


def state_crossing(project_root: Path, **overrides: Any) -> dict[str, Any]:
    """A stated, unconfirmed crossing record for the fixture crossing trait.

    Passes the project's own subject registry (as :func:`seed_project` left it, or as a caller
    updated it since) to the writer, the registry a crossing statement is checked against.
    """
    confirm_spec_statement(project_root, CROSSING_TRAIT)
    fields: dict[str, Any] = {
        "statement": "the date each plant reached the state the breeder scores in the field",
        "mechanism": "the calibrated state classifier over the isolated flowers of one plant",
        "measured_subject": "flower",
        "delivered_phenotypes": ["bloom_05per_date", "bloom_50per_date"],
    }
    fields.update(overrides)
    registry = cr.registry_for_dataset_root(project_root)
    return op.state_operationalization(
        project_root, CROSSING_TRAIT, op.STATE_CROSSING_DATES, registry=registry, **fields
    )


def state_count(project_root: Path, **overrides: Any) -> dict[str, Any]:
    """A stated, unconfirmed per-image count record for the fixture count trait."""
    confirm_spec_statement(project_root, COUNT_TRAIT)
    fields: dict[str, Any] = {
        "statement": "how many stems the model finds in one frame",
        "mechanism": "the calibrated detector over whole frames at the derived operating point",
        "measured_subject": "stem",
        "delivered_phenotypes": [],
    }
    fields.update(overrides)
    return op.state_operationalization(project_root, COUNT_TRAIT, op.PER_IMAGE_COUNT, **fields)


def state_aggregate(project_root: Path, delivery_kind: str, **overrides: Any) -> dict[str, Any]:
    """A stated, unconfirmed per-plant aggregate record for the fixture count trait."""
    confirm_spec_statement(project_root, COUNT_TRAIT)
    fields: dict[str, Any] = {
        "statement": "the aggregated value the breeder records for one plant",
        "mechanism": "the calibrated detector's per-plant assignment at the derived operating point",
        "measured_subject": "stem",
        "delivered_phenotypes": ["stem_count"],
        "delivered_value_keys": ["stem_count"],
    }
    fields.update(overrides)
    return op.state_operationalization(project_root, COUNT_TRAIT, delivery_kind, **fields)


def confirm(
    project_root: Path,
    trait: str,
    delivery_kind: str,
    record: dict[str, Any],
    *,
    user: str = "grüne",
    identity_from_request: bool = True,
) -> dict[str, Any]:
    """Confirm exactly the record a caller is holding, the way the surface posts it back."""
    return op.confirm_trait_operationalization(
        project_root,
        trait,
        delivery_kind,
        user=user,
        record_seen=op.record_seen_hash(record),
        identity_from_request=identity_from_request,
    )


def resolve(project_root: Path, trait: str, delivery_kind: str) -> op.ResolvedOperationalization:
    return op.resolve_trait_and_record(trait, delivery_kind, project_root=project_root)


def schema_basis() -> op.OperationalizationBasis:
    """A proof-of-precondition token for a test whose subject is the CSV writer's column schema.

    ``write_phenology_csv`` demands the basis a passing check returned, because it is handed a spec
    object rather than a project it could read the record from. A test about which columns reach
    the file has neither a project nor a record, so it produces the token directly.
    """
    return op.OperationalizationBasis(record_version=Version.ABSENT, constituting={})


def seed_delivery_traits(project_root: Path) -> Path:
    """Register every trait the count and aggregate delivery tests deliver under."""
    for spec in DELIVERY_SPECS:
        write_spec(project_root, spec)
        confirm_spec_statement(project_root, spec.name)
    return Path(project_root)


def seed_confirmed_count(
    project_root: Path, *, measured_subject: str = COUNT_SUBJECT, **overrides: Any
) -> dict[str, Any]:
    """Register the count trait at this root and confirm its per-image-count operationalization.

    What a module needs when its subject is the count delivery rather than the precondition in front
    of it. ``measured_subject`` is what the delivery's buckets recorded detecting, which the door
    checks against their own id maps wherever they recorded one.
    """
    write_spec(project_root, COUNT_SPEC)
    record = state_count(project_root, measured_subject=measured_subject, **overrides)
    return confirm(project_root, COUNT_TRAIT, op.PER_IMAGE_COUNT, record)


def confirm_aggregate(
    project_root: Path,
    trait: str,
    delivery_kind: str,
    *,
    delivered_phenotype: str,
    value_keys: Sequence[str],
    **overrides: Any,
) -> dict[str, Any]:
    """State and confirm one per-plant aggregate record, with its trait and kind named outright.

    The one writer behind every aggregate record these fixtures produce. It names both key parts
    itself rather than deriving either, so a test whose subject is the derivation can seed a record
    without going through it.
    """
    confirm_spec_statement(project_root, trait)
    fields: dict[str, Any] = {
        "statement": f"the {delivered_phenotype} the breeder records for one plant",
        "mechanism": "the calibrated model's per-plant assignment at the derived operating point",
        "measured_subject": COUNT_SUBJECT,
        "delivered_phenotypes": [delivered_phenotype],
        "delivered_value_keys": list(value_keys),
    }
    fields.update(overrides)
    record = op.state_operationalization(project_root, trait, delivery_kind, **fields)
    return confirm(project_root, trait, delivery_kind, record)


def seed_confirmed_aggregate(
    project_root: Path,
    delivered_phenotype: str,
    *,
    value_keys: Sequence[str],
    measurement_document: str = "operating_point",
    **overrides: Any,
) -> dict[str, Any]:
    """Confirm an aggregate record the way the door reaches it: by phenotype and by document.

    The traits are registered by :func:`seed_delivery_traits`; the phenotype names which of them
    this record belongs to and ``measurement_document`` names which kind, both resolved through the
    same functions the door resolves them with, so a module seeded this way is seeded for the call
    it makes.
    """
    return confirm_aggregate(
        project_root,
        op.resolve_trait_for_phenotype(delivered_phenotype, project_root=project_root),
        op.aggregate_delivery_kind(measurement_document),
        delivered_phenotype=delivered_phenotype,
        value_keys=value_keys,
        **overrides,
    )


def seed_confirmed_crossing(project_root: Path, trait: str, **overrides: Any) -> dict[str, Any]:
    """State and confirm a crossing operationalization for a trait already registered at this root.

    What a test needs when its subject is a delivery rather than the precondition: the crossing
    doors refuse without a confirmed record, so a module whose subject predates that rail seeds one
    and keeps testing what it was written to test. The delivered phenotypes come from the
    registered spec's own ``delivers``, so this works for any trait a test authors. Declares the
    spec's positive class for the measured subject in the project's own subject registry, adding
    both on first mention (see :func:`seed_positive_class`), so the registry the writer now
    requires is never a hole a test-only trait falls through.
    """
    from tcip_mcp.traits import get_trait_for

    confirm_spec_statement(project_root, trait)
    spec = get_trait_for(trait, project_root)
    fields: dict[str, Any] = {
        "statement": f"the date each plant reached the state {trait} scores in the field",
        "mechanism": f"the calibrated {spec.positive_value} classifier over one plant's objects",
        "measured_subject": trait,
        "delivered_phenotypes": list(spec.delivers),
    }
    fields.update(overrides)
    registry = seed_positive_class(project_root, fields["measured_subject"], spec.positive_value)
    record = op.state_operationalization(
        project_root, trait, op.STATE_CROSSING_DATES, registry=registry, **fields
    )
    return confirm(project_root, trait, op.STATE_CROSSING_DATES, record)
