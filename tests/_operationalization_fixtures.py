"""Registered trait specs and stated records a test needs before an operationalization exists.

Two traits, deliberately not the ones the first pilot uses: a crossing trait delivering bloom
dates and a count trait delivering a stem count, both real names in the crop vocabulary. A rail
shaped around one trait's vocabulary is the failure this platform keeps guarding against, so the
fixtures a rail's own tests run on name something else.

:func:`seed_confirmed_crossing` is the other half: it takes whatever trait a test registered and
gives it a confirmed record, for the many modules whose subject is a delivery rather than the
precondition standing in front of it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from tcip_store import Version

from tcip_mcp import operationalization as op
from tcip_mcp.traits import CENTER_MATCH, COUNT_UNBIASED, TraitSpec

CROSSING_TRAIT = "bloom"
COUNT_TRAIT = "stem"

CROSSING_SPEC = TraitSpec(
    name=CROSSING_TRAIT,
    positive_class_name="open",
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
    delivers=("stem_count",),
)


def write_spec(project_root: Path, spec: TraitSpec) -> None:
    """Author a spec file into a project's own registry, the way a breeder hand-authors one."""
    import yaml

    specs_dir = Path(project_root) / ".tcip" / "state" / "trait_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in dataclasses.asdict(spec).items()
    }
    (specs_dir / f"{spec.name}.yml").write_text(yaml.safe_dump(data), encoding="utf-8")


def seed_project(project_root: Path) -> Path:
    """A project whose registry carries both fixture traits."""
    write_spec(project_root, CROSSING_SPEC)
    write_spec(project_root, COUNT_SPEC)
    return Path(project_root)


def state_crossing(project_root: Path, **overrides: Any) -> dict[str, Any]:
    """A stated, unconfirmed crossing record for the fixture crossing trait."""
    fields: dict[str, Any] = {
        "statement": "the date each plant reached the state the breeder scores in the field",
        "mechanism": "the calibrated state classifier over the isolated flowers of one plant",
        "measured_subject": "flower",
        "delivered_phenotypes": ["bloom_05per_date", "bloom_50per_date"],
    }
    fields.update(overrides)
    return op.state_operationalization(
        project_root, CROSSING_TRAIT, op.STATE_CROSSING_DATES, **fields
    )


def state_count(project_root: Path, **overrides: Any) -> dict[str, Any]:
    """A stated, unconfirmed per-image count record for the fixture count trait."""
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


def seed_confirmed_crossing(project_root: Path, trait: str, **overrides: Any) -> dict[str, Any]:
    """State and confirm a crossing operationalization for a trait already registered at this root.

    What a test needs when its subject is a delivery rather than the precondition: the crossing
    doors refuse without a confirmed record, so a module whose subject predates that rail seeds one
    and keeps testing what it was written to test. The delivered phenotypes come from the
    registered spec's own ``delivers``, so this works for any trait a test authors.
    """
    from tcip_mcp.traits import get_trait_for

    spec = get_trait_for(trait, project_root)
    fields: dict[str, Any] = {
        "statement": f"the date each plant reached the state {trait} scores in the field",
        "mechanism": f"the calibrated {spec.positive_class_name} classifier over one plant's objects",
        "measured_subject": trait,
        "delivered_phenotypes": list(spec.delivers),
    }
    fields.update(overrides)
    record = op.state_operationalization(
        project_root, trait, op.STATE_CROSSING_DATES, **fields
    )
    return confirm(project_root, trait, op.STATE_CROSSING_DATES, record)
