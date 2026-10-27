"""Exact-name scope resolution.

A scope's subject, and an attribute a positive state names, resolve by exact name, and a
near-miss refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp.subject_registry import (
    Attribute,
    RegistryError,
    Subject,
    SubjectRegistry,
    positive_state_problem,
    read_registry,
    registry_to_dict,
)
from tcip_mcp.traits import PositiveState
from tests._producer_fixtures import registry_of, registry_over, scope_over


def _case_variant_subjects() -> SubjectRegistry:
    """Return two subject names differing in case alone, each with its own attribute vocabulary.

    ``bud`` declares a two-value categorical ``opening`` and ``Bud`` a three-value ordinal one.
    """
    return SubjectRegistry(
        subjects=(
            Subject(
                name="bud",
                description="",
                attributes=(
                    Attribute(
                        name="opening",
                        type="categorical",
                        values=("open", "closed"),
                    ),
                ),
            ),
            Subject(
                name="Bud",
                description="",
                attributes=(
                    Attribute(
                        name="opening",
                        type="ordinal",
                        values=("closed", "swelling", "partial"),
                    ),
                ),
            ),
        )
    )


def test_attribute_lookup_resolves_the_exactly_named_attribute(
    tmp_path: Path,
) -> None:
    """An attribute lookup resolves the exactly named attribute.

    With two attributes sharing a prefix, the scope carries both under their own names and a
    positive state resolves to the vocabulary it named.
    """
    reg = registry_of(
        "bud",
        Attribute(
            name="opening_stage",
            type="ordinal",
            values=("closed", "swelling", "partial", "shedding"),
        ),
        Attribute(
            name="opening", type="categorical", values=("open", "closed")
        ),
    )
    scope = scope_over(tmp_path, reg, "bud")
    assert [a.name for a in scope.attributes] == ["opening_stage", "opening"]

    assert scope.state_ids(
        PositiveState(attribute="opening", value="open")
    ) == (1, 0)
    assert scope.state_ids(
        PositiveState(attribute="opening_stage", value="partial")
    ) == (0, 2)
    assert (
        positive_state_problem(
            reg, "bud", PositiveState(attribute="opening", value="open")
        )
        is None
    )


def test_an_attribute_name_that_only_prefixes_a_declared_one_refuses() -> None:
    """An attribute name that is a strict prefix of a declared one is refused.

    The subject declares no attribute of the truncated name, and the positive state naming it
    is reported as naming no declared attribute.
    """
    reg = registry_of(
        "bud",
        Attribute(
            name="opening_stage",
            type="ordinal",
            values=("closed", "swelling", "partial", "shedding"),
        ),
    )
    bud = reg.subject("bud")
    assert bud is not None
    assert bud.attribute("opening") is None

    problem = positive_state_problem(
        reg, "bud", PositiveState(attribute="opening", value="closed")
    )
    assert problem is not None and "declares no attribute 'opening'" in problem


def test_subject_lookup_resolves_the_exactly_named_subject(
    tmp_path: Path,
) -> None:
    """A subject lookup resolves the exactly named subject.

    Subject names are matched verbatim: two names differing in case alone are two subjects with
    their own vocabularies, and a name declared by neither refuses.
    """
    reg = _case_variant_subjects()
    lower = scope_over(tmp_path, reg, "bud")
    upper = scope_over(tmp_path, reg, "Bud")
    assert [a.type for a in lower.attributes] == ["categorical"]
    assert [a.values for a in upper.attributes] == [
        ("closed", "swelling", "partial")
    ]

    with pytest.raises(
        RegistryError, match=r"subject 'BUD' is not in the registry"
    ):
        scope_over(tmp_path, reg, "BUD")


def test_case_variant_subjects_survive_a_file_roundtrip_as_distinct_subjects(
    tmp_path: Path,
) -> None:
    """Case-variant subjects survive a file round trip as distinct subjects.

    Both names are stored and read back verbatim, each with its own attribute vocabulary.
    """
    reg = _case_variant_subjects()
    path = tmp_path / "subjects.json"
    registry_over(tmp_path, reg)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == {"bud", "Bud"}

    back = read_registry(tmp_path)
    assert [s.name for s in back.subjects] == ["bud", "Bud"]
    assert back == reg
    assert registry_to_dict(back) == on_disk

    lower = back.subject("bud")
    upper = back.subject("Bud")
    assert lower is not None and upper is not None
    assert lower.attributes[0].values == ("open", "closed")
    assert upper.attributes[0].values == ("closed", "swelling", "partial")
