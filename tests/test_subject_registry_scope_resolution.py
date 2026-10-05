"""Exact-name scope resolution.

A scope's subject, and an attribute a positive state names, resolve by exact name: a near-miss must
refuse rather than land on some other declared name, whose value vocabulary belongs to a
different measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp.subject_registry import (
    Attribute,
    SubjectRegistry,
    RegistryError,
    Subject,
    positive_state_problem,
    read_registry,
    registry_to_dict,
)
from tcip_mcp.traits import PositiveState
from tests._producer_fixtures import registry_over


def _prefixed_attributes() -> SubjectRegistry:
    """One subject carrying two attributes whose names share a prefix, the longer declared first,
    with vocabularies of different size so the two cannot be confused for one another."""
    return SubjectRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening_stage", type="ordinal",
                      values=("closed", "swelling", "partial", "shedding")),
            Attribute(name="opening", type="categorical", values=("open", "closed")),
        )),
    ))


def _case_variant_subjects() -> SubjectRegistry:
    """Two subject names differing only in case, each with its own attribute vocabulary. The GUI
    accepts a free-text subject name, so a case variant is an ordinary registry state: the two are
    distinct subjects and each scope must resolve to its own vocabulary."""
    return SubjectRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("open", "closed")),)),
        Subject(name="Bud", attributes=(
            Attribute(name="opening", type="ordinal",
                      values=("closed", "swelling", "partial")),)),
    ))


def _scope(tmp_path: Path, registry: SubjectRegistry, subject: str):
    """The class space the admission reads for ``subject`` over a dataset declaring
    ``registry``."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope

    registry_over(tmp_path, registry)
    return registry_scope(tmp_path / "images", subject)


def test_attribute_lookup_resolves_the_exactly_named_attribute(tmp_path: Path):
    """With two attributes sharing a prefix, the scope carries both under their own names and a
    positive state resolves to the vocabulary it named, not to whichever declared name happens to
    start with it."""
    reg = _prefixed_attributes()
    scope = _scope(tmp_path, reg, "bud")
    assert [a.name for a in scope.attributes] == ["opening_stage", "opening"]

    assert scope.state_ids(PositiveState(attribute="opening", value="open")) == (1, 0)
    assert scope.state_ids(PositiveState(attribute="opening_stage", value="partial")) == (0, 2)
    assert positive_state_problem(reg, "bud", PositiveState(attribute="opening",
                                                            value="open")) is None


def test_an_attribute_name_that_only_prefixes_a_declared_one_refuses():
    """A truncated attribute name names no declared attribute and must refuse, rather than read
    the longer attribute's ranks while reporting the name the caller asked for."""
    reg = SubjectRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening_stage", type="ordinal",
                      values=("closed", "swelling", "partial", "shedding")),)),
    ))
    bud = reg.subject("bud")
    assert bud is not None
    assert bud.attribute("opening") is None

    problem = positive_state_problem(reg, "bud", PositiveState(attribute="opening",
                                                               value="closed"))
    assert problem is not None and "declares no attribute 'opening'" in problem


def test_subject_lookup_resolves_the_exactly_named_subject(tmp_path: Path):
    """Subject names are matched verbatim: two names differing only in case are two subjects with
    their own vocabularies, and a name declared by neither refuses."""
    reg = _case_variant_subjects()
    lower = _scope(tmp_path, reg, "bud")
    upper = _scope(tmp_path, reg, "Bud")
    assert [a.type for a in lower.attributes] == ["categorical"]
    assert [a.values for a in upper.attributes] == [("closed", "swelling", "partial")]

    with pytest.raises(RegistryError, match=r"subject 'BUD' is not in the registry"):
        _scope(tmp_path, reg, "BUD")


def test_case_variant_subjects_survive_a_file_roundtrip_as_distinct_subjects(tmp_path: Path):
    """Names are stored and read back verbatim: nothing folds two similar names into one subject,
    which would merge two label populations into a single scope."""
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
