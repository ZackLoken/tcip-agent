"""Exact-name scope resolution.

A training scope is a (subject, attribute) pair of names, and both halves resolve by exact name:
a near-miss must refuse rather than land on some other declared name, whose value vocabulary and
id map belong to a different measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp import subject_registry
from tcip_mcp.subject_registry import (
    Attribute,
    SubjectRegistry,
    RegistryError,
    Subject,
    assign_class_ids,
    num_classes,
    read_registry,
    registry_to_dict,
)
from tests._producer_fixtures import registry_over


def _prefixed_attributes() -> SubjectRegistry:
    """One subject carrying two attributes whose names share a prefix, the longer declared first,
    with vocabularies of different size so the two scopes cannot be confused for one another."""
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


def test_attribute_lookup_resolves_the_exactly_named_attribute():
    """With two attributes sharing a prefix, each scope resolves to the vocabulary it named, not to
    whichever declared name happens to start with it."""
    reg = _prefixed_attributes()
    bud = reg.subject("bud")
    assert bud is not None

    exact = bud.attribute("opening")
    assert exact is not None
    assert exact.name == "opening"
    assert exact.type == "categorical"
    assert exact.values == ("open", "closed")

    longer = bud.attribute("opening_stage")
    assert longer is not None
    assert longer.values == ("closed", "swelling", "partial", "shedding")

    assert assign_class_ids(reg, "bud", "opening") == {"open": 0, "closed": 1}
    assert assign_class_ids(reg, "bud", "opening_stage") == {
        "closed": 0, "swelling": 1, "partial": 2, "shedding": 3}
    assert num_classes(reg, "bud", "opening") == 2
    assert num_classes(reg, "bud", "opening_stage") == 4


def test_an_attribute_name_that_only_prefixes_a_declared_one_refuses():
    """A truncated scope name names no declared attribute and must refuse, rather than train over
    the longer attribute's ranks while reporting the name the caller asked for."""
    reg = SubjectRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening_stage", type="ordinal",
                      values=("closed", "swelling", "partial", "shedding")),)),
    ))
    bud = reg.subject("bud")
    assert bud is not None
    assert bud.attribute("opening") is None

    with pytest.raises(RegistryError, match=r"attribute 'opening' not on subject 'bud'"):
        assign_class_ids(reg, "bud", "opening")


def test_subject_lookup_resolves_the_exactly_named_subject():
    """Subject names are matched verbatim: two names differing only in case are two subjects with
    their own vocabularies, and a name declared by neither refuses."""
    reg = _case_variant_subjects()
    lower = reg.subject("bud")
    upper = reg.subject("Bud")
    assert lower is not None and upper is not None
    assert lower.name == "bud" and upper.name == "Bud"

    lower_attr = lower.attribute("opening")
    upper_attr = upper.attribute("opening")
    assert lower_attr is not None and upper_attr is not None
    assert lower_attr.type == "categorical" and upper_attr.type == "ordinal"

    assert assign_class_ids(reg, "bud", "opening") == {"open": 0, "closed": 1}
    assert assign_class_ids(reg, "Bud", "opening") == {
        "closed": 0, "swelling": 1, "partial": 2}
    assert num_classes(reg, "bud", "opening") == 2
    assert num_classes(reg, "Bud", "opening") == 3

    assert reg.subject("BUD") is None
    with pytest.raises(RegistryError, match=r"subject 'BUD' not in registry"):
        assign_class_ids(reg, "BUD")


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
    assert subject_registry.num_classes(back, "Bud", "opening") == 3
