"""The nested class registry and its deterministic name→id assignment.

The assignment is the measurement apex: a non-deterministic or reordered map ships predictions that
decode to the wrong class: a confident-wrong phenotype that passes every downstream test. These
pin the two properties that prevent it: assignment follows the registry's *declared* order (never
sorted, which would corrupt ordinal rank), and it is stable across calls and a file round-trip.
"""

from __future__ import annotations

import json

import pytest

from tcip_mcp.class_registry import (
    Attribute,
    ClassRegistry,
    RegistryError,
    Subject,
    assign_class_ids,
    decode_class_ids,
    num_classes,
    read_registry,
    registry_from_dict,
    registry_to_dict,
    write_registry,
)


def _catkin_bush() -> ClassRegistry:
    return ClassRegistry(subjects=(
        Subject(name="bush", description="one hazelnut bush crown", defined_by="user:breeder"),
        Subject(
            name="catkin",
            description="a hazelnut catkin",
            defined_by="user:breeder",
            attributes=(Attribute(name="elongation", type="categorical", values=("dormant", "elongated")),),
        ),
    ))


def test_dict_and_file_roundtrip_preserve_the_registry(tmp_path):
    reg = _catkin_bush()
    assert registry_from_dict(registry_to_dict(reg)) == reg

    path = tmp_path / "classes.json"
    write_registry(path, reg)
    assert read_registry(path) == reg
    # bush carries no attributes -> no 'attributes' key rather than an empty one
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "attributes" not in on_disk["bush"]
    assert on_disk["catkin"]["attributes"]["elongation"]["values"] == ["dormant", "elongated"]


def test_assign_ids_follow_declared_order_not_sorted():
    # Values declared reverse-alphabetically: a sorted/set-based assignment would flip these ids and
    # silently remap every prediction. Declared order must win.
    reg = ClassRegistry(subjects=(
        Subject(name="catkin", attributes=(
            Attribute(name="elongation", type="categorical", values=("elongated", "dormant")),)),
    ))
    assert assign_class_ids(reg, "catkin", "elongation") == {"elongated": 0, "dormant": 1}


def test_ordinal_assignment_is_rank_order():
    reg = ClassRegistry(subjects=(
        Subject(name="leaf", attributes=(
            Attribute(name="efb_severity", type="ordinal",
                      values=("none", "light", "moderate", "severe")),)),
    ))
    assert assign_class_ids(reg, "leaf", "efb_severity") == {
        "none": 0, "light": 1, "moderate": 2, "severe": 3}
    assert num_classes(reg, "leaf", "efb_severity") == 4


def test_assignment_is_stable_across_calls_and_a_file_roundtrip(tmp_path):
    reg = _catkin_bush()
    first = assign_class_ids(reg, "catkin", "elongation")
    assert first == assign_class_ids(reg, "catkin", "elongation")  # repeatable

    path = tmp_path / "classes.json"
    write_registry(path, reg)
    assert assign_class_ids(read_registry(path), "catkin", "elongation") == first  # survives round-trip


def test_no_attribute_scope_is_single_class():
    reg = _catkin_bush()
    assert assign_class_ids(reg, "bush") == {"bush": 0}  # subject with no attributes
    assert assign_class_ids(reg, "catkin") == {"catkin": 0}  # attributes exist but scope ignores them
    assert num_classes(reg, "bush") == 1


def test_decode_inverts_the_recorded_map():
    reg = _catkin_bush()
    id_map = assign_class_ids(reg, "catkin", "elongation")
    assert decode_class_ids(id_map) == {0: "dormant", 1: "elongated"}


def test_absent_subject_or_attribute_refuses():
    reg = _catkin_bush()
    with pytest.raises(RegistryError):
        assign_class_ids(reg, "acorn")
    with pytest.raises(RegistryError):
        assign_class_ids(reg, "catkin", "sex")  # not declared on catkin


def test_a_subject_with_no_attributes_is_valid():
    # rail admits valid work: bush (detection-only) loads and assigns, it is not treated as malformed
    reg = registry_from_dict({"bush": {"description": "a bush"}})
    assert reg.subject("bush") is not None
    assert assign_class_ids(reg, "bush") == {"bush": 0}


@pytest.mark.parametrize("bad", [
    {"catkin": {"attributes": {"elongation": {"type": "numeric", "values": ["1"]}}}},  # numeric not allowed
    {"catkin": {"attributes": {"elongation": {"type": "categorical", "values": []}}}},  # empty values
    {"catkin": {"attributes": {"elongation": {"type": "categorical", "values": ["a", "a"]}}}},  # dup values
    {"catkin": {"attributes": {"elongation": {"values": ["dormant"]}}}},  # missing type
])
def test_malformed_registry_refuses(bad):
    with pytest.raises(RegistryError):
        registry_from_dict(bad)


def test_ordinal_file_roundtrip_preserves_type_and_rank(tmp_path):
    # The apex on-disk risk: a writer that sorted or flattened ordinal metadata would reorder the
    # ranks. Declared order here is deliberately not alphabetical, so a sorting writer is caught.
    reg = ClassRegistry(subjects=(
        Subject(name="leaf", attributes=(
            Attribute(name="efb_severity", type="ordinal",
                      values=("none", "light", "moderate", "severe")),)),
    ))
    path = tmp_path / "classes.json"
    write_registry(path, reg)
    back = read_registry(path)
    assert back == reg  # type and declared value order both survive the round-trip
    leaf = back.subject("leaf")
    assert leaf is not None and leaf.attribute("efb_severity").type == "ordinal"  # type: ignore[union-attr]
    assert assign_class_ids(back, "leaf", "efb_severity") == {
        "none": 0, "light": 1, "moderate": 2, "severe": 3}


def test_attribute_refuses_bad_values_at_construction():
    # The invariant travels with the type, so assign_class_ids can never silently collapse a class,
    # however the Attribute was built, not only via the JSON parser.
    with pytest.raises(ValueError):
        Attribute(name="elongation", type="categorical", values=("dormant", "dormant"))  # duplicate
    with pytest.raises(ValueError):
        Attribute(name="elongation", type="categorical", values=())  # empty
    with pytest.raises(ValueError):
        Attribute(name="elongation", type="numeric", values=("dormant",))  # numeric not an attr type


def test_absent_or_null_attributes_valid_but_falsy_malformed_refused():
    for ok in ({"bush": {}}, {"bush": {"attributes": None}}):  # absent / null -> detection-only subject
        bush = registry_from_dict(ok).subject("bush")
        assert bush is not None and bush.attributes == ()
    for bad in (False, 0, "", []):  # a falsy but wrong-typed attributes is malformed, not "none"
        with pytest.raises(RegistryError):
            registry_from_dict({"bush": {"attributes": bad}})
