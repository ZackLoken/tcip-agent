"""Exact-name scope resolution and what the attribute-schema digest covers.

A training scope is a (subject, attribute) pair of names, and both halves resolve by exact name:
a near-miss must refuse rather than land on some other declared name, whose value vocabulary and
id map belong to a different measurement. The attribute-schema digest that quarantines a stale
confirmation covers the attribute's declared type as well as its values, since the same value
names mean an unordered vocabulary under ``categorical`` and ranks under ``ordinal``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_mcp import class_registry
from tcip_mcp.class_registry import (
    Attribute,
    ClassRegistry,
    RegistryError,
    Subject,
    assign_class_ids,
    attribute_schema_digest,
    num_classes,
    read_registry,
    registry_to_dict,
    write_registry,
)
from tcip_mcp.dataset_layout import record_image_statuses, status_bucket


def _severity_registry(attr_type: str) -> ClassRegistry:
    """A leaf severity vocabulary declared with the given attribute type, values held identical."""
    return ClassRegistry(subjects=(
        Subject(name="bush", description="one bush crown"),
        Subject(name="leaf", description="one leaf", attributes=(
            Attribute(name="efb_severity", type=attr_type,
                      values=("none", "light", "moderate", "severe")),)),
    ))


def _prefixed_attributes() -> ClassRegistry:
    """One subject carrying two attributes whose names share a prefix, the longer declared first,
    with vocabularies of different size so the two scopes cannot be confused for one another."""
    return ClassRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening_stage", type="ordinal",
                      values=("closed", "swelling", "partial", "shedding")),
            Attribute(name="opening", type="categorical", values=("open", "closed")),
        )),
    ))


def _case_variant_subjects() -> ClassRegistry:
    """Two subject names differing only in case, each with its own attribute vocabulary. The GUI
    accepts a free-text subject name, so a case variant is an ordinary registry state: the two are
    distinct subjects and each scope must resolve to its own vocabulary."""
    return ClassRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("open", "closed")),)),
        Subject(name="Bud", attributes=(
            Attribute(name="opening", type="ordinal",
                      values=("closed", "swelling", "partial")),)),
    ))


def test_digest_distinguishes_a_categorical_from_an_ordinal_vocabulary():
    """Redefining an attribute's type changes what every recorded value means, so the digest a
    confirmation is stamped under must move even when the value names are untouched."""
    categorical = attribute_schema_digest(_severity_registry("categorical"), "leaf")
    ordinal = attribute_schema_digest(_severity_registry("ordinal"), "leaf")
    assert categorical is not None and ordinal is not None
    assert categorical != ordinal, (
        "a categorical and an ordinal declaration of the same value names are different schemas; "
        "an equal digest leaves a confirmation stamped under the old definition reading as current"
    )
    assert attribute_schema_digest(_severity_registry("ordinal"), "leaf") == ordinal


def test_digest_ignores_free_text_provenance():
    """A rail that admits valid work: editing description or provenance says nothing about what an
    instance looks like, so it must not quarantine confirmations."""
    base = _severity_registry("ordinal")
    leaf = base.subject("leaf")
    assert leaf is not None
    reworded = ClassRegistry(subjects=(
        Subject(name="bush", description="a bush crown, reworded"),
        Subject(name="leaf", description="one currant leaf, reworded", defined_by="user:breeder",
                defined_at="2026-01-02", attributes=leaf.attributes),
    ))
    assert attribute_schema_digest(reworded, "leaf") == attribute_schema_digest(base, "leaf")


def test_confirmation_is_quarantined_when_an_attribute_type_is_redefined(tmp_path):
    """The consequence the digest exists to prevent: a negative confirmed while the attribute was
    categorical must not read as current after the same values are redeclared as ranks."""
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    root = tmp_path / "dataset"
    labels_dir = root / "annotations"
    labels_dir.mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (96, 40), color=(120, 120, 120)).save(root / "images" / "img_001.jpg")
    json_io.write_annotations(labels_dir / "img_001.json", [], 96, 40, keep_empty=True)

    write_registry(root / "classes.json", _severity_registry("categorical"))
    old_digest = attribute_schema_digest(read_registry(root / "classes.json"), "leaf")
    assert old_digest is not None
    bucket = status_bucket("leaf", None)
    record_image_statuses(root, bucket, {"img_001.jpg": "negative"}, recorded_by="user:breeder")
    import tcip_store
    from tcip_mcp.dataset_layout import image_status_digest_key
    tcip_store.replace(image_status_digest_key(root), {bucket: {"img_001.jpg": old_digest}},
                       expect=tcip_store.Version.ABSENT)

    write_registry(root / "classes.json", _severity_registry("ordinal"))

    quarantined: set[str] = set()
    admitted = confirmed_negative_names(labels_dir, subject="leaf", date=None, quarantined_out=quarantined)
    assert admitted == set()
    assert quarantined == {"img_001.jpg"}, (
        "a type-only redefinition is a schema change; admitting the old confirmation trains an "
        "image as empty under a definition the human never reviewed"
    )


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
    reg = ClassRegistry(subjects=(
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
    assert attribute_schema_digest(reg, "bud") != attribute_schema_digest(reg, "Bud")

    assert reg.subject("BUD") is None
    with pytest.raises(RegistryError, match=r"subject 'BUD' not in registry"):
        assign_class_ids(reg, "BUD")


def test_case_variant_subjects_survive_a_file_roundtrip_as_distinct_subjects(tmp_path: Path):
    """Names are stored and read back verbatim: nothing folds two similar names into one subject,
    which would merge two label populations into a single scope."""
    reg = _case_variant_subjects()
    path = tmp_path / "classes.json"
    write_registry(path, reg)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == {"bud", "Bud"}

    back = read_registry(path)
    assert [s.name for s in back.subjects] == ["bud", "Bud"]
    assert back == reg
    assert registry_to_dict(back) == on_disk

    lower = back.subject("bud")
    upper = back.subject("Bud")
    assert lower is not None and upper is not None
    assert lower.attributes[0].values == ("open", "closed")
    assert upper.attributes[0].values == ("closed", "swelling", "partial")
    assert class_registry.num_classes(back, "Bud", "opening") == 3
