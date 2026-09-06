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


def _bud_bush() -> ClassRegistry:
    return ClassRegistry(subjects=(
        Subject(name="bush", description="one currant bush crown", defined_by="user:breeder"),
        Subject(
            name="bud",
            description="a currant bud",
            defined_by="user:breeder",
            attributes=(Attribute(name="opening", type="categorical", values=("closed", "open")),),
        ),
    ))


def test_dict_and_file_roundtrip_preserve_the_registry(tmp_path):
    reg = _bud_bush()
    assert registry_from_dict(registry_to_dict(reg)) == reg

    path = tmp_path / "classes.json"
    write_registry(path, reg)
    assert read_registry(path) == reg
    # bush carries no attributes -> no 'attributes' key rather than an empty one
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "attributes" not in on_disk["bush"]
    assert on_disk["bud"]["attributes"]["opening"]["values"] == ["closed", "open"]


def test_assign_ids_follow_declared_order_not_sorted():
    # Values declared reverse-alphabetically: a sorted/set-based assignment would flip these ids and
    # silently remap every prediction. Declared order must win.
    reg = ClassRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("open", "closed")),)),
    ))
    assert assign_class_ids(reg, "bud", "opening") == {"open": 0, "closed": 1}


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
    reg = _bud_bush()
    first = assign_class_ids(reg, "bud", "opening")
    assert first == assign_class_ids(reg, "bud", "opening")  # repeatable

    path = tmp_path / "classes.json"
    write_registry(path, reg)
    assert assign_class_ids(read_registry(path), "bud", "opening") == first  # survives round-trip


def test_no_attribute_scope_is_single_class():
    reg = _bud_bush()
    assert assign_class_ids(reg, "bush") == {"bush": 0}  # subject with no attributes
    assert assign_class_ids(reg, "bud") == {"bud": 0}  # attributes exist but scope ignores them
    assert num_classes(reg, "bush") == 1


def test_decode_inverts_the_recorded_map():
    reg = _bud_bush()
    id_map = assign_class_ids(reg, "bud", "opening")
    assert decode_class_ids(id_map) == {0: "closed", 1: "open"}


def test_absent_subject_or_attribute_refuses():
    reg = _bud_bush()
    with pytest.raises(RegistryError):
        assign_class_ids(reg, "acorn")
    with pytest.raises(RegistryError):
        assign_class_ids(reg, "bud", "sex")  # not declared on bud


def test_a_subject_with_no_attributes_is_valid():
    # rail admits valid work: bush (detection-only) loads and assigns, it is not treated as malformed
    reg = registry_from_dict({"bush": {"description": "a bush"}})
    assert reg.subject("bush") is not None
    assert assign_class_ids(reg, "bush") == {"bush": 0}


@pytest.mark.parametrize("bad", [
    {"bud": {"attributes": {"opening": {"type": "numeric", "values": ["1"]}}}},  # numeric not allowed
    {"bud": {"attributes": {"opening": {"type": "categorical", "values": []}}}},  # empty values
    {"bud": {"attributes": {"opening": {"type": "categorical", "values": ["a", "a"]}}}},  # dup values
    {"bud": {"attributes": {"opening": {"values": ["closed"]}}}},  # missing type
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
        Attribute(name="opening", type="categorical", values=("closed", "closed"))  # duplicate
    with pytest.raises(ValueError):
        Attribute(name="opening", type="categorical", values=())  # empty
    with pytest.raises(ValueError):
        Attribute(name="opening", type="numeric", values=("closed",))  # numeric not an attr type


def test_absent_or_null_attributes_valid_but_falsy_malformed_refused():
    for ok in ({"bush": {}}, {"bush": {"attributes": None}}):  # absent / null -> detection-only subject
        bush = registry_from_dict(ok).subject("bush")
        assert bush is not None and bush.attributes == ()
    for bad in (False, 0, "", []):  # a falsy but wrong-typed attributes is malformed, not "none"
        with pytest.raises(RegistryError):
            registry_from_dict({"bush": {"attributes": bad}})


def test_a_present_but_undecodable_registry_refuses_and_an_absent_one_says_so(tmp_path):
    """Absence and corruption are different answers, and neither is an empty registry: reading an
    undecodable one as empty leaves every name-based label under it with nothing to resolve against.
    """
    path = tmp_path / "classes.json"
    path.write_bytes(b'{"bud": {"description": "a currant bud"')  # truncated mid-object

    with pytest.raises(RegistryError):
        read_registry(path)
    with pytest.raises(FileNotFoundError):
        read_registry(tmp_path / "absent" / "classes.json")

    # a registry that does decode still reads, so the refusal has not swallowed valid work.
    write_registry(path, ClassRegistry(subjects=(Subject(name="bud"),)))
    assert read_registry(path).subject("bud") is not None


def test_a_copied_registry_declares_the_same_document_in_the_same_order(tmp_path):
    """A registry placed beside another dataset's data declares exactly what its source declares,
    so a digest or an id assignment taken against either one reads the same declared order.

    Every record now shares one spelling, so equal bytes no longer distinguish a copy from a
    re-serialization; what this pins is that the document and its subject order survive, and
    that the copy carries the canonical bytes rather than a spelling of its own.
    """
    from tcip_mcp.class_registry import copy_registry

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    write_registry(source / "classes.json", ClassRegistry(subjects=(
        Subject(name="bud", description="a männlich flower"),
        Subject(name="bush"),
    )))

    copy_registry(source / "classes.json", destination / "classes.json")

    assert (destination / "classes.json").read_bytes() == (source / "classes.json").read_bytes()
    assert [s.name for s in read_registry(destination / "classes.json").subjects] == \
        ["bud", "bush"]


def test_copy_registry_refuses_when_the_destination_already_holds_one(tmp_path):
    """A materialization or split re-run must not replace a destination registry silently: the
    first copy is create-only, and a second copy attempt over the same destination refuses."""
    from tcip_mcp.class_registry import copy_registry

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    write_registry(source / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))
    write_registry(destination / "classes.json", ClassRegistry(subjects=(Subject(name="bush"),)))

    with pytest.raises(RegistryError):
        copy_registry(source / "classes.json", destination / "classes.json")

    assert read_registry(destination / "classes.json").subject("bush") is not None
    assert read_registry(destination / "classes.json").subject("bud") is None


def _leaf_bush() -> ClassRegistry:
    """A generic two-subject registry (a detection-only subject, a classified one), for the
    registry-write tests below."""
    return ClassRegistry(subjects=(
        Subject(name="bush", description="one plant crown", defined_by="user:breeder"),
        Subject(
            name="leaf",
            description="one leaf",
            defined_by="user:breeder",
            attributes=(Attribute(name="stage", type="categorical", values=("early", "late")),),
        ),
    ))


def test_replace_registry_refuses_an_empty_registry(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    with pytest.raises(RegistryError):
        replace_registry(path, ClassRegistry(subjects=()), expect=Version.ABSENT)


def test_replace_registry_first_write_succeeds_and_reports_no_sweep(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    # admits valid work: a first write over an absent registry, asserted with Version.ABSENT.
    path = tmp_path / "classes.json"
    result = replace_registry(path, _leaf_bush(), expect=Version.ABSENT)
    assert read_registry(path) == _leaf_bush()
    assert result["schema_change_sweep"] == {
        "newly_stamped": {}, "predating_vocabulary": {}, "warning": None}
    assert result["version"] == read_version(path)


def test_replace_registry_admits_growing_the_registry(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    # admits valid work: adding a subject/attribute/value never drops a declared name.
    path = tmp_path / "classes.json"
    replace_registry(path, ClassRegistry(subjects=(Subject(name="bush"),)), expect=Version.ABSENT)
    grown = ClassRegistry(subjects=(
        Subject(name="bush"),
        Subject(name="leaf", attributes=(
            Attribute(name="stage", type="categorical", values=("early", "late")),)),
    ))
    replace_registry(path, grown, expect=read_version(path))
    assert read_registry(path) == grown


def test_replace_registry_refuses_dropping_a_declared_name_without_allow_removals(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    replace_registry(path, _leaf_bush(), expect=Version.ABSENT)
    shrunk = ClassRegistry(subjects=(Subject(name="bush"),))  # drops leaf
    with pytest.raises(RegistryError, match="leaf"):
        replace_registry(path, shrunk, expect=read_version(path))
    assert read_registry(path) == _leaf_bush()  # refusal leaves the stored registry untouched


def test_replace_registry_allows_removals_when_stated(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    replace_registry(path, _leaf_bush(), expect=Version.ABSENT)
    shrunk = ClassRegistry(subjects=(Subject(name="bush"),))
    replace_registry(path, shrunk, expect=read_version(path), allow_removals=True)
    assert read_registry(path) == shrunk


def test_replace_registry_refuses_a_stale_expect_version(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version, VersionConflict

    path = tmp_path / "classes.json"
    replace_registry(path, ClassRegistry(subjects=(Subject(name="bush"),)), expect=Version.ABSENT)
    stale = read_version(path)
    replace_registry(
        path, ClassRegistry(subjects=(Subject(name="bush"), Subject(name="leaf"))), expect=stale)
    with pytest.raises(VersionConflict):
        replace_registry(
            path, ClassRegistry(subjects=(Subject(name="bush"), Subject(name="tip"))),
            expect=stale, allow_removals=True)


def test_replace_registry_refuses_undecodable_bytes_without_allow_removals(tmp_path):
    from tcip_mcp.class_registry import replace_registry

    path = tmp_path / "classes.json"
    path.write_bytes(b'{"leaf": {"description": "one leaf"')  # truncated mid-object
    with pytest.raises(RegistryError):
        replace_registry(path, ClassRegistry(subjects=(Subject(name="bush"),)), expect=None)
    assert path.read_bytes() == b'{"leaf": {"description": "one leaf"'


def test_replace_registry_repairs_undecodable_bytes_when_allow_removals(tmp_path):
    from tcip_mcp.class_registry import replace_registry

    path = tmp_path / "classes.json"
    path.write_bytes(b'{"leaf": {"description": "one leaf"')
    replace_registry(
        path, ClassRegistry(subjects=(Subject(name="bush"),)), expect=None, allow_removals=True)
    assert read_registry(path) == ClassRegistry(subjects=(Subject(name="bush"),))


def test_a_failed_compare_and_set_leaves_no_confirmation_stamp_behind(tmp_path):
    """A schema-changing write that loses its compare-and-set must not have swept the digest
    store first: the stamp is tied to the write actually landing, not to the attempt."""
    from tcip_mcp.class_registry import replace_registry
    from tcip_mcp.dataset_layout import (
        image_status_digest_key, record_image_statuses, status_bucket,
    )
    from tcip_store import Version, VersionConflict
    import tcip_store as ts

    path = tmp_path / "classes.json"
    two_states = ClassRegistry(subjects=(Subject(name="leaf", attributes=(
        Attribute(name="stage", type="categorical", values=("early", "late")),)),))
    replace_registry(path, two_states, expect=Version.ABSENT)
    record_image_statuses(
        tmp_path, status_bucket("leaf", None), {"img.jpg": "negative"}, recorded_by="user:breeder")
    assert not ts.exists(image_status_digest_key(tmp_path))

    three_states = ClassRegistry(subjects=(Subject(name="leaf", attributes=(
        Attribute(name="stage", type="categorical", values=("early", "mid", "late")),)),))
    with pytest.raises(VersionConflict):
        replace_registry(path, three_states, expect=Version("not-the-stored-token"))

    assert not ts.exists(image_status_digest_key(tmp_path))
    assert read_registry(path) == two_states


def test_replace_registry_refuses_a_same_values_type_flip_without_allow_type_changes(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    categorical = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    replace_registry(path, categorical, expect=Version.ABSENT)
    ordinal = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="ordinal", values=("closed", "open")),)),))

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(path, ordinal, expect=read_version(path))
    assert read_registry(path) == categorical


def test_replace_registry_refuses_the_reverse_type_flip_too(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    ordinal = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="ordinal", values=("closed", "open")),)),))
    replace_registry(path, ordinal, expect=Version.ABSENT)
    categorical = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),)),))

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(path, categorical, expect=read_version(path))
    assert read_registry(path) == ordinal


def test_replace_registry_allow_removals_alone_does_not_admit_a_type_flip(tmp_path):
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    categorical = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    replace_registry(path, categorical, expect=Version.ABSENT)
    ordinal = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="ordinal", values=("closed", "open")),)),))

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(path, ordinal, expect=read_version(path), allow_removals=True)


def test_replace_registry_admits_a_type_flip_with_allow_type_changes_and_sweeps_it(tmp_path):
    """A flip lands under the flag, and the confirmation-digest sweep stales a previously stamped
    finished status under the subject exactly as a value change would."""
    from tcip_mcp.class_registry import attribute_schema_digest, read_version, replace_registry
    from tcip_mcp.dataset_layout import (
        image_status_digest_key, record_image_statuses, stamp_image_status_digests, status_bucket,
    )
    from tcip_store import Version
    import tcip_store as ts

    path = tmp_path / "classes.json"
    categorical = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    replace_registry(path, categorical, expect=Version.ABSENT)
    record_image_statuses(
        tmp_path, status_bucket("bud", None), {"img.jpg": "complete"}, recorded_by="user:breeder")
    old_digest = attribute_schema_digest(categorical, "bud")
    stamp_image_status_digests(tmp_path, status_bucket("bud", None), ["img.jpg"], old_digest)

    ordinal = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="ordinal", values=("closed", "open")),)),))
    result = replace_registry(
        path, ordinal, expect=read_version(path), allow_type_changes=True)

    assert read_registry(path) == ordinal
    assert result["schema_change_sweep"]["predating_vocabulary"] == {"bud": 1}
    assert ts.read(image_status_digest_key(tmp_path)).get(
        status_bucket("bud", None), {}).get("img.jpg") == old_digest


def test_replace_registry_admits_a_values_only_growth_and_a_same_type_resave(tmp_path):
    """The type-flip refusal never fires over a growth or a re-save that restates the same type,
    the existing sweep tests' own coverage of those shapes."""
    from tcip_mcp.class_registry import read_version, replace_registry
    from tcip_store import Version

    path = tmp_path / "classes.json"
    two_values = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    replace_registry(path, two_values, expect=Version.ABSENT)
    grown = ClassRegistry(subjects=(Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "partial", "open")),)),))
    replace_registry(path, grown, expect=read_version(path))
    assert read_registry(path) == grown
    replace_registry(path, grown, expect=read_version(path))
    assert read_registry(path) == grown


def test_write_class_map_refuses_a_type_flip_without_the_flag(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "categorical", "values": ["closed", "open"]}}}})
    result = write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "ordinal", "values": ["closed", "open"]}}}})

    assert "error" in result and "allow_type_changes" in result["error"]
    assert read_registry(tmp_path / "classes.json").subject("bud").attribute("opening").type == \
        "categorical"


def test_write_class_map_allow_removals_alone_still_refuses_a_type_flip(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "categorical", "values": ["closed", "open"]}}}})
    result = write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "ordinal", "values": ["closed", "open"]}}}}, allow_removals=True)

    assert "error" in result and "allow_type_changes" in result["error"]


def test_write_class_map_admits_a_type_flip_with_allow_type_changes(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "categorical", "values": ["closed", "open"]}}}})
    result = write_class_map(str(tmp_path), {"bud": {"attributes": {
        "opening": {"type": "ordinal", "values": ["closed", "open"]}}}}, allow_type_changes=True)

    assert "error" not in result
    assert read_registry(tmp_path / "classes.json").subject("bud").attribute("opening").type == \
        "ordinal"


def test_write_class_map_refuses_dropping_a_declared_subject_without_allow_removals(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(tmp_path), {"leaf": {"description": "one leaf"}, "bush": {}})
    result = write_class_map(str(tmp_path), {"bush": {}})  # drops leaf

    assert "error" in result and "leaf" in result["error"]
    assert read_registry(tmp_path / "classes.json").subject("leaf") is not None


def test_write_class_map_admits_a_removal_stated_as_deliberate(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(tmp_path), {"leaf": {"description": "one leaf"}, "bush": {}})
    result = write_class_map(str(tmp_path), {"bush": {}}, allow_removals=True)

    assert "error" not in result
    assert read_registry(tmp_path / "classes.json").subject("leaf") is None


def test_write_class_map_refuses_undecodable_existing_bytes_without_allow_removals(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map

    path = tmp_path / "classes.json"
    path.write_bytes(b'{"leaf": {"description": "one leaf"')  # truncated mid-object

    result = write_class_map(str(tmp_path), {"bush": {}})

    assert "error" in result
    assert path.read_bytes() == b'{"leaf": {"description": "one leaf"'

    repaired = write_class_map(str(tmp_path), {"bush": {}}, allow_removals=True)
    assert "error" not in repaired
    assert read_registry(path) == ClassRegistry(subjects=(Subject(name="bush"),))
