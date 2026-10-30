"""The nested subject registry, whose declared value order is every value's id.

These pin that an id is the value's declared position, not its sorted one, and that the order
survives a file round-trip.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.subject_registry import (
    Attribute,
    RegistryError,
    Subject,
    SubjectRegistry,
    read_registry,
    read_versioned_registry,
    registry_from_dict,
    registry_to_dict,
)
from tests._producer_fixtures import registry_of, registry_over, scope_over

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_store import Version

OPENING_CATEGORICAL = Attribute(
    name="opening", type="categorical", values=("closed", "open")
)
"""``opening`` over ``closed`` and ``open``, categorical: one side of a type flip."""
OPENING_ORDINAL = Attribute(
    name="opening", type="ordinal", values=("closed", "open")
)
"""The same values declared ordinal: the other side of a type flip."""


def _version(dataset_root: Path) -> Version:
    """Return the version ``dataset_root``'s stored registry reads at."""
    return read_versioned_registry(dataset_root)[1]


def _bud_bush() -> SubjectRegistry:
    return SubjectRegistry(
        subjects=(
            Subject(name="bush", description="one currant bush crown"),
            Subject(
                name="bud",
                description="a currant bud",
                attributes=(
                    Attribute(
                        name="opening",
                        type="categorical",
                        values=("closed", "open"),
                    ),
                ),
            ),
        )
    )


def test_a_subject_stating_no_description_refuses_naming_the_field() -> None:
    """A subject missing ``description`` refuses naming the field."""
    with pytest.raises(RegistryError, match="description"):
        registry_from_dict({"bud": {}})


def test_dict_and_file_roundtrip_preserve_the_registry(tmp_path: Path) -> None:
    reg = _bud_bush()
    assert registry_from_dict(registry_to_dict(reg)) == reg

    path = tmp_path / "subjects.json"
    registry_over(tmp_path, reg)
    assert read_registry(tmp_path) == reg
    # bush carries no attributes -> no 'attributes' key rather than an empty one
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "attributes" not in on_disk["bush"]
    assert on_disk["bud"]["attributes"]["opening"]["values"] == [
        "closed",
        "open",
    ]


def test_a_value_id_is_its_declared_position_never_a_sorted_one(
    tmp_path: Path,
) -> None:
    # Values declared reverse-alphabetically: a sorted/set-based id would flip these and silently
    # remap every prediction. Declared order must win, every attribute of the subject carried.
    from tcip_annotation.json_io import UNASSESSED, attribute_ids
    from tcip_annotation.state import Annotation

    reg = registry_of(
        "bud",
        Attribute(
            name="opening", type="categorical", values=("open", "closed")
        ),
        Attribute(
            name="severity", type="ordinal", values=("none", "light", "severe")
        ),
    )
    scope = scope_over(tmp_path, reg, "bud")
    assert scope.attributes == reg.subjects[0].attributes
    assert attribute_ids(
        Annotation(
            subject="bud",
            attributes={"opening": "closed", "severity": "light"},
        ),
        "bud",
        scope.attributes,
    ) == [1, 1]
    assert attribute_ids(
        Annotation(subject="bud", attributes={"opening": "open"}),
        "bud",
        scope.attributes,
    ) == [0, UNASSESSED]


def test_a_subject_with_no_attributes_reads_none_and_an_absent_subject_refuses(
    tmp_path: Path,
) -> None:
    # rail admits valid work: bush (detection-only) loads and reads an empty attribute tuple
    reg = registry_from_dict({"bush": {"description": "a bush"}})
    assert scope_over(tmp_path, reg, "bush").attributes == ()
    with pytest.raises(RegistryError):
        scope_over(tmp_path, reg, "acorn")


@pytest.mark.parametrize(
    "bad",
    [
        # numeric not allowed
        {
            "bud": {
                "attributes": {"opening": {"type": "numeric", "values": ["1"]}}
            }
        },
        {
            "bud": {
                "attributes": {
                    "opening": {"type": "categorical", "values": []}
                }
            }
        },  # empty values
        # dup values
        {
            "bud": {
                "attributes": {
                    "opening": {"type": "categorical", "values": ["a", "a"]}
                }
            }
        },
        {
            "bud": {"attributes": {"opening": {"values": ["closed"]}}}
        },  # missing type
    ],
)
def test_malformed_registry_refuses(bad: dict) -> None:
    with pytest.raises(RegistryError):
        registry_from_dict({"bud": {**bad["bud"], "description": "one bud"}})


def test_ordinal_file_roundtrip_preserves_type_and_rank(
    tmp_path: Path,
) -> None:
    # The apex on-disk risk: a writer that sorted or flattened ordinal metadata would reorder the
    # ranks. Declared order here is deliberately not alphabetical, so a sorting writer is caught.
    reg = registry_of(
        "leaf",
        Attribute(
            name="efb_severity",
            type="ordinal",
            values=("none", "light", "moderate", "severe"),
        ),
    )
    registry_over(tmp_path, reg)
    back = read_registry(tmp_path)
    assert (
        back == reg
    )  # type and declared value order both survive the round-trip
    leaf = back.subject("leaf")
    assert (
        leaf is not None and leaf.attribute("efb_severity").type == "ordinal"
    )  # type: ignore[union-attr]
    assert leaf.attribute("efb_severity").values == (  # type: ignore[union-attr]
        "none",
        "light",
        "moderate",
        "severe",
    )


def test_attribute_refuses_bad_values_at_construction() -> None:
    # The invariant travels with the type, so a positional id can never silently collapse a value,
    # however the Attribute was built, not only via the JSON parser.
    with pytest.raises(ValueError):
        Attribute(
            name="opening", type="categorical", values=("closed", "closed")
        )  # duplicate
    with pytest.raises(ValueError):
        Attribute(name="opening", type="categorical", values=())  # empty
    with pytest.raises(ValueError):
        Attribute(
            name="opening", type="numeric", values=("closed",)
        )  # numeric not an attr type


def test_absent_attributes_are_valid_and_any_other_spelling_of_none_refuses() -> (
    None
):
    bush = registry_from_dict({"bush": {"description": ""}}).subject("bush")
    assert bush is not None and bush.attributes == ()
    for bad in (None, False, 0, "", []):
        with pytest.raises(RegistryError, match="attributes"):
            registry_from_dict(
                {"bush": {"description": "", "attributes": bad}}
            )


def test_a_present_but_undecodable_registry_refuses_and_an_absent_one_says_so(
    tmp_path: Path,
) -> None:
    """An undecodable registry refuses, and an absent one says it is absent.

    An undecodable registry raises ``RegistryError`` and an absent one ``FileNotFoundError``.
    """
    path = tmp_path / "subjects.json"
    path.write_bytes(
        b'{"bud": {"description": "a currant bud"'
    )  # truncated mid-object

    with pytest.raises(RegistryError):
        read_registry(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_registry(tmp_path / "absent")

    # a registry that does decode still reads, so the refusal has not swallowed valid work.
    registry_over(tmp_path, registry_of("bud"))
    assert read_registry(tmp_path).subject("bud") is not None


def test_a_copied_registry_declares_the_same_document_in_the_same_order(
    tmp_path: Path,
) -> None:
    """A copied registry declares the same document in the same order.

    A registry copied beside another dataset's data holds its source's bytes and reads back its
    subjects in the source's order.
    """
    from tcip_mcp.subject_registry import copy_registry

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    registry_over(
        source,
        SubjectRegistry(
            subjects=(
                Subject(
                    name="bud", description="a description with an ümlaut"
                ),
                Subject(name="bush", description=""),
            )
        ),
    )

    copy_registry(source, destination)

    assert (destination / "subjects.json").read_bytes() == (
        source / "subjects.json"
    ).read_bytes()
    assert [s.name for s in read_registry(destination).subjects] == [
        "bud",
        "bush",
    ]


def test_copy_registry_refuses_when_the_destination_already_holds_one(
    tmp_path: Path,
) -> None:
    """``copy_registry`` refuses when the destination already holds a registry.

    A copy over a destination holding its own registry raises ``RegistryError``, and the
    destination's registry still declares ``bush`` and does not declare ``bud``.
    """
    from tcip_mcp.subject_registry import copy_registry

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    registry_over(source, registry_of("bud"))
    registry_over(destination, registry_of("bush"))

    with pytest.raises(RegistryError):
        copy_registry(source, destination)

    assert read_registry(destination).subject("bush") is not None
    assert read_registry(destination).subject("bud") is None


def _leaf_bush() -> SubjectRegistry:
    """Return a generic two-subject registry.

    One subject declares no attribute and the other declares one.
    """
    return SubjectRegistry(
        subjects=(
            Subject(name="bush", description="one plant crown"),
            Subject(
                name="leaf",
                description="one leaf",
                attributes=(
                    Attribute(
                        name="stage",
                        type="categorical",
                        values=("early", "late"),
                    ),
                ),
            ),
        )
    )


def test_a_registry_declaring_no_subject_refuses_by_its_type() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least 1 item"):
        SubjectRegistry(subjects=())


def test_replace_registry_first_write_succeeds_and_reports_the_save(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    # admits valid work: a first write over an absent registry, asserted with Version.ABSENT.
    result = replace_registry(
        tmp_path, _leaf_bush(), expect=Version.ABSENT, actor=None
    )
    assert read_registry(tmp_path) == _leaf_bush()
    assert result.model_dump() == {
        "subjects_path": str(tmp_path / "subjects.json"),
        "n_subjects": 2,
        "version": _version(tmp_path).token,
    }


def test_replace_registry_admits_growing_the_registry(tmp_path: Path) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    # admits valid work: adding a subject/attribute/value never drops a declared name.
    replace_registry(
        tmp_path, registry_of("bush"), expect=Version.ABSENT, actor=None
    )
    grown = SubjectRegistry(
        subjects=(
            Subject(name="bush", description=""),
            Subject(
                name="leaf",
                description="",
                attributes=(
                    Attribute(
                        name="stage",
                        type="categorical",
                        values=("early", "late"),
                    ),
                ),
            ),
        )
    )
    replace_registry(tmp_path, grown, expect=_version(tmp_path), actor=None)
    assert read_registry(tmp_path) == grown


def test_replace_registry_refuses_dropping_a_declared_name_without_allow_removals(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    replace_registry(tmp_path, _leaf_bush(), expect=Version.ABSENT, actor=None)
    shrunk = registry_of("bush")  # drops leaf
    with pytest.raises(RegistryError, match="leaf"):
        replace_registry(
            tmp_path, shrunk, expect=_version(tmp_path), actor=None
        )
    assert (
        read_registry(tmp_path) == _leaf_bush()
    )  # refusal leaves the stored registry untouched


def test_replace_registry_allows_removals_when_stated(tmp_path: Path) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    replace_registry(tmp_path, _leaf_bush(), expect=Version.ABSENT, actor=None)
    shrunk = registry_of("bush")
    replace_registry(
        tmp_path,
        shrunk,
        expect=_version(tmp_path),
        allow_removals=True,
        actor=None,
    )
    assert read_registry(tmp_path) == shrunk


def test_replace_registry_refuses_a_stale_expect_version(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version, VersionConflictError

    replace_registry(
        tmp_path, registry_of("bush"), expect=Version.ABSENT, actor=None
    )
    stale = _version(tmp_path)
    replace_registry(
        tmp_path,
        SubjectRegistry(
            subjects=(
                Subject(name="bush", description=""),
                Subject(name="leaf", description=""),
            )
        ),
        expect=stale,
        actor=None,
    )
    with pytest.raises(VersionConflictError):
        replace_registry(
            tmp_path,
            SubjectRegistry(
                subjects=(
                    Subject(name="bush", description=""),
                    Subject(name="tip", description=""),
                )
            ),
            expect=stale,
            allow_removals=True,
            actor=None,
        )


def test_replace_registry_refuses_undecodable_bytes_without_allow_removals(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry

    path = tmp_path / "subjects.json"
    path.write_bytes(
        b'{"leaf": {"description": "one leaf"'
    )  # truncated mid-object
    with pytest.raises(RegistryError):
        replace_registry(
            tmp_path, registry_of("bush"), expect=None, actor=None
        )
    assert path.read_bytes() == b'{"leaf": {"description": "one leaf"'


def test_replace_registry_repairs_undecodable_bytes_when_allow_removals(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry

    path = tmp_path / "subjects.json"
    path.write_bytes(b'{"leaf": {"description": "one leaf"')
    replace_registry(
        tmp_path,
        registry_of("bush"),
        expect=None,
        allow_removals=True,
        actor=None,
    )
    assert read_registry(tmp_path) == registry_of("bush")


def test_replace_registry_refuses_a_same_values_type_flip_without_allow_type_changes(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    categorical = registry_of("bud", OPENING_CATEGORICAL)
    replace_registry(tmp_path, categorical, expect=Version.ABSENT, actor=None)
    ordinal = registry_of("bud", OPENING_ORDINAL)

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(
            tmp_path, ordinal, expect=_version(tmp_path), actor=None
        )
    assert read_registry(tmp_path) == categorical


def test_replace_registry_refuses_the_reverse_type_flip_too(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    ordinal = registry_of("bud", OPENING_ORDINAL)
    replace_registry(tmp_path, ordinal, expect=Version.ABSENT, actor=None)
    categorical = registry_of("bud", OPENING_CATEGORICAL)

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(
            tmp_path, categorical, expect=_version(tmp_path), actor=None
        )
    assert read_registry(tmp_path) == ordinal


def test_replace_registry_allow_removals_alone_does_not_admit_a_type_flip(
    tmp_path: Path,
) -> None:
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    categorical = registry_of("bud", OPENING_CATEGORICAL)
    replace_registry(tmp_path, categorical, expect=Version.ABSENT, actor=None)
    ordinal = registry_of("bud", OPENING_ORDINAL)

    with pytest.raises(RegistryError, match="bud.opening"):
        replace_registry(
            tmp_path,
            ordinal,
            expect=_version(tmp_path),
            allow_removals=True,
            actor=None,
        )


def test_replace_registry_admits_a_type_flip_with_allow_type_changes(
    tmp_path: Path,
) -> None:
    """``replace_registry`` with ``allow_type_changes`` lands a categorical-to-ordinal flip."""
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    categorical = registry_of("bud", OPENING_CATEGORICAL)
    replace_registry(tmp_path, categorical, expect=Version.ABSENT, actor=None)

    ordinal = registry_of("bud", OPENING_ORDINAL)
    replace_registry(
        tmp_path,
        ordinal,
        expect=_version(tmp_path),
        allow_type_changes=True,
        actor=None,
    )

    assert read_registry(tmp_path) == ordinal


def test_replace_registry_admits_a_values_only_growth_and_a_same_type_resave(
    tmp_path: Path,
) -> None:
    """``replace_registry`` admits a values-only growth and a same-type re-save.

    The type-flip refusal does not fire over a growth or a re-save that restates the same type.
    """
    from tcip_mcp.subject_registry import replace_registry
    from tcip_store import Version

    two_values = registry_of("bud", OPENING_CATEGORICAL)
    replace_registry(tmp_path, two_values, expect=Version.ABSENT, actor=None)
    grown = registry_of(
        "bud",
        Attribute(
            name="opening",
            type="categorical",
            values=("closed", "partial", "open"),
        ),
    )
    replace_registry(tmp_path, grown, expect=_version(tmp_path), actor=None)
    assert read_registry(tmp_path) == grown
    replace_registry(tmp_path, grown, expect=_version(tmp_path), actor=None)
    assert read_registry(tmp_path) == grown


def test_write_subject_registry_refuses_a_type_flip_without_the_flag(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "categorical",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        actor=None,
    )
    result = write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "ordinal",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        actor=None,
    )

    assert "error" in result and "allow_type_changes" in result["error"]
    assert (
        read_registry(tmp_path).subject("bud").attribute("opening").type
        == "categorical"
    )


def test_write_subject_registry_allow_removals_alone_still_refuses_a_type_flip(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "categorical",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        actor=None,
    )
    result = write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "ordinal",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        allow_removals=True,
        actor=None,
    )

    assert "error" in result and "allow_type_changes" in result["error"]


def test_write_subject_registry_admits_a_type_flip_with_allow_type_changes(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "categorical",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        actor=None,
    )
    result = write_subject_registry(
        tmp_path,
        str(tmp_path),
        {
            "bud": {
                "description": "one bud",
                "attributes": {
                    "opening": {
                        "type": "ordinal",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        allow_type_changes=True,
        actor=None,
    )

    assert "error" not in result
    assert (
        read_registry(tmp_path).subject("bud").attribute("opening").type
        == "ordinal"
    )


def test_write_subject_registry_refuses_dropping_a_declared_subject_without_allow_removals(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    write_subject_registry(
        tmp_path,
        str(tmp_path),
        {"leaf": {"description": "one leaf"}, "bush": {"description": "one bush"}},
        actor=None,
    )
    result = write_subject_registry(
        tmp_path, str(tmp_path), {"bush": {"description": "one bush"}}, actor=None
    )  # drops leaf

    assert "error" in result and "leaf" in result["error"]
    assert read_registry(tmp_path).subject("leaf") is not None


def test_write_subject_registry_admits_a_removal_stated_as_deliberate(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    write_subject_registry(
        tmp_path,
        str(tmp_path),
        {"leaf": {"description": "one leaf"}, "bush": {"description": "one bush"}},
        actor=None,
    )
    result = write_subject_registry(
        tmp_path,
        str(tmp_path),
        {"bush": {"description": "one bush"}},
        allow_removals=True,
        actor=None,
    )

    assert "error" not in result
    assert read_registry(tmp_path).subject("leaf") is None


def test_write_subject_registry_refuses_undecodable_existing_bytes_without_allow_removals(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    path = tmp_path / "subjects.json"
    path.write_bytes(
        b'{"leaf": {"description": "one leaf"'
    )  # truncated mid-object

    result = write_subject_registry(
        tmp_path, str(tmp_path), {"bush": {"description": "one bush"}}, actor=None
    )

    assert "error" in result
    assert path.read_bytes() == b'{"leaf": {"description": "one leaf"'

    stated = {"bush": {"description": "one bush"}}
    repaired = write_subject_registry(
        tmp_path, str(tmp_path), stated, allow_removals=True, actor=None
    )
    assert "error" not in repaired
    assert read_registry(tmp_path) == registry_from_dict(stated)
