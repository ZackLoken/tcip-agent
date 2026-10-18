"""``ClassScope.of`` over a run's recorded class space.

The class space is the ``scope`` the run's checkpoint config records on ``data``.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.subject_registry import Attribute


def _scope(scope: dict) -> ClassScope:
    """Return the class space ``ClassScope.of`` reads from a data section recording ``scope``."""
    return ClassScope.of({"scope": scope})


def test_the_decoder_reads_back_the_attribute_records_the_writer_wrote() -> (
    None
):
    written = ClassScope(
        "bud",
        (
            Attribute("color", "categorical", ("red", "blue")),
            Attribute("grade", "ordinal", ("low", "high")),
        ),
    )
    assert _scope(asdict(written)) == written


def test_the_decoder_reads_a_stated_bare_subject_as_attributes_not_yet_read() -> (
    None
):
    assert _scope({"subject": "bud"}) == ClassScope("bud", None)


def test_the_decoder_reads_an_explicit_empty_scope_as_an_empty_class_space() -> (
    None
):
    """An explicitly recorded empty scope decodes as an empty class space."""
    assert _scope(asdict(ClassScope())) == ClassScope()


def test_the_decoder_refuses_a_data_section_recording_no_scope() -> None:
    """A data section recording no scope is refused, naming the missing scope."""
    with pytest.raises(ValueError, match="carries no scope"):
        ClassScope.of({"num_channels": 3})
