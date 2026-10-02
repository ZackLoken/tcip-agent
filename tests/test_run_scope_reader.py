"""``ClassScope.of`` over a run's recorded class space, the ``scope`` its checkpoint's config
records on ``data``."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.subject_registry import Attribute


def _scope(scope: dict) -> ClassScope:
    """The run's own recorded class space, read the way every door reads it."""
    return ClassScope.of({"scope": scope})


def test_the_decoder_reads_back_the_attribute_records_the_writer_wrote() -> None:
    written = ClassScope("bud", (Attribute("color", "categorical", ("red", "blue")),
                                 Attribute("grade", "ordinal", ("low", "high"))))
    assert _scope(asdict(written)) == written


def test_the_decoder_reads_a_stated_bare_subject_as_attributes_not_yet_read() -> None:
    assert _scope({"subject": "bud"}) == ClassScope("bud", None)


def test_the_decoder_reads_an_explicit_empty_scope_as_an_empty_class_space() -> None:
    """A mask or table run's admission records the empty scope explicitly."""
    assert _scope(asdict(ClassScope())) == ClassScope()


def test_the_decoder_refuses_a_data_section_recording_no_scope() -> None:
    """A missing scope is not an empty one: it names nothing a reader could hold predictions to."""
    with pytest.raises(ValueError, match="carries no scope"):
        ClassScope.of({"num_channels": 3})
