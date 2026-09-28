"""``ClassScope.of`` over a run's recorded class space, the ``scope`` its checkpoint's config
records on ``data``."""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.data.selection import ClassScope


def _scope(scope: dict) -> ClassScope:
    """The run's own recorded class space, read the way every door reads it."""
    return ClassScope.of({"scope": scope})


def test_the_decoder_reads_the_recorded_class_space() -> None:
    scope = _scope({"subject": "bud", "attribute": "bud_opening", "id_map": {"open": 0}})
    assert scope == ClassScope("bud", "bud_opening", {"open": 0})


def test_the_decoder_reads_a_detector_runs_bare_subject() -> None:
    assert _scope({"subject": "bud"}) == ClassScope("bud", None, None)


def test_the_decoder_reads_an_explicit_empty_scope_as_an_empty_class_space() -> None:
    """A mask or table run's admission records the empty scope explicitly."""
    assert _scope({"subject": None, "attribute": None, "id_map": None}) == ClassScope()


def test_the_decoder_refuses_a_data_section_recording_no_scope() -> None:
    """A missing scope is not an empty one: it names nothing a reader could hold predictions to."""
    with pytest.raises(ValueError, match="carries no scope"):
        ClassScope.of({"num_channels": 3})


def test_the_decoder_refuses_a_recorded_attribute_with_no_subject() -> None:
    with pytest.raises(ValueError, match="attribute 'bud_opening' is stated with no subject"):
        _scope({"attribute": "bud_opening"})
