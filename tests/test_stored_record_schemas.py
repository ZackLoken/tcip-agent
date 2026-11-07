"""The schema of every stored record is held to a checked-in snapshot.

``tools/generate_record_schemas.py`` renders the JSON schema each stored record model validates
against into ``tests/stored_record_schemas.json``. A change to such a model moves the render, and
the snapshot is regenerated in the same change, which is where the change states how the records
already written are conformed.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter

from tcip_mcp.experiments import RunRecord
from tests import write_generated

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _generator() -> ModuleType:
    """Import and return the schema snapshot generator, ``tools.generate_record_schemas``."""
    return importlib.import_module("tools.generate_record_schemas")


def _assert_snapshot_is(rendered: str) -> None:
    """Assert the checked-in snapshot is ``rendered``, naming the command that regenerates it."""
    generator = _generator()
    assert generator.SNAPSHOT_PATH.read_text(encoding="utf-8") == rendered, (
        f"{generator.SNAPSHOT_PATH} is out of date; run python -m {generator.__name__}"
    )


class _RunRecordWithAnotherField(RunRecord):
    """A scratch extension of a stored record model: one more required field."""

    conform_marker: str


def test_stored_record_schemas_match_the_snapshot() -> None:
    """The checked-in snapshot equals the render of every stored record model."""
    generator = _generator()
    _assert_snapshot_is(generator.render(generator.stored_record_adapters()))


def test_a_schema_change_moves_the_snapshot() -> None:
    """A stored record model extended by a required field fails the snapshot at its entry alone."""
    generator = _generator()
    name = f"{__name__}._RunRecordWithAnotherField"
    extended = {
        **generator.stored_record_adapters(),
        name: TypeAdapter(_RunRecordWithAnotherField),
    }
    rendered = generator.render(extended)

    with pytest.raises(
        AssertionError,
        match="is out of date; run python -m tools.generate_record_schemas",
    ):
        _assert_snapshot_is(rendered)

    snapshot = json.loads(generator.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    moved = json.loads(rendered)
    assert moved.keys() - snapshot.keys() == {name}
    assert {k: v for k, v in moved.items() if k != name} == snapshot
    assert "conform_marker" in moved[name]["required"]


def test_regenerating_an_unchanged_tree_rewrites_the_snapshot_byte_for_byte(
    tmp_path: Path,
) -> None:
    """The generator's writer lays down exactly the bytes of the checked-in snapshot."""
    generator = _generator()
    written = tmp_path / "stored_record_schemas.json"
    write_generated(
        written, generator.render(generator.stored_record_adapters())
    )
    assert written.read_bytes() == generator.SNAPSHOT_PATH.read_bytes()
