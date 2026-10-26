"""The browser's server-declared types are a projection of the models that declare them.

``tools/generate_frontend_types.py`` renders the module at its ``GENERATED_PATH`` from the models
its ``declared_models`` lists and the constants and vocabularies it names. These tests hold that
projection to what the models produce, and check that no other frontend module declares an
interface with a generated field set or a bare union alias matching a generated vocabulary.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import BaseModel

from tests import REPO_ROOT
from tests._route_walk import route_generator

if TYPE_CHECKING:
    from types import ModuleType


def _generator() -> ModuleType:
    """Import and return the type-module generator, ``tools.generate_frontend_types``."""
    return importlib.import_module("tools.generate_frontend_types")


GENERATED = _generator().GENERATED_PATH


@pytest.fixture(scope="module")
def frontend() -> dict:
    """Return the route generator's own read of the frontend's sources, run once for the module.

    Its ``shapes`` and ``unions`` are the sources' declarations as the TypeScript compiler
    parses them.
    """
    return route_generator().frontend_reads()


def _source(declaration: dict) -> str:
    """Return the repository-relative path of the file ``declaration`` lies in."""
    return Path(declaration["file"]).relative_to(REPO_ROOT).as_posix()


def test_the_generated_types_module_is_what_the_declared_models_produce() -> (
    None
):
    """The checked-in types module equals what the generator renders from the declared models."""
    generated = _generator().render()
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        f"{GENERATED} is out of date; run python -m {_generator().__name__}"
    )


class _OneMemberLiteral(BaseModel):
    """A scratch model whose one field is a single-value ``Literal``."""

    kind: Literal["only"]


class _NullableStringList(BaseModel):
    """A scratch model whose one field is an array of a nullable union."""

    values: list[str | None]


class _FreeFormObject(BaseModel):
    """A scratch model whose one field is a bare ``dict``."""

    payload: dict


def test_a_one_member_literal_renders_its_own_value_as_the_type() -> None:
    """A one-member ``Literal`` renders its own value as the type."""
    generator = _generator()
    schema = _OneMemberLiteral.model_json_schema()
    assert (
        generator._ts_type(
            schema["properties"]["kind"], "_OneMemberLiteral.kind"
        )
        == '"only"'
    )


def test_a_union_used_as_an_array_item_is_parenthesized() -> None:
    generator = _generator()
    schema = _NullableStringList.model_json_schema()
    assert (
        generator._ts_type(
            schema["properties"]["values"], "_NullableStringList.values"
        )
        == "(string | null)[]"
    )


def test_a_free_form_object_renders_as_a_record_of_unknown() -> None:
    """A bare ``dict`` field renders as ``Record<string, unknown>``."""
    generator = _generator()
    schema = _FreeFormObject.model_json_schema()
    assert (
        generator._ts_type(
            schema["properties"]["payload"], "_FreeFormObject.payload"
        )
        == "Record<string, unknown>"
    )


def test_reordering_platform_panel_events_does_not_change_a_named_constants_value() -> (
    None
):
    """A named ``PANEL_EVENT_*`` constant keeps its value when the event tuple is reordered."""
    from tcip_mcp import web_client

    generator = _generator()
    before = generator.platform_panel_event_constants()
    original = web_client.PLATFORM_PANEL_EVENTS
    try:
        web_client.PLATFORM_PANEL_EVENTS = tuple(reversed(original))
        after = generator.platform_panel_event_constants()
    finally:
        web_client.PLATFORM_PANEL_EVENTS = original
    assert before == after


def test_an_object_with_its_own_fixed_property_set_still_refuses() -> None:
    """A fixed-shape object schema outside a ``$ref`` refuses."""
    generator = _generator()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(SystemExit):
        generator._ts_type(schema, "_fixed_object")


def test_no_other_frontend_module_declares_an_interface_with_a_generated_field_set(
    frontend: dict,
) -> None:
    """No other frontend module declares an interface with a generated field set."""
    generated_sets = [
        set(shape["fields"])
        for shape in frontend["shapes"]
        if Path(shape["file"]) == GENERATED and shape["fields"]
    ]
    assert generated_sets, "no interfaces read from the generated module"
    others = [
        shape
        for shape in frontend["shapes"]
        if Path(shape["file"]) != GENERATED
    ]
    assert others, "no interface read from a source but the generated module"

    offenders = [
        _source(shape)
        for shape in others
        if shape["fields"] and set(shape["fields"]) in generated_sets
    ]
    assert not offenders, (
        "these declare an interface with the same field set as a generated coverage type:\n"
        + "\n".join(offenders)
    )


def test_no_frontend_module_hand_declares_a_generated_vocabulary(
    frontend: dict,
) -> None:
    """No frontend module hand-declares a union alias matching a generated vocabulary.

    The vocabularies are each string literal union the generated module declares, each tuple
    constant and each record constant's keys; a type alias of a string literal union whose
    members exactly match one is an offender.
    """
    generator = _generator()
    unions = [
        set(union["members"])
        for union in frontend["unions"]
        if Path(union["file"]) == GENERATED
    ]
    assert unions, "no string literal union read from the generated module"
    vocabularies = (
        unions
        + [set(v) for v in generator.tuple_constants().values()]
        + [set(v) for v in generator.record_constants().values()]
    )
    generated_sets = [s for s in vocabularies if len(s) > 1]

    offenders = [
        f"{_source(union)}: {union['alias']}"
        for union in frontend["unions"]
        if Path(union["file"]) != GENERATED
        and union["alias"] is not None
        and set(union["members"]) in generated_sets
    ]
    assert not offenders, (
        "these declare a union alias matching a generated vocabulary by hand:\n"
        + "\n".join(offenders)
    )
