"""The browser's server-declared types are a projection of the models that declare them.

``tools/generate_frontend_types.py`` renders ``frontend/src/api/types.generated.ts`` from
``routes/images.py``, ``routes/training.py``, ``routes/terminal.py`` and
``tcip_web.state.GuiVocabulary`` (the declared pydantic models), the server's ``Literal``
vocabularies the browser names, plus a handful of runtime constants from ``routes/images.py``,
``tcip_mcp.web_client`` and ``tcip_web.jobstore``. These tests hold that projection to what the
models produce, and check that no other frontend module declares an interface with a generated
field set or a bare union alias matching a generated vocabulary.
"""

from __future__ import annotations

import importlib.util
import re
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import BaseModel

from tests import FRONTEND_SRC, REPO_ROOT

if TYPE_CHECKING:
    from types import ModuleType

GENERATED = FRONTEND_SRC / "api" / "types.generated.ts"
GENERATOR = REPO_ROOT / "tools" / "generate_frontend_types.py"

_TS_BLOCK_RE = re.compile(r"(?:interface|type)\s+\w+\s*=?\s*\{(.*?)\n\}", re.S)
_TS_FIELD_RE = re.compile(r"^\s+(\w+)(\??):", re.M)
_FIELD_UNION_RE = re.compile(
    r'\w+\??:\s*((?:"[^"]+"\s*\|\s*)+"[^"]+")(?:\s*\|\s*null)?[;,]'
)
_TYPE_ALIAS_UNION_RE = re.compile(
    r'^\s*(?:export\s+)?type\s+(\w+)\s*=\s*((?:"[^"]+"\s*\|\s*)+"[^"]+")\s*;',
    re.M,
)
_QUOTED_RE = re.compile(r'"([^"]+)"')


def _generator() -> ModuleType:
    """Load and return the type-module generator, ``tools/generate_frontend_types.py``."""
    spec = importlib.util.spec_from_file_location(
        "tcip_frontend_type_generator", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generated_types_module_is_what_the_declared_models_produce() -> (
    None
):
    """The checked-in types module equals what the generator renders from the declared models."""
    generated = _generator().render()
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        "packages/tcip-web/frontend/src/api/types.generated.ts is out of date; "
        "run python tools/generate_frontend_types.py"
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


def _generated_field_sets() -> list[set[str]]:
    text = GENERATED.read_text(encoding="utf-8")
    return [
        {name for name, _ in _TS_FIELD_RE.findall(block)}
        for block in _TS_BLOCK_RE.findall(text)
    ]


def test_no_other_frontend_module_declares_an_interface_with_a_generated_field_set() -> (
    None
):
    """No other frontend module declares an interface with a generated field set."""
    generated_sets = [s for s in _generated_field_sets() if s]
    assert generated_sets, "no interfaces parsed out of the generated module"

    sources = [
        p
        for p in sorted(
            list(FRONTEND_SRC.rglob("*.ts"))
            + list(FRONTEND_SRC.rglob("*.tsx"))
        )
        if ".test." not in p.name
        and "test" not in p.parent.name
        and p != GENERATED
    ]
    assert sources, "no frontend sources found, so nothing was checked"
    offenders = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for block in _TS_BLOCK_RE.findall(text):
            fields = {name for name, _ in _TS_FIELD_RE.findall(block)}
            if fields and fields in generated_sets:
                offenders.append(source.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        "these declare an interface with the same field set as a generated coverage type:\n"
        + "\n".join(offenders)
    )


def _generated_vocabularies() -> list[set[str]]:
    """Return the vocabularies the generator emits.

    Each generated field's literal union, each runtime constant, and each literal alias.
    """
    text = GENERATED.read_text(encoding="utf-8")
    generator = _generator()
    return (
        [
            set(_QUOTED_RE.findall(m.group(1)))
            for m in _FIELD_UNION_RE.finditer(text)
        ]
        + [set(v) for v in generator.tuple_constants().values()]
        + [set(v) for v in generator.literal_aliases().values()]
    )


def test_no_frontend_module_hand_declares_a_generated_vocabulary() -> None:
    """No frontend module hand-declares a union alias matching a generated vocabulary.

    A bare ``type X = "a" | "b" | ...;`` alias whose members exactly match a vocabulary the
    generator emits is an offender.
    """
    generated_sets = [s for s in _generated_vocabularies() if len(s) > 1]
    assert generated_sets, "no vocabularies read from the generator"

    sources = [
        p
        for p in sorted(
            list(FRONTEND_SRC.rglob("*.ts"))
            + list(FRONTEND_SRC.rglob("*.tsx"))
        )
        if ".test." not in p.name
        and "test" not in p.parent.name
        and p != GENERATED
    ]
    offenders = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for m in _TYPE_ALIAS_UNION_RE.finditer(text):
            if set(_QUOTED_RE.findall(m.group(2))) in generated_sets:
                offenders.append(
                    f"{source.relative_to(REPO_ROOT).as_posix()}: {m.group(1)}"
                )
    assert not offenders, (
        "these declare a union alias matching a generated vocabulary by hand:\n"
        + "\n".join(offenders)
    )
