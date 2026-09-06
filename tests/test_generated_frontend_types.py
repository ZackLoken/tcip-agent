"""The browser's coverage-record types are a projection of the pydantic models that declare them.

``tools/generate_frontend_types.py`` renders ``frontend/src/api/types.generated.ts`` from
``routes/_coverage_models.py``, ``routes/coverage.py``, ``routes/review.py``,
``routes/training.py``, ``routes/terminal.py`` and ``tcip_web.state.GuiVocabulary`` (the declared
pydantic models), plus a handful of runtime constants from ``routes/images.py``,
``tcip_mcp.web_client`` and ``tcip_web.jobstore``; these tests hold that projection to what the
models produce, and keep another frontend module from declaring an interface with the same
field set as a generated one, or a bare union alias with the same members as one of a generated
interface's own literal-union fields.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
GENERATED = FRONTEND_SRC / "api" / "types.generated.ts"
GENERATOR = REPO_ROOT / "tools" / "generate_frontend_types.py"

_TS_BLOCK_RE = re.compile(r"(?:interface|type)\s+\w+\s*=?\s*\{(.*?)\n\}", re.S)
_TS_FIELD_RE = re.compile(r"^\s+(\w+)(\??):", re.M)
_FIELD_UNION_RE = re.compile(r'\w+\??:\s*((?:"[^"]+"\s*\|\s*)+"[^"]+")(?:\s*\|\s*null)?[;,]')
_TYPE_ALIAS_UNION_RE = re.compile(
    r'^\s*export\s+type\s+(\w+)\s*=\s*((?:"[^"]+"\s*\|\s*)+"[^"]+")\s*;', re.M
)
_QUOTED_RE = re.compile(r'"([^"]+)"')


def _generator():
    """The type-module generator loaded as a module, so the test regenerates the same way CI does."""
    spec = importlib.util.spec_from_file_location("tcip_frontend_type_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generated_types_module_is_what_the_declared_models_produce() -> None:
    """The checked-in module is a projection of the declared models' JSON schema, not a
    hand-edited copy: a field renamed, added or retyped on a model that never reached the module
    would leave the browser reading a shape the backend no longer sends."""
    generated = _generator().render()
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        "packages/tcip-web/frontend/src/api/types.generated.ts is out of date; "
        "run python tools/generate_frontend_types.py"
    )


class _OneMemberLiteral(BaseModel):
    """A scratch model whose one field is a single-value ``Literal``, which pydantic's own JSON
    schema renders as ``const`` rather than ``enum``."""

    kind: Literal["only"]


class _NullableStringList(BaseModel):
    """A scratch model whose one field is an array of a nullable union, exercising the mapper's
    array-of-union case independently of any declared coverage model."""

    values: list[Optional[str]]


class _FreeFormObject(BaseModel):
    """A scratch model whose one field is a bare ``dict``, the free-form object shape a
    discriminated frame carries beside its ``Literal`` type field."""

    payload: dict


def test_a_one_member_literal_renders_its_own_value_as_the_type() -> None:
    """A ``const`` schema is a single-member ``Literal``: its own value is the type TypeScript
    narrows a discriminated union on, so it renders rather than dropping the literal."""
    generator = _generator()
    schema = _OneMemberLiteral.model_json_schema()
    assert generator._ts_type(schema["properties"]["kind"], "_OneMemberLiteral.kind") == '"only"'


def test_a_union_used_as_an_array_item_is_parenthesized() -> None:
    generator = _generator()
    schema = _NullableStringList.model_json_schema()
    assert (generator._ts_type(schema["properties"]["values"], "_NullableStringList.values")
            == "(string | null)[]")


def test_a_free_form_object_renders_as_a_record_of_unknown() -> None:
    """A bare ``dict`` field carries no fixed property set, so it maps to ``Record<string,
    unknown>`` rather than refusing or dropping to an untyped ``object``."""
    generator = _generator()
    schema = _FreeFormObject.model_json_schema()
    assert (generator._ts_type(schema["properties"]["payload"], "_FreeFormObject.payload")
            == "Record<string, unknown>")


def test_reordering_platform_panel_events_does_not_change_a_named_constants_value() -> None:
    """Each named ``PANEL_EVENT_*`` constant is read off its own module attribute, never off
    ``PLATFORM_PANEL_EVENTS``'s position, so a consumer importing one by name is unaffected by
    how that tuple is ordered."""
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
    """The free-form-object case must not swallow a plain fixed-shape object schema that arrives
    outside a ``$ref``: that shape is still not one the mapper covers."""
    generator = _generator()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(SystemExit):
        generator._ts_type(schema, "_fixed_object")


def _generated_field_sets() -> list[set[str]]:
    text = GENERATED.read_text(encoding="utf-8")
    return [{name for name, _ in _TS_FIELD_RE.findall(block)} for block in _TS_BLOCK_RE.findall(text)]


def test_no_other_frontend_module_declares_an_interface_with_a_generated_field_set() -> None:
    """One declaration per generated shape, so the backend has one browser-side counterpart to
    stay equal to. A second interface naming the same fields is what drifts: the module that was
    updated keeps working and the one that was not reads a key the models no longer emit."""
    generated_sets = [s for s in _generated_field_sets() if s]
    assert generated_sets, "no interfaces parsed out of the generated module"

    sources = [
        p for p in sorted(list(FRONTEND_SRC.rglob("*.ts")) + list(FRONTEND_SRC.rglob("*.tsx")))
        if ".test." not in p.name and "test" not in p.parent.name and p != GENERATED
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


def _generated_field_unions() -> list[set[str]]:
    text = GENERATED.read_text(encoding="utf-8")
    return [set(_QUOTED_RE.findall(m.group(1))) for m in _FIELD_UNION_RE.finditer(text)]


def test_no_frontend_module_hand_declares_a_generated_fields_literal_union() -> None:
    """A bare ``export type X = "a" | "b" | ...;`` alias parses as neither an interface nor an
    object type, so the field-set check above never sees it: this catches one whose members
    exactly match a generated interface field's literal union, the shape a field-set comparison
    misses entirely."""
    generated_sets = [s for s in _generated_field_unions() if len(s) > 1]
    assert generated_sets, "no literal unions parsed out of the generated module's fields"

    sources = [
        p for p in sorted(list(FRONTEND_SRC.rglob("*.ts")) + list(FRONTEND_SRC.rglob("*.tsx")))
        if ".test." not in p.name and "test" not in p.parent.name and p != GENERATED
    ]
    offenders = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for m in _TYPE_ALIAS_UNION_RE.finditer(text):
            if set(_QUOTED_RE.findall(m.group(2))) in generated_sets:
                offenders.append(f"{source.relative_to(REPO_ROOT).as_posix()}: {m.group(1)}")
    assert not offenders, (
        "these declare a union alias matching a generated field's literal union by hand:\n"
        + "\n".join(offenders)
    )
