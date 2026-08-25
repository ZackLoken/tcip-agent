"""The browser's coverage-record types are a projection of the pydantic models that declare them.

``scripts/generate_frontend_types.py`` renders ``frontend/src/api/types.generated.ts`` from
``routes/_coverage_models.py`` and ``routes/coverage.py``; these tests hold that projection to
what the models produce now, and keep another frontend module from declaring an interface with
the same field set as a generated one.
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
GENERATOR = REPO_ROOT / "scripts" / "generate_frontend_types.py"

_TS_BLOCK_RE = re.compile(r"(?:interface|type)\s+\w+\s*=?\s*\{(.*?)\n\}", re.S)
_TS_FIELD_RE = re.compile(r"^\s+(\w+)(\??):", re.M)


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
        "run python scripts/generate_frontend_types.py"
    )


class _OneMemberLiteral(BaseModel):
    """A scratch model whose one field is a single-value ``Literal``, which pydantic's own JSON
    schema renders as ``const`` rather than ``enum``."""

    kind: Literal["only"]


class _NullableStringList(BaseModel):
    """A scratch model whose one field is an array of a nullable union, exercising the mapper's
    array-of-union case independently of any declared coverage model."""

    values: list[Optional[str]]


def test_a_one_member_literal_refuses_rather_than_render_the_wrapping_type() -> None:
    """A ``const`` schema is a shape the mapper does not cover; it must say so by name instead of
    silently rendering the primitive type underneath and dropping the literal value."""
    generator = _generator()
    schema = _OneMemberLiteral.model_json_schema()
    with pytest.raises(SystemExit):
        generator._ts_type(schema["properties"]["kind"], "_OneMemberLiteral.kind")


def test_a_union_used_as_an_array_item_is_parenthesized() -> None:
    generator = _generator()
    schema = _NullableStringList.model_json_schema()
    assert (generator._ts_type(schema["properties"]["values"], "_NullableStringList.values")
            == "(string | null)[]")


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
