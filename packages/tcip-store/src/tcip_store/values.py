"""What a value must be before a store will carry it, and its two stored spellings.

The module also states how a producer says a number is not finite. A record is spelled by
:func:`encode_record` and a log line by :func:`encode_log_line`; both are read back by
:func:`decode_value`. :func:`check_json_value` refuses, naming the field, a value pydantic's
``JsonValue`` does not admit or a non-finite number. :func:`stored_number` keeps a non-finite
number as a JSON null beside a sibling field naming the state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Annotated, Any, TypeGuard

from pydantic import (
    ConfigDict,
    JsonValue,
    PlainValidator,
    TypeAdapter,
    ValidationError,
)


def encode_record(value: JsonValue) -> bytes:
    """Return a record's bytes: UTF-8 JSON indented by two, keys in their authored order.

    The bytes end in a trailing newline. Refuses what :func:`check_json_value` refuses, before
    anything is spelled.
    """
    check_json_value(value)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)
    return f"{text}\n".encode()


def encode_log_line(value: JsonValue) -> bytes:
    """Return a log entry's bytes: UTF-8 JSON on one line, keys in their authored order.

    Refuses what :func:`check_json_value` refuses, before anything is spelled.
    """
    check_json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=False).encode(
        "utf-8"
    )


def decode_value(data: bytes) -> JsonValue:
    """Return the value a record's or a log entry's bytes spell.

    Raises ``ValueError`` when they do not decode.
    """
    return json.loads(data.decode("utf-8"))


NOT_FINITE_SUFFIX = "_state"
"""Appended to a numeric field's name for the sibling that says why it is null."""


def scalar_number(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a number other than a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a finite :func:`scalar_number`."""
    return scalar_number(value) and math.isfinite(value)


def non_finite_state(value: float) -> str | None:
    """Return the token naming why a float cannot be stored, or None when it can."""
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return None


def stored_number[T](field: str, value: T) -> dict[str, T | str | None]:
    """Return one numeric field as a record carries it, with a state field when it is not a number.

    A finite number, or None, is carried as itself. A non-finite one becomes null plus
    ``<field>_state``.
    """
    if (
        isinstance(value, float)
        and (state := non_finite_state(value)) is not None
    ):
        return {field: None, f"{field}{NOT_FINITE_SUFFIX}": state}
    return {field: value}


def stored_numbers(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flat mapping of measurements as a record carries it, each via :func:`stored_number`.

    Each entry of ``values`` is one named number; a state field sits beside the value it explains.
    """
    row: dict[str, Any] = {}
    for name, value in values.items():
        row.update(stored_number(name, value))
    return row


def finite_or_none[T](value: T) -> T | None:
    """Return ``None`` for a non-finite float, else ``value``; no state companion is written."""
    if isinstance(value, float) and non_finite_state(value) is not None:
        return None
    return value


_FINITE_JSON_RULE = ConfigDict(allow_inf_nan=False, strict=True)
"""Refuse a non-finite number, a ``bytes`` key (never decoded to text) and a type JSON has no form
for; a ``str`` subclass or a string enum is admitted as the plain string it spells."""

_FINITE_JSON: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue, config=_FINITE_JSON_RULE
)

FiniteJsonValue = Annotated[
    JsonValue,
    PlainValidator(
        _FINITE_JSON.validate_python, json_schema_input_type=JsonValue
    ),
]
"""A JSON value a store carries, one :func:`check_json_value` admits."""

FINITE_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(
    dict[str, JsonValue], config=_FINITE_JSON_RULE
)
"""The adapter admitting a :data:`FiniteJsonObject`, refusing (``pydantic.ValidationError``)
anything else, naming the position."""

FiniteJsonObject = Annotated[
    dict[str, JsonValue],
    PlainValidator(
        FINITE_JSON_OBJECT.validate_python,
        json_schema_input_type=dict[str, JsonValue],
    ),
]
"""A JSON object a store carries: string keys, every value one :func:`check_json_value` admits."""


def check_json_value(value: object, *, path: str = "value") -> None:
    """Refuse a payload that is not a finite JSON value, naming each position and its type.

    A JSON value is pydantic's ``JsonValue``: a ``dict`` keyed by strings, a list, a string,
    number, boolean or None, every number finite (:data:`_FINITE_JSON_RULE`). A non-finite number,
    a ``bytes`` key or a type JSON has no form for refuses (``ValueError``) under ``path``, naming
    the position and the type it holds, and naming :func:`stored_number` as how a non-finite number
    is stored.
    """
    try:
        _FINITE_JSON.validate_python(value)
    except ValidationError as exc:
        raise ValueError(
            f"{path} is not a finite JSON value: {exc}\nConvert what it holds to a JSON type at "
            "the writer, a non-finite number to null beside a field naming its state "
            "(stored_number)."
        ) from exc
