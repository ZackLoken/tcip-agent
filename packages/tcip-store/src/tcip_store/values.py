"""What a value must be before a store will carry it, its two stored spellings, and how a
producer says a number is not finite.

A record is spelled by :func:`encode_record` and a log line by :func:`encode_log_line`; both are
read back by :func:`decode_value`. :func:`check_json_value` refuses, naming the field, an object
JSON has no type for or a non-finite number. :func:`stored_number` keeps a non-finite number as a
JSON null beside a sibling field naming the state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def encode_record(value: Any) -> bytes:
    """A record's bytes: UTF-8 JSON indented by two, keys in their authored order, a trailing
    newline. Refuses what :func:`check_json_value` refuses, before anything is spelled."""
    check_json_value(value)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    return f"{text}\n".encode("utf-8")


def encode_log_line(value: Any) -> bytes:
    """A log entry's bytes: UTF-8 JSON on one line, keys in their authored order. Refuses as
    :func:`encode_record` does."""
    check_json_value(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=False).encode("utf-8")


def decode_value(data: bytes) -> Any:
    """The value a record's or a log entry's bytes spell; ``ValueError`` when they do not decode."""
    return json.loads(data.decode("utf-8"))


_SCALARS = (str, bool, int, float, type(None))

NOT_FINITE_SUFFIX = "_state"
"""Appended to a numeric field's name for the sibling that says why it is null."""


def non_finite_state(value: float) -> str | None:
    """The token naming why a float cannot be stored, or None when it can."""
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return None


def stored_number(field: str, value: Any) -> dict[str, Any]:
    """One numeric field as a record carries it, with a state field when it is not a number.

    A finite number, or None, is carried as itself. A non-finite one becomes null plus
    ``<field>_state``.
    """
    if isinstance(value, float) and (state := non_finite_state(value)) is not None:
        return {field: None, f"{field}{NOT_FINITE_SUFFIX}": state}
    return {field: value}


def stored_numbers(values: Mapping[str, Any]) -> dict[str, Any]:
    """A flat mapping of measurements as a record carries it, each through :func:`stored_number`.

    For a metrics row, where every entry is one named number and a state field can sit
    beside the value it explains.
    """
    row: dict[str, Any] = {}
    for name, value in values.items():
        row.update(stored_number(name, value))
    return row


def finite_or_none(value: Any) -> Any:
    """The value a record carries in place of a non-finite number, with no state beside it.

    For a number nested where a sibling field has nowhere to go, inside a list of sweep rows
    a reader indexes positionally. Prefer :func:`stored_number` wherever a sibling fits.
    """
    if isinstance(value, float) and non_finite_state(value) is not None:
        return None
    return value


def check_json_value(value: Any, *, path: str = "value") -> None:
    """Refuse a payload the canonical codec cannot encode, naming the field and the type.

    Raises ``TypeError`` for an object JSON has no type for and ``ValueError`` for a non-finite
    number. ``path`` names the position inside the payload.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} is keyed by {type(key).__name__} {key!r}: a JSON object's keys "
                    "are strings, so convert the key before storing it"
                )
            check_json_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, float):
        state = non_finite_state(value)
        if state is not None:
            raise ValueError(
                f"{path} is {state}, which JSON cannot spell and no strict parser will read. "
                "Represent it as null beside a field naming the state, at the writer that "
                "produced it."
            )
        return
    if isinstance(value, _SCALARS):
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            check_json_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} is a {type(value).__name__}, which JSON has no type for. Convert it to a "
        "string, number, boolean, list or object at the writer, so the record states what it "
        "holds rather than a repr of it."
    )
