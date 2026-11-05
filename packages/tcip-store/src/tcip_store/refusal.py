"""Why a record or value will not read or a check refuses it, the two read boundaries, the reader.

:func:`validated` validates a raw value against its model and, when the model refuses it, raises
:class:`ReadRefusedError` carrying every error the model raised, each by its place and message
(:class:`Invalid`), at where the value lives (:class:`RefusedRead`). :func:`unreadable` raises
the same error carrying :class:`Unreadable`, the sentence of any other error that stops a read.
:func:`read_raw` reads a record under the one set of errors that stops a store read, and
:func:`read_record` validates what it read. A check that refuses a value states each objection as
an issue of an :class:`Invalid`, its place empty when it names none. Each reads as text where a
sentence carries it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, Literal, overload

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from tcip_store.errors import StoreError
from tcip_store.model import REQUIRED, Key, ReadDefault


class Issue(BaseModel):
    """One error a model's validation raised, or one objection a check states.

    ``loc`` is where in the input it lies, empty for the input as a whole or an objection that
    names no place, and ``msg`` is what it is.
    """

    loc: list[str | int]
    msg: str

    def __str__(self) -> str:
        """Return the message behind its dotted place, or alone when its place is empty."""
        return (
            f"{'.'.join(map(str, self.loc))}: {self.msg}"
            if self.loc
            else self.msg
        )


_ISSUES: TypeAdapter[list[Issue]] = TypeAdapter(list[Issue])


class Invalid(BaseModel):
    """A value its model refused, with every error the refusal carried, or a check's objections.

    A check's objection is an issue of its own, its sentence naming any place it has.
    """

    kind: Literal["invalid"] = "invalid"
    issues: list[Issue]

    @classmethod
    def of(cls, exc: ValidationError) -> Invalid:
        """Return ``exc``'s errors as ``exc.errors(include_url=False)`` reports them.

        A validator's own refusal (``value_error``) is its raised error's message, which pydantic
        carries under the error's ``ctx``.
        """
        return cls(
            issues=_ISSUES.validate_python(
                [
                    {**e, "msg": str(e["ctx"]["error"])}
                    if e["type"] == "value_error"
                    else e
                    for e in exc.errors(include_url=False)
                ]
            )
        )

    def __str__(self) -> str:
        """Return each issue as text, in the order the model raised or the check stated them."""
        return "; ".join(map(str, self.issues))


class Unreadable(BaseModel):
    """A record that would not read for a reason other than its model's: the error's sentence."""

    kind: Literal["unreadable"] = "unreadable"
    message: str

    def __str__(self) -> str:
        """Return the error's own sentence."""
        return self.message


type ReadRefusal = Annotated[Invalid | Unreadable, Field(discriminator="kind")]
"""Why a record will not read or a check refuses a value, told apart by its ``kind``."""


class RefusedRead(BaseModel):
    """A record or value that will not read: the place ``where`` names, and why (``refusal``)."""

    where: str
    refusal: ReadRefusal

    def __str__(self) -> str:
        """Return ``where`` before the refusal's text; it never quotes the value."""
        return f"{self.where}: {self.refusal}"


class ReadRefusedError(ValueError):
    """A record or value that will not read, raised carrying its :class:`RefusedRead` (``served``).

    Its message is the refused read's text.
    """

    def __init__(self, refusal: ReadRefusal, where: str) -> None:
        """Hold ``refusal`` at ``where`` as the refused read, stated as the error's message."""
        self.served = RefusedRead(where=where, refusal=refusal)
        super().__init__(str(self.served))


@overload
def validated[M: BaseModel](
    model: type[M],
    raw: object,
    where: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> M: ...


@overload
def validated[T](
    model: TypeAdapter[T],
    raw: object,
    where: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> T: ...


def validated(
    model: type[BaseModel] | TypeAdapter[Any],
    raw: object,
    where: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> Any:
    """Return ``raw`` validated as ``model``, a model or the adapter of a type, under ``context``.

    A model validates through its own ``model_validate``. ``raw`` is the decoded value, or the
    JSON bytes that spell it. ``where`` names where the value lives. A value the model refuses
    raises :class:`ReadRefusedError` carrying the model's errors as :class:`Invalid`, behind
    ``where``.
    """
    try:
        if isinstance(model, TypeAdapter):
            if isinstance(raw, bytes):
                return model.validate_json(raw, context=context)
            return model.validate_python(raw, context=context)
        if isinstance(raw, bytes):
            return model.model_validate_json(raw, context=context)
        return model.model_validate(raw, context=context)
    except ValidationError as exc:
        raise ReadRefusedError(Invalid.of(exc), where) from exc


@contextmanager
def unreadable(where: str, *errors: type[Exception]) -> Iterator[None]:
    """Raise :class:`ReadRefusedError` for any of ``errors`` raised inside.

    The error carries the caught error's sentence as :class:`Unreadable`, behind ``where``; a
    ``ReadRefusedError`` raised inside passes as it is.
    """
    try:
        yield
    except ReadRefusedError:
        raise
    except errors as exc:
        raise ReadRefusedError(Unreadable(message=str(exc)), where) from exc


def read_raw[T](
    read: Callable[..., T],
    key: Key,
    where: str,
    *,
    default: ReadDefault = REQUIRED,
) -> T:
    """Return what ``read(key, default=default)`` answers, read under :func:`unreadable`.

    ``read`` is a store read or a transaction's (``tcip_store.read``, ``Txn.read``,
    ``tcip_store.read_versioned``). Any ``StoreError`` (an absent record with no ``default``, bytes
    that do not decode, each sentence naming the key) and any ``OSError`` (a database that does
    not read from disk) raises :class:`ReadRefusedError` carrying its sentence behind ``where``.
    """
    with unreadable(where, StoreError, OSError):
        return read(key, default=default)


@overload
def read_record[M](
    read: Callable[..., JsonValue],
    key: Key,
    model: type[M] | TypeAdapter[M],
    where: str,
) -> M: ...


@overload
def read_record[M](
    read: Callable[..., JsonValue],
    key: Key,
    model: type[M] | TypeAdapter[M],
    where: str,
    *,
    default: None,
) -> M | None: ...


def read_record(
    read: Callable[..., JsonValue],
    key: Key,
    model: type[Any] | TypeAdapter[Any],
    where: str,
    *,
    default: None = REQUIRED,
) -> Any:
    """Return the record at ``key`` read by ``read`` (:func:`read_raw`) and :func:`validated`.

    ``model`` is a model or the adapter of a type, ``where`` the place a refusal names. With
    ``default=None`` an absent record answers ``None``; without it, absence refuses as
    :func:`read_raw` does. Every refusal is :class:`ReadRefusedError` behind ``where``.
    """
    raw = read_raw(read, key, where, default=default)
    if raw is None and default is None:
        return None
    return validated(model, raw, where)
