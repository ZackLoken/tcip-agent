"""Why a record or value will not read, and the two boundaries that state it.

:func:`validated` validates a raw value against its model and, when the model refuses it, raises
:class:`~tcip_store.errors.ReadRefusedError` carrying every error the model raised, each by its
place and message (:class:`Invalid`), behind where the value lives. :func:`unreadable` raises the
same error carrying :class:`Unreadable`, the sentence of any other error that stops a read. Each
reads as text where a sentence carries it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, Literal, overload

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from tcip_store.errors import ReadRefusedError


class Issue(BaseModel):
    """One error a model's validation raised: where in the input (``loc``) and what (``msg``)."""

    loc: list[str | int]
    msg: str

    def __str__(self) -> str:
        """Return the message behind its dotted place, or alone for the input as a whole."""
        return (
            f"{'.'.join(map(str, self.loc))}: {self.msg}"
            if self.loc
            else self.msg
        )


_ISSUES: TypeAdapter[list[Issue]] = TypeAdapter(list[Issue])


class Invalid(BaseModel):
    """A value its model refused, with every error the refusal carried."""

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
        """Return each issue as text, in the order the model raised them."""
        return "; ".join(map(str, self.issues))


class Unreadable(BaseModel):
    """A record that would not read for a reason other than its model's: the error's sentence."""

    kind: Literal["unreadable"] = "unreadable"
    message: str

    def __str__(self) -> str:
        """Return the error's own sentence."""
        return self.message


type ReadRefusal = Annotated[Invalid | Unreadable, Field(discriminator="kind")]
"""Why a record will not read, told apart by its ``kind``."""


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
    raises :class:`~tcip_store.errors.ReadRefusedError` carrying the model's errors as
    :class:`Invalid`, behind ``where``.
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
    """Raise :class:`~tcip_store.errors.ReadRefusedError` for any of ``errors`` raised inside.

    The error carries the caught error's sentence as :class:`Unreadable`, behind ``where``; a
    ``ReadRefusedError`` raised inside passes as it is.
    """
    try:
        yield
    except ReadRefusedError:
        raise
    except errors as exc:
        raise ReadRefusedError(Unreadable(message=str(exc)), where) from exc
