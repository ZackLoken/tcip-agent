"""Why a record or value will not read, served as pydantic's own errors carry it.

:func:`read_refusal` is where a caught error becomes a served reason: a ``ValidationError`` as
every error its model raised, each by its place and message (:class:`Invalid`), any other error
as its own sentence (:class:`Unreadable`). Each reads as text where a sentence carries it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


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


def read_refusal(exc: Exception) -> ReadRefusal:
    """Return why a read raised ``exc``: its model's errors, or its own sentence."""
    if isinstance(exc, ValidationError):
        return Invalid.of(exc)
    return Unreadable(message=str(exc))
