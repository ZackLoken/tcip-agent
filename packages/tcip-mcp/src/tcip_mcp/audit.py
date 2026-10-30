"""The audit log: one append-only store under a dataset root or a project root.

The log's key is :func:`audit_log_key`. :func:`audited` records a decorated door's calls
(timestamp, tool name, arguments, status and duration); :func:`record_event_or_raise` records for
code that is not a door.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypeVar, overload

from pydantic import AwareDatetime, BeforeValidator, ConfigDict, JsonValue
from pydantic_core import to_jsonable_python

from tcip_mcp import agent_identity
from tcip_mcp.registry_paths import located
from tcip_store import FiniteJsonValue, Key, append

logger = logging.getLogger(__name__)

AUDIT_LOG_STORE = "audit_log"
_AUDIT_PARTS = ("audit",)

# Fields to redact from logged arguments
_REDACTED_FIELDS = {"api_key", "token", "password", "secret"}


class _CommittedUnrecordedError(RuntimeError):
    """A call's change committed and the audit entry recording it could not be appended."""

    def __init__(self, tool: str, cause: BaseException) -> None:
        """Take the tool whose entry did not land and the append's exception."""
        super().__init__(
            f"{tool} completed and its audit entry could not be written: {cause}. Whatever the "
            "call changed is committed and unrecorded, so do not retry it blind: repair the "
            "audit log's destination, then reconcile the trail against what the call did."
        )
        self.tool = tool


class MutationCommittedWithoutAuditLineError(_CommittedUnrecordedError):
    """A tool body ran to completion and the audit entry that follows it did not land."""


class AuditEntryNotWrittenError(_CommittedUnrecordedError):
    """A mutation recorded outside ``@audited`` committed, and the append for it failed.

    ``arguments`` are the facts the unwritten line would have carried.
    """

    def __init__(
        self,
        tool: str,
        cause: BaseException,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Take the tool, the append's exception and the unwritten line's ``arguments``."""
        super().__init__(tool, cause)
        self.arguments = arguments or {}


def audit_log_key(scope: str | Path) -> Key:
    """Return the key of the audit log under ``scope``.

    ``scope`` is the root the event's subject hangs off: a dataset root when the event changed a
    record that travels with the data, the project root otherwise.
    """
    return Key(AUDIT_LOG_STORE, str(scope), _AUDIT_PARTS)


def audit_entries(
    scope: str | Path, *, after: str | None = None
) -> tuple[list[AuditLine], str]:
    """Return every entry of ``scope``'s log past the cursor ``after``, and the log's cursor.

    Entries are read as :class:`AuditLine`, in the order they landed. A log carrying undecodable
    entries raises ``ValueError`` naming how many, and an entry that is no audit line
    ``pydantic.ValidationError``.
    """
    from tcip_store import read_log

    key = audit_log_key(scope)
    page = read_log(key, after=after)
    if page.corrupt:
        raise ValueError(
            f"the audit log at {key.root} carries {len(page.corrupt)} undecodable "
            "entries; repair the log before trusting a read of it"
        )
    return [AuditLine.model_validate(r) for r in page.records], page.cursor


def acts_of(
    scope: str | Path, tools: tuple[str, ...], *, after: str | None = None
) -> tuple[list[AuditLine], str]:
    """Return the ``ok`` entries of ``tools`` among :func:`audit_entries`, and the log's cursor."""
    entries, cursor = audit_entries(scope, after=after)
    return [e for e in entries if e.status == "ok" and e.tool in tools], cursor


def now() -> datetime:
    """Return the current instant, in UTC."""
    return datetime.now(timezone.utc)


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from tool arguments."""
    return {
        k: "***REDACTED***" if k in _REDACTED_FIELDS else v
        for k, v in args.items()
    }


ACTOR_KEY = "actor"
"""The entry key naming the person who performed the act, absent for an act no person made."""


class AuditLine(agent_identity.RecordedIdentity):
    """One audit log entry.

    ``arguments`` is held as the JSON form of what it was given
    (``pydantic_core.to_jsonable_python``), a snapshot no later change to the given values
    reaches; a value with no JSON form refuses. ``actor`` is absent for an act no person made,
    the identity fields outside an agent session, and ``duration_ms`` and ``error`` for a line no
    door call wrote. Any other key a writer adds, a caller's extra facts, is kept as an extra
    field.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    __pydantic_extra__: dict[str, FiniteJsonValue]

    timestamp: AwareDatetime
    tool: str
    arguments: Annotated[
        dict[str, FiniteJsonValue], BeforeValidator(to_jsonable_python)
    ]
    status: str
    actor: str | None = None
    duration_ms: float | None = None
    error: str | None = None

    def stored(self) -> dict[str, JsonValue]:
        """Return the line as the log stores it, each field it does not hold absent."""
        return self.model_dump(mode="json", exclude_none=True)


def audit_line(
    tool: str,
    arguments: dict[str, Any] | None,
    actor: str | None,
    status: str,
    extra: dict[str, FiniteJsonValue] | None = None,
) -> AuditLine:
    """Return the audit line of one act, stamped now.

    The line holds the clock, the tool, its redacted arguments, the person who performed the act
    when one did (``actor``, :func:`~tcip_mcp.identity.actor`'s spelling), ``status``, a caller's
    extra facts, and the agent identity this process established at its MCP handshake, if it has
    one. A key of ``extra`` the line declares as a field of its own is no extra fact and is
    dropped. An argument with no JSON form refuses (``pydantic.ValidationError``).
    """
    return AuditLine.model_validate(
        {
            **{k: v for k, v in (extra or {}).items() if k not in AuditLine.model_fields},
            **dict(agent_identity.current() or agent_identity.RecordedIdentity()),
            "timestamp": now(),
            "tool": tool,
            "arguments": _redact(arguments) if arguments else {},
            "status": status,
            "actor": actor,
        }
    )


def record_event_or_raise(
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    actor: str | None,
    status: str = "ok",
    scope: str | Path,
    **extra: FiniteJsonValue,
) -> None:
    """Emit one audit line (:func:`audit_line`), performed by ``actor``, for a mutation made.

    The line is appended by :func:`append_or_raise`.
    """
    append_or_raise(audit_line(tool, arguments, actor, status, extra), scope=scope)


def append_or_raise(line: AuditLine, *, scope: str | Path) -> None:
    """Append ``line``, the audit line of a mutation made, to the log of the root ``scope`` names.

    The log is :func:`audit_log_key`'s. A failed append is raised as
    :class:`AuditEntryNotWrittenError`, naming the mutation that committed unrecorded.
    """
    try:
        append(audit_log_key(scope), line.stored())
    except Exception as exc:
        raise AuditEntryNotWrittenError(
            line.tool, exc, arguments=line.arguments
        ) from exc


def dataset_scope_of(value: Path) -> Path | None:
    """Return the dataset root ``value`` names, or ``None`` when it does not name one.

    ``value`` is an established location naming a root: a directory that carries dataset or
    project state of its own (:func:`dataset_layout.carries_root_state`) is that root; otherwise
    it is :func:`dataset_layout.dataset_root_of` the capture directory or image tree ``value`` is,
    or, for a file, the one it lies in. Anything else yields ``None``.
    """
    from tcip_mcp.dataset_layout import carries_root_state, dataset_root_of

    if value.is_dir() and carries_root_state(value):
        return value
    return dataset_root_of(value.parent if value.is_file() else value)


_Door = TypeVar("_Door", bound=Callable[..., Any])


@overload
def audited(fn: _Door) -> _Door: ...
@overload
def audited(*, scope_arg: str | None = None) -> Callable[[_Door], _Door]: ...
def audited(
    fn: Callable | None = None,
    *,
    scope_arg: str | None = None,
) -> Callable:
    """Decorate a door so each call that acts leaves one audit line.

    The door takes a ``project`` parameter (refused without one). A call that returns a result
    leaves one ``ok`` line, a call that raises leaves one ``exception`` line, and a call returning
    its error dict leaves none, in one log: the project's, or with ``scope_arg`` the dataset root
    that argument resolves to at call time (:func:`dataset_scope_of`; no root means the
    project's). A set ``scope_arg`` reaches the door as the location it names against ``project``
    (:func:`~tcip_mcp.registry_paths.located`), and the entry keeps the caller's spelling.
    ``project`` and ``workspace`` are not recorded as arguments, the rest are recorded as their
    JSON form taken before the body runs (:class:`AuditLine`), so an argument with no JSON form
    refuses the call unrun, and an ``actor`` parameter is recorded as the entry's actor. A
    failed append after a return raises :class:`MutationCommittedWithoutAuditLineError`; after a
    raise the body's exception propagates and the failed append is logged. A declared scope that
    cannot be resolved refuses the call.
    """

    def decorate(func: Callable) -> Callable:
        sig = inspect.signature(func)
        if "project" not in sig.parameters:
            raise ValueError(
                f"@audited on {func.__name__} needs a project parameter to record under"
            )
        if scope_arg is not None and scope_arg not in sig.parameters:
            raise ValueError(
                f"@audited(scope_arg={scope_arg!r}) on {func.__name__} names no parameter of it; "
                f"it takes {tuple(sig.parameters)}"
            )

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            tool_name = func.__name__
            t0 = time.monotonic()
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            logged_args: dict[str, Any] = dict(bound.arguments)
            project = logged_args.pop("project")
            logged_args.pop("workspace", None)
            begun = audit_line(
                tool_name,
                logged_args,
                logged_args.pop(ACTOR_KEY, None),
                "exception",
            )
            where = bound.arguments.get(scope_arg) if scope_arg else None
            if scope_arg and where is not None:
                bound.arguments[scope_arg] = where = located(where, project)

            def record(status: str, error: str | None = None) -> None:
                """Resolve the scope, stamp the outcome, and append. Raises what it cannot do."""
                # Resolved after the body, so a tool that creates the dataset it names is
                # recorded in that dataset's own log rather than the project's.
                scope = dataset_scope_of(where) if where is not None else None
                line = begun.model_copy(
                    update={
                        "status": status,
                        "error": error,
                        "duration_ms": round((time.monotonic() - t0) * 1000, 1),
                    }
                )
                append(
                    audit_log_key(scope if scope is not None else project),
                    line.stored(),
                )

            # At most one entry per call by construction: the two writing paths are exclusive,
            # and neither writer sits inside a handler that could run the other.
            try:
                result = func(*bound.args, **bound.kwargs)
            except Exception as body_exc:
                try:
                    record("exception", str(body_exc))
                except Exception:
                    # The body's exception is the caller's answer; this one never displaces it.
                    logger.warning(
                        "Failed to audit the failed %s call",
                        tool_name,
                        exc_info=True,
                    )
                raise
            if isinstance(result, dict) and result.get("error") is not None:
                return result  # a refusal returned as the error dict every tool returns: no act
            try:
                record("ok")
            except Exception as exc:
                logger.warning(
                    "Failed to write the audit entry for %s",
                    tool_name,
                    exc_info=True,
                )
                raise MutationCommittedWithoutAuditLineError(
                    tool_name, exc
                ) from exc
            return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
