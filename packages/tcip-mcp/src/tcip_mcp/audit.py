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
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel, JsonValue

from tcip_mcp import agent_identity
from tcip_mcp.registry_paths import located
from tcip_store import Key, append

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
) -> tuple[list[Mapping[str, Any]], str]:
    """Return every entry of ``scope``'s log past the cursor ``after``, and the log's cursor.

    Entries are in :func:`audit_entry`'s shape, in the order they landed. A log carrying
    undecodable entries raises ``ValueError`` naming how many.
    """
    from tcip_store import read_log

    key = audit_log_key(scope)
    page = read_log(key, after=after)
    if page.corrupt:
        raise ValueError(
            f"the audit log at {key.root} carries {len(page.corrupt)} undecodable "
            "entries; repair the log before trusting a read of it"
        )
    return list(page.records), page.cursor


def acts_of(
    scope: str | Path, tools: tuple[str, ...], *, after: str | None = None
) -> tuple[list[Mapping[str, Any]], str]:
    """Return the ``ok`` entries of ``tools`` among :func:`audit_entries`, and the log's cursor."""
    entries, cursor = audit_entries(scope, after=after)
    return [
        e for e in entries if e["status"] == "ok" and e["tool"] in tools
    ], cursor


def now() -> datetime:
    """Return the current instant, in UTC."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Return :func:`now` as an ISO-8601 string."""
    return now().isoformat()


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from tool arguments."""
    return {
        k: "***REDACTED***" if k in _REDACTED_FIELDS else v
        for k, v in args.items()
    }


ACTOR_KEY = "actor"
"""The entry key naming the person who performed the act, absent for an act no person made."""


def audit_entry(
    tool: str,
    arguments: dict[str, Any] | None,
    actor: str | None,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shape every audit entry starts from.

    The entry holds the clock, the tool, its redacted arguments, the person who performed the act
    when one did (``actor``, :func:`~tcip_mcp.identity.actor`'s spelling), ``status`` when given,
    a caller's extra facts, and the agent identity this process established at its MCP
    handshake, if it has one.

    The actor and identity keys are reserved: a caller's ``extra`` cannot set one.
    """
    entry: dict[str, Any] = {
        "timestamp": now_iso(),
        "tool": tool,
        "arguments": _redact(arguments) if arguments else {},
    }
    if actor is not None:
        entry[ACTOR_KEY] = actor
    if status is not None:
        entry["status"] = status
    reserved = {ACTOR_KEY, *agent_identity.RECORD_FIELDS}
    entry.update({k: v for k, v in (extra or {}).items() if k not in reserved})
    entry.update(agent_identity.audit_fields())
    return entry


def record_event_or_raise(
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    actor: str | None,
    status: str = "ok",
    scope: str | Path,
    **extra: JsonValue,
) -> None:
    """Emit one audit line, performed by ``actor``, for a mutation already made.

    The line lands in the log of the root ``scope`` names (:func:`audit_log_key`). A failed
    append is raised as :class:`AuditEntryNotWrittenError`, naming the mutation that
    committed unrecorded.
    """
    entry = audit_entry(tool, arguments, actor, status, extra)
    try:
        append(audit_log_key(scope), entry)
    except Exception as exc:
        raise AuditEntryNotWrittenError(
            tool, exc, arguments=arguments
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
    ``project`` and ``workspace`` are not recorded as arguments, a model argument is recorded as
    its JSON dump, and an ``actor`` parameter is recorded as the entry's actor. A failed append
    after a return raises :class:`MutationCommittedWithoutAuditLineError`; after a raise the
    body's exception propagates and the failed append is logged. A declared scope that cannot be
    resolved refuses the call.
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
            logged_args: dict[str, Any] = {
                k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                for k, v in bound.arguments.items()
            }
            project = logged_args.pop("project")
            logged_args.pop("workspace", None)
            entry = audit_entry(
                tool_name, logged_args, logged_args.pop(ACTOR_KEY, None)
            )
            where = bound.arguments.get(scope_arg) if scope_arg else None
            if scope_arg and where is not None:
                bound.arguments[scope_arg] = where = located(where, project)

            def record() -> None:
                """Resolve the scope, stamp the duration, and append. Raises what it cannot do."""
                # Resolved after the body, so a tool that creates the dataset it names is
                # recorded in that dataset's own log rather than the project's.
                scope = dataset_scope_of(where) if where is not None else None
                entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
                append(
                    audit_log_key(scope if scope is not None else project),
                    entry,
                )

            # At most one entry per call by construction: the two writing paths are exclusive,
            # and neither writer sits inside a handler that could run the other.
            try:
                result = func(*bound.args, **bound.kwargs)
            except Exception as body_exc:
                entry["status"] = "exception"
                entry["error"] = str(body_exc)
                try:
                    record()
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
            entry["status"] = "ok"
            try:
                record()
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
