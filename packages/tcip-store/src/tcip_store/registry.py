"""The store catalogue: what each store is, how its values encode, and how it may be written.

A store is declared once, by the module that owns it, next to that store's key constructor.
Importing that module is what registers the store, so an ``UnknownStore`` is answered by an
import rather than by a configuration file.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from tcip_store.errors import BadKey, UnknownStore, WrongKind
from tcip_store.model import Key

if TYPE_CHECKING:  # a database backend imports this module and must not import a path type
    from tcip_store.file_backend import Locator

Kind = Literal["record", "log", "blob"]
Concurrency = Literal["cas", "last_writer_wins"]


class Codec(Protocol):
    """How one store's values become bytes and come back.

    There is no global codec. The platform's current writers genuinely disagree on
    separators, indentation and trailing newlines, and byte compatibility means preserving
    that disagreement per store rather than rewriting every file on its next touch.
    """

    def encode(self, value: Any) -> bytes: ...

    def decode(self, data: bytes) -> Any: ...


@dataclass(frozen=True)
class StoreDescriptor:
    """Everything the seam knows about one store, declared by the module that owns it.

    ``kind`` decides which operations apply: a record is replaced, a log is appended to, a
    blob is streamed. ``key_fields`` names the parts of the key's identity, coarse to fine,
    and fixes its arity. ``codec`` is required for records and logs and unused for blobs.

    ``concurrency`` is required for records and rejected for the other kinds. A ``cas``
    store is one more than one writer read-modify-writes, so the unconditional write form
    is refused there and only the compare-and-set and transactional forms remain. A
    ``last_writer_wins`` store is written single-shot, so the unconditional form is legal.

    ``durable`` decides whether a write flushes before returning. It never affects locking:
    there is no way to declare a store unlocked. ``enumerable`` decides whether ``keys``
    answers or refuses. ``path_readable`` decides whether a blob store will hand out a real
    filesystem path for a library that cannot take a file object.

    ``locator`` is the file backend's identity map for this store. Other backends key on
    (store, scope, parts) and ignore it.
    """

    name: str
    kind: Kind
    key_fields: tuple[str, ...]
    codec: Codec | None = None
    concurrency: Concurrency | None = None
    durable: bool = True
    enumerable: bool = False
    path_readable: bool = False
    locator: "Locator | None" = None
    declared_in: str = ""


_registry: dict[str, StoreDescriptor] = {}


def register_store(descriptor: StoreDescriptor) -> StoreDescriptor:
    """Add one store to the catalogue and return the registered descriptor.

    Refuses a name that is already registered: two declarations of one name are two stores
    wearing the same identity, and whichever imported second would silently win. Refuses a
    record with no concurrency policy, and a log or blob that declares one, because the
    policy is a statement about read-modify-write that only a record can make.
    """
    if descriptor.name in _registry:
        owner = _registry[descriptor.name].declared_in
        raise ValueError(
            f"store {descriptor.name!r} is already registered (declared in {owner}); "
            "a store name identifies one store"
        )
    if not descriptor.key_fields:
        raise ValueError(f"store {descriptor.name!r} must declare at least one key field")
    if descriptor.kind == "record" and descriptor.concurrency is None:
        raise ValueError(
            f"record store {descriptor.name!r} must declare concurrency='cas' or "
            "concurrency='last_writer_wins'"
        )
    if descriptor.kind != "record" and descriptor.concurrency is not None:
        raise ValueError(
            f"{descriptor.kind} store {descriptor.name!r} must not declare a concurrency "
            "policy: only a record is read-modify-written"
        )
    if descriptor.kind in ("record", "log") and descriptor.codec is None:
        raise ValueError(f"{descriptor.kind} store {descriptor.name!r} must declare a codec")
    if descriptor.kind == "blob" and descriptor.codec is not None:
        raise ValueError(f"blob store {descriptor.name!r} takes bytes, so it has no codec")
    if descriptor.kind == "log" and not descriptor.durable:
        raise ValueError(
            f"log store {descriptor.name!r} cannot relax durability: an append returns only "
            "once the entry will survive a crash, so the declaration would be ignored"
        )

    frame = sys._getframe(1)
    registered = replace(descriptor, declared_in=str(frame.f_globals.get("__name__", "?")))
    _registry[descriptor.name] = registered
    return registered


def get_descriptor(store: str) -> StoreDescriptor:
    """The named store's descriptor, or ``UnknownStore`` naming what is registered."""
    try:
        return _registry[store]
    except KeyError:
        known = ", ".join(
            f"{name} (declared in {d.declared_in})" for name, d in sorted(_registry.items())
        )
        raise UnknownStore(
            f"store {store!r} is not registered. Import the module that declares it before "
            f"using its keys. Registered: {known or 'nothing'}."
        ) from None


def registered_stores() -> tuple[str, ...]:
    """Every registered store name, sorted."""
    return tuple(sorted(_registry))


def validate_key(key: Key, *, expect_kind: Kind | tuple[Kind, ...], operation: str) -> StoreDescriptor:
    """The key's descriptor, once the key is known to fit the store and the operation.

    Raises ``UnknownStore`` for an unregistered store, ``WrongKind`` when the operation does
    not apply to the store's kind, and ``BadKey`` when the parts do not match the declared
    ``key_fields``.
    """
    descriptor = get_descriptor(key.store)
    kinds = (expect_kind,) if isinstance(expect_kind, str) else expect_kind
    if descriptor.kind not in kinds:
        raise WrongKind(
            f"{operation} does not apply to {key.store!r}, which is declared kind "
            f"{descriptor.kind!r}: use {_calls_for(descriptor.kind)}"
        )
    if len(key.parts) != len(descriptor.key_fields):
        raise BadKey(
            f"{key.store!r} is keyed by {list(descriptor.key_fields)} "
            f"({len(descriptor.key_fields)} parts); got {list(key.parts)}"
        )
    for field, part in zip(descriptor.key_fields, key.parts, strict=True):
        if not isinstance(part, str) or not part:
            raise BadKey(f"{key.store!r} key field {field!r} must be a non-empty string; got {part!r}")
    if not key.scope:
        raise BadKey(f"{key.store!r} key carries no scope: name the root the entry hangs off")
    return descriptor


def _calls_for(kind: Kind) -> str:
    if kind == "record":
        return "read / replace / delete / transaction"
    if kind == "log":
        return "append / read_log"
    return "put_blob / write_blob / open_blob"


@dataclass(frozen=True)
class _JsonCodec:
    """A JSON codec whose exact spelling is fixed per store."""

    indent: int | None
    ensure_ascii: bool
    default: Callable[[Any], Any] | None
    separators: tuple[str, str] | None
    allow_nan: bool
    trailing_newline: bool

    def encode(self, value: Any) -> bytes:
        text = json.dumps(
            value,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            default=self.default,
            separators=self.separators,
            allow_nan=self.allow_nan,
        )
        if self.trailing_newline:
            text += "\n"
        return text.encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


def json_codec(
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = str,
    separators: tuple[str, str] | None = None,
    allow_nan: bool = True,
    trailing_newline: bool = False,
) -> Codec:
    """A JSON codec with every knob the platform's existing writers actually differ on.

    ``trailing_newline`` is a knob rather than a convention because three of today's writers
    emit one and a codec without it would drop a byte on the first write through the seam.
    A log's codec must leave ``indent`` at None: an entry is one line.
    """
    return _JsonCodec(
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
        separators=separators,
        allow_nan=allow_nan,
        trailing_newline=trailing_newline,
    )


@dataclass(frozen=True)
class _TextCodec:
    """A plain-text codec."""

    encoding: str
    trailing_newline: bool

    def encode(self, value: Any) -> bytes:
        text = str(value)
        if self.trailing_newline and not text.endswith("\n"):
            text += "\n"
        return text.encode(self.encoding)

    def decode(self, data: bytes) -> Any:
        return data.decode(self.encoding)


def text_codec(*, encoding: str = "utf-8", trailing_newline: bool = False) -> Codec:
    """A codec for stores whose value is text and nothing else."""
    return _TextCodec(encoding=encoding, trailing_newline=trailing_newline)
