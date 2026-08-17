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

    JSON stores do not choose a spelling. Every record encodes through ``RECORD_JSON`` and
    every log through ``LOG_JSON``, and ``register_store`` refuses anything else unless the
    descriptor names its exemption, so a bespoke spelling is a decision someone wrote down
    rather than a knob someone reached for.
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

    ``locator`` is the file backend's identity map for this store, required of a record or a
    log and unused by a blob. Another backend keys on (store, scope, parts) and does not
    consult it to read or write, but the file layout it names is what a record held anywhere
    else is written back out as, so a store without one could only ever be half exported.

    ``codec_exemption`` is why this store does not encode through the canonical JSON codec.
    It is required of any record or log carrying some other JSON spelling, and reading it is
    how a reviewer finds every store that opted out.
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
    codec_exemption: str = ""
    declared_in: str = ""


_registry: dict[str, StoreDescriptor] = {}


def register_store(descriptor: StoreDescriptor) -> StoreDescriptor:
    """Add one store to the catalogue and return the registered descriptor.

    Refuses a name that is already registered: two declarations of one name are two stores
    wearing the same identity, and whichever imported second would silently win. Refuses a
    record with no concurrency policy, and a log or blob that declares one, because the
    policy is a statement about read-modify-write that only a record can make. Refuses a
    record or log that declares no locator, which is the store's own statement of the file it
    owns and the only thing that can put its bytes back on disk in the layout the tools
    reading that layout expect. Refuses a JSON spelling that is not the canonical one for the
    kind, unless the descriptor states why in ``codec_exemption``, so a module nothing has
    imported cannot quietly hold a bespoke codec that no test enumerating the registry would
    ever see.
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
    if descriptor.kind in ("record", "log") and descriptor.locator is None:
        raise ValueError(
            f"{descriptor.kind} store {descriptor.name!r} must declare a locator: it is the "
            "only statement of which file this store owns, so without it the file backend "
            "cannot place the bytes and nothing can write a database-held record back out "
            "as the file the tools that read the layout expect"
        )
    if descriptor.kind == "log" and not descriptor.durable:
        raise ValueError(
            f"log store {descriptor.name!r} cannot relax durability: an append returns only "
            "once the entry will survive a crash, so the declaration would be ignored"
        )
    _check_canonical_codec(descriptor)

    frame = sys._getframe(1)
    registered = replace(descriptor, declared_in=str(frame.f_globals.get("__name__", "?")))
    _registry[descriptor.name] = registered
    return registered


def _check_canonical_codec(descriptor: StoreDescriptor) -> None:
    """Refuse a record or log whose codec is neither canonical nor a stated exemption.

    Text stores are exempt by kind rather than by declaration: a text codec has no spelling
    to choose, since its only knobs are the encoding and a trailing newline, both of which
    are the stored value itself. Everything else that is not ``RECORD_JSON`` or ``LOG_JSON``
    must say why in ``codec_exemption``.
    """
    if descriptor.kind == "blob":
        if descriptor.codec_exemption:
            raise ValueError(
                f"blob store {descriptor.name!r} declares a codec exemption but has no codec"
            )
        return
    canonical = RECORD_JSON if descriptor.kind == "record" else LOG_JSON
    exempt_by_kind = isinstance(descriptor.codec, _TextCodec)
    if descriptor.codec is canonical or exempt_by_kind:
        if descriptor.codec_exemption:
            raise ValueError(
                f"store {descriptor.name!r} declares a codec exemption it does not use: its "
                "codec is already the one every store of its kind carries"
            )
        return
    if not descriptor.codec_exemption:
        name = "RECORD_JSON" if descriptor.kind == "record" else "LOG_JSON"
        raise ValueError(
            f"{descriptor.kind} store {descriptor.name!r} carries a codec that is not "
            f"{name}. Encode through {name} so every store of its kind reads the same, or "
            "state why this one cannot in codec_exemption."
        )


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
    return "read_blob_versioned / put_blob / write_blob / open_blob / delete"


@dataclass(frozen=True)
class _JsonCodec:
    """A JSON codec with one fixed spelling, instantiated twice and never per store."""

    indent: int | None
    ensure_ascii: bool
    default: Callable[[Any], Any] | None
    allow_nan: bool
    sort_keys: bool
    trailing_newline: bool

    def encode(self, value: Any) -> bytes:
        text = json.dumps(
            value,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            default=self.default,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
        )
        if self.trailing_newline:
            text += "\n"
        return text.encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


RECORD_JSON = _JsonCodec(
    indent=2,
    ensure_ascii=False,
    default=None,
    allow_nan=False,
    sort_keys=False,
    trailing_newline=True,
)
"""Every JSON record's spelling.

``default=None`` rather than ``str``: converting an unserializable object to its ``repr``
turns a measurement into a string that reads as valid forever after, so the encode raises
and the writer converts the value explicitly instead. ``allow_nan=False`` because NaN and
Infinity are not JSON and no strict parser, the breeder's browser included, will read them;
a non-finite measurement is represented by its producer with a reason attached.
``sort_keys=False`` because two records carry meaning in their key order: ``classes.json``'s
subject and attribute sequences are read back as ordered tuples, and re-ordering them would
be the codec changing content rather than spelling.
"""

LOG_JSON = _JsonCodec(
    indent=None,
    ensure_ascii=False,
    default=None,
    allow_nan=False,
    sort_keys=False,
    trailing_newline=False,
)
"""Every JSON log entry's spelling.

An entry is one line, so ``indent`` stays None and the backend supplies the terminating
newline. ``sort_keys=False`` keeps the authored field order a human tailing the file reads.
"""


@dataclass(frozen=True)
class _TextCodec:
    """A codec for a store whose value is the text itself.

    ``encode`` refuses anything but ``str``: calling ``str()`` on an arbitrary object here
    would fabricate a value out of a repr exactly the way a JSON ``default`` does. A caller
    holding a number formats it before it reaches the store.
    """

    encoding: str
    trailing_newline: bool

    def encode(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise TypeError(
                f"a text store holds text, so it cannot encode {type(value).__name__}; "
                "format the value into a string before writing it"
            )
        text = value
        if self.trailing_newline and not text.endswith("\n"):
            text += "\n"
        return text.encode(self.encoding)

    def decode(self, data: bytes) -> Any:
        return data.decode(self.encoding)


def text_codec(*, encoding: str = "utf-8", trailing_newline: bool = False) -> Codec:
    """A codec for stores whose value is text and nothing else."""
    return _TextCodec(encoding=encoding, trailing_newline=trailing_newline)
