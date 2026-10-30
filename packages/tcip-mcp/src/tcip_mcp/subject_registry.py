"""The dataset's subject registry: its subjects and the attributes each declares.

The on-disk registry (``<dataset_root>/subjects.json``) is self-describing and name-based; for
example, with ``tree`` and ``fruit`` as the subjects::

    {
        "tree": {"description": "one tree crown"},
        "fruit": {
            "description": "one fruit",
            "attributes": {
                "condition": {
                    "type": "categorical",
                    "values": ["healthy", "diseased"],
                }
            },
        },
    }

An attribute is an axis a subject's instances carry, ``categorical`` (unordered) or ``ordinal``
(the ``values`` order is the rank). A value's id is its position in that declared order.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    SerializerFunctionWrapHandler,
    Strict,
    TypeAdapter,
    WrapSerializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tcip_annotation.state import Attribute

if TYPE_CHECKING:
    from pydantic import JsonValue
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema

    from tcip_mcp.traits import PositiveState
    from tcip_store import Version


class _Named(Protocol):
    @property
    def name(self) -> str: ...


_N = TypeVar("_N", bound=_Named)


def named(records: Iterable[_N], name: str) -> _N | None:
    """Return the first of ``records`` named ``name``, or ``None`` when none is."""
    return next((r for r in records if r.name == name), None)


def _entries(document: object) -> object:
    """Return a name-keyed mapping as its entries in order, each stating its key as ``name``.

    Anything else is returned as given: a tuple of entries is read as itself, and any other value
    is refused by the tuple it is validated as.
    """
    if not isinstance(document, dict):
        return document
    return tuple(
        {**body, "name": name} if isinstance(body, dict) else body
        for name, body in document.items()
    )


def _keyed(
    entries: tuple[Any, ...], handler: SerializerFunctionWrapHandler
) -> dict[str, Any]:
    """Return ``entries`` serialized as a mapping from each entry's ``name`` to its other fields."""
    return {
        serialized.pop("name"): serialized for serialized in handler(entries)
    }


class _KeyedSchema:
    """The JSON schema of a :data:`ByName` field: the mapping it is stored and sent as.

    The mapping's value schema is an entry's own, its ``name`` dropped, since the key names it.
    """

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Return the mapping's schema from the entry schema of the tuple ``schema`` wraps."""
        held: Any = schema
        while held["type"] != "tuple":
            held = held["schema"]
        entry = dict(handler(held["items_schema"][0]))
        entry["properties"] = {
            k: v for k, v in entry["properties"].items() if k != "name"
        }
        entry["required"] = [k for k in entry["required"] if k != "name"]
        return {"type": "object", "additionalProperties": entry}


type ByName[T] = Annotated[
    tuple[T, ...],
    Strict(),
    BeforeValidator(_entries),
    WrapSerializer(_keyed),
    _KeyedSchema(),
]
"""Entries stored as one mapping from each entry's name to its other fields, in declared order;
held as a tuple, and read from that mapping or a tuple of entries, never a list."""


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Subject:
    """The object a label set is about.

    ``attributes`` may be empty, such a subject is only detected, and is then not written.
    """

    name: str
    description: str
    attributes: ByName[Attribute] = Field(
        default=(), exclude_if=lambda attributes: not attributes
    )

    def attribute(self, name: str) -> Attribute | None:
        """Return the named attribute of this subject (:func:`named`), or ``None``."""
        return named(self.attributes, name)


type RegistrySubjects = Annotated[ByName[Subject], Field(min_length=1)]
"""A registry's subjects: at least one, by name, in declared order."""


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SubjectRegistry:
    """The whole ``subjects.json``, the dataset's subjects (:data:`RegistrySubjects`)."""

    subjects: RegistrySubjects

    def subject(self, name: str) -> Subject | None:
        """Return the named subject of this registry (:func:`named`), or ``None``."""
        return named(self.subjects, name)


_REGISTRY = TypeAdapter(SubjectRegistry)


class RegistryError(ValueError):
    """A registry that cannot be read as a valid subject registry, or a scope it lacks."""


def registry_from_dict(data: object) -> SubjectRegistry:
    """Parse the nested registry mapping into a :class:`SubjectRegistry`, in declared order.

    Refuses (:class:`RegistryError`) what does not validate as the registry, naming each field.
    """
    try:
        return _REGISTRY.validate_python({"subjects": data})
    except ValueError as exc:
        raise RegistryError(
            f"the subject registry does not read: {exc}"
        ) from exc


def registry_to_dict(registry: SubjectRegistry) -> dict:
    """Serialize a :class:`SubjectRegistry` back to the nested mapping.

    The inverse of :func:`registry_from_dict`.
    """
    return _REGISTRY.dump_python(registry, mode="json")["subjects"]


def _checked_registry_document(data: bytes, *, path: str | Path) -> JsonValue:
    """Return the stored registry's decoded document.

    Raises :class:`RegistryError` for bytes that do not decode as JSON.
    """
    import tcip_store

    try:
        return tcip_store.decode_value(data)
    except ValueError as exc:
        raise RegistryError(f"{path} does not decode as JSON: {exc}") from exc


def read_versioned_registry(
    dataset_root: str | Path,
) -> tuple[SubjectRegistry, Version]:
    """Read ``dataset_root``'s subject registry and the version of the bytes it decoded.

    Both come from one read; the registry is a :class:`SubjectRegistry`.

    No registry raises ``FileNotFoundError``, and a registry whose bytes are present but will not
    decode raises :class:`RegistryError`, the same refusal a structurally invalid registry raises.
    """
    import tcip_store
    from tcip_mcp.dataset_layout import subjects_path

    path = subjects_path(dataset_root)
    try:
        versioned = tcip_store.read_blob_versioned(path)
    except tcip_store.NotFoundError as exc:
        raise FileNotFoundError(f"no subject registry at {path}") from exc
    document = _checked_registry_document(versioned.value, path=path)
    return registry_from_dict(document), versioned.version


def read_registry(dataset_root: str | Path) -> SubjectRegistry:
    """Return :func:`read_versioned_registry`'s registry alone."""
    return read_versioned_registry(dataset_root)[0]


def _dropped_names(
    outgoing: SubjectRegistry, incoming: SubjectRegistry
) -> list[str]:
    """Return every subject, attribute or value ``outgoing`` declares that ``incoming`` does not.

    Each is dotted: ``subject``, ``subject.attribute``, ``subject.attribute=value``.
    """
    dropped: list[str] = []
    for s in outgoing.subjects:
        new_s = incoming.subject(s.name)
        if new_s is None:
            dropped.append(s.name)
            continue
        for a in s.attributes:
            new_a = new_s.attribute(a.name)
            if new_a is None:
                dropped.append(f"{s.name}.{a.name}")
                continue
            dropped.extend(
                f"{s.name}.{a.name}={v}"
                for v in a.values
                if v not in new_a.values
            )
    return dropped


class RegistrySave(BaseModel):
    """A committed registry save as its audit line records it: the file, its subjects, its token."""

    subjects_path: str
    n_subjects: int
    version: str


def replace_registry(
    dataset_root: str | Path,
    registry: SubjectRegistry,
    *,
    expect: Version | None,
    allow_removals: bool = False,
    allow_type_changes: bool = False,
    actor: str | None,
) -> RegistrySave:
    """Write ``dataset_root``'s registry, reading what it replaces and refusing a silent drop.

    Refuses an empty ``registry`` outright, whether or not ``allow_removals`` is set. Reads the
    stored registry (absent reads as no prior registry, not a refusal) and refuses a write that
    drops a subject, an attribute, or an attribute value the stored one declares, unless
    ``allow_removals`` is true. Stored bytes present but undecodable are likewise refused unless
    ``allow_removals`` is true.

    Refuses, independently of ``allow_removals``, a write that keeps an attribute's name and values
    but changes its ``type`` (categorical to ordinal or back), unless ``allow_type_changes`` is
    set.

    ``expect`` is compare-and-set against the blob's actual version at write time
    (``tcip_store.VersionConflictError`` on a mismatch, nothing written): a version read before,
    or ``Version.ABSENT`` asserting no registry exists yet; ``None`` checks against the version
    this call read.

    The write's one audit line by ``actor`` follows it in the dataset's log
    (``AuditEntryNotWrittenError`` when it cannot be appended, carrying that line's arguments).
    Returns the committed save, the new token its ``version``.
    """
    import tcip_store
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.dataset_layout import subjects_path

    path = subjects_path(dataset_root)
    versioned = tcip_store.read_blob_versioned(path, default=None)
    outgoing: SubjectRegistry | None = None
    if versioned.value is not None:
        try:
            outgoing = registry_from_dict(
                _checked_registry_document(versioned.value, path=path)
            )
        except ValueError as exc:
            if not allow_removals:
                raise RegistryError(
                    f"the stored registry at {path} does not decode ({exc}); pass allow_removals "
                    "to replace it anyway, since a repair drops whatever the stored bytes held"
                ) from exc

    if outgoing is not None and not allow_removals:
        dropped = _dropped_names(outgoing, registry)
        if dropped:
            raise RegistryError(
                f"this write drops {dropped} from the registry at {path}, and labels or "
                "confirmations may still reference them; pass allow_removals to drop them "
                "deliberately"
            )

    if outgoing is not None and not allow_type_changes:
        for s in outgoing.subjects:
            new_s = registry.subject(s.name)
            if new_s is None:
                continue
            for a in s.attributes:
                new_a = new_s.attribute(a.name)
                if new_a is not None and new_a.type != a.type:
                    raise RegistryError(
                        f"this write changes {s.name}.{a.name}'s type from {a.type!r} to "
                        f"{new_a.type!r} at {path}; pass allow_type_changes to state the flip as "
                        "deliberate. Every recorded value of that attribute acquires the other "
                        "type's meaning (a rank, or an unordered label)"
                    )

    new_version = tcip_store.put_blob(
        path,
        tcip_store.encode_record(registry_to_dict(registry)),
        expect=versioned.version if expect is None else expect,
    )

    committed = RegistrySave(
        subjects_path=str(path),
        n_subjects=len(registry.subjects),
        version=new_version.token,
    )
    record_event_or_raise(
        "replace_registry",
        committed.model_dump(),
        actor=actor,
        scope=dataset_root,
    )
    return committed


def copy_registry(source: str | Path, destination: str | Path) -> None:
    """Place dataset root ``source``'s registry beside dataset root ``destination``'s data, once.

    The stored document is copied byte-for-byte, written create-only (``expect=Version.ABSENT``),
    refusing (:class:`RegistryError`) when the destination already holds a registry.
    """
    import tcip_store
    from tcip_mcp.dataset_layout import subjects_path

    try:
        tcip_store.put_blob(
            subjects_path(destination),
            tcip_store.read_blob_versioned(subjects_path(source)).value,
            expect=tcip_store.Version.ABSENT,
        )
    except tcip_store.VersionConflictError as exc:
        raise RegistryError(
            f"a subject registry already exists at {destination}; copy_registry places a first "
            "copy only and never replaces one"
        ) from exc


def positive_state_problem(
    registry: SubjectRegistry, subject_name: str, state: PositiveState
) -> str | None:
    """Return why ``state`` cannot be ``subject_name``'s positive state in ``registry``.

    Returns ``None`` when it can: that subject declares ``state``'s attribute and the attribute
    lists ``state``'s value.
    """
    subject = registry.subject(subject_name)
    if subject is None:
        known = [s.name for s in registry.subjects]
        return (
            f"no subject {subject_name!r} in the registry (subjects: {known})"
        )
    attribute = subject.attribute(state.attribute)
    if attribute is None:
        return (
            f"subject {subject_name!r} declares no attribute {state.attribute!r} (it declares "
            f"{[a.name for a in subject.attributes]})"
        )
    if state.value not in attribute.values:
        return (
            f"attribute {state.attribute!r} of {subject_name!r} declares no value "
            f"{state.value!r} (it declares {list(attribute.values)})"
        )
    return None


def registry_for_dataset_root(
    dataset_root: str | Path,
) -> SubjectRegistry | None:
    """Return the registry at ``dataset_root``, or ``None`` when none has been written there yet.

    A dataset with no ``subjects.json`` is not corrupt, only unregistered so far.
    """
    try:
        return read_registry(dataset_root)
    except FileNotFoundError:
        return None
