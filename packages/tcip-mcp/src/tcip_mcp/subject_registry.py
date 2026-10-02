"""The dataset's subject registry: its subjects and the attributes each declares.

The on-disk registry (``<dataset_root>/subjects.json``) is self-describing and name-based; for
example, with ``tree`` and ``fruit`` as the subjects::

    {
      "tree": {"description": "one tree crown", "defined_by": "...", "defined_at": "..."},
      "fruit": {"description": "one fruit", "defined_by": "...", "defined_at": "...",
                "attributes": {
                  "condition": {"type": "categorical", "values": ["healthy", "diseased"]}
                }}
    }

An attribute is an axis a subject's instances carry, ``categorical`` (unordered) or ``ordinal``
(the ``values`` order is the rank). A value's id is its position in that declared order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from tcip_store import Version

    from tcip_mcp.traits import PositiveState

#: The attribute kinds a subject may carry. Numeric is deliberately absent, see the module docstring.
ATTR_TYPES = ("categorical", "ordinal")


@dataclass(frozen=True)
class Attribute:
    """One classification axis of a subject's instances. ``values`` are ordered; for an ``ordinal``
    attribute that order is the rank (severity 0 < 1 < 2). A known ``type`` and a non-empty list of
    distinct names are enforced here.
    """

    name: str
    type: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.type not in ATTR_TYPES:
            raise ValueError(f"type {self.type!r} not in {ATTR_TYPES}")
        if not self.values or not all(isinstance(v, str) and v for v in self.values):
            raise ValueError("values must be a non-empty list of names")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"duplicate values: {list(self.values)}")


class _Named(Protocol):
    @property
    def name(self) -> str: ...


_N = TypeVar("_N", bound=_Named)


def named(records: Iterable[_N], name: str) -> _N | None:
    """The one of ``records`` (subjects or attributes) named ``name``, or ``None`` when none is."""
    return next((r for r in records if r.name == name), None)


@dataclass(frozen=True)
class Subject:
    """The object a label set is about. ``attributes`` may be empty, such a subject is only detected."""

    name: str
    description: str = ""
    defined_by: str = ""
    defined_at: str = ""
    attributes: tuple[Attribute, ...] = ()

    def attribute(self, name: str) -> Attribute | None:
        """:func:`named` over this subject's attributes."""
        return named(self.attributes, name)


@dataclass(frozen=True)
class SubjectRegistry:
    """The whole ``subjects.json``, the dataset's subjects, in declared order."""

    subjects: tuple[Subject, ...] = ()

    def subject(self, name: str) -> Subject | None:
        """:func:`named` over this registry's subjects."""
        return named(self.subjects, name)


class RegistryError(ValueError):
    """A registry that cannot be read as a valid subject registry, or a scope it does not contain."""


def registry_from_dict(data: object) -> SubjectRegistry:
    """Parse the nested registry mapping into a :class:`SubjectRegistry`, preserving declared
    order.

    Refuses a malformed shape: an attribute whose ``type`` is unknown, or whose ``values`` are
    absent/empty/non-string/duplicated.
    """
    if not isinstance(data, dict):
        raise RegistryError(f"registry must be a JSON object of subjects, got {type(data).__name__}")
    subjects: list[Subject] = []
    for sname, sbody in data.items():
        if not isinstance(sbody, dict):
            raise RegistryError(f"subject {sname!r} must be an object, got {type(sbody).__name__}")
        attrs: list[Attribute] = []
        raw_attrs = sbody.get("attributes")
        if raw_attrs is None:  # absent/null attributes -> a detection-only subject (valid)
            raw_attrs = {}
        if not isinstance(raw_attrs, dict):  # a falsy non-object (false/0/""/[]) is malformed, not "none"
            raise RegistryError(
                f"subject {sname!r} 'attributes' must be an object, got {type(raw_attrs).__name__}")
        for aname, abody in raw_attrs.items():
            if not isinstance(abody, dict):
                raise RegistryError(f"attribute {sname}.{aname} must be an object")
            values = abody.get("values")
            if not isinstance(values, list):
                raise RegistryError(f"attribute {sname}.{aname} 'values' must be a list")
            # The value invariant (known type, non-empty, distinct) lives on Attribute, one guard,
            # shared by the parser and any code that constructs an Attribute directly.
            try:
                # abody is arbitrary decoded JSON; Attribute.__post_init__ is the real type gate
                # (raises on a missing/invalid type), so the value is cast rather than validated here.
                attrs.append(
                    Attribute(name=aname, type=cast(str, abody.get("type")), values=tuple(values)))
            except ValueError as exc:
                raise RegistryError(f"attribute {sname}.{aname}: {exc}") from exc
        subjects.append(Subject(
            name=sname,
            description=str(sbody.get("description", "")),
            defined_by=str(sbody.get("defined_by", "")),
            defined_at=str(sbody.get("defined_at", "")),
            attributes=tuple(attrs),
        ))
    return SubjectRegistry(subjects=tuple(subjects))


def registry_to_dict(registry: SubjectRegistry) -> dict:
    """Serialize a :class:`SubjectRegistry` back to the nested mapping (inverse of
    :func:`registry_from_dict`; empty description/provenance fields are still written for legibility)."""
    out: dict[str, dict] = {}
    for s in registry.subjects:
        body: dict = {"description": s.description, "defined_by": s.defined_by, "defined_at": s.defined_at}
        if s.attributes:
            body["attributes"] = {
                a.name: {"type": a.type, "values": list(a.values)} for a in s.attributes
            }
        out[s.name] = body
    return out


def _checked_registry_document(data: bytes, *, path: str | Path) -> dict:
    """The stored registry's decoded document, version-checked.

    Raises :class:`RegistryError` for bytes that do not decode as JSON. Propagates
    :class:`tcip_store.SchemaVersionRefused`, uncaught.
    """
    import tcip_store

    try:
        document = tcip_store.RECORD_JSON.decode(data)
    except ValueError as exc:
        raise RegistryError(f"{path} does not decode as JSON: {exc}") from exc
    from tcip_mcp.dataset_layout import SUBJECT_REGISTRY_STORE

    tcip_store.check_schema_version(tcip_store.get_descriptor(SUBJECT_REGISTRY_STORE), document)
    return document


def read_versioned_registry(dataset_root: str | Path) -> tuple[SubjectRegistry, "Version"]:
    """Read ``dataset_root``'s subject registry into a :class:`SubjectRegistry`, beside the
    version of the bytes it decoded, both from one read.

    No registry raises ``FileNotFoundError``, and a registry whose bytes are present but will not
    decode raises :class:`RegistryError`, the same refusal a structurally invalid registry raises.
    A ``schema_version`` this reader does not accept propagates as
    :class:`tcip_store.SchemaVersionRefused`, uncaught.
    """
    import tcip_store

    from tcip_mcp.dataset_layout import subject_registry_key, subjects_path

    path = subjects_path(dataset_root)
    try:
        versioned = tcip_store.read_blob_versioned(subject_registry_key(dataset_root))
    except tcip_store.NotFound as exc:
        raise FileNotFoundError(f"no subject registry at {path}") from exc
    document = _checked_registry_document(versioned.value, path=path)
    return registry_from_dict(document), versioned.version


def read_registry(dataset_root: str | Path) -> SubjectRegistry:
    """:func:`read_versioned_registry`'s registry alone."""
    return read_versioned_registry(dataset_root)[0]


def _dropped_names(outgoing: SubjectRegistry, incoming: SubjectRegistry) -> list[str]:
    """Every subject, attribute or attribute value ``outgoing`` declares that ``incoming`` does
    not, dotted (``subject``, ``subject.attribute``, ``subject.attribute=value``)."""
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
                f"{s.name}.{a.name}={v}" for v in a.values if v not in new_a.values
            )
    return dropped


def replace_registry(
    dataset_root: str | Path, registry: SubjectRegistry, *, expect: "Version | None",
    allow_removals: bool = False, allow_type_changes: bool = False,
) -> dict:
    """Write ``dataset_root``'s registry, reading what it replaces and refusing a silent drop.

    Refuses an empty ``registry`` outright, whether or not ``allow_removals`` is set. Reads the
    stored registry (absent reads as no prior registry, not a refusal) and refuses a write that
    drops a subject, an attribute, or an attribute value the stored one declares, unless
    ``allow_removals`` is true. Stored bytes present but undecodable are likewise refused unless
    ``allow_removals`` is true. A stored registry whose ``schema_version`` this reader does not
    accept refuses the whole write regardless of ``allow_removals``:
    :class:`tcip_store.SchemaVersionRefused` propagates uncaught from
    :func:`_checked_registry_document`.

    Refuses, independently of ``allow_removals``, a write that keeps an attribute's name and values
    but changes its ``type`` (categorical to ordinal or back), unless ``allow_type_changes`` is
    set.

    ``expect`` is compare-and-set against the blob's actual version at write time
    (``tcip_store.VersionConflict`` on a mismatch, nothing written): pass the version the caller
    read, or ``Version.ABSENT`` for a caller asserting no registry exists yet. ``None`` checks
    against the version this call read.

    The write's one audit line follows it in the dataset's log (``AuditEntryNotWritten`` when it
    cannot be appended, carrying that line's arguments). Returns the committed save as its audit
    line records it: ``{"subjects_path", "n_subjects", "version"}`` (the new token).
    """
    import tcip_store

    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.dataset_layout import subject_registry_key, subjects_path

    if not registry.subjects:
        raise RegistryError("a subject registry write must declare at least one subject")

    path = subjects_path(dataset_root)
    key = subject_registry_key(dataset_root)
    versioned = tcip_store.read_blob_versioned(key, default=None)
    outgoing: SubjectRegistry | None = None
    if versioned.value is not None:
        try:
            outgoing = registry_from_dict(_checked_registry_document(versioned.value, path=path))
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
        key, tcip_store.RECORD_JSON.encode(registry_to_dict(registry)),
        expect=versioned.version if expect is None else expect,
    )

    committed = {"subjects_path": str(path), "n_subjects": len(registry.subjects),
                 "version": new_version.token}
    record_event_or_raise("replace_registry", committed, scope=dataset_root)
    return committed


def copy_registry(source: str | Path, destination: str | Path) -> None:
    """Place dataset root ``source``'s registry beside dataset root ``destination``'s data, once:
    the stored document byte-for-byte, written create-only (``expect=Version.ABSENT``), refusing
    when the destination already holds a registry.
    """
    import tcip_store

    from tcip_mcp.dataset_layout import subject_registry_key

    try:
        tcip_store.put_blob(
            subject_registry_key(destination),
            tcip_store.read_blob_versioned(subject_registry_key(source)).value,
            expect=tcip_store.Version.ABSENT,
        )
    except tcip_store.VersionConflict as exc:
        raise RegistryError(
            f"a subject registry already exists at {destination}; copy_registry places a first "
            "copy only and never replaces one"
        ) from exc


def positive_state_problem(registry: SubjectRegistry, subject_name: str,
                           state: "PositiveState") -> str | None:
    """Why ``state`` cannot be ``subject_name``'s positive state in ``registry``, or ``None`` when
    that subject declares ``state``'s attribute and the attribute lists ``state``'s value."""
    subject = registry.subject(subject_name)
    if subject is None:
        known = [s.name for s in registry.subjects]
        return f"no subject {subject_name!r} in the registry (subjects: {known})"
    attribute = subject.attribute(state.attribute)
    if attribute is None:
        return (f"subject {subject_name!r} declares no attribute {state.attribute!r} (it declares "
                f"{[a.name for a in subject.attributes]})")
    if state.value not in attribute.values:
        return (f"attribute {state.attribute!r} of {subject_name!r} declares no value "
                f"{state.value!r} (it declares {list(attribute.values)})")
    return None


def registry_for_dataset_root(dataset_root: str | Path) -> SubjectRegistry | None:
    """The registry at ``dataset_root``, or ``None`` when no ``subjects.json`` has been written there
    yet (a dataset with no registry is not corrupt, only unregistered so far)."""
    try:
        return read_registry(dataset_root)
    except FileNotFoundError:
        return None


def distinct_dataset_root(pred_dirs: Sequence[str | Path]) -> Path | None:
    """The single dataset root every one of ``pred_dirs`` resolves under, or ``None`` when none
    do. Refuses (``RegistryError``) when the directories span more than one.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    roots: set[Path] = {r for d in pred_dirs
                        if d and (r := dataset_root_of(Path(d).resolve())) is not None}
    if len(roots) > 1:
        raise RegistryError(
            "a delivery's prediction directories resolve to more than one dataset root "
            f"({sorted(str(r) for r in roots)}); no delivery this platform ships spans more than "
            "one dataset, so this cannot be reconciled to a single registry"
        )
    return next(iter(roots), None)


def dataset_root_for_pred_dirs(pred_dirs: Sequence[str | Path]) -> Path:
    """The single dataset root every one of ``pred_dirs`` resolves under.

    Refuses (``RegistryError``) when the directories span more than one dataset root, and refuses
    by name when none resolves to a dataset root at all.
    """
    root = distinct_dataset_root(pred_dirs)
    if root is None:
        raise RegistryError(
            f"none of the prediction directories {[str(d) for d in pred_dirs]} resolves to a "
            "dataset root, so there is no dataset for a plant-mapping delivery to attribute "
            "these predictions to; re-export under the dataset's own predictions tree "
            "(images/<date>/ sibling), or register this bucket's root as a dataset through "
            "register_dataset"
        )
    return root


def registry_for_pred_dirs(pred_dirs: Sequence[str | Path]) -> SubjectRegistry | None:
    """The registry for the single dataset every one of ``pred_dirs`` resolves under.

    ``None`` when none of the directories resolves to a dataset root, or the one they do resolve to
    carries no registry yet. Refuses (``RegistryError``) when the directories span more than one
    dataset root.
    """
    root = distinct_dataset_root(pred_dirs)
    if root is None:
        return None
    return registry_for_dataset_root(root)
