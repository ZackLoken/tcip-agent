"""The dataset's class registry — subjects, their attributes, and the deterministic
name→id assignment a training run uses (and records, so predictions stay decodable).

The on-disk registry (``<dataset_root>/classes.json``) is self-describing and name-based::

    {
      "bush":   {"description": "one hazelnut bush crown", "defined_by": "...", "defined_at": "..."},
      "catkin": {"description": "a hazelnut catkin", "defined_by": "...", "defined_at": "...",
                 "attributes": {
                   "elongation": {"type": "categorical", "values": ["dormant", "elongated"]}
                 }}
    }

A *subject* is the object a label set is about (catkin, bush, efb). A subject with no
``attributes`` is simply detected. An *attribute* is an independent axis a subject's instances
carry — ``categorical`` (unordered) or ``ordinal`` (ordered; the ``values`` order is the rank).
Numeric is not an attribute type; measured/field values live in the plant-keyed field CSVs.

Labels reference these names, never integer ids. Integer class ids exist only inside a training
run: :func:`assign_class_ids` maps the names in a training scope to contiguous 0-indexed ids in
their *declared order*, deterministically and re-derivably — so the loader that builds targets,
the model that predicts, and the code that later decodes a prediction all agree by construction.
Ordering is the declared ``values`` order and never sorted: ordinal values carry rank, which
sorting would corrupt. Because the registry file's order could change between training and decode,
a run also *records* the map it used (the return value of :func:`assign_class_ids`) beside its
predictions; decode reads that recorded map, never a fresh derivation against an edited registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: The attribute kinds a subject may carry. Numeric is deliberately absent — see the module docstring.
ATTR_TYPES = ("categorical", "ordinal")


@dataclass(frozen=True)
class Attribute:
    """One classification axis of a subject's instances. ``values`` are ordered; for an ``ordinal``
    attribute that order is the rank (severity 0 < 1 < 2). The value invariant — a known ``type`` and a
    non-empty list of distinct names — is enforced here, so no ``Attribute``, however it is built, can
    hold values that would silently collapse the name→id map in :func:`assign_class_ids`."""

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


@dataclass(frozen=True)
class Subject:
    """The object a label set is about. ``attributes`` may be empty — such a subject is only detected."""

    name: str
    description: str = ""
    defined_by: str = ""
    defined_at: str = ""
    attributes: tuple[Attribute, ...] = ()

    def attribute(self, name: str) -> Attribute | None:
        for a in self.attributes:
            if a.name == name:
                return a
        return None


@dataclass(frozen=True)
class ClassRegistry:
    """The whole ``classes.json`` — the dataset's subjects, in declared order."""

    subjects: tuple[Subject, ...] = ()

    def subject(self, name: str) -> Subject | None:
        for s in self.subjects:
            if s.name == name:
                return s
        return None


class RegistryError(ValueError):
    """A registry that cannot be read as a valid class registry, or a scope it does not contain."""


def registry_from_dict(data: object) -> ClassRegistry:
    """Parse the nested registry mapping into a :class:`ClassRegistry`, preserving declared order.

    Refuses a malformed shape rather than guessing (an attribute whose ``type`` is unknown, or whose
    ``values`` are absent/empty/non-string/duplicated), so a bad registry fails loudly instead of
    silently assigning ids over garbage.
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
            # The value invariant (known type, non-empty, distinct) lives on Attribute — one guard,
            # shared by the parser and any code that constructs an Attribute directly.
            try:
                attrs.append(Attribute(name=aname, type=abody.get("type"), values=tuple(values)))
            except ValueError as exc:
                raise RegistryError(f"attribute {sname}.{aname}: {exc}") from exc
        subjects.append(Subject(
            name=sname,
            description=str(sbody.get("description", "")),
            defined_by=str(sbody.get("defined_by", "")),
            defined_at=str(sbody.get("defined_at", "")),
            attributes=tuple(attrs),
        ))
    return ClassRegistry(subjects=tuple(subjects))


def registry_to_dict(registry: ClassRegistry) -> dict:
    """Serialize a :class:`ClassRegistry` back to the nested mapping (inverse of
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


def read_registry(path: str | Path) -> ClassRegistry:
    """Read ``classes.json`` into a :class:`ClassRegistry`."""
    return registry_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_registry(path: str | Path, registry: ClassRegistry) -> None:
    """Write a :class:`ClassRegistry` to ``classes.json`` (atomic)."""
    from tcip_mcp.utils.atomic_io import atomic_write_text

    atomic_write_text(Path(path), json.dumps(registry_to_dict(registry), indent=2) + "\n")


def assign_class_ids(registry: ClassRegistry, subject: str, attribute: str | None = None) -> dict[str, int]:
    """The deterministic name→id map for one training scope, in the registry's *declared* order.

    - ``attribute`` given: one class per value of that attribute (``{value: 0..N-1}``), in the order
      the registry declares them — the rank order for an ordinal attribute.
    - ``attribute`` is ``None``: the subject is trained as a single detection class (``{subject: 0}``),
      whether or not it carries attributes — a plain detector that does not classify instances.

    Same registry + scope → identical map, every call: the assignment iterates the declared ``values``
    tuple, never a set/dict, so nothing depends on hashing or insertion iteration. Callers that need
    the map to survive a later registry edit must record this return value (see the module docstring),
    not re-derive it. Raises :class:`RegistryError` for an absent subject/attribute rather than guessing.
    """
    subj = registry.subject(subject)
    if subj is None:
        known = [s.name for s in registry.subjects]
        raise RegistryError(f"subject {subject!r} not in registry (subjects: {known})")
    if attribute is None:
        return {subject: 0}
    attr = subj.attribute(attribute)
    if attr is None:
        known = [a.name for a in subj.attributes]
        raise RegistryError(f"attribute {attribute!r} not on subject {subject!r} (attributes: {known})")
    return {value: idx for idx, value in enumerate(attr.values)}


def num_classes(registry: ClassRegistry, subject: str, attribute: str | None = None) -> int:
    """Class count for a training scope — the size of :func:`assign_class_ids` (0 = background is the
    detector's own offset, applied by the loader, not counted here)."""
    return len(assign_class_ids(registry, subject, attribute))


def decode_class_ids(id_map: dict[str, int]) -> dict[int, str]:
    """Invert a recorded name→id map to id→name, for decoding a run's predictions."""
    return {cid: name for name, cid in id_map.items()}
