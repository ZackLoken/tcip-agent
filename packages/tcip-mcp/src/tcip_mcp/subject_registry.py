"""The dataset's subject registry, subjects, their attributes, and the deterministic name→id
assignment a training run uses (and records, so predictions stay decodable).

The on-disk registry (``<dataset_root>/subjects.json``) is self-describing and name-based; for
example, with ``tree`` and ``fruit`` as the subjects::

    {
      "tree": {"description": "one tree crown", "defined_by": "...", "defined_at": "..."},
      "fruit": {"description": "one fruit", "defined_by": "...", "defined_at": "...",
                "attributes": {
                  "condition": {"type": "categorical", "values": ["healthy", "diseased"]}
                }}
    }

A subject is the object a label set is about. A subject with no ``attributes`` is simply detected.
An attribute is an independent axis a subject's instances carry, ``categorical`` (unordered) or
``ordinal`` (ordered; the ``values`` order is the rank). Numeric is not an attribute type;
measured/field values live in the plant-keyed field CSVs.

Labels reference these names, never integer ids. Integer class ids exist only inside a training
run: :func:`assign_class_ids` maps the names in a training scope to contiguous 0-indexed ids in
their declared order, deterministically and re-derivably. Ordering is the declared ``values`` order
and never sorted. A run records the map it used: the producer that admitted its samples states the
scope on the run's own data config, which travels onto the checkpoint via the run's own config
object and, best-effort, onto the durable experiment record. Decode reads that recorded map first
(``inference_tools.resolve_decode_id_map``), falling back to a fresh derivation from the inference
dataset's registry only for a checkpoint with no recorded map, a run whose ground truth carries its
own classes and which no registry scopes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tcip_store import Key, Version

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


@dataclass(frozen=True)
class Subject:
    """The object a label set is about. ``attributes`` may be empty, such a subject is only detected."""

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
class SubjectRegistry:
    """The whole ``subjects.json``, the dataset's subjects, in declared order."""

    subjects: tuple[Subject, ...] = ()

    def subject(self, name: str) -> Subject | None:
        for s in self.subjects:
            if s.name == name:
                return s
        return None


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


def attribute_schema_digest(registry: SubjectRegistry, subject: str) -> str | None:
    """Digest over ``subject``'s attribute vocabulary (name -> {type, declared-order values}) only.

    ``None`` if ``subject`` is not in the registry at all. Excludes
    ``description``/``defined_by``/``defined_at``. An attribute-less subject gets a real, stable
    digest of ``{}``.
    """
    s = registry.subject(subject)
    if s is None:
        return None
    canonical = json.dumps(
        {a.name: {"type": a.type, "values": list(a.values)} for a in s.attributes},
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _registry_key(path: str | Path) -> "Key":
    """The stored registry a ``subjects.json`` path names, addressed by the dataset root holding it."""
    from tcip_mcp.dataset_layout import subject_registry_key

    return subject_registry_key(Path(path).absolute().parent)


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


def read_registry(path: str | Path) -> SubjectRegistry:
    """Read ``subjects.json`` into a :class:`SubjectRegistry`.

    No registry raises ``FileNotFoundError``, and a registry whose bytes are present but will not
    decode raises :class:`RegistryError`, the same refusal a structurally invalid registry raises.
    A ``schema_version`` this reader does not accept propagates as
    :class:`tcip_store.SchemaVersionRefused`, uncaught.
    """
    import tcip_store

    try:
        data = tcip_store.read_blob_versioned(_registry_key(path)).value
    except tcip_store.NotFound as exc:
        raise FileNotFoundError(f"no subject registry at {path}") from exc
    document = _checked_registry_document(data, path=path)
    return registry_from_dict(document)


def write_registry(path: str | Path, registry: SubjectRegistry) -> None:
    """Write a :class:`SubjectRegistry` to ``subjects.json``, unconditionally.

    Encoded through the canonical record codec. A plain overwrite, with no compare-and-set and no
    refusal for a dropped name; :func:`replace_registry` is the checked write.
    """
    import tcip_store

    tcip_store.put_blob(
        _registry_key(path), tcip_store.RECORD_JSON.encode(registry_to_dict(registry))
    )


def read_version(path: str | Path) -> "Version":
    """The subject registry blob's current version token (``Version.ABSENT`` if it does not exist).

    Reads the version alone, never the content, so it never raises on bytes that will not decode.
    """
    import tcip_store

    return tcip_store.read_blob_versioned(_registry_key(path), default=None).version


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


def _sweep_schema_change(
    dataset_root: Path, outgoing: SubjectRegistry | None, incoming: SubjectRegistry
) -> dict:
    """Stamp the outgoing attribute-schema digest onto every confirmation of an affected subject
    that carries no stamp yet, before ``incoming`` is what a later read sees.

    A confirmation and its digest stamp are two transactions, so unstamped confirmations exist, and
    an unstamped confirmation reads as valid. ``outgoing`` (``None`` for a first-ever write, or a
    stored registry :func:`replace_registry` could not decode) is the last state that digest is
    recoverable from, so it is recorded here, and those confirmations then read as stale exactly
    like the stamped ones.

    Stamps every status in the subject's buckets, not the negatives alone. Already-stamped
    confirmations, and subjects whose digest is unchanged, are left alone.

    Also counts, per affected subject, its finished confirmations
    (:func:`~tcip_mcp.dataset_layout.is_finished_status`, complete or negative) whose stamped
    digest, once this write's own stamping has landed, still disagrees with the subject's new
    digest, computed with :func:`~tcip_mcp.pipelines.data.label_queries.stale_stamped_names`.

    Never blocks the registry write, which has already landed by the time this runs: an absent
    ``outgoing`` is a no-op, and a failing sweep returns a ``warning`` for the caller to surface.
    Returns ``{"newly_stamped": {subject: count}, "predating_vocabulary": {subject: count},
    "warning": str | None}``, each count over finished statuses only.
    """
    import tcip_store

    from tcip_mcp.dataset_layout import (
        bucket_digest_stamps, bucket_subject_date, image_status_digest_key, image_status_key,
        is_finished_status, status_tokens, stamp_image_status_digests,
    )
    from tcip_mcp.pipelines.data.label_queries import stale_stamped_names

    newly_stamped: dict[str, int] = {}
    predating_vocabulary: dict[str, int] = {}
    empty = {"newly_stamped": newly_stamped, "predating_vocabulary": predating_vocabulary}
    if outgoing is None:
        return {**empty, "warning": None}
    changed = {
        s.name: digest for s in outgoing.subjects
        if (digest := attribute_schema_digest(outgoing, s.name)) is not None
        and digest != attribute_schema_digest(incoming, s.name)
    }
    if not changed:
        return {**empty, "warning": None}
    try:
        statuses = status_tokens(
            tcip_store.read(image_status_key(dataset_root), default={}))
        for bucket, entries in statuses.items():
            subject, _ = bucket_subject_date(bucket)
            outgoing_digest = changed.get(subject)
            if outgoing_digest is None:
                continue
            stamped = stamp_image_status_digests(
                dataset_root, bucket, sorted(entries), outgoing_digest, only_unstamped=True)
            finished_stamped = [name for name in stamped if is_finished_status(entries.get(name))]
            if finished_stamped:
                newly_stamped[subject] = newly_stamped.get(subject, 0) + len(finished_stamped)
        stamps_after = tcip_store.read(image_status_digest_key(dataset_root), default={})
        for bucket, entries in statuses.items():
            subject, _ = bucket_subject_date(bucket)
            if subject not in changed:
                continue
            new_digest = attribute_schema_digest(incoming, subject)
            if new_digest is None:
                continue
            finished_names = [name for name, status in entries.items() if is_finished_status(status)]
            stale = stale_stamped_names(
                bucket_digest_stamps(stamps_after, bucket), new_digest, finished_names)
            if stale:
                predating_vocabulary[subject] = predating_vocabulary.get(subject, 0) + len(stale)
    except (OSError, tcip_store.StoreError) as exc:
        return {**empty, "warning":
                f"could not stamp the outgoing attribute schema onto the confirmations under "
                f"{dataset_root} ({exc}); the unstamped ones will read as made under the new "
                f"schema, so re-review them before they train"}
    return {**empty, "warning": None}


def replace_registry(
    path: str | Path, registry: SubjectRegistry, *, expect: "Version | None", allow_removals: bool = False,
    allow_type_changes: bool = False,
) -> dict:
    """Write the registry, reading what it replaces and refusing a silent drop.

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
    set; landing it runs the same confirmation-digest sweep a value change does.

    ``expect`` is compare-and-set against the blob's actual version at write time
    (``tcip_store.VersionConflict`` on a mismatch, nothing written): pass the version the caller
    read, or ``Version.ABSENT`` for a caller asserting no registry exists yet. ``None`` skips the
    check.

    The confirmation-digest sweep (:func:`_sweep_schema_change`) runs only once the write has
    actually landed, against the registry this call read before writing. A crash between the put
    landing and the sweep completing leaves the affected confirmations unstamped under the registry
    that did land. Returns ``{"version": Version, "schema_change_sweep": dict}``.
    """
    import tcip_store

    if not registry.subjects:
        raise RegistryError("a subject registry write must declare at least one subject")

    key = _registry_key(path)
    versioned = tcip_store.read_blob_versioned(key, default=None)
    outgoing: SubjectRegistry | None = None
    decode_warning: str | None = None
    if versioned.value is not None:
        try:
            outgoing = registry_from_dict(_checked_registry_document(versioned.value, path=path))
        except ValueError as exc:
            if not allow_removals:
                raise RegistryError(
                    f"the stored registry at {path} does not decode ({exc}); pass allow_removals "
                    "to replace it anyway, since a repair drops whatever the stored bytes held"
                ) from exc
            decode_warning = (
                f"the outgoing registry at {path} does not read ({exc}), so confirmations made "
                "under it stay unstamped and will read as made under the new schema; re-review "
                "them before they train"
            )

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
                        "deliberate. Every recorded value of that attribute, on confirmed and "
                        "unconfirmed images alike, acquires the other type's meaning (a rank, or "
                        "an unordered label), and only the finished statuses under the subject "
                        "are quarantined by the sweep that follows"
                    )

    new_version = tcip_store.put_blob(
        key, tcip_store.RECORD_JSON.encode(registry_to_dict(registry)), expect=expect
    )

    sweep = _sweep_schema_change(Path(path).absolute().parent, outgoing, registry)
    if decode_warning and sweep["warning"] is None:
        sweep = {**sweep, "warning": decode_warning}
    return {"version": new_version, "schema_change_sweep": sweep}


def copy_registry(source: str | Path, destination: str | Path) -> None:
    """Place one dataset's registry beside another dataset's data, once: the stored document
    byte-for-byte, written create-only (``expect=Version.ABSENT``), refusing when the destination
    already holds a registry.
    """
    import tcip_store

    dest_key = _registry_key(destination)
    try:
        tcip_store.put_blob(
            dest_key,
            tcip_store.read_blob_versioned(_registry_key(source)).value,
            expect=tcip_store.Version.ABSENT,
        )
    except tcip_store.VersionConflict as exc:
        raise RegistryError(
            f"a subject registry already exists at {destination}; copy_registry places a first "
            "copy only and never replaces one"
        ) from exc


def assign_class_ids(registry: SubjectRegistry, subject: str, attribute: str | None = None) -> dict[str, int]:
    """The deterministic name→id map for one training scope, in the registry's declared order.

    - ``attribute`` given: one class per value of that attribute (``{value: 0..N-1}``), in the
      order the registry declares them, the rank order for an ordinal attribute.
    - ``attribute`` is ``None``: the subject is trained as a single detection class (``{subject:
      0}``), whether or not it carries attributes.

    Same registry + scope → identical map, every call. Raises :class:`RegistryError` for an absent
    subject/attribute.
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


def num_classes(registry: SubjectRegistry, subject: str, attribute: str | None = None) -> int:
    """Class count for a training scope, the size of :func:`assign_class_ids` (0 = background is the
    detector's own offset, applied by the loader, not counted here)."""
    return len(assign_class_ids(registry, subject, attribute))


def decode_class_ids(id_map: dict[str, int]) -> dict[int, str]:
    """Invert a recorded name→id map to id→name, for decoding a run's predictions."""
    return {cid: name for name, cid in id_map.items()}


def positive_value_problem(registry: SubjectRegistry, subject_name: str, value: str) -> str | None:
    """Why ``value`` cannot be ``subject_name``'s positive value in ``registry``, or ``None`` when
    some attribute of that subject lists it among its values. A subject with no attributes cannot
    carry a positive value.
    """
    subject = registry.subject(subject_name)
    if subject is None:
        known = [s.name for s in registry.subjects]
        return f"no subject {subject_name!r} in the registry (subjects: {known})"
    if not subject.attributes:
        return (
            f"subject {subject_name!r} has no attributes, so a bare detector decodes it as a "
            "single class keyed by the subject's own name; a single-class detector with no "
            "classification axis never assessed a trait's positive state, so it cannot carry one"
        )
    values = sorted({v for a in subject.attributes for v in a.values})
    if value not in values:
        return f"value {value!r} is not among subject {subject_name!r}'s attributes' values {values}"
    return None


def registry_for_dataset_root(dataset_root: str | Path) -> SubjectRegistry | None:
    """The registry at ``dataset_root``, or ``None`` when no ``subjects.json`` has been written there
    yet (a dataset with no registry is not corrupt, only unregistered so far)."""
    from tcip_mcp.dataset_layout import subjects_path

    try:
        return read_registry(subjects_path(dataset_root))
    except FileNotFoundError:
        return None


def distinct_dataset_root(pred_dirs: Sequence[str | Path]) -> Path | None:
    """The single dataset root every one of ``pred_dirs`` resolves under, or ``None`` when none
    do. Refuses (``RegistryError``) when the directories span more than one: no delivery this
    platform ships mixes datasets, so that can only be a caller error.
    """
    from tcip_mcp.dataset_layout import bucket_dataset_root

    roots: set[Path] = {r for d in pred_dirs if d and (r := bucket_dataset_root(d)) is not None}
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
