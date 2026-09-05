"""The dataset's class registry, subjects, their attributes, and the deterministic
name→id assignment a training run uses (and records, so predictions stay decodable).

The on-disk registry (``<dataset_root>/classes.json``) is self-describing and name-based::

    {
      "bush": {"description": "one bush crown", "defined_by": "...", "defined_at": "..."},
      "leaf": {"description": "one leaf", "defined_by": "...", "defined_at": "...",
               "attributes": {
                 "condition": {"type": "categorical", "values": ["healthy", "diseased"]}
               }}
    }

A *subject* is the object a label set is about (leaf, bush, efb). A subject with no
``attributes`` is simply detected. An *attribute* is an independent axis a subject's instances
carry, ``categorical`` (unordered) or ``ordinal`` (ordered; the ``values`` order is the rank).
Numeric is not an attribute type; measured/field values live in the plant-keyed field CSVs.

Labels reference these names, never integer ids. Integer class ids exist only inside a training
run: :func:`assign_class_ids` maps the names in a training scope to contiguous 0-indexed ids in
their *declared order*, deterministically and re-derivably, so the loader that builds targets,
the model that predicts, and the code that later decodes a prediction all agree by construction.
Ordering is the declared ``values`` order and never sorted: ordinal values carry rank, which
sorting would corrupt. The registry file's order could change between training and decode, so a run
*records* the map it used (``subprocess_worker.py::run`` resolves it via
``_resolve_run_id_map`` right after the dataset is built and stamps it onto ``config["data"]
["id_map"]``, which travels onto the checkpoint via the run's own config object, and, best-effort,
onto the durable experiment record via ``_patch_experiment_config_id_map``) and decode reads that
recorded map first (``inference_tools.resolve_decode_id_map``, the one resolution both writers
that persist predictions to disk, ``inference_tools.run_inference`` and the web GUI's inference worker,
call), falling back to a fresh derivation from the inference dataset's registry only for a
checkpoint with no recorded map, a bespoke ``dataset_source`` with no registry scope, or a run
trained from a pre-built COCO source whose id space isn't registry-derived.
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
    attribute that order is the rank (severity 0 < 1 < 2). The value invariant, a known ``type`` and a
    non-empty list of distinct names, is enforced here, so no ``Attribute``, however it is built, can
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
class ClassRegistry:
    """The whole ``classes.json``, the dataset's subjects, in declared order."""

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


def attribute_schema_digest(registry: ClassRegistry, subject: str) -> str | None:
    """Digest over ``subject``'s attribute vocabulary (name -> {type, declared-order values}) only.

    ``None`` if ``subject`` is not in the registry at all. Deliberately excludes ``description``/
    ``defined_by``/``defined_at``, free-text provenance whose editing (a typo fix, a citation
    update) says nothing about what an instance of the subject looks like, so hashing it would
    quarantine confirmations over changes that never affected them. This is attribute-*schema* drift
    detection, not full subject-redefinition detection: a ``description``-only redefinition of what
    the subject *is* is real but is a domain-expert judgment call, not something a hash can catch,
    an attribute-less subject (e.g. ``bush``) still gets a real, stable digest of ``{}``.
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
    """The stored registry a ``classes.json`` path names, addressed by the dataset root holding it."""
    from tcip_mcp.dataset_layout import class_registry_key

    return class_registry_key(Path(path).absolute().parent)


def _checked_registry_document(data: bytes, *, path: str | Path) -> dict:
    """The stored registry's decoded document, version-checked: the one implementation
    :func:`read_registry` and :func:`replace_registry` both call, so an undecodable registry and
    a version-refused one cannot read as two different facts depending which caller hit them.

    Raises :class:`RegistryError` for bytes that do not decode as JSON. Propagates
    :class:`tcip_store.SchemaVersionRefused`, uncaught: a newer writer's document is a policy
    fact, never the same fact as corruption, so a caller (:func:`replace_registry`'s
    ``allow_removals`` repair path, ``dataset_fingerprint._registry_term``) that tolerates or
    folds ``RegistryError`` away must not do the same to this.
    """
    import tcip_store

    try:
        document = tcip_store.RECORD_JSON.decode(data)
    except ValueError as exc:
        raise RegistryError(f"{path} does not decode as JSON: {exc}") from exc
    from tcip_mcp.dataset_layout import CLASS_REGISTRY_STORE

    tcip_store.check_schema_version(tcip_store.get_descriptor(CLASS_REGISTRY_STORE), document)
    return document


def read_registry(path: str | Path) -> ClassRegistry:
    """Read ``classes.json`` into a :class:`ClassRegistry`.

    Absence and corruption are different answers: no registry raises ``FileNotFoundError``, and
    a registry whose bytes are present but will not decode raises :class:`RegistryError`, the
    same refusal a structurally invalid registry raises. Reading an undecodable registry as an
    empty one would let every name-based label under it train as an unknown subject. A
    ``schema_version`` this reader does not accept propagates as
    :class:`tcip_store.SchemaVersionRefused`, uncaught, distinguishable from ``RegistryError``: a
    newer writer's registry must refuse whatever reads it (``dataset_fingerprint._registry_term``'s
    own fingerprint computation included), never fold into the "no registry" answer a genuinely
    unregistered dataset gets.
    """
    import tcip_store

    try:
        data = tcip_store.read_blob_versioned(_registry_key(path)).value
    except tcip_store.NotFound as exc:
        raise FileNotFoundError(f"no class registry at {path}") from exc
    document = _checked_registry_document(data, path=path)
    return registry_from_dict(document)


def write_registry(path: str | Path, registry: ClassRegistry) -> None:
    """Write a :class:`ClassRegistry` to ``classes.json``, unconditionally.

    Encoded through the canonical record codec object rather than a spelling of its own, so
    the ordered subject and attribute sequences land exactly as every other JSON document does.
    A plain overwrite, with no compare-and-set and no refusal for a dropped name: a fixture or a
    repair that means to place a registry outright uses this; the two doors a breeder or agent
    actually authors a registry through call :func:`replace_registry` instead.
    """
    import tcip_store

    tcip_store.put_blob(
        _registry_key(path), tcip_store.RECORD_JSON.encode(registry_to_dict(registry))
    )


def read_version(path: str | Path) -> "Version":
    """The class registry blob's current version token (``Version.ABSENT`` if it does not exist).

    Reads the version alone, never the content, so it never raises on bytes that will not
    decode: a caller that only wants a version to pass as :func:`replace_registry`'s ``expect``
    does not need the stored registry to be readable first.
    """
    import tcip_store

    return tcip_store.read_blob_versioned(_registry_key(path), default=None).version


def _dropped_names(outgoing: ClassRegistry, incoming: ClassRegistry) -> list[str]:
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
    dataset_root: Path, outgoing: ClassRegistry | None, incoming: ClassRegistry
) -> dict:
    """Stamp the outgoing attribute-schema digest onto every confirmation of an affected subject
    that carries no stamp yet, before ``incoming`` is what a later read sees.

    A confirmation and its digest stamp are two transactions, status first, so a stamp that could not
    be written never rejects the human's confirmation and unstamped confirmations legitimately exist
    (see ``tcip_web.routes.classes._stamp_digest``). An unstamped confirmation reads as valid
    (``tcip_mcp.pipelines.data.label_queries.confirmed_negative_names`` quarantines only a stamp that
    positively disagrees), which is right until the vocabulary changes underneath it: from then on
    nothing distinguishes it from a confirmation made under the new schema, and it trains against a
    definition its human never saw. ``outgoing`` (``None`` for a first-ever write, or a stored
    registry :func:`replace_registry` could not decode) is the last state that digest is
    recoverable from, so it is recorded here, and those confirmations then read as stale exactly
    like the stamped ones.

    Stamps the same set the confirmation-time writer stamps, every status in the subject's buckets
    rather than the negatives alone, since the quarantine question is asked of whatever a later
    reader takes from the store. Already-stamped confirmations, and subjects whose digest is
    unchanged, are left alone.

    Also counts, per affected subject, its confirmations (every status, not the negatives alone)
    whose stamped digest, once this write's own stamping above has landed, still disagrees with
    the subject's new digest: a confirmation stamped under an outgoing schema this write just
    recorded, and one already stamped under a still-earlier schema that a prior sweep never
    touched because it was not unstamped, read alike as made before the vocabulary in effect now.
    Computed with :func:`~tcip_mcp.pipelines.data.label_queries.stale_stamped_names`, the same
    comparison :func:`~tcip_mcp.pipelines.data.label_queries.confirmed_negative_records` quarantines
    a stale negative by, so the two readers cannot disagree about what "stale" means.

    Never blocks the registry write, which has already landed by the time this runs: an absent
    ``outgoing`` is a no-op, and a failing sweep returns a ``warning`` for the caller to surface.
    Returns ``{"newly_stamped": {subject: count}, "predating_vocabulary": {subject: count},
    "warning": str | None}``.
    """
    import tcip_store

    from tcip_mcp.dataset_layout import (
        bucket_digest_stamps, bucket_subject_date, image_status_digest_key, image_status_key,
        normalize_status_store, stamp_image_status_digests,
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
        statuses = normalize_status_store(
            tcip_store.read(image_status_key(dataset_root), default={}))
        for bucket, entries in statuses.items():
            subject, _ = bucket_subject_date(bucket)
            outgoing_digest = changed.get(subject)
            if outgoing_digest is None:
                continue
            stamped = stamp_image_status_digests(
                dataset_root, bucket, sorted(entries), outgoing_digest, only_unstamped=True)
            if stamped:
                newly_stamped[subject] = newly_stamped.get(subject, 0) + len(stamped)
        stamps_after = tcip_store.read(image_status_digest_key(dataset_root), default={})
        for bucket, entries in statuses.items():
            subject, _ = bucket_subject_date(bucket)
            if subject not in changed:
                continue
            new_digest = attribute_schema_digest(incoming, subject)
            if new_digest is None:
                continue
            stale = stale_stamped_names(
                bucket_digest_stamps(stamps_after, bucket), new_digest, entries)
            if stale:
                predating_vocabulary[subject] = predating_vocabulary.get(subject, 0) + len(stale)
    except (OSError, tcip_store.StoreError) as exc:
        return {**empty, "warning":
                f"could not stamp the outgoing attribute schema onto the confirmations under "
                f"{dataset_root} ({exc}); the unstamped ones will read as made under the new "
                f"schema, so re-review them before they train"}
    return {**empty, "warning": None}


def replace_registry(
    path: str | Path, registry: ClassRegistry, *, expect: "Version | None", allow_removals: bool = False,
) -> dict:
    """The one write both registry doors call: read what it replaces, refuse a silent drop.

    Refuses an empty ``registry`` outright, whether or not ``allow_removals`` is set: a registry
    write states subjects, never clears them. Reads the stored registry (absent reads as no
    prior registry, not a refusal) and refuses a write that drops a subject, an attribute, or an
    attribute value the stored one declares, unless ``allow_removals`` is true, since labels and
    confirmations may still reference the dropped name. Stored bytes present but undecodable are
    likewise refused unless ``allow_removals`` is true, since replacing them is how such a
    registry is repaired and a repair drops whatever the bytes held. A stored registry whose
    ``schema_version`` this reader does not accept is a different fact, never a repair
    candidate: the read routes through :func:`_checked_registry_document`, whose
    :class:`tcip_store.SchemaVersionRefused` propagates uncaught here regardless of
    ``allow_removals``, refusing the whole write rather than letting the repair path overwrite a
    newer writer's document.

    ``expect`` is compare-and-set against the blob's actual version at write time
    (``tcip_store.VersionConflict`` on a mismatch, nothing written): pass the version the caller
    read, or ``Version.ABSENT`` for a caller asserting no registry exists yet. ``None`` skips the
    check (an unconditional write), for a caller with no version to assert against.

    The confirmation-digest sweep (:func:`_sweep_schema_change`) runs only once the write has
    actually landed, against the registry this call read before writing, so a write that loses
    the compare-and-set leaves no stamp against a registry that never landed. The accepted
    residual is the reverse case: a crash between the put landing and the sweep completing leaves
    the affected confirmations unstamped under the registry that did land, which the training
    carry's quarantine then reads as predating the change rather than made under it. Returns
    ``{"version": Version, "schema_change_sweep": dict}``.
    """
    import tcip_store

    if not registry.subjects:
        raise RegistryError("a class registry write must declare at least one subject")

    key = _registry_key(path)
    versioned = tcip_store.read_blob_versioned(key, default=None)
    outgoing: ClassRegistry | None = None
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

    new_version = tcip_store.put_blob(
        key, tcip_store.RECORD_JSON.encode(registry_to_dict(registry)), expect=expect
    )

    sweep = _sweep_schema_change(Path(path).absolute().parent, outgoing, registry)
    if decode_warning and sweep["warning"] is None:
        sweep = {**sweep, "warning": decode_warning}
    return {"version": new_version, "schema_change_sweep": sweep}


def copy_registry(source: str | Path, destination: str | Path) -> None:
    """Place one dataset's registry beside another dataset's data, once.

    Carries the stored document across rather than a re-serialization of a parsed registry, so a
    materialized copy declares exactly what its source declares and a digest taken against either
    one agrees. Writes create-only (``expect=Version.ABSENT``) and refuses when the destination
    already holds a registry: a materialization or split run repeated over an existing root must
    not replace a registry silently, since a breeder or a later write may have changed it since.
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
            f"a class registry already exists at {destination}; copy_registry places a first "
            "copy only and never replaces one"
        ) from exc


def assign_class_ids(registry: ClassRegistry, subject: str, attribute: str | None = None) -> dict[str, int]:
    """The deterministic name→id map for one training scope, in the registry's *declared* order.

    - ``attribute`` given: one class per value of that attribute (``{value: 0..N-1}``), in the order
      the registry declares them, the rank order for an ordinal attribute.
    - ``attribute`` is ``None``: the subject is trained as a single detection class (``{subject: 0}``),
      whether or not it carries attributes, a plain detector that does not classify instances.

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
    """Class count for a training scope, the size of :func:`assign_class_ids` (0 = background is the
    detector's own offset, applied by the loader, not counted here)."""
    return len(assign_class_ids(registry, subject, attribute))


def decode_class_ids(id_map: dict[str, int]) -> dict[int, str]:
    """Invert a recorded name→id map to id→name, for decoding a run's predictions."""
    return {cid: name for name, cid in id_map.items()}


def positive_class_problem(registry: ClassRegistry, subject_name: str, class_name: str) -> str | None:
    """Why ``class_name`` cannot be ``subject_name``'s positive class in ``registry``, or ``None``
    when some attribute of that subject lists it among its values.

    Deliberately stricter than :func:`assign_class_ids`'s own id-map shape: a subject with no
    attributes decodes as a single class keyed by its own name, a bare detector with no
    classification axis at all, and a bare single-class detector never assessed a trait's positive
    state (the precondition ``count_by_class`` checks), so a subject with no attributes cannot
    carry a positive class here even though it decodes fine as a training scope.
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
    if class_name not in values:
        return f"class {class_name!r} is not among subject {subject_name!r}'s attributes' values {values}"
    return None


def registry_for_dataset_root(dataset_root: str | Path) -> ClassRegistry | None:
    """The registry at ``dataset_root``, or ``None`` when no ``classes.json`` has been written there
    yet (a dataset with no registry is not corrupt, only unregistered so far)."""
    from tcip_mcp.dataset_layout import classes_path

    try:
        return read_registry(classes_path(dataset_root))
    except FileNotFoundError:
        return None


def _distinct_dataset_root(pred_dirs: Sequence[str | Path]) -> Path | None:
    """The single dataset root every one of ``pred_dirs`` resolves under, or ``None`` when none
    do. Refuses (``RegistryError``) when the directories span more than one: no delivery this
    platform ships mixes datasets, so that can only be a caller error, never a case to silently
    resolve by picking one. The one computation ``registry_for_pred_dirs`` and
    ``dataset_root_for_pred_dirs`` both build on, so the two cannot disagree about it.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    roots: set[Path] = {r for d in pred_dirs if d and (r := dataset_root_of(d)) is not None}
    if len(roots) > 1:
        raise RegistryError(
            "a delivery's prediction directories resolve to more than one dataset root "
            f"({sorted(str(r) for r in roots)}); no delivery this platform ships spans more than "
            "one dataset, so this cannot be reconciled to a single registry"
        )
    return next(iter(roots), None)


def dataset_root_for_pred_dirs(pred_dirs: Sequence[str | Path]) -> Path:
    """The single dataset root every one of ``pred_dirs`` resolves under.

    Refuses (``RegistryError``) when the directories span more than one dataset root, the same
    check ``registry_for_pred_dirs`` makes, and refuses by name when none resolves to a dataset
    root at all: a plant-mapping delivery needs the dataset the buckets belong to, and buckets
    under no dataset root cannot supply one. This is a mapping-delivery-only requirement:
    ``run_inference`` itself writes and reads a bucket under no dataset root fine, only this
    dataset-bound delivery refuses it.
    """
    root = _distinct_dataset_root(pred_dirs)
    if root is None:
        raise RegistryError(
            f"none of the prediction directories {[str(d) for d in pred_dirs]} resolves to a "
            "dataset root, so there is no dataset for a plant-mapping delivery to attribute "
            "these predictions to; re-export under the dataset's own predictions tree "
            "(images/<date>/ sibling), or register this bucket's root as a dataset through "
            "register_dataset"
        )
    return root


def registry_for_pred_dirs(pred_dirs: Sequence[str | Path]) -> ClassRegistry | None:
    """The registry for the single dataset every one of ``pred_dirs`` resolves under.

    ``None`` when none of the directories resolves to a dataset root, or the one they do resolve
    to carries no registry yet. Refuses (``RegistryError``) when the directories span more than one
    dataset root: no delivery this platform ships mixes datasets, so that can only be a caller error,
    never a case to silently resolve by picking one.
    """
    root = _distinct_dataset_root(pred_dirs)
    if root is None:
        return None
    return registry_for_dataset_root(root)
