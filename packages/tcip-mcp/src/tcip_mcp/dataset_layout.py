"""Canonical dataset layout: an image's path, ``<dataset_root>/images/<capture>/<stem>.<imgext>``,
and the keys of its label document (capture and stem), a prediction bucket (its name) and the
bucket's documents (name and stem); and the one label save."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import tcip_store
from tcip_store import Key, decode_value

if TYPE_CHECKING:
    from tcip_annotation.flags import FlagRequest
    from tcip_annotation.json_io import LabelDocument

    from tcip_mcp.buckets import Bucket

#: Geometry kinds a task authors, kept as a selector, not a label-path segment.
TASKS = ("detect", "segment")
SUBJECTS_FILENAME = "subjects.json"

UNDATED_BUCKET = "undated"
"""The capture name of a dateless capture."""

LABEL_DOCUMENTS = "label_documents"
PREDICTION_DOCUMENTS = "prediction_documents"
PREDICTION_BUCKETS = "prediction_buckets"


def label_key(dataset_root: str | Path, capture: str, stem: str) -> Key:
    """The label document of the image ``stem`` of ``capture`` under ``dataset_root``, the triple
    :func:`parse_image_path` answers for an image path."""
    return Key(LABEL_DOCUMENTS, str(Path(dataset_root)), (capture, stem))


def prediction_key(dataset_root: str | Path, bucket: str, stem: str) -> Key:
    """The prediction document of the image ``stem`` in the bucket named ``bucket``."""
    return Key(PREDICTION_DOCUMENTS, str(Path(dataset_root)), (bucket, stem))


def bucket_key(dataset_root: str | Path, bucket: str) -> Key:
    """The record of the prediction bucket named ``bucket`` under ``dataset_root``."""
    return Key(PREDICTION_BUCKETS, str(Path(dataset_root)), (bucket,))


def is_bucket_name(name: str) -> bool:
    """Whether ``name`` is legal as a capture bucket directory name under ``images/``: a single
    safe path segment (see
    ``workspace.is_valid_name``) that does not start with a dot, so a hidden directory (an
    editor's swap file, platform cruft) is never mistaken for one."""
    from tcip_mcp.workspace import is_valid_name

    return is_valid_name(name) and not name.startswith(".")


def parse_capture_dir(images_dir: str | Path) -> tuple[Path, str]:
    """``(dataset_root, capture)`` of a capture directory ``<root>/images/<capture>``; any other
    directory, the image tree ``<root>/images`` itself included, raises ``ValueError`` naming it."""
    directory = Path(images_dir)
    if directory.parent.name != "images":
        raise ValueError(
            f"{str(directory)!r} is not a capture: a capture's images sit at "
            f"<root>/images/<capture>/<stem>, a dateless capture's under {UNDATED_BUCKET!r}.")
    return directory.parent.parent, directory.name


def parse_image_path(image_path: str | Path) -> tuple[Path, str, str]:
    """``(dataset_root, capture, stem)`` of an image at ``<root>/images/<capture>/<stem>``
    (:func:`parse_capture_dir` of its directory)."""
    img = Path(image_path)
    return (*parse_capture_dir(img.parent), img.stem)


def capture_of(source: str | Path) -> tuple[Optional[str], Optional[str]]:
    """``(dataset_id, capture)`` of an image source: the identity record's id of the dataset root
    whose image tree holds it (``None`` when the root records none) and its capture; both
    ``None`` for a source that is no image of a capture."""
    try:
        root, capture, _stem = parse_image_path(source)
    except ValueError:
        return None, None
    identity = read_dataset_identity(root)
    return (identity["id"] if identity is not None else None), capture


def image_root(dataset_root: str | Path) -> Path:
    """``<dataset_root>/images/``: the whole image tree, every capture under it."""
    return Path(dataset_root, "images")


def image_dir(dataset_root: str | Path, capture: str) -> Path:
    """``<dataset_root>/images/<capture>/``: where a capture's image bytes live; ``ValueError``
    for a capture that is not a single safe path segment."""
    from tcip_mcp.workspace import is_valid_name

    if not is_valid_name(capture):
        raise ValueError(
            f"a capture must be a single safe path segment (no separators/'..'), got {capture!r}")
    return image_root(dataset_root) / capture


def image_filename(stem: str, ext: str) -> str:
    """The file name one capture's bytes are stored under (``ext`` includes the leading dot)."""
    return f"{stem}{ext}"


def image_path(dataset_root: str | Path, capture: str, stem: str, ext: str) -> Path:
    """Canonical write path for an image (``ext`` includes the leading dot)."""
    return image_dir(dataset_root, capture) / image_filename(stem, ext)


def list_dates(dataset_root: str | Path) -> list[str]:
    """Sorted capture names under ``images/``, each a directory whose name
    :func:`is_bucket_name` admits."""
    imgs = image_root(dataset_root)
    if not imgs.is_dir():
        return []
    return sorted(p.name for p in imgs.iterdir() if p.is_dir() and is_bucket_name(p.name))


def dataset_root_of(path: str | Path) -> Optional[Path]:
    """The ``<dataset_root>`` a path under ``<dataset_root>/images/`` lives under, or ``None`` if
    it lives under none.

    Anchors on the *last* ``images`` segment in the path, so a dataset physically nested under an
    ancestor named ``images`` still resolves to the real root rather than the ancestor. A bare
    segment with nothing above it is not inside a dataset -> ``None``.
    """
    parts = Path(path).parts
    idxs = [k for k, p in enumerate(parts) if p == "images"]
    if not idxs:
        return None
    i = max(idxs)
    return Path(*parts[:i]) if i > 0 else None


def subjects_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/subjects.json``: the one nested registry that decodes the dataset's
    labels, written through ``tcip_store.encode_record`` in declared order."""
    return Path(dataset_root, SUBJECTS_FILENAME)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """``<dataset_root>/dataset.json``: the dataset's identity ({crop, id, fingerprint}), written
    compare-and-set through ``tcip_store.encode_record``.

    The stored fingerprint is a cache; recompute-on-read
    (``dataset_fingerprint.dataset_fingerprint``) is authority.
    """
    return Path(dataset_root, "dataset.json")


def decode_dataset_identity_document(data: bytes, *, dataset_root: str | Path) -> dict:
    """A dataset identity document's bytes, decoded and shape-checked.

    Raises ``ValueError`` for bytes that do not decode, or that decode to something other than a
    dict; a dict lacking ``id``, ``crop`` or ``fingerprint`` raises ``KeyError``.
    """
    try:
        identity = decode_value(data)
    except ValueError as exc:
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            f"identity ({exc}); re-register with register_dataset") from exc
    if not isinstance(identity, dict):
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            "identity; re-register with register_dataset")
    _id, _crop, _fingerprint = identity["id"], identity["crop"], identity["fingerprint"]
    return identity


def read_dataset_identity(dataset_root: str | Path) -> dict | None:
    """The dataset's identity record (``{crop, id, fingerprint}``) decoded through
    :func:`decode_dataset_identity_document`, or ``None`` for a dataset never registered. A record
    that does not decode raises ``ValueError``."""
    stored = tcip_store.read_blob_versioned(dataset_identity_path(dataset_root), default=None)
    if stored.value is None:
        return None
    return decode_dataset_identity_document(stored.value, dataset_root=dataset_root)


def require_dataset_identity(dataset_root: str | Path) -> dict:
    """:func:`read_dataset_identity`, refusing with ``ValueError`` naming ``register_dataset`` a
    dataset never registered."""
    identity = read_dataset_identity(dataset_root)
    if identity is None:
        raise ValueError(
            f"{dataset_root} carries no dataset identity record "
            f"({dataset_identity_path(dataset_root)} absent); register it first with "
            "register_dataset")
    return identity


@dataclass(frozen=True)
class Gestures:
    """What one save decides beyond the annotations it writes: the proposals of the bucket named
    ``bucket``, under the image's own dataset root, it accepts and rejects, each by its index in
    that bucket's document for the image;
    each subject of ``complete`` marked complete over ``rect`` (pixel ``[x, y, w, h]``, the whole
    image when ``None``) or, mapped to ``False``, its marks withdrawn; whether proposals were
    hidden while the person annotated; each flag it raises
    (:class:`~tcip_annotation.flags.FlagRequest`); and each open flag it resolves, by id, with
    the reply given."""

    bucket: Optional[str] = None
    accept: frozenset[int] = frozenset()
    reject: frozenset[int] = frozenset()
    complete: Mapping[str, bool] = field(default_factory=dict)
    rect: Optional[tuple[float, float, float, float]] = None
    proposals_hidden: bool = False
    flag: tuple["FlagRequest", ...] = ()
    resolve: Mapping[str, str] = field(default_factory=dict)


def label_key_of(image_path: str | Path) -> Key:
    """The label document of the image at ``image_path`` (:func:`parse_image_path`, then
    :func:`label_key`); an image under no dataset image tree refuses (``ValueError``)."""
    return label_key(*parse_image_path(image_path))


def image_proposals(bucket: str, key: Key) -> tuple["Bucket", list]:
    """The published bucket named ``bucket`` under the root of the label document ``key`` and its
    proposals for that image, in document order; a bucket that names no document for the image
    refuses (``ValueError``)."""
    from tcip_annotation.json_io import read_predictions

    from tcip_mcp.buckets import read_bucket

    found = read_bucket(key.root, bucket)
    document = found.document_key(key.parts[-1])
    if document is None:
        raise ValueError(f"bucket {bucket!r} holds no proposals for {key.parts[-1]}")
    return found, read_predictions(document)


def verdict_key_of(key: Key, bucket: str) -> Key:
    """The verdict shard of the proposals the bucket named ``bucket`` holds for the image whose
    label document ``key`` names."""
    from tcip_annotation.verdicts import verdict_key

    return verdict_key(key.root, bucket, key.parts[-1])


def proposal_pairs(project: str | Path | None, bucket: "Bucket", annotations: list,
                   proposals: list) -> dict[int, int]:
    """Which annotation each of ``bucket``'s proposals pairs with, proposal index to annotation
    index, by the one matcher (:func:`~tcip_annotation.matching.pair_proposals`) under the
    localization criterion the assessment ``bucket`` was published under measured its count by;
    a bucket published under none, or under one that measured no count, pairs under the
    platform's comparability convention. Refuses (``ValueError``) a bucket's assessment with no
    ``project`` to read it from."""
    from tcip_annotation.matching import pair_proposals

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    criterion = None
    if bucket.assessment_id is not None:
        if project is None:
            raise ValueError(f"bucket {bucket.name!r} was published under assessment "
                             f"{bucket.assessment_id}, which only its project holds; open it")
        count = read_assessment(project, bucket.assessment_id).criterion.get("count")
        criterion = count["localization"] if count else None
    m = pair_proposals(annotations, proposals, criterion or resolve_match_criterion(None, []))
    return {p: g for g, p in m.pairs}


def save_label_document(
    project: str | Path | None, key: Key, payloads: Iterable[dict], *,
    width: int, height: int, author: str, actor: Optional[str],
    expect: Optional[tcip_store.Version] = None, gestures: Gestures = Gestures(),
) -> "tuple[tcip_store.Version, LabelDocument]":
    """Write the label document ``key`` names (:func:`label_key_of` an image), the verdicts its
    ``gestures`` decide and the save's one audit line by ``actor``, in one commit under the
    document's dataset root. Returns the document's new version and the document as written.

    The document holds every annotation parsed from ``payloads``, provenance stamped
    (:func:`~tcip_annotation.json_io.stamped`): a record unchanged since it was stored keeps its
    own, any other is ``author``'s at the save's time. Each accepted proposal of ``gestures``
    pairing no annotation (:func:`proposal_pairs`) joins it as ground truth, its geometry and
    subject only, every attribute unassessed, authored by its producer and accepted by
    ``author``; one that pairs confirms that annotation and adds nothing.
    The document's completion marks still live over the new annotations stay, beside the marks
    ``gestures`` makes; a subject mapped to ``False`` loses its marks. Each accepted and rejected
    proposal appends one entry to the image's verdict shard under that bucket. The image's flags
    record gains each flag ``gestures`` raises, by ``author``, and marks each it resolves; an
    open flag on an annotation the save removes is resolved as removed
    (:func:`~tcip_annotation.flags.resolved_by_removal`).

    Raises with nothing written: ``ValueError`` for a payload that does not parse, a proposal
    index the bucket's document does not hold or that is both accepted and rejected, a flag with
    no comment or on a proposal its bucket does not hold, or a resolved flag that is not open;
    ``UnreadableLabelDocumentError`` for a stored document that does not decode;
    ``VersionConflictError`` when ``expect`` is not the version the commit reads.
    """
    from tcip_annotation.flags import (
        flag_key, flags_of, flags_record, raised, resolved_by_removal,
    )
    from tcip_annotation.json_io import (
        CompletionMark, LabelDocument, annotation_from_payload, document_at, document_payload,
        stamped, subject_digest,
    )
    from tcip_annotation.verdicts import Verdict, VerdictAction, record_verdicts

    from tcip_mcp.audit import audit_entry, audit_log_key, now_iso

    audit = audit_log_key(key.root)
    capture, stem = key.parts
    now = now_iso()
    contents = []
    for i, payload in enumerate(payloads):
        try:
            contents.append(annotation_from_payload(payload))
        except ValueError as exc:
            raise ValueError(f"annotation {i} {exc}") from exc
    decides = bool(gestures.accept or gestures.reject)
    if decides:
        if gestures.bucket is None:
            raise ValueError("accepting or rejecting a proposal names the bucket it came from")
        both = gestures.accept & gestures.reject
        if both:
            raise ValueError(f"proposal(s) {sorted(both)} are both accepted and rejected; decide "
                             "each once")
        bucket, proposals = image_proposals(gestures.bucket, key)
        verdict = verdict_key_of(key, gestures.bucket)
    new_flags = [raised(request, by=author, at=now) for request in gestures.flag]
    for flag in new_flags:
        if flag.proposal is not None:
            named, index = flag.proposal
            held = len(image_proposals(named, key)[1])
            if index >= held:
                raise ValueError(f"bucket {named!r} holds {held} proposals for {stem}, not one "
                                 f"at index {index} to flag")
    flags_key = flag_key(key)
    with tcip_store.transaction(key, audit, flags_key, *([verdict] if decides else [])) as txn:
        read = txn.read_versioned(key, default=None)
        stored = document_at(key, read)
        if expect is not None and expect != read.version:
            raise tcip_store.VersionConflictError(key, expect, read.version)
        annotations = stamped(contents, stored.annotations, author=author, now=now)
        held_flags = flags_of(txn.read(flags_key, default=None))
        open_ids = {flag.id for flag in held_flags if flag.open}
        unknown = sorted(set(gestures.resolve) - open_ids)
        if unknown:
            raise ValueError(f"flag(s) {unknown} are not open on {stem}; resolve an open flag")
        flags = [flag.resolved(by=author, at=now, reply=gestures.resolve[flag.id])
                 if flag.id in gestures.resolve else flag for flag in held_flags] + new_flags
        verdicts: list[Verdict] = []
        if decides:
            paired = proposal_pairs(project, bucket, annotations, proposals)
            decided: tuple[tuple[VerdictAction, frozenset[int]], ...] = (
                ("accepted", gestures.accept), ("rejected", gestures.reject))
            for action, indices in decided:
                for i in sorted(indices):
                    if not 0 <= i < len(proposals):
                        raise ValueError(f"bucket {bucket.name!r} holds {len(proposals)} "
                                         f"proposals for {stem}, not one at index {i}")
                    if action == "accepted" and i not in paired:
                        annotations.append(replace(proposals[i], score=None, attributes={},
                                                   accepted_by=author, accepted_at=now))
                    verdicts.append(Verdict(proposal=i, action=action, by=author, at=now))
            record_verdicts(txn, verdict, verdicts)
        marks = {s: held for s, held in stored.marks.items() if gestures.complete.get(s, True)}
        for subject in (s for s, made in gestures.complete.items() if made):
            marks.setdefault(subject, []).append(CompletionMark(
                rect=gestures.rect or (0, 0, width, height), by=author, at=now,
                digest=subject_digest(annotations, subject),
                proposals_hidden=gestures.proposals_hidden))
        version = txn.write(key, document_payload(annotations, width, height, keep_empty=True,
                                                  marks=marks))
        flags = resolved_by_removal(flags, stored.annotations, annotations, by=author, at=now)
        if flags != held_flags:
            txn.write(flags_key, flags_record(flags))
        txn.append(audit, audit_entry("save_label_document", {
            "capture": capture, "stem": stem, "n_annotations": len(annotations),
            "version": version.token, "accepted": sorted(gestures.accept),
            "rejected": sorted(gestures.reject), "complete": dict(gestures.complete),
            "flagged": [flag.id for flag in new_flags],
            "resolved": sorted(flag.id for flag in flags
                               if not flag.open and flag.id in open_ids),
        }, actor, "ok"))
    return version, LabelDocument(annotations, width, height, marks)


def list_subjects(dataset_root: str | Path) -> list[str]:
    """The dataset's subjects, in the registry's declared order. ``[]`` when there is no registry.

    A registry that is present but unreadable raises rather than reading as no subjects.
    """
    from tcip_mcp import subject_registry

    if not subjects_path(dataset_root).is_file():
        return []
    return [s.name for s in subject_registry.read_registry(dataset_root).subjects]


def capture_label_keys(dataset_root: str | Path, capture: str) -> list[Key]:
    """Every label document of ``capture`` under ``dataset_root``, in stem order."""
    return tcip_store.keys(LABEL_DOCUMENTS, str(dataset_root), (capture,))


def capture_subjects(dataset_root: str | Path, capture: str) -> tuple[list[str], list[str]]:
    """The distinct subjects the label documents of ``capture`` hold, sorted, and the stem of
    every one of them that will not read."""
    from tcip_annotation.json_io import UnreadableLabelDocumentError, read_label_document

    found: set[str] = set()
    unreadable: list[str] = []
    for key in capture_label_keys(dataset_root, capture):
        try:
            found.update(a.subject for a in read_label_document(key).annotations)
        except UnreadableLabelDocumentError:
            unreadable.append(key.parts[-1])
    return sorted(found), unreadable
