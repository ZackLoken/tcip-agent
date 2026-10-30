"""Canonical dataset layout: an image's path, its documents' keys, and its label save.

An image's path is ``<dataset_root>/images/<capture>/<stem>.<imgext>``; the keys are its label
document's (capture and stem), a prediction bucket's (its name) and the bucket's documents' (name
and stem).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

import tcip_store
from tcip_annotation.flags import FlagRequest, ProposalTarget
from tcip_annotation.verdicts import VerdictAction
from tcip_store import Key, decode_value

if TYPE_CHECKING:
    from tcip_annotation.json_io import LabelDocument
    from tcip_annotation.state import Annotation
    from tcip_mcp.buckets import Bucket

#: Geometry kinds a task authors, kept as a selector, not a label-path segment.
TASKS = ("detect", "segment")
SUBJECTS_FILENAME = "subjects.json"

UNDATED_BUCKET = "undated"
"""The capture name of a dateless capture."""


def capture_date(capture: str) -> date | None:
    """Return the calendar date the capture name ``capture`` is, or ``None``.

    A dated capture is named by its date's ISO ``YYYY-MM-DD`` spelling; any other name (the
    undated bucket, a folder that is no calendar date such as ``2026-13-01``) is dated by nothing.
    """
    try:
        day = date.fromisoformat(capture)
    except ValueError:
        return None
    return day if day.isoformat() == capture else None


LABEL_DOCUMENTS = "label_documents"
PREDICTION_DOCUMENTS = "prediction_documents"
PREDICTION_BUCKETS = "prediction_buckets"


def label_key(dataset_root: str | Path, capture: str, stem: str) -> Key:
    """Return the key of the label document of image ``stem`` of ``capture`` under ``dataset_root``.

    The triple is the one :func:`parse_image_path` answers for an image path.
    """
    return Key(LABEL_DOCUMENTS, str(Path(dataset_root)), (capture, stem))


def prediction_key(dataset_root: str | Path, bucket: str, stem: str) -> Key:
    """Return the key of the prediction document of image ``stem`` in the bucket ``bucket``."""
    return Key(PREDICTION_DOCUMENTS, str(Path(dataset_root)), (bucket, stem))


def bucket_key(dataset_root: str | Path, bucket: str) -> Key:
    """Return the key of the record of the prediction bucket ``bucket`` under ``dataset_root``."""
    return Key(PREDICTION_BUCKETS, str(Path(dataset_root)), (bucket,))


def is_bucket_name(name: str) -> bool:
    """Return whether ``name`` is legal as a capture bucket directory name under ``images/``.

    It is legal as a :data:`~tcip_mcp.workspace.PathSegment` that does not start with a dot, so a
    hidden directory (an editor's swap file, platform cruft) is never mistaken for one.
    """
    from pydantic import ValidationError

    from tcip_mcp.workspace import path_segment

    try:
        path_segment(name)
    except ValidationError:
        return False
    return not name.startswith(".")


def image_tree_root(images_root: str | Path) -> Path | None:
    """Return the dataset root whose image tree (:func:`image_root`) is ``images_root``.

    ``None`` for a directory that is no image tree.
    """
    tree = Path(images_root)
    return tree.parent if tree.name == image_root(tree.parent).name else None


def carries_root_state(directory: str | Path) -> bool:
    """Return whether ``directory`` carries dataset or project state of its own.

    That is its own ``.tcip/`` store, a subject registry (:func:`subjects_path`) or an identity
    record (:func:`dataset_identity_path`).
    """
    path = Path(directory)
    return (path / ".tcip").is_dir() or any(
        f(path).is_file() for f in (subjects_path, dataset_identity_path)
    )


def _capture(directory: str | Path) -> tuple[Path, str] | None:
    """Return ``(dataset_root, capture)`` of a capture ``<root>/images/<capture>``, else ``None``.

    A directory named ``images`` inside one also named ``images`` is by position either a capture
    or the image tree of a root named ``images``: it is the one whose root carries its own state
    (:func:`carries_root_state`), and neither or both refuses (``ValueError``) naming both.
    """
    path = Path(directory)
    root = image_tree_root(path.parent)
    if root is None:
        return None
    if image_tree_root(path) is not None:
        as_capture, as_tree = (
            carries_root_state(r) for r in (root, path.parent)
        )
        if as_capture == as_tree:
            raise ValueError(
                f"{path} is by position the capture {path.name!r} of the dataset {root} or the "
                f"image tree of the dataset {path.parent}, and "
                f"{'both carry' if as_capture else 'neither carries'} state of its own; "
                "register the one it belongs to with register_dataset."
            )
        if as_tree:
            return None
    return root, path.name


def parse_capture_dir(images_dir: str | Path) -> tuple[Path, str]:
    """Return ``(dataset_root, capture)`` of a capture directory ``<root>/images/<capture>``.

    Any other directory raises ``ValueError`` naming it, the image tree ``<root>/images`` itself
    included (:func:`_capture`).
    """
    found = _capture(images_dir)
    if found is None:
        raise ValueError(
            f"{str(Path(images_dir))!r} is not a capture: a capture's images sit at "
            f"<root>/images/<capture>/<stem>, a dateless capture's under {UNDATED_BUCKET!r}."
        )
    return found


def parse_image_path(image_path: str | Path) -> tuple[Path, str, str]:
    """Return ``(dataset_root, capture, stem)`` of an image at ``<root>/images/<capture>/<stem>``.

    The first two are :func:`parse_capture_dir` of its directory.
    """
    img = Path(image_path)
    return (*parse_capture_dir(img.parent), img.stem)


def image_capture(source: str | Path) -> tuple[Path | None, str | None]:
    """Return ``(dataset_root, capture)`` of an image source, :func:`parse_image_path`'s first two.

    Both are ``None`` for a source that is no image of a capture (:func:`_capture` of its
    directory, whose refusal of an ambiguous one propagates).
    """
    return _capture(Path(source).parent) or (None, None)


def dataset_id_of(root: Path | None) -> str | None:
    """Return the id ``root``'s identity record states, ``None`` for no root or no record."""
    identity = None if root is None else read_dataset_identity(root)
    return None if identity is None else identity.id


def image_root(dataset_root: str | Path) -> Path:
    """Return the whole image-tree path ``<dataset_root>/images/``, every capture under it."""
    return Path(dataset_root, "images")


def dataset_root_of(directory: str | Path) -> Path | None:
    """Return the dataset root of the capture directory or the image tree ``directory`` is.

    Both are read by position, a directory position reads both ways taking the reading whose
    root carries its own state (:func:`_capture`, which refuses one it cannot tell). ``None``
    for any other directory.
    """
    found = _capture(directory)
    return image_tree_root(directory) if found is None else found[0]


def image_dir(dataset_root: str | Path, capture: str) -> Path:
    """Return ``<dataset_root>/images/<capture>/``, where a capture's image bytes live.

    Refuses a capture that is no :data:`~tcip_mcp.workspace.PathSegment`.
    """
    from tcip_mcp.workspace import path_segment

    return image_root(dataset_root) / path_segment(capture)


def image_filename(stem: str, ext: str) -> str:
    """Return the file name one capture's bytes are stored under; ``ext`` has the leading dot."""
    return f"{stem}{ext}"


def image_path(
    dataset_root: str | Path, capture: str, stem: str, ext: str
) -> Path:
    """Return the canonical write path for an image (``ext`` includes the leading dot)."""
    return image_dir(dataset_root, capture) / image_filename(stem, ext)


def capture_order(captures: Iterable[str]) -> list[str]:
    """Return ``captures`` in the order a dataset lists its captures, by name."""
    return sorted(captures)


def list_dates(dataset_root: str | Path) -> list[str]:
    """Return the capture names under ``images/``, in :func:`capture_order`.

    Each is a directory whose name :func:`is_bucket_name` admits.
    """
    imgs = image_root(dataset_root)
    if not imgs.is_dir():
        return []
    return capture_order(
        p.name for p in imgs.iterdir() if p.is_dir() and is_bucket_name(p.name)
    )


def subjects_path(dataset_root: str | Path) -> Path:
    """Return ``<dataset_root>/subjects.json``, the registry that decodes the dataset's labels."""
    return Path(dataset_root, SUBJECTS_FILENAME)


def dataset_identity_path(dataset_root: str | Path) -> Path:
    """Return ``<dataset_root>/dataset.json``, the dataset's identity ({crop, id, fingerprint}).

    The stored fingerprint is a cache; recompute-on-read
    (``dataset_fingerprint.dataset_fingerprint``) is authority.
    """
    return Path(dataset_root, "dataset.json")


class DatasetRecord(BaseModel):
    """A dataset's identity document, ``dataset.json``: its crop, its minted id, its fingerprint.

    ``fingerprint`` is ``None`` for a dataset that yielded none when it was registered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    crop: str
    id: str
    fingerprint: str | None


def decode_dataset_identity_document(
    data: bytes, *, dataset_root: str | Path
) -> DatasetRecord:
    """Decode a dataset identity document's bytes as its :class:`DatasetRecord`.

    Bytes that do not decode, or decode to no :class:`DatasetRecord`, raise ``ValueError`` naming
    the document and ``register_dataset``.
    """
    try:
        return DatasetRecord.model_validate(decode_value(data))
    except ValueError as exc:
        raise ValueError(
            f"{dataset_identity_path(dataset_root)} exists but does not decode as a dataset "
            f"identity ({exc}); re-register with register_dataset"
        ) from exc


def read_dataset_identity(dataset_root: str | Path) -> DatasetRecord | None:
    """Return the dataset's identity record, or ``None``.

    The record is decoded through :func:`decode_dataset_identity_document`; ``None`` is a dataset
    never registered. A record that does not decode raises ``ValueError``.
    """
    stored = tcip_store.read_blob_versioned(
        dataset_identity_path(dataset_root), default=None
    )
    if stored.value is None:
        return None
    return decode_dataset_identity_document(
        stored.value, dataset_root=dataset_root
    )


def require_dataset_identity(dataset_root: str | Path) -> DatasetRecord:
    """Return :func:`read_dataset_identity`'s record, refusing a dataset never registered.

    The refusal is ``ValueError`` naming ``register_dataset``.
    """
    identity = read_dataset_identity(dataset_root)
    if identity is None:
        raise ValueError(
            f"{dataset_root} carries no dataset identity record "
            f"({dataset_identity_path(dataset_root)} absent); register it first with "
            "register_dataset"
        )
    return identity


@dataclass(frozen=True)
class Decisions:
    """The proposals of one bucket a save decides.

    ``bucket`` names the bucket under the image's own dataset root, and ``verdicts`` each decided
    proposal's verdict by its index in that bucket's document for the image.
    """

    __pydantic_config__ = ConfigDict(extra="forbid")

    bucket: str
    verdicts: Mapping[int, VerdictAction]


@dataclass(frozen=True)
class Gestures:
    """What one save decides beyond the annotations it writes.

    It holds the proposals it ``decide``s; the
    annotations it confirms as the person's own call, each by its position among the annotations
    the save writes (``confirm``); each subject of ``complete`` marked complete over ``rect``
    (pixel ``[x, y, w, h]``, the whole image when ``None``) or, mapped to ``False``, its marks
    withdrawn; whether proposals were hidden while the person annotated; each flag it raises
    (:class:`~tcip_annotation.flags.FlagRequest`); and each open flag it resolves, by id, with
    the reply given.
    """

    __pydantic_config__ = ConfigDict(extra="forbid")

    decide: Decisions | None = None
    confirm: frozenset[int] = frozenset()
    complete: Mapping[str, bool] = field(default_factory=dict)
    rect: tuple[float, float, float, float] | None = None
    proposals_hidden: bool = False
    flag: tuple[FlagRequest, ...] = ()
    resolve: Mapping[str, str] = field(default_factory=dict)


def label_key_of(image_path: str | Path) -> Key:
    """Return the key of the label document of the image at ``image_path``.

    The key is :func:`parse_image_path`, then :func:`label_key`; an image under no dataset image
    tree refuses (``ValueError``).
    """
    return label_key(*parse_image_path(image_path))


def image_proposals(bucket: str, key: Key) -> tuple[Bucket, list]:
    """Return the published bucket ``bucket`` and its proposals for the image ``key`` names.

    The bucket is the one under the root of the label document ``key``; the proposals are in
    document order, empty for an image the bucket holds no document of.
    """
    from tcip_annotation.json_io import read_predictions
    from tcip_mcp.buckets import read_bucket

    found = read_bucket(key.root, bucket)
    document = found.document_key(key.parts[-1])
    if document is None:
        return found, []
    return found, read_predictions(document).annotations


def verdict_key_of(key: Key, bucket: str) -> Key:
    """Return the key of the verdict shard of ``bucket``'s proposals for the image ``key`` names.

    ``key`` is the image's label document key.
    """
    from tcip_annotation.verdicts import verdict_key

    return verdict_key(key.root, bucket, key.parts[-1])


def proposal_pairs(
    project: str | Path | None,
    bucket: Bucket,
    annotations: list,
    proposals: list,
) -> dict[int, int]:
    """Return which annotation each of ``bucket``'s proposals pairs with, by index.

    The mapping is proposal index to annotation index, by
    :func:`~tcip_annotation.matching.pair_proposals` under the localization criterion the
    assessment ``bucket`` was published under measured its count by; a bucket published under
    none, or under one that measured no count, pairs under the platform's comparability
    convention (``schemas.COMPARABILITY_IOU``). Refuses (``ValueError``) a bucket's assessment
    with no ``project`` to read it from.
    """
    from tcip_annotation.matching import pair_proposals
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.pipelines.schemas import COMPARABILITY_IOU
    from tcip_mcp.pipelines.training.evaluation import comparability_criterion

    criterion = None
    if bucket.assessment_id is not None:
        if project is None:
            raise ValueError(
                f"bucket {bucket.name!r} was published under assessment "
                f"{bucket.assessment_id}, which only its project holds; open it"
            )
        assessed = read_assessment(project, bucket.assessment_id)
        match assessed.criterion.get("count"):
            case None:
                pass
            case {"localization": dict() as localization}:
                criterion = localization
            case count:
                raise ValueError(
                    f"assessment {bucket.assessment_id} records a count criterion "
                    f"{count!r} that states no localization"
                )
    m = pair_proposals(
        annotations,
        proposals,
        criterion or comparability_criterion(COMPARABILITY_IOU, by_mask=False),
    )
    return {p: g for g, p in m.pairs}


def signed_off(annotation: Annotation, by: str, at: str) -> Annotation:
    """Return ``annotation`` taken responsibility for by the person ``by`` at ``at``.

    Its ``accepted_by`` and ``accepted_at`` are set, everything else as it was.
    """
    return replace(annotation, accepted_by=by, accepted_at=at)


def save_label_document(
    project: str | Path | None,
    key: Key,
    payloads: Iterable[dict],
    *,
    width: int,
    height: int,
    author: str,
    actor: str | None,
    expect: tcip_store.Version | None = None,
    gestures: Gestures = Gestures(),
) -> tuple[tcip_store.Version, LabelDocument, dict[int, int]]:
    """Write a label document, the verdicts its gestures decide and one audit line, in one commit.

    The document is the one ``key`` names (:func:`label_key_of` an image); the verdicts are the
    ones ``gestures`` decide and the audit line is by ``actor``, all under the document's dataset
    root. Returns the document's new version, the document decoded from the very record the
    commit wrote (rounded geometry, only the completion marks still live) and, for each accepted
    proposal, the index in that document of the annotation it resolved to.

    The document holds every annotation parsed from ``payloads``, provenance stamped
    (:func:`~tcip_annotation.json_io.stamped`): a record unchanged since it was stored keeps its
    own, any other is ``author``'s at the save's time. Each annotation ``gestures`` confirms
    (:attr:`Gestures.confirm`) is signed off by ``author`` at the save's time
    (:func:`signed_off`). Each accepted proposal of ``gestures`` pairing no annotation
    (:func:`proposal_pairs`) joins it as ground truth, its geometry and subject only, every
    attribute unassessed, authored by its producer and accepted by ``author``; one that pairs
    signs that annotation off the same way and adds nothing. The document's completion marks
    still live over the new annotations stay, beside the marks ``gestures`` makes; a subject
    mapped to ``False`` loses its marks. Each accepted and rejected proposal appends one entry to
    the image's verdict shard under that bucket. The image's flags record gains each flag
    ``gestures`` raises, by ``author``, and marks each it resolves; an open flag on an annotation
    the save removes is resolved as removed by ``author``
    (:func:`~tcip_annotation.flags.resolved_by_removal`), so a save whose ``author`` names no
    person (a tool's) cannot remove a flagged annotation.

    Raises with nothing written: ``ValueError`` for a payload that does not parse, a confirmed
    position the save writes no annotation at, a removed annotation an open flag sits on when
    ``author`` names no person, a proposal index the bucket's document does not hold, a flag
    with no comment or on a proposal its bucket does not hold,
    or a resolved flag that is not open; ``UnreadableLabelDocumentError`` for a stored document
    that does not decode; ``VersionConflictError`` when ``expect`` is not the version the commit
    reads.
    """
    from tcip_annotation.flags import (
        flag_key,
        flags_of,
        flags_record,
        raised,
        resolved_by_removal,
    )
    from tcip_annotation.json_io import (
        CompletionMark,
        annotation_from_payload,
        document_at,
        document_payload,
        label_document,
        stamped,
        subject_digest,
    )
    from tcip_annotation.verdicts import Verdict, record_verdicts
    from tcip_mcp.audit import audit_line, audit_log_key, now

    audit = audit_log_key(key.root)
    capture, stem = key.parts
    instant = now()
    # The label document's own stamps are strings (its reader is json_io's hand parser).
    stamp = instant.isoformat()
    contents = []
    for i, payload in enumerate(payloads):
        try:
            contents.append(annotation_from_payload(payload))
        except ValueError as exc:
            raise ValueError(f"annotation {i} {exc}") from exc
    decided = gestures.decide
    if decided is not None:
        bucket, proposals = image_proposals(decided.bucket, key)
        verdict = verdict_key_of(key, decided.bucket)
    new_flags = [
        raised(request, by=author, at=instant) for request in gestures.flag
    ]
    for flag in new_flags:
        target = flag.request.target
        if isinstance(target, ProposalTarget):
            held = len(image_proposals(target.bucket, key)[1])
            if target.index >= held:
                raise ValueError(
                    f"bucket {target.bucket!r} holds {held} proposals for {stem}, not "
                    f"one at index {target.index} to flag"
                )
    flags_key = flag_key(key)
    with tcip_store.transaction(
        key, audit, flags_key, *([] if decided is None else [verdict])
    ) as txn:
        read = txn.read_versioned(key, default=None)
        stored = document_at(key, read)
        if expect is not None and expect != read.version:
            raise tcip_store.VersionConflictError(key, expect, read.version)
        annotations = stamped(
            contents, stored.annotations, author=author, now=stamp
        )
        for i in sorted(gestures.confirm):
            if not 0 <= i < len(annotations):
                raise ValueError(
                    f"the save writes {len(annotations)} annotations for {stem}, "
                    f"none at position {i} to confirm"
                )
            annotations[i] = signed_off(annotations[i], author, stamp)
        held_flags = flags_of(txn.read(flags_key, default=None))
        open_ids = {flag.id for flag in held_flags if flag.open}
        unknown = sorted(set(gestures.resolve) - open_ids)
        if unknown:
            raise ValueError(
                f"flag(s) {unknown} are not open on {stem}; resolve an open flag"
            )
        flags = [
            flag.resolved(
                by=author, at=instant, reply=gestures.resolve[flag.id]
            )
            if flag.id in gestures.resolve
            else flag
            for flag in held_flags
        ] + new_flags
        verdicts: list[Verdict] = []
        accepted: dict[int, int] = {}
        if decided is not None:
            paired = proposal_pairs(project, bucket, annotations, proposals)
            for i, action in sorted(decided.verdicts.items()):
                if not 0 <= i < len(proposals):
                    raise ValueError(
                        f"bucket {bucket.name!r} holds {len(proposals)} "
                        f"proposals for {stem}, not one at index {i}"
                    )
                if action == "accepted" and i in paired:
                    annotations[paired[i]] = signed_off(
                        annotations[paired[i]], author, stamp
                    )
                    accepted[i] = paired[i]
                elif action == "accepted":
                    annotations.append(
                        replace(
                            signed_off(proposals[i], author, stamp),
                            score=None,
                            attributes={},
                        )
                    )
                    accepted[i] = len(annotations) - 1
                verdicts.append(
                    Verdict(proposal=i, action=action, by=author, at=instant)
                )
            record_verdicts(txn, verdict, verdicts)
        marks = {
            s: held
            for s, held in stored.marks.items()
            if gestures.complete.get(s, True)
        }
        for subject in (s for s, made in gestures.complete.items() if made):
            marks.setdefault(subject, []).append(
                CompletionMark(
                    rect=gestures.rect or (0, 0, width, height),
                    by=author,
                    at=stamp,
                    digest=subject_digest(annotations, subject),
                    proposals_hidden=gestures.proposals_hidden,
                )
            )
        written = document_payload(annotations, width, height, marks=marks)
        version = txn.write(key, written)
        flags = resolved_by_removal(
            flags, stored.annotations, annotations, by=author, at=instant
        )
        if flags != held_flags:
            txn.write(flags_key, flags_record(flags))
        txn.append(
            audit,
            audit_line(
                "save_label_document",
                {
                    "capture": capture,
                    "stem": stem,
                    "n_annotations": len(annotations),
                    "version": version.token,
                    "accepted": sorted(v.proposal for v in verdicts if v.action == "accepted"),
                    "rejected": sorted(v.proposal for v in verdicts if v.action == "rejected"),
                    "confirmed": sorted(gestures.confirm),
                    "complete": dict(gestures.complete),
                    "flagged": [flag.id for flag in new_flags],
                    "resolved": sorted(
                        flag.id
                        for flag in flags
                        if not flag.open and flag.id in open_ids
                    ),
                },
                actor,
                "ok",
            ).stored(),
        )
    return version, label_document(written), accepted


def list_subjects(dataset_root: str | Path) -> list[str]:
    """Return the dataset's subjects, in the registry's declared order.

    ``[]`` when there is no registry. A registry that is present but unreadable raises rather than
    reading as no subjects.
    """
    from tcip_mcp import subject_registry

    if not subjects_path(dataset_root).is_file():
        return []
    return [
        s.name for s in subject_registry.read_registry(dataset_root).subjects
    ]


def capture_label_keys(dataset_root: str | Path, capture: str) -> list[Key]:
    """Return the label-document keys of ``capture`` under ``dataset_root``, in stem order."""
    return tcip_store.keys(LABEL_DOCUMENTS, str(dataset_root), (capture,))


def capture_subjects(
    dataset_root: str | Path, capture: str
) -> tuple[list[str], list[str]]:
    """Return the distinct subjects the label documents of ``capture`` hold, sorted.

    Beside them, the stem of every one of those documents that will not read.
    """
    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_label_document,
    )

    found: set[str] = set()
    unreadable: list[str] = []
    for key in capture_label_keys(dataset_root, capture):
        try:
            found.update(
                a.subject for a in read_label_document(key).annotations
            )
        except UnreadableLabelDocumentError:
            unreadable.append(key.parts[-1])
    return sorted(found), unreadable
