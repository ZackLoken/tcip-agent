"""A prediction bucket: its per-image prediction documents and the record of who made them and how.

Each document and the record lie under the dataset root the source images belong to.

The record states who produced the documents (a checkpoint's sha256 and the run that produced
it, or the engine or agent that proposed them), the class scope their labels decode under, the
execution record a model's pass ran under, the capture the source images belong to, the source
image of every document, how many detections were dropped for a box with no extent, the
assessment the bucket was published under, if any, and when it was published. A bucket is
published once, in one commit over its record, every document and the publication's audit line.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter

import tcip_store
from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.execution import Execution
from tcip_mcp.pipelines.raster_source import RasterIdentity

if TYPE_CHECKING:
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Pass

_PATHS = (("raster_path",),)


class Document(NamedTuple):
    """One document a publication writes.

    It holds its source image, its record, and the number of detections its encoding dropped for
    a box with no extent.
    """

    source: str
    data: dict
    dropped: int = 0


class BucketExistsError(ValueError):
    """A publication named a bucket whose record already exists under its dataset root."""


class NotABucketError(ValueError):
    """A bucket name names no published bucket under a dataset root, or one that does not decode."""


class Proposed(BaseModel):
    """The engine or agent that proposed a bucket's documents, by the name it stages under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposed_by: str


@dataclass(frozen=True)
class Bucket:
    """One published bucket, as its record states it.

    ``root`` is the dataset root its source images belong to and ``name`` the name it was
    published under. ``producer`` is a checkpoint's sha256 and run
    (:attr:`~tcip_mcp.model_registry.VerifiedCheckpoint.producer`), or the engine or agent that
    proposed the documents (:class:`Proposed`), whose ``execution`` is the record of the inference
    pass an engine's proposal ran through (``None`` for a proposal no such pass made) and which no
    class scope decodes (``scope`` empty). ``documents`` maps each
    document's stem to the source file name it was predicted from, ``raster_path`` is the
    raster as stored against the project (:meth:`raster`), and ``published_at`` the instant its
    publication committed.
    """

    root: Path
    name: str
    published_at: AwareDatetime
    producer: Producer | Proposed
    scope: ClassScope
    execution: Execution | None
    dataset_id: str | None
    date: str | None
    raster_path: str | None
    raster_identity: RasterIdentity | None
    documents: dict[str, str]
    dropped_boxes: int
    assessment_id: str | None
    __pydantic_config__ = ConfigDict(ser_json_inf_nan="constants")

    @cached_property
    def stem_keys(self) -> dict[str, tcip_store.Key]:
        """The key of each document the record names, by its source image's stem, in stem order."""
        from tcip_mcp.dataset_layout import prediction_key

        return {
            stem: prediction_key(self.root, self.name, stem)
            for stem in sorted(self.documents)
        }

    def document_key(self, stem: str) -> tcip_store.Key | None:
        """Return the key of the document this bucket holds for the source image of stem ``stem``.

        The key is as :attr:`stem_keys` names it, or ``None`` when the record names none for that
        image.
        """
        return self.stem_keys.get(stem)

    @property
    def document_keys(self) -> list[tcip_store.Key]:
        """Every document the record names, in stem order (:attr:`stem_keys`)."""
        return list(self.stem_keys.values())

    def raster(self, project: Path) -> str | None:
        """Return the raster this bucket was predicted on, resolved against ``project``.

        Returns ``None`` for a bucket of per-image predictions.
        """
        from tcip_mcp.registry_paths import resolved_registry_path

        if self.raster_path is None:
            return None
        return str(resolved_registry_path(project, self.raster_path))


_BUCKET = TypeAdapter(Bucket)
"""The bucket record: every field of :class:`Bucket` but its ``root`` and ``name``, which its key
states."""
_KEYED = {"root", "name"}


def read_bucket(dataset_root: str | Path, name: str) -> Bucket:
    """Read the bucket named ``name`` under ``dataset_root``, as its record states it.

    ``dataset_root`` is taken as given, not located, and carried as the bucket's root. No record
    object, or one that does not decode, refuses (:class:`NotABucketError`) naming it; a record
    that does not validate as the bucket's refuses (``pydantic.ValidationError``) naming each
    field.
    """
    from tcip_mcp.dataset_layout import bucket_key

    dataset_root = Path(dataset_root)
    try:
        record = tcip_store.read(bucket_key(dataset_root, name), default=None)
    except tcip_store.DecodeError as exc:
        raise NotABucketError(str(exc)) from exc
    if not isinstance(record, dict):
        raise NotABucketError(
            f"no bucket {name!r} is published under {dataset_root}, so nothing states who "
            "produced its documents or how. Publish predictions through run_inference, or stage "
            "proposals through stage_proposals."
        )
    return _BUCKET.validate_python(
        {**record, "root": dataset_root, "name": name}
    )


def bucket_names(dataset_root: str | Path) -> list[str]:
    """Return the name of every bucket published under ``dataset_root``, in name order."""
    from tcip_mcp.dataset_layout import PREDICTION_BUCKETS

    return [
        key.parts[0]
        for key in tcip_store.keys(PREDICTION_BUCKETS, str(dataset_root))
    ]


def buckets_under(dataset_root: str | Path) -> list[Bucket]:
    """Return every bucket published under ``dataset_root``, in publication order."""
    return sorted(
        (read_bucket(dataset_root, n) for n in bucket_names(dataset_root)),
        key=lambda bucket: bucket.published_at,
    )


def by_recorded_date(buckets: Iterable[Bucket]) -> dict[str, Bucket]:
    """Return ``buckets`` keyed by the capture date each record states.

    A bucket recording no date, and two recording one date, refuse (``ValueError``) naming them.
    """
    dated: dict[str, Bucket] = {}
    for bucket in buckets:
        if bucket.date is None:
            raise ValueError(
                f"bucket {bucket.name!r} records no capture date, so it stands for no "
                "date of a series."
            )
        if bucket.date in dated:
            raise ValueError(
                f"buckets {dated[bucket.date].name!r} and {bucket.name!r} both record "
                f"capture date {bucket.date}: a series takes one bucket per date."
            )
        dated[bucket.date] = bucket
    return dated


def shared_root(buckets: Iterable[Bucket]) -> Path:
    """Return the one dataset root every bucket of ``buckets`` is published under.

    Buckets under more than one root, or no bucket, refuse (``ValueError``) naming them.
    """
    return _one_root((b.root for b in buckets), "a delivery's buckets")


def buckets_by_date(
    dataset_root: str | Path, dates: list[str]
) -> dict[str, list[str]]:
    """Return, for each of ``dates``, the names of the buckets under ``dataset_root`` of that date.

    A bucket is of a date when its record states that capture date; each date's are in
    publication order (:func:`buckets_under`).
    """
    grouped: dict[str, list[str]] = {d: [] for d in dates}
    for bucket in buckets_under(dataset_root):
        if bucket.date in grouped:
            grouped[bucket.date].append(bucket.name)
    return grouped


def pass_documents(p: Pass, results: Iterable[dict]) -> Iterator[Document]:
    """Yield each of a pass's ``results`` encoded as the document it publishes.

    A detector's (a pass whose execution record states a conf) is read as its
    :class:`~tcip_mcp.pipelines.postprocessing.export.DetectionResult`, refusing
    (``ValidationError``) a result that is none, and encoded as its prediction document under
    the pass's scope, carrying the cap its frame was predicted under and stamped with the
    checkpoint that predicted it
    (:func:`~tcip_mcp.pipelines.postprocessing.export.encode_predictions`); any other head's is
    its own output (:func:`~tcip_mcp.pipelines.postprocessing.export.encode_head_output`).
    """
    from tcip_mcp.pipelines.postprocessing.export import (
        DetectionResult,
        encode_head_output,
        encode_predictions,
    )

    created_by = prediction_producer(p.checkpoint)
    detector = p.execution.conf is not None
    for r in results:
        if detector:
            detected = DetectionResult.model_validate(r)
            yield Document(
                detected.image,
                *encode_predictions(detected, created_by=created_by, scope=p.scope),
            )
        else:
            yield Document(r["image"], *encode_head_output(r, task=p.checkpoint.task))


def _one_root(roots: Iterable[Path], what: str) -> Path:
    """Return the one dataset root among ``roots``.

    None, or more than one, refuses (``ValueError``) naming ``what`` and the roots.
    """
    found = {tcip_store.canonical_path(root): root for root in roots}
    if len(found) != 1:
        raise ValueError(
            f"{what} lie under {sorted(found) or 'no'} dataset root(s), and they "
            "belong to one."
        )
    return next(iter(found.values()))


def source_root(sources: Iterable[str | Path]) -> Path:
    """Return the one dataset root every source image of a bucket lies under.

    Each source's root is :func:`~tcip_mcp.dataset_layout.dataset_root_of` its directory: the
    capture's, or the image tree's for a raster in no capture, a directory position reads both
    ways read by which root carries its own state. A source under no dataset image tree, one
    whose directory neither or both roots claim, and sources under more than one root, refuse
    (``ValueError``) naming them.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    def root_of(source: str | Path) -> Path:
        root = dataset_root_of(Path(source).parent)
        if root is None:
            raise ValueError(
                f"{source} lies under no dataset image tree, so no dataset root holds "
                "a bucket of its predictions."
            )
        return root

    return _one_root(map(root_of, sources), "a bucket's source images")


def publish(
    project: Path,
    root: Path,
    name: str,
    documents: Iterable[Document],
    *,
    producer: Producer | Proposed,
    scope: ClassScope,
    execution: Execution | None,
    raster_path: str | None,
    raster_identity: RasterIdentity | None,
    assessment_id: str | None,
    actor: str | None,
) -> Bucket:
    """Publish ``documents`` as the bucket ``name`` under the dataset root ``root``, by ``actor``.

    ``root`` is the one :func:`source_root` answers for their sources. One commit covers the
    bucket's record, every document and the publication's audit line,
    ``prediction_bucket_published``, naming the bucket and its ``dropped_boxes``. Returns the
    bucket as read back.

    The record states ``producer``, ``scope``, ``execution``, the raster and the assessment as
    given, and the capture of the raster, else the one capture every document's source image is
    of (:func:`~tcip_mcp.dataset_layout.image_capture`), with its root's dataset id. Refuses
    with nothing written
    (``ValueError``): no document, two documents of one stem, documents of two captures, and a
    bucket of that name already published under the root (:class:`BucketExistsError`).
    """
    from tcip_mcp.audit import audit_line, audit_log_key
    from tcip_mcp.dataset_layout import (
        bucket_key,
        dataset_id_of,
        image_capture,
    )
    from tcip_mcp.registry_paths import recorded_paths

    documents = list(documents)
    if not documents:
        raise ValueError(f"bucket {name!r} has no document to publish.")
    stems = [Path(d.source).stem for d in documents]
    repeated = sorted(s for s, n in Counter(stems).items() if n > 1)
    if repeated:
        raise ValueError(
            f"two documents of one publication name the stems {repeated}, and a "
            "document is written once."
        )
    by_stem = dict(zip(stems, documents, strict=True))
    captures = (
        {image_capture(raster_path)}
        if raster_path is not None
        else {image_capture(d.source) for d in documents}
    )
    if len(captures) > 1:
        raise ValueError(
            f"bucket {name!r} holds documents of the captures "
            f"{sorted(str(c) for _, c in captures)}, and a publication states the one capture "
            "its documents are of: publish each capture's own."
        )
    ((captured_root, date),) = captures
    dataset_id = dataset_id_of(captured_root)
    dropped_boxes = sum(d.dropped for d in documents)
    line = audit_line(
        "prediction_bucket_published",
        {"dataset_root": str(root), "bucket": name, "dropped_boxes": dropped_boxes},
        actor,
        "ok",
    )
    bucket = Bucket(
        root=root,
        name=name,
        published_at=line.timestamp,
        producer=producer,
        scope=scope,
        execution=execution,
        dataset_id=dataset_id,
        date=date,
        raster_path=raster_path,
        raster_identity=raster_identity,
        documents={stem: Path(d.source).name for stem, d in by_stem.items()},
        dropped_boxes=dropped_boxes,
        assessment_id=assessment_id,
    )
    keys = bucket.stem_keys
    held, audit = bucket_key(root, name), audit_log_key(root)
    with tcip_store.transaction(held, audit, *keys.values()) as txn:
        if (
            txn.read_versioned(held, default=None).version
            != tcip_store.Version.ABSENT
        ):
            raise BucketExistsError(
                f"bucket {name!r} is already published under {root}: a bucket is published "
                "once, so a new publication, a resumed raster pass included, names a bucket that "
                "does not exist yet."
            )
        for stem, key in keys.items():
            txn.write(key, by_stem[stem].data)
        txn.write(
            held,
            recorded_paths(
                _BUCKET.dump_python(bucket, mode="json", exclude=_KEYED),
                _PATHS,
                project,
            ),
        )
        txn.append(audit, line.stored())
    return read_bucket(root, name)


def prediction_producer(checkpoint: VerifiedCheckpoint) -> str:
    """Return the ``created_by`` spelling for a prediction ``checkpoint`` wrote."""
    return f"model:{Path(checkpoint.path).stem}@{checkpoint.sha256[:12]}"


def detection_rows(bucket: Bucket) -> list[dict[str, Any]]:
    """Return every document of ``bucket``, in stem order, as ``{"image", "count", "scores"}``.

    Each row holds the source filename its record names, and the count of real detections it
    holds (a ``Point`` and a crowd region excluded) with their scores.
    """
    from tcip_annotation.json_io import (
        detection_annotations,
        read_label_document,
    )
    from tcip_annotation.state import prediction_score

    rows = []
    for stem, key in bucket.stem_keys.items():
        annotations = detection_annotations(
            read_label_document(key).annotations
        )
        rows.append(
            {
                "image": bucket.documents[stem],
                "count": len(annotations),
                "scores": [prediction_score(a) for a in annotations],
            }
        )
    return rows
