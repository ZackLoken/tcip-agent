"""A prediction bucket: a directory of per-image prediction documents and one ``bucket.json``.

``bucket.json`` states who produced the documents (a checkpoint's sha256 and the run that produced
it, or the engine or agent that proposed them), the class scope their labels decode under, the
execution record a model's pass ran under, the capture the source images belong to, the source
image of every document, how many detections were dropped for a box with no extent, and the
assessment the bucket was published under, if any. A bucket is published once: the publication
creates its directory, refusing one that exists, before it writes any document, writes every
document once, and writes ``bucket.json`` last, once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from tcip_annotation.json_io import BUCKET_RECORD

from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.execution import Execution

if TYPE_CHECKING:
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Pass

_PATHS = (("raster_path",),)


class Document(NamedTuple):
    """One document a publication writes: its source image, its encoded bytes, and the number of
    detections its encoding dropped for a box with no extent."""

    source: str
    data: bytes
    dropped: int = 0


class BucketExists(FileExistsError):
    """A publication named a bucket directory that already exists."""


class NotABucket(ValueError):
    """A directory read as a bucket holds no ``bucket.json``, or one that does not decode."""


@dataclass(frozen=True)
class Bucket:
    """One published bucket, as its ``bucket.json`` states it.

    ``producer`` is a checkpoint's sha256 and run
    (:attr:`~tcip_mcp.model_registry.VerifiedCheckpoint.producer`), or ``{"proposed_by"}``, the
    engine or agent that proposed the documents, which no execution record governs
    (``execution`` ``None``) and no class scope decodes (``scope`` empty). ``documents`` maps each
    document's stem to the source file name it was predicted from, and ``raster_path`` is the
    raster as stored against the project (:meth:`raster`)."""

    path: Path
    producer: dict[str, str | None]
    scope: ClassScope
    execution: Execution | None
    dataset_id: str | None
    date: str | None
    raster_path: str | None
    raster_identity: dict | None
    documents: dict[str, str]
    dropped_boxes: int
    assessment_id: str | None

    def document(self, image: str | Path) -> Path | None:
        """The document this bucket holds for the source image ``image`` (a path or a file name),
        as its record names it, or ``None`` when the record names none for that image."""
        from tcip_mcp.dataset_layout import label_filename

        name = Path(image).name
        stem = Path(name).stem
        return self.path / label_filename(stem) if self.documents.get(stem) == name else None

    @property
    def document_paths(self) -> list[Path]:
        """Every document the record names, in stem order."""
        from tcip_mcp.dataset_layout import label_filename

        return [self.path / label_filename(stem) for stem in sorted(self.documents)]

    def raster(self, project: Path) -> str | None:
        """The raster this bucket was predicted on, resolved against ``project``, or ``None`` for
        a bucket of per-image predictions."""
        from tcip_mcp.registry_paths import runtime_paths

        return runtime_paths({"raster_path": self.raster_path}, _PATHS, project)["raster_path"]


def read_bucket(path: str | Path) -> Bucket:
    """The bucket at ``path`` as its ``bucket.json`` states it. A directory holding no
    ``bucket.json``, or one that does not decode, refuses (:class:`NotABucket`) naming it."""
    from tcip_mcp.experiments import read_record
    from tcip_store import DecodeError

    directory = Path(path)
    record_path = directory / BUCKET_RECORD
    if not record_path.is_file():
        raise NotABucket(
            f"{directory} holds no {BUCKET_RECORD}: it is not a published prediction bucket, so "
            "nothing states who produced its documents or how. Publish predictions through "
            "run_inference, or stage proposals through stage_proposals.")
    try:
        record = read_record(record_path)
    except DecodeError as exc:
        raise NotABucket(str(exc)) from exc
    execution = record.pop("execution")
    return Bucket(path=directory, scope=ClassScope(**record.pop("scope")),
                  execution=Execution.of(execution) if execution is not None else None, **record)


def by_recorded_date(buckets: Iterable[Bucket]) -> dict[str, Bucket]:
    """``buckets`` keyed by the capture date each record states. A bucket recording no date, and
    two recording one date, refuse (``ValueError``) naming them."""
    dated: dict[str, Bucket] = {}
    for bucket in buckets:
        if bucket.date is None:
            raise ValueError(f"{bucket.path} records no capture date, so it stands for no date of "
                             "a series.")
        if bucket.date in dated:
            raise ValueError(f"{dated[bucket.date].path} and {bucket.path} both record capture "
                             f"date {bucket.date}: a series takes one bucket per date.")
        dated[bucket.date] = bucket
    return dated


def input_scope(bucket: Bucket | None, subject: str | None, attribute: str | None) -> ClassScope:
    """The class space documents are read under: a model's ``bucket`` states its own, and a
    subject or attribute stated beside it refuses (``ValueError``). With no bucket, or a bucket of
    proposals, which decodes no labels, it is the stated subject, and a stated attribute refuses,
    since only a model's record carries a classified map."""
    if bucket is None or bucket.scope == ClassScope():
        if attribute is not None:
            raise ValueError("these documents record no class map; a classified read takes a "
                             "model bucket's own.")
        return ClassScope(subject=subject)
    if subject is not None or attribute is not None:
        raise ValueError(f"{bucket.path}'s record states its own class space; a subject or "
                         "attribute stated beside it would be a second one.")
    return bucket.scope


def bucket_dirs(dataset_root: str | Path) -> list[Path]:
    """Every published bucket under ``dataset_root``'s ``predictions/`` tree: each directory at
    any depth holding a ``bucket.json``, sorted."""
    from tcip_mcp.dataset_layout import prediction_root

    root = prediction_root(dataset_root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob(BUCKET_RECORD))


def buckets_by_date(dataset_root: str | Path, dates: list[str]) -> dict[str, dict[str, str]]:
    """For each of ``dates``, the published buckets under ``dataset_root`` whose record states that
    capture date: each bucket's directory, named by its path under the ``predictions/`` tree."""
    from tcip_mcp.dataset_layout import prediction_root

    root = prediction_root(dataset_root)
    grouped: dict[str, dict[str, str]] = {d: {} for d in dates}
    for directory in bucket_dirs(dataset_root):
        date = read_bucket(directory).date
        if date in grouped:
            grouped[date][directory.relative_to(root).as_posix()] = str(directory)
    return grouped


def pass_documents(p: Pass, results: Iterable[dict]) -> Iterator[Document]:
    """Each of a pass's ``results`` encoded as the document it publishes: a detector's (a pass
    whose execution record states a conf) as its prediction document under the pass's scope,
    stamped with the checkpoint that predicted it
    (:func:`~tcip_mcp.pipelines.postprocessing.export.encode_predictions`), any other head's as
    its own output (:func:`~tcip_mcp.pipelines.postprocessing.export.encode_head_output`)."""
    from tcip_mcp.pipelines.postprocessing.export import encode_head_output, encode_predictions

    created_by = prediction_producer(p.checkpoint)
    detector = p.execution.conf is not None
    for r in results:
        yield Document(r["image"], *(
            encode_predictions(r, created_by=created_by, scope=p.scope) if detector
            else encode_head_output(r, task=p.checkpoint.task)))


def publish(
    project: Path, out: Path, documents: Iterable[Document], *, producer: dict[str, str | None],
    scope: ClassScope, execution: Execution | None, raster_path: str | None,
    raster_identity: dict | None, assessment_id: str | None,
) -> Bucket:
    """Publish the bucket ``out``: create its directory, refusing one that already exists
    (:class:`BucketExists`) before anything is consumed or written, then write each of
    ``documents`` once as ``documents`` yields them, a stem already written refusing
    (``ValueError``), then ``bucket.json`` once, then the publication's one audit line,
    ``prediction_bucket_published``.

    The record states ``producer``, ``scope``, ``execution``, the raster and the assessment as
    given, and the capture of the raster, else of the first document's source image
    (:func:`~tcip_mcp.dataset_layout.capture_of`). A raise after the first document lands, before
    ``bucket.json`` does, leaves the directory without it (never a bucket) and one
    ``prediction_bucket_published`` line under status ``failed`` naming the documents written,
    then propagates.
    """
    import tcip_store
    from tcip_annotation.json_io import annotation_record_key

    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
    from tcip_mcp.dataset_layout import capture_of, dataset_root_of
    from tcip_mcp.experiments import RunDirectoryExists, create_run_directory, write_once
    from tcip_mcp.registry_paths import recorded_paths

    try:
        create_run_directory(out)
    except RunDirectoryExists:
        raise BucketExists(
            f"{out} already exists: a bucket is published once, so a new publication, a resumed "
            "raster pass included, names a bucket that does not exist yet.") from None
    audit_scope = dataset_root_of(out.resolve()) or project
    written: dict[str, str] = {}
    dropped_boxes = 0
    capture: tuple[str | None, str | None] = (
        capture_of(raster_path) if raster_path is not None else (None, None))
    try:
        for document in documents:
            image = Path(document.source)
            if raster_path is None and not written:
                capture = capture_of(image)
            try:
                tcip_store.put_blob(annotation_record_key(out, image.stem), document.data,
                                    expect=tcip_store.Version.ABSENT)
            except tcip_store.VersionConflict:
                raise ValueError(f"two documents of one publication name the stem "
                                 f"{image.stem!r}, and a document is written once.") from None
            dropped_boxes += document.dropped
            written[image.stem] = image.name
        record = {
            "producer": producer, "scope": asdict(scope),
            "execution": execution.record() if execution is not None else None,
            "dataset_id": capture[0], "date": capture[1], "raster_path": raster_path,
            "raster_identity": raster_identity, "documents": written,
            "dropped_boxes": dropped_boxes, "assessment_id": assessment_id,
        }
        write_once(out / BUCKET_RECORD, recorded_paths(record, _PATHS, project))
    except AuditEntryNotWritten:
        raise
    except Exception as exc:
        if written:
            record_event_or_raise(
                "prediction_bucket_published",
                {"predictions_dir": str(out), "written": sorted(written), "error": str(exc)},
                status="failed", scope=audit_scope)
        raise
    record_event_or_raise("prediction_bucket_published", {"predictions_dir": str(out)},
                          scope=audit_scope)
    return read_bucket(out)


def prediction_producer(checkpoint: VerifiedCheckpoint) -> str:
    """The ``created_by`` spelling for a prediction ``checkpoint`` wrote."""
    return f"model:{Path(checkpoint.path).stem}@{checkpoint.sha256[:12]}"


def bucket_key_of(bucket_dir: str | Path | None) -> str:
    """The verdict store's key for the prediction bucket at ``bucket_dir``: its path relative to
    the dataset root it sits in, its own resolved path under none, and
    :data:`~tcip_annotation.review_engine.NO_BUCKET` for no directory at all."""
    from tcip_annotation.review_engine import NO_BUCKET

    from tcip_mcp.dataset_layout import dataset_root_of

    if not bucket_dir:
        return NO_BUCKET
    d = Path(bucket_dir).resolve()
    root = dataset_root_of(d)
    return d.as_posix() if root is None else d.relative_to(root).as_posix()


def detection_rows(bucket: Bucket) -> list[dict[str, Any]]:
    """Every document of ``bucket``, in stem order, as ``{"image", "count", "scores"}``: the
    source filename its record names, and the real detections it holds (a ``Point`` and a crowd
    region excluded) with their scores."""
    from tcip_annotation.json_io import detection_annotations
    from tcip_annotation.state import prediction_score

    rows = []
    for path in bucket.document_paths:
        annotations = detection_annotations(path)
        rows.append({"image": bucket.documents[path.stem], "count": len(annotations),
                     "scores": [prediction_score(a) for a in annotations]})
    return rows
