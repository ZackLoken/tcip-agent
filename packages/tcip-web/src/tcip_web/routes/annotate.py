"""The Annotate tab's routes: an image's label document read and saved, the proposals a
prediction bucket offers for it, and the review queue.

Reads the canonical per-image label document (every subject's annotations by name and their
completion marks) by the key of the image the caller names; its one save is
:func:`~tcip_mcp.dataset_layout.save_label_document`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation.json_io import (
    LabelDocument,
    UnreadableLabelDocumentError,
    authorship_of,
    client_annotation,
    read_document_versioned,
)
from tcip_annotation.state import Annotation
from tcip_store import Key, Version, VersionConflictError
from tcip_mcp.pipelines.active_learning import DEFAULT_REVIEW_BUDGET, DEFAULT_SCORER
from tcip_web import jobstore
from tcip_web.paths import allowed_image, allowed_path
from tcip_web.state import store

router = APIRouter(prefix="/api/annotate", tags=["annotate"])
logger = logging.getLogger(__name__)


class AnnotationPayload(BaseModel):
    """One annotation's content: its ``subject`` (the object it is about), a geometry (``bbox``
    corners, ``points`` for the one ring the canvas draws by hand, ``rings`` for a loaded
    multi-ring shape round-tripping, ``point`` for a placed prompt or keypoint, or none of them for
    an image-level label), its attribute values by name and its crowd flag.

    Every field but the subject is carried uninterpreted: its one reading is
    :func:`~tcip_annotation.json_io.annotation_from_payload`.
    """

    subject: str
    bbox: Any = None
    points: Any = None
    rings: Any = None
    point: Any = None
    attributes: Any = None
    iscrowd: Any = None


class FlagPayload(BaseModel):
    """One flag a save raises: its comment and its target, a ``point`` with the ``subject``
    flagged there, a ``proposal`` (bucket and index), or neither for the image as a whole
    (:class:`~tcip_annotation.flags.FlagRequest`)."""

    text: str
    point: Optional[tuple[float, float]] = None
    subject: Optional[str] = None
    proposal: Optional[tuple[str, int]] = None


class SavePayload(BaseModel):
    """One save: the image's annotations and the gestures it adjudicates beside them
    (:class:`~tcip_mcp.dataset_layout.Gestures`), by the person ``user`` names."""

    image_path: str
    annotations: list[AnnotationPayload] = []
    # The label document's version token as the client loaded it: with one, the save is a
    # compare-and-set and a 409 says it changed underneath. Omit to skip the comparison.
    base_mtime: Optional[str] = None
    user: str
    bucket: Optional[str] = None
    accept: list[int] = []
    reject: list[int] = []
    complete: dict[str, bool] = {}
    rect: Optional[tuple[float, float, float, float]] = None
    proposals_hidden: bool = False
    flag: list[FlagPayload] = []
    # Each open flag resolved, by id, with the reply given.
    resolve: dict[str, str] = {}


def _flags(key: Key) -> list[dict]:
    """The image's flags as the editor reads them, oldest first
    (:func:`~tcip_annotation.flags.read_flags`); a record that will not read answers 400."""
    from tcip_annotation.flags import encode_flag, flag_key, read_flags

    try:
        return [encode_flag(flag) for flag in read_flags(flag_key(key))]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def annotation_dict(a: Annotation) -> dict:
    """An :class:`Annotation` for the canvas: the library's client projection
    (:func:`~tcip_annotation.json_io.client_annotation`) plus its ``authorship``
    (:func:`authorship_of`).
    """
    return {**client_annotation(a), "authorship": authorship_of(a)}


def _completion(doc: LabelDocument) -> dict:
    """Each subject ``doc`` holds or marks, with its state
    (:meth:`~tcip_annotation.json_io.LabelDocument.state`)."""
    return {subject: doc.state(subject)
            for subject in sorted({a.subject for a in doc.annotations} | set(doc.marks))}


def _admitted(image_path: str) -> tuple[Key, int, int]:
    """The image at ``image_path`` admitted once (:func:`~tcip_web.paths.allowed_image`): the key
    of its label document (:func:`~tcip_mcp.dataset_layout.label_key_of`) and its dimensions. An
    image under no dataset image tree answers 400."""
    from tcip_mcp.dataset_layout import label_key_of

    path, w, h = allowed_image(image_path)
    try:
        return label_key_of(path), w, h
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _read(key: Key) -> tuple[LabelDocument, Version]:
    """The label document ``key`` names and its version; one that will not read answers 400."""
    try:
        return read_document_versioned(key)
    except UnreadableLabelDocumentError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/labels")
def load_labels(image_path: str) -> dict:
    """An image's label document in pixel coords: its annotations, each with its ``index`` in the
    document (the index a proposal's ``paired`` names), each subject's completion
    (:func:`_completion`), the image's flags (:func:`_flags`) and the version token a save
    echoes back, the empty token for an image with no document yet."""
    key, w, h = _admitted(image_path)
    doc, version = _read(key)
    return {"image_path": image_path, "img_width": w, "img_height": h,
            "annotations": [{**annotation_dict(a), "index": i}
                            for i, a in enumerate(doc.annotations)],
            "completion": _completion(doc), "flags": _flags(key),
            "base_mtime": version.token}


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Save an image's label document through
    :func:`~tcip_mcp.dataset_layout.save_label_document`, by the person
    :func:`~tcip_mcp.identity.actor` makes of ``user``, and answer the saved document's version
    and completion and the image's flags. A stale ``base_mtime`` answers 409 and a refusal 400,
    nothing written.
    """
    from tcip_annotation.flags import FlagRequest

    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tcip_mcp.identity import actor

    key, w, h = _admitted(payload.image_path)
    expect = Version(payload.base_mtime) if payload.base_mtime is not None else None
    for name, indices in (("accept", payload.accept), ("reject", payload.reject)):
        if len(set(indices)) != len(indices):
            raise HTTPException(400, f"{name} names a proposal more than once: {indices}")
    gestures = Gestures(
        bucket=payload.bucket, accept=frozenset(payload.accept),
        reject=frozenset(payload.reject), complete=payload.complete, rect=payload.rect,
        proposals_hidden=payload.proposals_hidden,
        flag=tuple(FlagRequest(**f.model_dump()) for f in payload.flag),
        resolve=payload.resolve)
    person = actor(payload.user)
    try:
        version, doc = save_label_document(
            store.project_root, key, [ap.model_dump() for ap in payload.annotations],
            width=w, height=h, author=person, actor=person, expect=expect, gestures=gestures)
    except VersionConflictError as exc:
        raise HTTPException(409, {"error": "label document changed since it was loaded"}) from exc
    except (ValueError, UnreadableLabelDocumentError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "ok", "image_path": payload.image_path,
            "n_annotations": len(doc.annotations), "base_mtime": version.token,
            "completion": _completion(doc), "flags": _flags(key)}


@router.get("/proposals")
def load_proposals(image_path: str, bucket: str) -> dict:
    """The proposals the bucket named ``bucket`` under the image's dataset root offers for
    ``image_path``, in document order: each with its ``index``, the index of the annotation of
    the image's label document it pairs with (``paired``,
    :func:`~tcip_mcp.dataset_layout.proposal_pairs`) and the last ``decision`` its verdict shard
    records; beside them the bucket's ``operating_point``, the ``conf`` a review may accept at on
    its assessment's authority or ``None`` with the ``reason``
    (:func:`~tcip_mcp.delivery.admitted_conf`). A bucket that names no document for the image, or
    one that will not read, answers 400."""
    from tcip_annotation.verdicts import read_verdicts

    from tcip_mcp.delivery import admitted_conf
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs, verdict_key_of

    key, _w, _h = _admitted(image_path)
    annotations = _read(key)[0].annotations
    try:
        published, proposals = image_proposals(bucket, key)
        decisions = {v.proposal: v.action for v in read_verdicts(verdict_key_of(key, bucket))}
        paired = proposal_pairs(store.project_root, published, annotations, proposals)
    except (ValueError, UnreadableLabelDocumentError) as exc:
        raise HTTPException(400, str(exc)) from exc
    conf, reason = admitted_conf(store.open_root(), published)
    return {"bucket": published.name, "operating_point": {"conf": conf, "reason": reason},
            "proposals": [
                {**annotation_dict(p), "index": i, "paired": paired.get(i),
                 "decision": decisions.get(i)}
                for i, p in enumerate(proposals)]}


@dataclass
class PriorityQueueJob:
    job_id: str
    # The project open when the job launched, which it runs for.
    project: str
    checkpoint_path: str
    images_dir: str
    subject: Optional[str]
    method: str
    budget: int
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None
    # [{image, score, reference_member?}], highest first.
    queue: list[dict] = field(default_factory=list)
    total_candidates: int = 0
    reviewed_skipped: int = 0


_pq_registry = jobstore.JobRegistry()
"""The review queue's own live jobs (``jobstore.JobRegistry``)."""


def _pq_worker(job: PriorityQueueJob) -> None:
    try:
        job.status = "running"
        from tcip_mcp.tools.feedback_tools import prioritize_review_queue

        result = prioritize_review_queue(
            Path(job.project), checkpoint_path=job.checkpoint_path, images_dir=job.images_dir,
            method=job.method, budget=job.budget, subject=job.subject)
        if "error" in result:
            job.status, job.error = "failed", result["error"]
        else:
            job.status = "completed"
            job.queue = result["queue"]
            job.total_candidates = result["total_candidates"]
            job.reviewed_skipped = result["reviewed_skipped"]
    except Exception as exc:
        logger.exception("priority-queue job %s failed", job.job_id)
        job.status, job.error = "failed", str(exc)


class LaunchPriorityQueuePayload(BaseModel):
    checkpoint_path: str
    images_dir: str
    subject: Optional[str] = None
    method: str = DEFAULT_SCORER
    budget: int = DEFAULT_REVIEW_BUDGET


@router.post("/queue/launch")
def launch_priority_queue(payload: LaunchPriorityQueuePayload) -> dict:
    """Launch :func:`~tcip_mcp.tools.feedback_tools.prioritize_review_queue` for the open project
    on a background thread; a checkpoint or images directory outside the allowed roots is refused
    and a missing one answers 404."""
    checkpoint_path = allowed_path(payload.checkpoint_path)
    images_dir = allowed_path(payload.images_dir)
    if not checkpoint_path.is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")
    job = PriorityQueueJob(
        job_id=f"pq-{uuid.uuid4().hex[:8]}", project=str(store.open_root()),
        checkpoint_path=str(checkpoint_path), images_dir=str(images_dir),
        subject=payload.subject, method=payload.method, budget=payload.budget)
    _pq_registry.register(job.job_id, job)
    threading.Thread(target=_pq_worker, args=(job,), daemon=True).start()
    return {"status": "launched", "job_id": job.job_id}


@router.get("/queue/{job_id}")
def get_priority_queue_job(job_id: str) -> dict:
    """The queue job ``job_id``'s state and, once completed, its ranked images; an unknown job
    answers 404."""
    job = _pq_registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return {"job_id": job.job_id, "status": job.status, "error": job.error, "queue": job.queue,
            "total_candidates": job.total_candidates, "reviewed_skipped": job.reviewed_skipped}
