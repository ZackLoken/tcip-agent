"""The Annotate tab's routes: an image's label document read and saved, the proposals a
prediction bucket offers for it, and the review queue.

Reads the canonical per-image label document (one JSON per image, every subject's annotations by
name and their completion marks) at the label path the caller supplies; its one save is
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
from pydantic import BaseModel, Field

from tcip_annotation.json_io import (
    LabelDocument,
    UnreadableLabelDocument,
    authorship_of,
    client_annotation,
    label_document,
    read_document_versioned,
)
from tcip_annotation.state import Annotation
from tcip_store import Version, VersionConflict
from tcip_mcp.pipelines.active_learning import DEFAULT_REVIEW_BUDGET, DEFAULT_SCORER
from tcip_web import jobstore
from tcip_web.paths import allowed_image_dimensions, allowed_optional, allowed_path
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


class SavePayload(BaseModel):
    """One save: the image's annotations and the gestures it adjudicates beside them
    (:class:`~tcip_mcp.dataset_layout.Gestures`), by the person ``user`` names."""

    image_path: str
    label_path: str = Field(min_length=1)
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


def _read(label_path: Path) -> tuple[LabelDocument, Version]:
    """The label document at ``label_path`` and its version, a document that will not read
    answering 400."""
    try:
        return read_document_versioned(label_path)
    except UnreadableLabelDocument as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/labels")
def load_labels(image_path: str, label_path: Optional[str] = None) -> dict:
    """An image's label document in pixel coords: its annotations, each subject's completion
    (:func:`_completion`) and the version token a save echoes back; no ``label_path`` reads as an
    empty document."""
    w, h = allowed_image_dimensions(image_path)
    label = allowed_optional(label_path)
    doc, token = (label_document(None), None)
    if label:
        doc, version = _read(Path(label))
        token = version.token
    return {"image_path": image_path, "img_width": w, "img_height": h,
            "annotations": [annotation_dict(a) for a in doc.annotations],
            "completion": _completion(doc), "base_mtime": token}


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Save an image's label document through
    :func:`~tcip_mcp.dataset_layout.save_label_document`, by the person
    :func:`~tcip_mcp.identity.actor` makes of ``user``, and answer the saved document's version
    and completion. A stale ``base_mtime`` answers 409, a refusal 400, and a write that commits
    and cannot be recorded 409 with the marker and the save's recorded facts.
    """
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tcip_mcp.identity import actor
    from tcip_web.routes.audit_gap import audit_gap_409

    def saved() -> dict:
        doc, version = _read(label_path)
        return {"status": "ok", "image_path": payload.image_path,
                "n_annotations": len(doc.annotations), "base_mtime": version.token,
                "completion": _completion(doc)}

    w, h = allowed_image_dimensions(payload.image_path)
    label_path = allowed_path(payload.label_path)
    expect = Version(payload.base_mtime) if payload.base_mtime is not None else None
    for name, indices in (("accept", payload.accept), ("reject", payload.reject)):
        if len(set(indices)) != len(indices):
            raise HTTPException(400, f"{name} names a proposal more than once: {indices}")
    gestures = Gestures(
        bucket=str(allowed_path(payload.bucket)) if payload.bucket else None,
        accept=frozenset(payload.accept), reject=frozenset(payload.reject),
        complete=payload.complete, rect=payload.rect, proposals_hidden=payload.proposals_hidden)
    person = actor(payload.user)
    try:
        save_label_document(
            store.project_root, payload.image_path, label_path,
            [ap.model_dump() for ap in payload.annotations], width=w, height=h,
            author=person, actor=person, expect=expect, gestures=gestures)
    except VersionConflict as exc:
        raise HTTPException(409, {"error": "label file changed since it was loaded"}) from exc
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, saved()) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write labels: {exc}") from exc
    except (ValueError, UnreadableLabelDocument) as exc:
        raise HTTPException(400, str(exc)) from exc
    return saved()


@router.get("/proposals")
def load_proposals(image_path: str, bucket: str, label_path: Optional[str] = None) -> dict:
    """The proposals the published bucket at ``bucket`` offers for ``image_path``, in document
    order: each with its ``index``, the annotation of the label document at ``label_path`` it
    pairs with (``paired``, :func:`~tcip_mcp.dataset_layout.proposal_pairs`), the last
    ``decision`` its verdict shard records, and whether the bucket's assessment admits it
    (``admitted``, :func:`~tcip_mcp.delivery.admitted_conf`). A bucket that names no document for
    the image, or one that will not read, answers 400."""
    from tcip_annotation.verdicts import read_verdicts

    from tcip_mcp.delivery import admitted_conf
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs, verdict_key_of

    label = allowed_optional(label_path)
    try:
        published, proposals = image_proposals(allowed_path(bucket), image_path)
        annotations = _read(Path(label))[0].annotations if label else []
        decisions = {v.proposal: v.action for v in read_verdicts(verdict_key_of(
            store.project_root, image_path, published.path))}
        paired = proposal_pairs(store.project_root, published, annotations, proposals)
    except (ValueError, UnreadableLabelDocument) as exc:
        raise HTTPException(400, str(exc)) from exc
    conf, _reason = admitted_conf(store.open_root(), published)
    return {"bucket": str(published.path), "proposals": [
        {**annotation_dict(p), "index": i, "paired": paired.get(i), "decision": decisions.get(i),
         "admitted": conf is not None and p.score is not None and p.score >= conf}
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
