"""Annotation routes: an image's label document, its bucket proposals and the review queue.

The label document is read and saved; the proposals are those a prediction bucket offers for
the image, staged there by an engine the person prompts from the canvas or by the agent.

Reads the canonical per-image label document (every subject's annotations by name and their
completion marks) by the key of the image the request names; its one save is
:func:`~tcip_mcp.dataset_layout.save_label_document`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from tcip_annotation.json_io import (
    LabelDocument,
    UnreadableLabelDocumentError,
    authorship_of,
    client_annotation,
    read_document_versioned,
)
from tcip_annotation.state import Annotation
from tcip_mcp.pipelines.active_learning import ScorerSpec
from tcip_store import Key, Version, VersionConflictError
from tcip_web import jobstore
from tcip_web.paths import allowed_file, allowed_image, allowed_path
from tcip_web.state import store

router = APIRouter(prefix="/api/annotate", tags=["annotate"])
logger = logging.getLogger(__name__)


class AnnotationPayload(BaseModel):
    """One annotation's content: its subject, a geometry, its attribute values and its crowd flag.

    The ``subject`` is the object it is about; the geometry is ``bbox`` corners, ``points`` for
    one ring, ``rings`` for a multi-ring shape, ``point`` for a keypoint or landmark, or
    none of them for an image-level label; the attribute values are by name.

    Every field but the subject is carried uninterpreted.
    """

    subject: str
    bbox: Any = None
    points: Any = None
    rings: Any = None
    point: Any = None
    attributes: Any = None
    iscrowd: Any = None


class FlagPayload(BaseModel):
    """One flag a save raises: its comment and its target.

    The target is a ``point`` with the ``subject`` flagged there, a ``proposal`` (bucket and
    index), or neither for the image as a whole (:class:`~tcip_annotation.flags.FlagRequest`).
    """

    text: str
    point: tuple[float, float] | None = None
    subject: str | None = None
    proposal: tuple[str, int] | None = None


class SavePayload(BaseModel):
    """One save: the image's annotations and the gestures it adjudicates beside them.

    The gestures are :class:`~tcip_mcp.dataset_layout.Gestures`; the save is by the person
    ``user`` names.
    """

    image_path: str
    annotations: list[AnnotationPayload] = []
    # The label document's version token as the client loaded it: with one, the save is a
    # compare-and-set and a 409 says it changed underneath. Omit to skip the comparison.
    base_mtime: str | None = None
    user: str
    bucket: str | None = None
    accept: list[int] = []
    reject: list[int] = []
    # Each annotation the person signs off as their own call, by its position in ``annotations``.
    confirm: list[int] = []
    complete: dict[str, bool] = {}
    rect: tuple[float, float, float, float] | None = None
    proposals_hidden: bool = False
    flag: list[FlagPayload] = []
    # Each open flag resolved, by id, with the reply given.
    resolve: dict[str, str] = {}


def _flags(key: Key) -> list[dict]:
    """Return the image's flags, oldest first.

    Read through :func:`~tcip_annotation.flags.read_flags`; a record that will not read answers
    400.
    """
    from tcip_annotation.flags import encode_flag, flag_key, read_flags

    try:
        return [encode_flag(flag) for flag in read_flags(flag_key(key))]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def annotation_dict(a: Annotation) -> dict:
    """Return an annotation's client projection with authorship.

    The projection is the library's (:func:`~tcip_annotation.json_io.client_annotation`)
    plus its ``authorship`` (:func:`authorship_of`).
    """
    return {**client_annotation(a), "authorship": authorship_of(a)}


def _completion(doc: LabelDocument) -> dict:
    """Map each subject ``doc`` holds or marks to its state.

    The state is :meth:`~tcip_annotation.json_io.LabelDocument.state`'s.
    """
    return {
        subject: doc.state(subject)
        for subject in sorted(
            {a.subject for a in doc.annotations} | set(doc.marks)
        )
    }


def _admitted(image_path: str) -> tuple[Key, int, int]:
    """Admit the image at ``image_path`` once and return its label key and dimensions.

    The admission is :func:`~tcip_web.paths.allowed_image`'s and the key is its label
    document's (:func:`~tcip_mcp.dataset_layout.label_key_of`). An image under no dataset image
    tree answers 400.
    """
    from tcip_mcp.dataset_layout import label_key_of

    path, w, h = allowed_image(image_path)
    try:
        return label_key_of(path), w, h
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _read(key: Key) -> tuple[LabelDocument, Version]:
    """Read the label document ``key`` names and its version; one that will not read answers 400."""
    try:
        return read_document_versioned(key)
    except UnreadableLabelDocumentError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/labels")
def load_labels(image_path: str) -> dict:
    """Load an image's label document in pixel coords.

    It answers the annotations, each with its ``index`` in the document (the index a proposal's
    ``paired`` names), each subject's completion (:func:`_completion`), the image's flags
    (:func:`_flags`) and the version token a save echoes back, the empty token for an image with
    no document yet.
    """
    key, w, h = _admitted(image_path)
    doc, version = _read(key)
    return _labels_body(image_path, key, w, h, doc, version)


def _labels_body(
    image_path: str,
    key: Key,
    w: int,
    h: int,
    doc: LabelDocument,
    version: Version,
) -> dict:
    """Return the label document with its version token.

    The token is answered only together with the document it names.
    """
    return {
        "image_path": image_path,
        "img_width": w,
        "img_height": h,
        "annotations": [
            {**annotation_dict(a), "index": i}
            for i, a in enumerate(doc.annotations)
        ],
        "completion": _completion(doc),
        "flags": _flags(key),
        "base_mtime": version.token,
    }


@router.post("/labels")
def save_labels(payload: SavePayload) -> dict:
    """Save an image's label document and answer the saved document with its version.

    The save goes through :func:`~tcip_mcp.dataset_layout.save_label_document`, by the person
    :func:`~tcip_mcp.identity.actor` makes of ``user``. The answer is the document as a load
    answers it (:func:`_labels_body`) and ``accepted``, each accepted proposal's index mapped to
    the index of the annotation it resolved to, its audit line in the open project's scope. A
    stale ``base_mtime``, or no project open, answers 409 and a refusal 400, nothing written.
    """
    from tcip_annotation.flags import FlagRequest
    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tcip_mcp.identity import actor

    person = actor(payload.user)
    key, w, h = _admitted(payload.image_path)
    expect = (
        Version(payload.base_mtime) if payload.base_mtime is not None else None
    )
    for name, noun, indices in (
        ("accept", "a proposal", payload.accept),
        ("reject", "a proposal", payload.reject),
        ("confirm", "an annotation position", payload.confirm),
    ):
        if len(set(indices)) != len(indices):
            raise HTTPException(
                400, f"{name} names {noun} more than once: {indices}"
            )
    gestures = Gestures(
        bucket=payload.bucket,
        accept=frozenset(payload.accept),
        reject=frozenset(payload.reject),
        confirm=frozenset(payload.confirm),
        complete=payload.complete,
        rect=payload.rect,
        proposals_hidden=payload.proposals_hidden,
        flag=tuple(FlagRequest(**f.model_dump()) for f in payload.flag),
        resolve=payload.resolve,
    )
    try:
        version, doc, accepted = save_label_document(
            store.held().root,
            key,
            [ap.model_dump() for ap in payload.annotations],
            width=w,
            height=h,
            author=person,
            actor=person,
            expect=expect,
            gestures=gestures,
        )
    except VersionConflictError as exc:
        raise HTTPException(
            409, {"message": "label document changed since it was loaded"}
        ) from exc
    except (ValueError, UnreadableLabelDocumentError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "status": "ok",
        **_labels_body(payload.image_path, key, w, h, doc, version),
        "accepted": accepted,
    }


@router.get("/proposals")
def load_proposals(image_path: str, bucket: str) -> dict:
    """Load the proposals the bucket ``bucket`` offers for ``image_path``, in document order.

    The bucket is the one so named under the image's dataset root. Each proposal carries its
    ``index``, the index of the annotation of the image's label document it pairs with (``paired``,
    :func:`~tcip_mcp.dataset_layout.proposal_pairs`) and the last ``decision`` its verdict shard
    records; beside them the bucket's ``operating_point``, the ``conf`` a review may accept at on
    its assessment's authority or ``None`` with the ``reason``
    (:func:`~tcip_mcp.delivery.admitted_conf`), both read against the open project. A bucket that
    names no document for the image answers no proposals; one that will not read answers 400; no
    project open answers 409.
    """
    from tcip_annotation.verdicts import read_verdicts
    from tcip_mcp.dataset_layout import (
        image_proposals,
        proposal_pairs,
        verdict_key_of,
    )
    from tcip_mcp.delivery import admitted_conf

    root = store.held().root
    key, _w, _h = _admitted(image_path)
    annotations = _read(key)[0].annotations
    try:
        published, proposals = image_proposals(bucket, key)
        decisions = {
            v.proposal: v.action
            for v in read_verdicts(verdict_key_of(key, bucket))
        }
        paired = proposal_pairs(root, published, annotations, proposals)
    except (ValueError, UnreadableLabelDocumentError) as exc:
        raise HTTPException(400, str(exc)) from exc
    conf, reason = admitted_conf(root, published)
    return {
        "bucket": published.name,
        "operating_point": {"conf": conf, "reason": reason},
        "proposals": [
            {
                **annotation_dict(p),
                "index": i,
                "paired": paired.get(i),
                "decision": decisions.get(i),
            }
            for i, p in enumerate(proposals)
        ],
    }


class SegmenterPrompt(BaseModel):
    """The points the person placed and the boxes they drew, in the image's pixels.

    Each ``(x, y)`` point and each ``(x1, y1, x2, y2)`` box is a prompt of its own to the
    segmenter; the engine refuses a prompt stating neither.
    """

    model_config = ConfigDict(extra="forbid")

    points: list[tuple[float, float]] = []
    boxes: list[tuple[float, float, float, float]] = []


class PhrasePrompt(BaseModel):
    """A phrase describing what to find: a detector prompt."""

    model_config = ConfigDict(extra="forbid")

    phrase: str


class ProposePayload(BaseModel):
    """One prompt the person ``user`` names sends about an image, and its proposals' subject.

    The prompt is the placed points and drawn boxes, which the segmenter answers, or a phrase,
    which the detector answers under its libraries' own thresholds and merge
    (:func:`~tcip_mcp.pipelines.proposal_engines.detector_defaults`); one of the two, by the
    keys it carries.
    """

    image_path: str
    subject: str
    user: str
    prompt: SegmenterPrompt | PhrasePrompt


class ProposeAnswer(BaseModel):
    """What one prompt staged: the proposal bucket's name and how many proposals it holds.

    ``bucket`` is ``None`` and the count zero when the engine proposed nothing.
    """

    bucket: str | None
    proposal_count: int


def _door(result: dict) -> dict:
    """Return a door's ``result``; one carrying an ``error`` answers 400 with its text."""
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/propose")
def propose(payload: ProposePayload) -> ProposeAnswer:
    """Run the engine ``payload``'s prompt names on the image and stage every candidate it answers.

    The image is admitted by :func:`~tcip_web.paths.allowed_file`. The points and boxes prompt
    the segmenter for one best mask each, a phrase the detector, through
    :func:`~tcip_mcp.tools.proposal_tools.propose_annotations`, and every candidate of that run,
    named by the ``staging_id`` it answers, is staged under ``subject`` through
    :func:`~tcip_mcp.tools.proposal_tools.stage_proposals`, both on the open project and both by
    the person :func:`~tcip_mcp.identity.actor` makes of ``user``. Either door's error, and
    candidates the first door could not stage, answer 400 carrying its text; no project open
    answers 409.
    """
    from tcip_mcp.identity import actor
    from tcip_mcp.pipelines.proposal_engines import (
        DetectorParams,
        PromptableSegmenter,
        TextPromptedDetector,
        detector_defaults,
    )
    from tcip_mcp.tools.proposal_tools import (
        Assignment,
        StagedAssignments,
        propose_annotations,
        stage_proposals,
    )

    person = actor(payload.user)
    project = store.held().root
    image = str(allowed_file(payload.image_path))
    match payload.prompt:
        case SegmenterPrompt(points=points, boxes=boxes):
            engine = PromptableSegmenter.name
            params: dict = {
                "points": points,
                "boxes": boxes,
                "multimask_output": False,
            }
        case PhrasePrompt(phrase=phrase):
            engine = TextPromptedDetector.name
            params = DetectorParams(
                phrase=phrase, **detector_defaults()
            ).model_dump()
    proposed = _door(
        propose_annotations(
            project, image, engine, engine_params=params, actor=person
        )
    )
    if proposed["staging_id"] is None:
        if proposed["candidates"]:
            raise HTTPException(400, proposed["summary"])
        return ProposeAnswer(bucket=None, proposal_count=0)
    staged = _door(
        stage_proposals(
            project,
            image,
            assignments=StagedAssignments(
                staging_id=proposed["staging_id"],
                candidates=[
                    Assignment(candidate_id=c["id"], subject=payload.subject)
                    for c in proposed["candidates"]
                ],
            ),
            actor=person,
        )
    )
    return ProposeAnswer(
        bucket=staged["bucket"], proposal_count=staged["proposal_count"]
    )


class PriorityQueueJob(BaseModel):
    """A review-queue job's record.

    It carries what the job was launched under, its state, and once completed its ``queue``
    (``{image, score, reference_member?}``, highest first) with the candidate and skipped counts.
    """

    job_id: str
    project: str
    checkpoint_path: str
    images_dir: str
    subject: str | None
    scorer: ScorerSpec
    budget: int
    rationale: str
    status: jobstore.JobStatus
    error: str | None
    queue: list[dict]
    total_candidates: int
    reviewed_skipped: int

    @classmethod
    def launched(
        cls,
        *,
        job_id: str,
        project: str,
        checkpoint_path: str,
        images_dir: str,
        subject: str | None,
        scorer: ScorerSpec,
        budget: int,
        rationale: str,
    ) -> PriorityQueueJob:
        """Return the record of a job just launched under these values.

        It is pending, with no error, no queue and zero counts.
        """
        return cls(
            job_id=job_id,
            project=project,
            checkpoint_path=checkpoint_path,
            images_dir=images_dir,
            subject=subject,
            scorer=scorer,
            budget=budget,
            rationale=rationale,
            status="pending",
            error=None,
            queue=[],
            total_candidates=0,
            reviewed_skipped=0,
        )


_pq_registry: jobstore.JobRegistry[PriorityQueueJob] = jobstore.JobRegistry()
"""The review queue's own live jobs (``jobstore.JobRegistry``)."""


def _pq_worker(job: PriorityQueueJob) -> None:
    try:
        job.status = "running"
        from tcip_mcp.tools.feedback_tools import ranked_review_queue

        result = ranked_review_queue(
            Path(job.project),
            Path(job.checkpoint_path),
            Path(job.images_dir),
            scorer=job.scorer,
            budget=job.budget,
            rationale=job.rationale,
            subject=job.subject,
        )
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
    """A review-queue launch: the checkpoint, images directory, subject, scorer, budget, reason."""

    checkpoint_path: str
    images_dir: str
    subject: str | None = None
    scorer: ScorerSpec
    budget: int
    rationale: str


@router.post("/queue/launch")
def launch_priority_queue(
    payload: LaunchPriorityQueuePayload,
) -> PriorityQueueJob:
    """Launch the open project's review queue on a background thread and answer its record.

    The queue is :func:`~tcip_mcp.tools.feedback_tools.ranked_review_queue`'s; the record is the
    job's as it stands before the thread starts. A checkpoint or images directory outside the
    allowed roots is refused and a missing one answers 404.
    """
    checkpoint_path = allowed_path(payload.checkpoint_path)
    images_dir = allowed_path(payload.images_dir)
    if not checkpoint_path.is_file():
        raise HTTPException(
            404, f"checkpoint not found: {payload.checkpoint_path}"
        )
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")
    job = PriorityQueueJob.launched(
        job_id=f"pq-{uuid.uuid4().hex[:8]}",
        project=str(store.held().root),
        checkpoint_path=str(checkpoint_path),
        images_dir=str(images_dir),
        subject=payload.subject,
        scorer=payload.scorer,
        budget=payload.budget,
        rationale=payload.rationale,
    )
    _pq_registry.register(job)
    launched = job.model_copy(deep=True)
    threading.Thread(target=_pq_worker, args=(job,), daemon=True).start()
    return launched


@router.get("/queue/{job_id}")
def get_priority_queue_job(job_id: str) -> PriorityQueueJob:
    """Return the queue job ``job_id``'s record; an unknown job answers 404.

    The record carries what it was launched under (the scorer block, budget and reason among
    it), its state and, once completed, its ranked images.
    """
    job = _pq_registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return job
