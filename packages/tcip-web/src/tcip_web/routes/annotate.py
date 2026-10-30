"""Annotation routes: an image's label document, its bucket proposals and the review queue.

The label document is read and saved; the proposals are those a prediction bucket offers for
the image, staged there by an engine the person prompts from the canvas or by the agent.

Reads the canonical per-image label document (every subject's annotations by name and their
completion marks) by the key of the image the request names; its one save is
:func:`~tcip_mcp.dataset_layout.save_label_document`.
"""

from __future__ import annotations

import logging
import uuid
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, computed_field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tcip_annotation.flags import Flag, PointTarget, covers
from tcip_annotation.json_io import (
    Authorship,
    ClientAnnotation,
    LabelDocument,
    SubjectState,
    UnreadableLabelDocumentError,
    authorship_of,
    read_document_versioned,
)
from tcip_annotation.verdicts import VerdictAction
from tcip_mcp.dataset_layout import Gestures, list_subjects
from tcip_mcp.delivery import OperatingPoint
from tcip_mcp.pipelines.active_learning import (
    ScorerCatalog,
    ScorerSpec,
    scorer_catalog,
)
from tcip_mcp.pipelines.image_utils import logical_image_name
from tcip_mcp.tools.feedback_tools import ReviewQueue
from tcip_mcp.tools.proposal_tools import (
    Assignment,
    Staged,
    StagedAssignments,
    propose_annotations,
    stage_assignments,
)
from tcip_store import Key, Version, VersionConflictError
from tcip_web import jobstore
from tcip_web.paths import allowed_file, allowed_image, allowed_path
from tcip_web.state import selected, store

if TYPE_CHECKING:
    from tcip_annotation.state import Annotation

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


class SavePayload(BaseModel):
    """One save by the person ``user`` names: the image's annotations and the gestures beside them.

    The gestures are what the save decides beyond the annotations
    (:class:`~tcip_mcp.dataset_layout.Gestures`).
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    annotations: list[AnnotationPayload]
    # The label document's version token as the client loaded it: with one, the save is a
    # compare-and-set and a 409 says it changed underneath. Omit to skip the comparison.
    base_mtime: str | None = None
    user: str
    gestures: Gestures = Gestures()


class ServedAnnotation(ClientAnnotation):
    """An annotation of an image's label document or a bucket's, as the annotate routes serve it.

    It is the client projection with its ``index`` in its document, the index a proposal's
    ``paired`` names.
    """

    index: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def authorship(self) -> Authorship:
        """Who stands behind it (:func:`~tcip_annotation.json_io.authorship_of`)."""
        return authorship_of(self.created_by, self.accepted_by)


class ServedProposal(ServedAnnotation):
    """A bucket's proposal for an image, with what it pairs with and the last decision on it.

    ``paired`` is the index of the annotation of the image's label document it pairs with
    (:func:`~tcip_mcp.dataset_layout.proposal_pairs`), and ``decision`` the last its verdict
    shard records, each ``None`` when there is none.
    """

    paired: int | None
    decision: VerdictAction | None


@pydantic_dataclass(frozen=True)
class ServedFlag(Flag):
    """A flag of an image as the annotate routes serve it, with the annotations it is on.

    ``covered_by`` is the index of every annotation of the served document, in document order,
    that :func:`~tcip_annotation.flags.covers` the flag's point; empty for a flag on the image
    as a whole or on a proposal, or one whose point no annotation covers.
    """

    covered_by: list[int]

    @classmethod
    def of(cls, flag: Flag, annotations: list[Annotation]) -> ServedFlag:
        """Return ``flag`` served beside the document whose annotations are ``annotations``."""
        target = flag.request.target
        return cls(
            **vars(flag),
            covered_by=[
                i
                for i, a in enumerate(annotations)
                if isinstance(target, PointTarget)
                and covers(a, target.subject, target.point)
            ],
        )


class LabelDocumentAnswer(BaseModel):
    """An image's label document in pixel coordinates, with the version token naming it.

    It holds the image's frame, the document's annotations, the completion of each subject the
    dataset's registry declares or the document names
    (:meth:`~tcip_annotation.json_io.LabelDocument.state`), the image's flags oldest first, each
    with the annotations it is on, and the token a save echoes back, the empty token for an image
    with no document yet.
    """

    image_path: str
    img_width: int
    img_height: int
    annotations: list[ServedAnnotation]
    completion: dict[str, SubjectState]
    flags: list[ServedFlag]
    base_mtime: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """The image's name (:func:`~tcip_mcp.pipelines.image_utils.logical_image_name`)."""
        return logical_image_name(self.image_path)

    @classmethod
    def of(
        cls,
        image_path: str,
        key: Key,
        w: int,
        h: int,
        doc: LabelDocument,
        version: Version,
    ) -> LabelDocumentAnswer:
        """Return the document ``key`` names at ``version``.

        The image's flags are read through :func:`~tcip_annotation.flags.read_flags` and the
        registry's subjects through :func:`~tcip_mcp.dataset_layout.list_subjects`; either
        record that will not read answers 400.
        """
        from tcip_annotation.flags import flag_key, read_flags

        try:
            flags = read_flags(flag_key(key))
            declared = list_subjects(key.root)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return cls(
            image_path=image_path,
            img_width=w,
            img_height=h,
            annotations=[
                ServedAnnotation.of(a, index=i)
                for i, a in enumerate(doc.annotations)
            ],
            completion={
                subject: doc.state(subject)
                for subject in sorted(
                    set(declared)
                    | {a.subject for a in doc.annotations}
                    | set(doc.marks)
                )
            },
            flags=[ServedFlag.of(f, doc.annotations) for f in flags],
            base_mtime=version.token,
        )


class SavedLabels(BaseModel):
    """A save's answer: the document it wrote and what each accepted proposal resolved to.

    ``accepted`` maps each accepted proposal's index to the index of the annotation it resolved
    to.
    """

    document: LabelDocumentAnswer
    accepted: dict[int, int]


class VersionConflictRefusal(BaseModel):
    """A save refused because the label document changed since the client loaded it."""

    kind: Literal["version_conflict"] = "version_conflict"
    message: str


class ServedProposals(BaseModel):
    """The proposals a bucket offers for an image, in document order, with its operating point."""

    operating_point: OperatingPoint
    proposals: list[ServedProposal]


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
def load_labels(image_path: str) -> LabelDocumentAnswer:
    """Load an image's label document in pixel coords (:class:`LabelDocumentAnswer`)."""
    key, w, h = _admitted(image_path)
    doc, version = _read(key)
    return LabelDocumentAnswer.of(image_path, key, w, h, doc, version)


@router.post("/labels")
def save_labels(payload: SavePayload) -> SavedLabels:
    """Save an image's label document and answer the saved document with its version.

    The save goes through :func:`~tcip_mcp.dataset_layout.save_label_document`, by the person
    :func:`~tcip_mcp.identity.actor` makes of ``user``, under the request's gestures, its audit
    line in the open project's scope. The answer is :class:`SavedLabels`. A stale ``base_mtime``,
    or no project open, answers 409 and a refusal 400, nothing written.
    """
    from tcip_mcp.dataset_layout import save_label_document
    from tcip_mcp.identity import actor

    person = actor(payload.user)
    key, w, h = _admitted(payload.image_path)
    expect = (
        Version(payload.base_mtime) if payload.base_mtime is not None else None
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
            gestures=payload.gestures,
        )
    except VersionConflictError as exc:
        raise HTTPException(
            409,
            VersionConflictRefusal(
                message="label document changed since it was loaded"
            ).model_dump(),
        ) from exc
    except (ValueError, UnreadableLabelDocumentError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return SavedLabels(
        document=LabelDocumentAnswer.of(
            payload.image_path, key, w, h, doc, version
        ),
        accepted=accepted,
    )


@router.get("/proposals")
def load_proposals(image_path: str, bucket: str) -> ServedProposals:
    """Load the proposals the bucket ``bucket`` offers for ``image_path``, in document order.

    The bucket is the one so named under the image's dataset root. Each proposal is a
    :class:`ServedProposal`; beside them the bucket's operating point, the ``conf`` a review may
    accept at on its assessment's authority or ``None`` with the ``reason``
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
    return ServedProposals(
        operating_point=admitted_conf(root, published),
        proposals=[
            ServedProposal.of(
                p, index=i, paired=paired.get(i), decision=decisions.get(i)
            )
            for i, p in enumerate(proposals)
        ],
    )


class ImageStagings(BaseModel):
    """The buckets staged by an engine or the agent that hold a document for one image."""

    buckets: list[str]


@router.get("/stagings")
def image_stagings(image_path: str) -> ImageStagings:
    """List the staged proposal buckets holding a document for ``image_path``, in publication order.

    A staged bucket is one a proposal engine or the agent published
    (:class:`~tcip_mcp.buckets.Proposed`); a model's prediction bucket is never listed. The
    buckets are read under the image's dataset root; no project open answers 409.
    """
    from tcip_mcp.buckets import Proposed, buckets_under

    store.held()
    key, _w, _h = _admitted(image_path)
    stem = key.parts[-1]
    return ImageStagings(
        buckets=[
            b.name
            for b in buckets_under(key.root)
            if isinstance(b.producer, Proposed) and stem in b.documents
        ]
    )


class ImageStates(BaseModel):
    """Each image of the selected capture, by name, with what its document says of a subject."""

    states: dict[str, SubjectState]


@router.get("/states")
def image_states(subject: str) -> ImageStates:
    """Read what every image of the selected capture's label document says about ``subject``.

    The state is :meth:`~tcip_annotation.json_io.LabelDocument.state`'s, keyed by the image's
    name in the selection's list. No project open answers 409; no capture selected
    (:func:`~tcip_web.state.selected`) answers 400; a document that will not read answers 400.
    """
    from tcip_mcp.dataset_layout import label_key_of

    states: dict[str, SubjectState] = {}
    for image in selected(store.held_state()[1]).images:
        try:
            key = label_key_of(image.path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        states[image.name] = _read(key)[0].state(subject)
    return ImageStates(states=states)


class SegmenterPrompt(BaseModel):
    """The points the person placed and the boxes they drew, in the image's pixels.

    Each ``(x, y)`` point and each ``(x1, y1, x2, y2)`` box is a prompt of its own to the
    segmenter; the engine refuses a prompt stating neither.
    """

    model_config = ConfigDict(extra="forbid")

    points: list[tuple[float, float]] = []
    boxes: list[tuple[float, float, float, float]] = []


class ProposePayload(BaseModel):
    """One prompt the person ``user`` names sends about an image, and its proposals' subject.

    The prompt is the placed points and drawn boxes, which the segmenter answers.
    """

    image_path: str
    subject: str
    user: str
    prompt: SegmenterPrompt


class FindPayload(BaseModel):
    """A phrase the person ``user`` names describes what to find on an image."""

    image_path: str
    phrase: str
    user: str


class FindAnswer(BaseModel):
    """The boxes the detector found for a phrase, ``(x1, y1, x2, y2)`` in the image's pixels.

    They are the person's to keep, drop or add to before a prompt sends them to the segmenter.
    """

    boxes: list[tuple[float, float, float, float]]


def _door[Answer: BaseModel | None](result: Answer | dict) -> Answer:
    """Return a door's answer; its refusal, a dict naming an ``error``, answers 400 with it."""
    if isinstance(result, dict):
        raise HTTPException(400, result["error"])
    return result


@router.post("/propose")
def propose(payload: ProposePayload) -> Staged | None:
    """Run the segmenter on the image's prompts and stage every candidate it answers.

    The image is admitted by :func:`~tcip_web.paths.allowed_file`. The points and boxes prompt
    the segmenter for one best mask each through
    :func:`~tcip_mcp.tools.proposal_tools.propose_annotations`, and every candidate of that run,
    named by the ``staging_id`` it answers, is staged under ``subject`` through
    :func:`~tcip_mcp.tools.proposal_tools.stage_assignments`, both on the open project and both by
    the person :func:`~tcip_mcp.identity.actor` makes of ``user``; the answer is that staging, or
    ``None`` when the engine proposed nothing. Either door's error answers 400 carrying its text;
    no project open answers 409.
    """
    from tcip_mcp.identity import actor
    from tcip_mcp.pipelines.proposal_engines import PromptableSegmenter

    person = actor(payload.user)
    project = store.held().root
    image = str(allowed_file(payload.image_path))
    proposed = _door(
        propose_annotations(
            project,
            image,
            PromptableSegmenter.name,
            engine_params={
                "points": payload.prompt.points,
                "boxes": payload.prompt.boxes,
                "multimask_output": False,
            },
            actor=person,
        )
    )
    if proposed is None:
        return None
    return _door(
        stage_assignments(
            project,
            image,
            StagedAssignments(
                staging_id=proposed.staging_id,
                candidates=[
                    Assignment(candidate_id=c.id, subject=payload.subject)
                    for c in proposed.candidates
                ],
            ),
            actor=person,
        )
    )


@router.post("/find")
def find(payload: FindPayload) -> FindAnswer:
    """Run the detector on the image for the phrase and answer the boxes it found, unstaged.

    The image is admitted by :func:`~tcip_web.paths.allowed_file`. The phrase prompts the
    detector under its libraries' own thresholds and merge
    (:func:`~tcip_mcp.pipelines.proposal_engines.detector_defaults`) through
    :func:`~tcip_mcp.tools.proposal_tools.propose_annotations`, on the open project and by the
    person :func:`~tcip_mcp.identity.actor` makes of ``user``; nothing is staged, the boxes being
    prompts for the person to send on. The door's error answers 400 carrying its text; no project
    open answers 409.
    """
    from tcip_mcp.identity import actor
    from tcip_mcp.pipelines.proposal_engines import (
        DetectorParams,
        TextPromptedDetector,
        detector_defaults,
    )

    person = actor(payload.user)
    project = store.held().root
    image = str(allowed_file(payload.image_path))
    proposed = _door(
        propose_annotations(
            project,
            image,
            TextPromptedDetector.name,
            engine_params=DetectorParams(
                phrase=payload.phrase, **detector_defaults()
            ).model_dump(),
            actor=person,
        )
    )
    return FindAnswer(
        boxes=[] if proposed is None else [c.bbox for c in proposed.candidates]
    )


class LaunchPriorityQueuePayload(BaseModel):
    """A review-queue launch over the selected capture: the checkpoint, scorer, budget, reason."""

    checkpoint_path: str
    scorer: ScorerSpec
    budget: int
    rationale: str


@router.get("/scorers")
def list_scorers() -> ScorerCatalog:
    """Return the scorers a review queue may name, as ``scorer_catalog`` lists them."""
    return scorer_catalog()


class PriorityQueueJob(jobstore.ProjectJob):
    """A review-queue job's record.

    It carries the checkpoint it ranks under, the capture it ranks and its subject, its state,
    and once completed the queue the ranking answered (``ranked``, which states the scorer block,
    budget and reason it ranked under).
    """

    kind: ClassVar[str] = "review priority-queue scoring pass"
    subject: str | None
    ranked: ReviewQueue | None = None


_pq_registry: jobstore.JobRegistry[PriorityQueueJob] = jobstore.JobRegistry()
"""The review queue's own live jobs (``jobstore.JobRegistry``)."""


def _pq_worker(
    job: PriorityQueueJob, payload: LaunchPriorityQueuePayload
) -> None:
    try:
        job.status = "running"
        from tcip_mcp.tools.feedback_tools import ranked_review_queue

        result = ranked_review_queue(
            Path(job.project),
            Path(job.checkpoint_path),
            Path(job.images_dir),
            scorer=payload.scorer,
            budget=payload.budget,
            rationale=payload.rationale,
            subject=job.subject,
        )
        if isinstance(result, ReviewQueue):
            job.status, job.ranked = "completed", result
        else:
            job.status, job.error = "failed", result["error"]
    except Exception as exc:
        logger.exception("priority-queue job %s failed", job.job_id)
        job.status, job.error = "failed", str(exc)


@router.post("/queue/launch")
def launch_priority_queue(
    payload: LaunchPriorityQueuePayload,
) -> PriorityQueueJob:
    """Launch the open project's review queue on a background thread and answer its record.

    The queue is :func:`~tcip_mcp.tools.feedback_tools.ranked_review_queue`'s over the selected
    capture's image directory (:func:`~tcip_mcp.dataset_layout.image_dir`) and its subject
    (:func:`~tcip_web.state.selected`, no capture selected answering 400); the record is the
    job's as it stands before the thread starts. A checkpoint outside the allowed roots is
    refused and a missing one answers 404.
    """
    from tcip_mcp.dataset_layout import image_dir

    project, held = store.held_state()
    selection = selected(held)
    checkpoint_path = allowed_path(payload.checkpoint_path)
    if not checkpoint_path.is_file():
        raise HTTPException(
            404, f"checkpoint not found: {payload.checkpoint_path}"
        )
    return _pq_registry.launch(
        PriorityQueueJob(
            job_id=f"pq-{uuid.uuid4().hex[:8]}",
            project=str(project.root),
            checkpoint_path=str(checkpoint_path),
            images_dir=str(image_dir(selection.dataset_root, selection.date)),
            subject=selection.subject,
        ),
        partial(_pq_worker, payload=payload),
    )


@router.get("/queue/{job_id}")
def get_priority_queue_job(job_id: str) -> PriorityQueueJob:
    """Return the queue job ``job_id``'s record; an unknown job answers 404.

    The record carries what it ranks, its state and, once completed, the ranked queue.
    """
    job = _pq_registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return job
