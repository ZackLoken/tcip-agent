"""Review routes: verdict/GT recording (compute matches, walk detections, record actions, save GT)
plus the image-status group (mark_complete, backup_labels, image_statuses, generation_conf) and the
priority queue.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance lives in memory per
dataset (keyed by dataset_root). Review state is persisted via the engine to per-image shards under
``<dataset_root>/.tcip/state/review/``.

Ground truth and predictions are each one JSON file per image holding every subject's annotations
by name (a prediction is an :class:`~tcip_annotation.state.Annotation` whose ``score`` is set); a
class is named by its ``subject``, never an integer id, so the recorded verdict carries the real
subject name and a resolved ``class_id``: the producing bucket's own recorded name->id map, read
once at record time (``_verdict_class_id``).
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation import (
    BBox,
    ReviewContext,
    ReviewDetection,
    ReviewEngine,
    compute_classified_trait_matches,
    compute_matches,
)
from tcip_annotation.json_io import (
    UnreadableLabelDocument, annotation_from_payload, check_box_extent,
    prediction_documents, read_annotations, read_predictions,
)
from tcip_annotation.matching import REVIEW_CONF_FLOOR
from tcip_annotation.review_engine import capture_label_baseline
from tcip_annotation.state import Annotation, prediction_score
from tcip_annotation.verdicts import (
    ACCEPTED_ACTION, EDITED_ACTION, REJECTED_ACTION, SWEPT_ACTION, VerdictAction, decode_verdict,
)
from tcip_mcp.buckets import Bucket
from tcip_mcp.dataset_layout import annotations_hold_subject, derive_status
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_web import jobstore
from tcip_web.identity import resolve_user, user_id
from tcip_web.label_annotations_cache import cached_label_annotations
from tcip_web.paths import allowed_image_dimensions, allowed_optional, allowed_path
from tcip_web.routes.annotate import annotation_dict
from tcip_web.state import store

router = APIRouter(prefix="/api/review", tags=["review"])
logger = logging.getLogger(__name__)


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _current_user() -> str:
    """Reviewer fallback when the GUI request omits ``user``: env override else the OS login."""
    from tcip_web.identity import current_user

    return current_user()


def _get_engine(dataset_root: str) -> ReviewEngine:
    """The review engine anchored on a client-supplied dataset root, confined first (403)."""
    from tcip_mcp.project_paths import project_state_dir

    key = str(allowed_path(dataset_root))
    if key not in _engines:
        _engines[key] = ReviewEngine(state_dir=project_state_dir(key), current_user=_current_user())
    return _engines[key]


def _audit(scope: str, tool: str, arguments: dict) -> None:
    """Record a GUI review mutation in the audit log of the dataset root the request states,
    confined before the append. A failed append raises ``AuditEntryNotWritten``.
    """
    from tcip_web.routes.audit_gap import record_committed

    record_committed(tool, arguments, scope=str(allowed_path(scope)))


def _published(pred_dir: Optional[str]) -> Optional[Bucket]:
    """The published bucket at ``pred_dir`` (:func:`~tcip_mcp.buckets.read_bucket`), or ``None``
    for no directory."""
    from tcip_mcp.buckets import read_bucket

    return read_bucket(pred_dir) if pred_dir else None


def _producer_identity(bucket: Optional[Bucket]) -> Optional[dict]:
    """The producing model's identity a verdict records: the bucket's ``checkpoint_sha256`` and
    ``experiment_id``, beside its directory. ``None`` for no published bucket."""
    return None if bucket is None else {**bucket.producer, "bucket_dir": str(bucket.path)}


def _bucket_of_dir(pred_dir: Optional[str]) -> str:
    """The verdict store's key for the prediction bucket dir this request names. No dir is a review
    with no bucket at all.
    """
    from tcip_mcp.buckets import bucket_key_of

    return bucket_key_of(pred_dir)


def _bucket_of_file(pred_path: Optional[str]) -> str:
    """Same, from a per-image prediction file path: the bucket dir is its parent."""
    return _bucket_of_dir(str(Path(pred_path).parent) if pred_path else None)


def _verdict_class_id(scope: ClassScope, class_name: str) -> Optional[int]:
    """The 0-indexed class identity ``class_name`` resolves to under the review scope's map, the
    producing bucket's own when it has a stamp, resolved at verdict-record time. ``None`` for a
    scope with no map or a ``class_name`` the map does not carry; never defaults to 0.
    """
    return (scope.id_map or {}).get(class_name)


def _ensure_original_backup(label_path: Optional[str]) -> None:
    """Capture one label file's pristine bytes before its first mutation, if none is held yet.

    Same baseline record and create-only capture as :meth:`ReviewEngine.backup_original_labels`,
    per file. New GT files a verdict is creating have no original to preserve, so they are skipped,
    and an already-held baseline is kept.
    """
    if not label_path:
        return
    if not Path(label_path).is_file():
        return
    capture_label_baseline(label_path)


def _review_scope(
    bucket: Optional[Bucket], stated_subject: Optional[str], stated_attribute: Optional[str],
) -> ClassScope:
    """The class space this review reads under (:func:`~tcip_mcp.buckets.input_scope` over the
    prediction file's bucket); each of its refusals answers 400.
    """
    from tcip_mcp.buckets import input_scope

    try:
        return input_scope(bucket, stated_subject, stated_attribute)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _bucket_of_prediction(pred_path: Optional[str]) -> Optional[Bucket]:
    """The published bucket a per-image prediction file sits in (:func:`_published`)."""
    return _published(str(Path(pred_path).parent) if pred_path else None)


def _compute_matches(
    gt: list, preds: list, *, iou_threshold: float, conf_threshold: float, scope: ClassScope,
) -> dict:
    """Dispatch to plain detection matching, or classified-trait matching under a classified
    ``scope``, held to its map's names. A record the matching refuses answers 400.
    """
    try:
        if not scope.classified:
            return compute_matches(gt, preds, iou_threshold, conf_threshold)
        return compute_classified_trait_matches(
            gt, preds, subject=cast(str, scope.subject), attribute=cast(str, scope.attribute),
            vocabulary=set(scope.id_map or {}),
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _read_annotations_or_400(read, path: str) -> list:
    """``read(path)``, refused (400) naming the file when the document will not read."""
    try:
        return read(path)
    except UnreadableLabelDocument as exc:
        raise HTTPException(400, str(exc)) from exc


def _load_ctx(image_name: str, image_path: str, *, gt_path: Optional[str],
              pred_path: Optional[str]) -> ReviewContext:
    w, h = allowed_image_dimensions(image_path)
    ctx = ReviewContext(img_name=image_name, img_width=w, img_height=h)
    gt_path = allowed_optional(gt_path)
    pred_path = allowed_optional(pred_path)
    if gt_path:
        ctx.gt = _read_annotations_or_400(read_annotations, gt_path)
    if pred_path:
        ctx.preds = _read_annotations_or_400(read_predictions, pred_path)
    return ctx


# ── Request/response schemas ──────────────────────────────────────────────


class MatchesRequest(BaseModel):
    dataset_root: str
    image_name: str
    image_path: str
    gt_path: Optional[str] = None      # the per-image ground-truth label file
    pred_path: Optional[str] = None    # the per-image prediction file
    iou_threshold: float = 0.5
    conf_threshold: float = REVIEW_CONF_FLOOR
    filter_type: str = "all"
    filter_class: str = "all"          # a class name (an annotation's subject) or "all"
    # A fallback only: a classified bucket's own scope (`_review_scope`) always governs pred_path,
    # whether or not these are given, and a stated pair that disagrees with it refuses.
    subject: Optional[str] = None
    attribute: Optional[str] = None


class Detection(BaseModel):
    det_type: str
    class_name: str
    conf: Optional[float]
    iou: Optional[float]
    gt_idx: Optional[int]
    pred_idx: Optional[int]
    bbox: tuple[float, float, float, float]
    reviewed: bool = False
    reviewed_action: Optional[VerdictAction] = None


class MatchesResponse(BaseModel):
    img_width: int
    img_height: int
    n_tp: int
    n_fp: int
    n_fn: int
    detections: list[Detection]
    gt: list[dict]      # every GT annotation (subject + geometry + attributes + provenance)
    preds: list[dict]   # every prediction annotation (carries score)
    image_status: str   # "not_started" | "started" | "completed"
    n_reviewed: int      # current detections with a stored verdict, review_progress's own count
    n_total: int         # current detections, the same count's denominator
    # The resolved review scope (`_review_scope`): both None means a bare directory, nothing else.
    subject: Optional[str] = None
    attribute: Optional[str] = None


def _matches_response(
    ctx: ReviewContext,
    matches: dict,
    engine: ReviewEngine,
    image_name: str,
    *,
    bucket: str,
    filter_type: str,
    filter_class: str,
    scope: ClassScope,
) -> MatchesResponse:
    """Build the canvas payload (filtered + review-decorated detections, GT/pred annotations,
    status) from an already-computed match set.

    ``n_reviewed``/``n_total`` come from :meth:`ReviewEngine.review_progress` over the unfiltered
    ``matches``, before ``filter_type``/``filter_class`` narrow ``detections`` to what is rendered.
    """
    # Built once for the whole image: the wheel's progress is over the unfiltered set, so a filter
    # that narrows the rendered list below must not narrow what review_progress counts.
    all_dets = engine.build_detection_list(ctx, matches)
    if filter_type == "all" and filter_class == "all":
        dets = all_dets
    else:
        dets = engine.build_detection_list(
            ctx, matches, filter_type=filter_type, filter_class=filter_class
        )
    out_dets: list[Detection] = []
    for d in dets:
        entry = engine.find_reviewed_entry(bucket, d, ctx)
        out_dets.append(Detection(
            det_type=d.det_type,
            class_name=d.class_name,
            conf=d.conf,
            iou=d.iou,
            gt_idx=d.gt_idx,
            pred_idx=d.pred_idx,
            bbox=d.bbox,
            reviewed=entry is not None,
            reviewed_action=decode_verdict(entry).action if entry else None,
        ))

    n_reviewed, n_total = engine.review_progress(bucket, ctx, all_dets)
    return MatchesResponse(
        img_width=ctx.img_width,
        img_height=ctx.img_height,
        n_tp=len(matches["tp"]),
        n_fp=len(matches["fp"]),
        n_fn=len(matches["fn"]),
        detections=out_dets,
        gt=[annotation_dict(a) for a in ctx.gt],
        preds=[annotation_dict(a) for a in ctx.preds],
        image_status=engine.get_image_review_status(bucket, image_name),
        n_reviewed=n_reviewed,
        n_total=n_total,
        subject=scope.subject,
        attribute=scope.attribute,
    )


@router.post("/matches")
def compute_image_matches(req: MatchesRequest) -> MatchesResponse:
    """Compute TP/FP/FN, decorate with review status, and return everything the canvas needs."""
    ctx = _load_ctx(req.image_name, req.image_path, gt_path=req.gt_path, pred_path=req.pred_path)
    engine = _get_engine(req.dataset_root)
    scope = _review_scope(_bucket_of_prediction(req.pred_path), req.subject, req.attribute)
    matches = _compute_matches(
        ctx.gt, ctx.preds, iou_threshold=req.iou_threshold, conf_threshold=req.conf_threshold,
        scope=scope)
    return _matches_response(
        ctx, matches, engine, req.image_name, bucket=_bucket_of_file(req.pred_path),
        filter_type=req.filter_type, filter_class=req.filter_class, scope=scope,
    )


class ActionPayload(BaseModel):
    dataset_root: str
    image_name: str
    image_path: str
    gt_path: Optional[str] = None
    pred_path: Optional[str] = None
    # The detection being acted on (same shape as the Detection response)
    det_type: str
    class_name: str
    conf: Optional[float] = None
    iou: Optional[float] = None
    gt_idx: Optional[int] = None
    pred_idx: Optional[int] = None
    bbox: tuple[float, float, float, float]
    # "swept" is an explicit "checked this image for missed objects, found none" attestation: no
    # geometry, never mutates GT, see _apply_gt_mutation.
    action: VerdictAction
    # GUI-set reviewer identity (bare name, e.g. "breeder"); stamped as accepted_by/created_by
    # ("user:<name>"). Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None
    # Edited shape committed from the Review canvas (only for action="edited"): a box, or a
    # polygon's points. Accept/Reject don't carry these: they act on the loaded pred/gt by index.
    # Carried uninterpreted: the save conversion (annotation_from_payload) is their one reading.
    edited_box: Any = None
    edited_points: Any = None
    # Review thresholds so the route can decide (at the same op point as the GUI) whether
    # this verdict was the last one and the image should flip to 'completed'.
    iou_threshold: float = 0.5
    conf_threshold: float = REVIEW_CONF_FLOOR
    # Active review filters, so the fresh matches this route returns are scoped identically to
    # what /matches would have returned (the client installs them without a second fetch).
    filter_type: str = "all"
    filter_class: str = "all"
    # Same (subject, attribute) meaning as MatchesRequest: set together when this verdict is on a
    # classified trait rather than plain detection.
    subject: Optional[str] = None
    attribute: Optional[str] = None
    # A claim, not a fact: the route verifies it against the bucket's assessment and refuses by name.
    rule_admitted: bool = False


def _names_prediction(payload: "ActionPayload", ctx: ReviewContext) -> bool:
    """Whether ``payload.pred_idx`` names a prediction of this image's loaded document."""
    return payload.pred_idx is not None and 0 <= payload.pred_idx < len(ctx.preds)


def _is_reviewer_drawn_new_shape(payload: "ActionPayload") -> bool:
    """A reviewer drew a brand-new shape from scratch: no matched GT, no matched prediction, and
    not a sweep attestation.
    """
    return payload.gt_idx is None and payload.pred_idx is None and payload.action != SWEPT_ACTION


def _check_classified_value(class_name: str, scope: ClassScope) -> None:
    """Refuse a value a classified bucket's own ``id_map`` does not declare."""
    vocabulary = set(scope.id_map or {})
    if class_name not in vocabulary:
        raise ValueError(
            f"{class_name!r} is not a value this bucket's own id_map declares "
            f"({sorted(vocabulary)}): a reviewer can only confirm a value the vocabulary has."
        )


def _check_target_subject(existing: Annotation, scope) -> None:
    """Refuse editing a record of a subject other than the classified scope's own object class."""
    if existing.subject != scope.subject:
        raise ValueError(
            f"this record is of subject {existing.subject!r}, not {scope.subject!r}, this "
            "bucket's own object class: a classified review only edits records of its own class."
        )


def _apply_gt_mutation(
    ctx: ReviewContext, payload: "ActionPayload", reviewer: str, now_iso: str, *, scope,
    accepted_by_rule: Optional[str] = None,
) -> tuple[bool, Optional[int]]:
    """Author GT from a verdict; return ``(gt_changed, index the written annotation landed at in
    ctx.gt)``: the index is set only for edited/accepted writes. ``action="swept"`` (an explicit
    "checked this image for missed objects, found none" attestation) matches none of the branches
    below and always no-ops.

    ``scope`` is the resolved review scope (``_review_scope``), never
    ``payload.subject``/``payload.attribute`` directly. Under a classified scope a verdict judges
    the value of an object a person already placed, checking a written ``payload.class_name``
    against the scope's own map and an edited record's own subject against ``scope.subject``,
    refusing by name; under a detector review it judges the object's presence.

    Accept on a false positive: a paired one (``payload.gt_idx`` set, its partner a ground-truth
    record of the subject whose value differs) replaces the confirmed value on the person's own
    record, keeping their geometry and authorship; an unpaired one appends a fresh GT record from
    the prediction, its origin traveling with it. Under a detector review, accept always appends
    with empty ``attributes``. ``accepted_by_rule`` is written only there; the unpaired classified
    accept and both edit branches write ``None`` explicitly, and the paired classified accept keeps
    the record's own value. Reject on a false positive leaves ground truth untouched under either
    regime. Reject on a true positive or false negative under a classified scope refuses.

    Edit authors the edited geometry onto the record it edits (a true positive/false negative, or a
    paired false positive) with the reviewer as author, keeping the record's other attribute values
    and dropping any sign-off and rule marker; a stated ``gt_idx`` out of range refuses. An
    unpaired false positive edited into ground truth is a fresh record, no score and no sign-off. A
    reviewer-drawn new shape (:func:`_is_reviewer_drawn_new_shape`) refuses under a classified
    scope.
    """
    dt, act = payload.det_type, payload.action
    classifying = scope.classified

    if classifying and _is_reviewer_drawn_new_shape(payload):
        raise ValueError(
            "a reviewer-drawn new shape cannot be authored under a classified-trait review: it "
            "would fabricate a state nobody assessed. Add the object through the Annotate tab, "
            "then review its value here."
        )

    if act == EDITED_ACTION:
        # The edited box or the one contour the reviewer drew, through the save routes' own
        # conversion, so an edit is checked exactly as a saved shape is.
        geom = annotation_from_payload(
            {"subject": payload.class_name, "points": payload.edited_points,
             "bbox": payload.edited_box}, author=None, now=now_iso).geometry
        if geom is None:
            return False, None
        # An existing record is edited only for a tp/fn or a paired fp; any other gt_idx-carrying
        # det_type falls to the fresh-record path below, matching this branch's own design.
        edits_existing = payload.gt_idx is not None and (dt in ("tp", "fn")
                                                         or (classifying and dt == "fp"))
        if edits_existing:
            assert payload.gt_idx is not None
            if not (0 <= payload.gt_idx < len(ctx.gt)):
                raise ValueError(
                    f"gt_idx {payload.gt_idx} is out of range for this image's {len(ctx.gt)} "
                    "ground-truth record(s): the record this edit targets no longer exists."
                )
            existing = ctx.gt[payload.gt_idx]
            if classifying:
                _check_target_subject(existing, scope)
                _check_classified_value(payload.class_name, scope)
            attrs = dict(existing.attributes)
            if classifying:
                attrs[scope.attribute] = payload.class_name
            ctx.gt[payload.gt_idx] = replace(
                existing, geometry=geom, attributes=attrs, created_by=reviewer,
                created_at=now_iso, accepted_by=None, accepted_at=None, accepted_by_rule=None)
            return True, payload.gt_idx
        # An unpaired false positive edited into ground truth: a fresh record.
        if classifying:
            _check_classified_value(payload.class_name, scope)
        reviewed = {scope.attribute: payload.class_name} if classifying else {}
        new_subject = scope.subject if classifying else payload.class_name
        ctx.gt.append(Annotation(subject=new_subject, geometry=geom, attributes=reviewed,
                                 created_by=reviewer, created_at=now_iso, accepted_by_rule=None))
        return True, len(ctx.gt) - 1

    if act == REJECTED_ACTION and dt in ("tp", "fn"):
        if classifying:
            raise ValueError(
                "reject on a true positive or false negative is a detector-scope act: it removes "
                "the object, and a classified-trait review never adjudicated whether the object "
                "is present. Review this bucket under its object class, or remove the object "
                "through the Annotate tab."
            )
        if payload.gt_idx is not None and 0 <= payload.gt_idx < len(ctx.gt):
            ctx.gt.pop(payload.gt_idx)
            return True, None
        return False, None

    if act == ACCEPTED_ACTION and dt == "fp" and _names_prediction(payload, ctx):
        assert payload.pred_idx is not None  # _names_prediction's own guard
        pred = ctx.preds[payload.pred_idx]
        if isinstance(pred.geometry, BBox):
            check_box_extent(pred.geometry, where=f"accepting {payload.class_name!r}")
        if classifying and payload.gt_idx is not None and 0 <= payload.gt_idx < len(ctx.gt):
            # A paired false positive: the person's object, a wrong value confirmed. Keep
            # their geometry and authorship; replace only the confirmed value.
            existing = ctx.gt[payload.gt_idx]
            _check_target_subject(existing, scope)
            _check_classified_value(payload.class_name, scope)
            attrs = dict(existing.attributes)
            attrs[scope.attribute] = payload.class_name
            ctx.gt[payload.gt_idx] = replace(
                existing, attributes=attrs, accepted_by=reviewer, accepted_at=now_iso)
            return True, payload.gt_idx
        if classifying:
            _check_classified_value(payload.class_name, scope)
            accepted = replace(pred, score=None, attributes={scope.attribute: payload.class_name},
                               accepted_by=reviewer, accepted_at=now_iso, accepted_by_rule=None)
        else:
            # The one arm a verified rule-admitted claim can reach: a classified scope refuses one.
            accepted = replace(pred, score=None, attributes={},
                               accepted_by=reviewer, accepted_at=now_iso,
                               accepted_by_rule=accepted_by_rule)
        ctx.gt.append(accepted)
        return True, len(ctx.gt) - 1

    return False, None  # accept TP/FN and reject FP leave GT untouched


def _verify_rule_admitted_claim(
    payload: "ActionPayload", ctx: ReviewContext, bucket: Optional[Bucket], scope: ClassScope,
) -> str:
    """Verify a client's ``rule_admitted`` claim against the prediction's own ``bucket`` and answer
    the id of the assessment whose conf admitted it (:func:`_admission`), or refuse by name.

    Refuses 400 for a condition the claim itself fails (wrong action or det_type, no named
    prediction, a classified scope, a below-conf prediction, no admitting assessment), and
    409 when a fresh recompute over the pristine, unmutated ``ctx`` no longer holds the submitted
    detection. The claim is the client's; the identity is the bucket record's own, never one the
    client supplied.
    """
    if payload.action != ACCEPTED_ACTION:
        raise HTTPException(
            400, f"rule_admitted refuses action {payload.action!r}: only an accept can be rule-admitted")
    if payload.det_type not in ("tp", "fp"):
        raise HTTPException(
            400, f"rule_admitted refuses det_type {payload.det_type!r}: only tp or fp can be rule-admitted")
    if not _names_prediction(payload, ctx):
        raise HTTPException(
            400, "rule_admitted needs pred_idx naming a prediction of this image's loaded document")
    if scope.classified:
        raise HTTPException(
            400, "rule_admitted refuses a classified scope: the count operating point admits "
                 "detections of the object class, and a classified review judges values")
    assert payload.pred_idx is not None  # _names_prediction's own guard
    pred = ctx.preds[payload.pred_idx]
    conf, reason = _admission(bucket)
    if conf is None:
        raise HTTPException(400, reason)
    assert bucket is not None and bucket.assessment_id is not None, "an admitting bucket"
    if prediction_score(pred) < conf:
        raise HTTPException(
            400, f"rule_admitted refuses a prediction scored {pred.score}, below the rule's own "
                 f"conf {conf}")
    pristine_matches = _compute_matches(
        ctx.gt, ctx.preds, iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold,
        scope=scope,
    )
    if not any(d.get("pred_idx") == payload.pred_idx for d in pristine_matches[payload.det_type]):
        raise HTTPException(
            409, "this image's matches changed since they were loaded; reload before confirming")
    return bucket.assessment_id


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision; auto-complete the image when done.

    A true ``rule_admitted`` claim is verified (:func:`_verify_rule_admitted_claim`) before
    anything is mutated; a false one never reaches ground truth.
    """
    if not payload.dataset_root:
        raise HTTPException(
            400,
            "record_action requires the dataset root this verdict is scoped to; name one "
            "rather than leaving it unstated.",
        )
    gt_path = allowed_optional(payload.gt_path)
    pred_path = allowed_optional(payload.pred_path)
    published = _bucket_of_prediction(pred_path)
    scope = _review_scope(published, payload.subject, payload.attribute)
    ctx = _load_ctx(payload.image_name, payload.image_path, gt_path=gt_path, pred_path=pred_path)
    engine = _get_engine(payload.dataset_root)
    # GUI-set reviewer drives both the verdict log (reviewed_by, bare) and the GT provenance
    # (accepted_by/created_by, "user:<name>") so the two never disagree on who acted.
    reviewer_name = resolve_user(payload.user)
    engine.current_user = reviewer_name
    reviewer = user_id(reviewer_name)
    now_iso = datetime.now(timezone.utc).isoformat()
    det = ReviewDetection(
        det_type=payload.det_type,
        class_name=payload.class_name,
        conf=payload.conf,
        iou=payload.iou,
        gt_idx=payload.gt_idx,
        pred_idx=payload.pred_idx,
        bbox=payload.bbox,
    )

    # A verified claim before any mutation: a false one never reaches _apply_gt_mutation.
    accepted_by_rule: Optional[str] = None
    if payload.rule_admitted:
        accepted_by_rule = _verify_rule_admitted_claim(payload, ctx, published, scope)

    # Author GT on a copy so the guard can 400 before anything is recorded, and so the verdict
    # entry is recorded against the pristine ctx (its bbox lookups read gt_idx).
    work = replace(ctx, gt=list(ctx.gt))
    try:
        changed, landed_idx = _apply_gt_mutation(
            work, payload, reviewer, now_iso, scope=scope, accepted_by_rule=accepted_by_rule)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if changed and not gt_path:
        raise HTTPException(
            400, "this verdict writes ground truth, but no annotations path was provided")

    # Recompute over the mutated in-memory documents before any write, so a refusal here leaves
    # neither the verdict log nor the GT file touched, never a write on disk behind a 400.
    bucket = _bucket_of_file(pred_path)
    matches = _compute_matches(
        work.gt, ctx.preds,
        iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold, scope=scope,
    )

    # An edited verdict rewrites the GT geometry, so key the entry to the post-edit geometry;
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == EDITED_ACTION and changed and landed_idx is not None:
        norm_det = replace(det, gt_idx=landed_idx)
        norm_ctx = work
    producer_identity = _producer_identity(published)
    class_id = _verdict_class_id(scope, payload.class_name)
    engine.record_detection_action(
        bucket, det, ctx, action=payload.action, norm_det=norm_det, norm_ctx=norm_ctx,
        producer_identity=producer_identity, conf_threshold=payload.conf_threshold,
        class_id=class_id,
    )

    # Write the single per-image GT file (keep_empty: an emptied GT stays an {"annotations": []}
    # record, not deleted). accept-TP/FN and reject-FP are no-ops.
    if changed and gt_path:
        _ensure_original_backup(gt_path)  # baseline this file before its first mutation
        engine.save_gt(work, path=gt_path)

    # Annotation status to sync client-side (only when GT changed); an emptied GT reads as
    # "unannotated": a negative needs an explicit Complete, not just an empty file.
    annotation_status: Optional[str] = None
    if changed:
        annotation_status = "partial" if work.gt else "unannotated"

    # Promote to 'completed' once every detection at these thresholds is reviewed, the only path
    # by which a GUI review reaches 'completed'.
    engine.check_image_review_complete(bucket, work, matches)

    def _committed() -> dict:
        # The fresh matches this verdict just recomputed (gt_idx/pred_idx rebuilt against the
        # written GT), so the client installs them without a second /matches round-trip.
        fresh = _matches_response(
            work, matches, engine, payload.image_name, bucket=bucket,
            filter_type=payload.filter_type, filter_class=payload.filter_class, scope=scope,
        )
        return {
            "status": "ok",
            "image_status": engine.get_image_review_status(bucket, payload.image_name),
            "annotation_status": annotation_status,
            "matches": fresh,
        }

    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        _audit(payload.dataset_root, "gui_review_action", {
            "image_name": payload.image_name,
            "det_type": payload.det_type,
            "class_name": payload.class_name,
            "action": payload.action,
            "gt_changed": changed,
            "rule_admitted": payload.rule_admitted,
            "accepted_by_rule": accepted_by_rule,
        })
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, _committed()) from exc
    return _committed()


class MarkCompletePayload(BaseModel):
    dataset_root: str
    image_name: str
    gt_path: Optional[str] = None
    # A confirmed negative carries zero verdict entries, so this stamps the producer identity
    # on the image-level record instead of a verdict.
    pred_dir: Optional[str] = None
    completed: bool = True  # False reverses a manual mark (verdicts are kept)
    # The subject this Complete confirms; absent, completion is still recorded but no
    # subject-scoped status is derived or mirrored into the classes store.
    subject: Optional[str] = None


def _is_negative_for_subject(
    image_name: str, subject: Optional[str], bucket: Optional[Bucket],
) -> Optional[bool]:
    """Whether ``bucket``'s predictions for ``image_name`` hold nothing for ``subject``; ``None``
    when the bucket cannot answer for ``subject`` at all.

    No bucket at all is unconditionally negative. The document read is the one the bucket's record
    names for the image; an image it names none for was not predicted, so the bucket answers
    ``None``. A subject-less Complete checks the whole document. A named subject is answered only
    by a bucket whose scope names it among the object classes it detects
    (:attr:`~tcip_mcp.pipelines.data.selection.ClassScope.subjects`), compared by decoded name.
    """
    if bucket is None:
        return True
    pred_file = bucket.document(image_name)
    if pred_file is None or (subject is not None and subject not in bucket.scope.subjects):
        return None
    if subject is None:
        return not _has_objects(pred_file)
    return not any(a.subject == subject for a in cached_label_annotations(pred_file))


@router.post("/mark_complete")
def mark_complete(payload: MarkCompletePayload) -> dict:
    """Mark (or unmark) an image fully reviewed; covers negatives / bulk-accept cases.

    Adjudication coverage is recorded per subject: a map from subject name (or ``"*"`` for a
    subject-less Complete, a claim about every subject) to whether that zero-verdict completion was
    a genuine negative for it, so a later Complete under another subject on the same image adds its
    own entry. A subject the bucket's own recorded class map cannot resolve writes no entry at all;
    the Complete and its status write still proceed. A bucket record that will not read refuses
    400 before anything is written.
    """
    if not payload.dataset_root:
        raise HTTPException(
            400,
            "mark_complete requires the dataset root this completion is scoped to; name one "
            "rather than leaving it unstated.",
        )
    gt_path = allowed_optional(payload.gt_path)
    pred_dir = allowed_optional(payload.pred_dir)
    engine = _get_engine(payload.dataset_root)
    bucket = _bucket_of_dir(pred_dir)
    published = _published(pred_dir)
    is_negative: Optional[bool] = None
    if payload.completed:
        # Adjudication-covered only for a genuine negative: a bulk-accept with no individual
        # verdicts on an image the bucket did predict on is not covered.
        try:
            is_negative = _is_negative_for_subject(payload.image_name, payload.subject, published)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from None
    # The annotation status is derived from the GT file, scoped to the confirmed subject; the
    # read runs before either engine write, so an unreadable document persists nothing.
    annotations: list = []
    if payload.subject and gt_path:
        annotations = _read_annotations_or_400(read_annotations, gt_path)
    if payload.completed:
        producer_identity = _producer_identity(published)
        # An unresolvable subject omits the entry rather than refusing the Complete; the reader
        # fails closed on the missing entry at validation time.
        adjudication_covered = (
            {(payload.subject or "*"): is_negative} if is_negative is not None else None
        )
        engine.mark_image_reviewed(bucket, payload.image_name,
                                   producer_identity=producer_identity,
                                   adjudication_covered=adjudication_covered)
    else:
        engine.unmark_image_reviewed(bucket, payload.image_name)
    annotation_status = None
    if payload.subject:
        has_content = annotations_hold_subject(annotations, payload.subject)
        annotation_status = derive_status(completed=payload.completed, has_content=has_content)
    committed = {
        "status": "ok",
        "image_status": engine.get_image_review_status(bucket, payload.image_name),
        "annotation_status": annotation_status,
    }
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        _audit(payload.dataset_root, "gui_review_mark_complete", {
            "image_name": payload.image_name,
            "completed": payload.completed,
            "subject": payload.subject,
            "annotation_status": annotation_status,
        })
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, committed) from exc
    return committed


class BackupPayload(BaseModel):
    dataset_root: str
    label_dirs: list[str]


@router.post("/backup_labels")
def backup_labels(payload: BackupPayload) -> dict:
    """Top up ``<dir>/.original/``: capture any label file that has no baseline yet."""
    label_dirs = [d for d in (allowed_optional(d) for d in payload.label_dirs) if d]
    engine = _get_engine(payload.dataset_root)
    n = engine.backup_original_labels(*label_dirs)
    return {"status": "ok", "files_backed_up": n}


class ImageStatusesResponse(BaseModel):
    # image_name -> "not_started" | "started" | "completed"; images the engine has never
    # touched are absent (the client defaults them to "not_started").
    statuses: dict[str, str]
    # Stems (filename without extension) whose GT or prediction file holds >=1 annotation; a
    # stem absent here contributes no TP/FP/FN, so Review navigation skips it.
    detection_stems: list[str]
    # Absolute paths of GT or prediction documents that would not read; a stem named here can
    # also be in detection_stems, and the tab keeps it visible rather than navigating past it.
    unreadable: list[str]


def _has_objects(path: Path) -> bool:
    """True if ``path`` holds at least one annotation record. An empty (confirmed-negative) or
    missing file has nothing to review.
    """
    return bool(cached_label_annotations(path))


def _stems_with_objects(documents: list[Path]) -> tuple[set[str], set[str]]:
    """Stems of ``documents`` holding >=1 annotation record, and the absolute paths of documents
    that would not read (per file). A stem can appear in both sets at once."""
    stems: set[str] = set()
    unreadable: set[str] = set()
    for f in documents:
        try:
            has_objects = _has_objects(f)
        except UnreadableLabelDocument:
            unreadable.add(str(f))
            continue
        if has_objects:
            stems.add(f.stem)
    return stems, unreadable


@router.get("/image_statuses")
def image_statuses(
    dataset_root: str,
    gt_dir: Optional[str] = None,
    pred_dir: Optional[str] = None,
) -> ImageStatusesResponse:
    """Batch review status + detection presence for a whole (date): ``gt_dir`` is the date's
    annotations directory, ``pred_dir`` the published bucket whose recorded documents are read.
    """
    gt_dir = allowed_optional(gt_dir)
    bucket = _published(allowed_optional(pred_dir))
    engine = _get_engine(dataset_root)
    stems, unreadable = _stems_with_objects(
        [*(prediction_documents(gt_dir) if gt_dir else []),
         *(bucket.document_paths if bucket is not None else [])])
    return ImageStatusesResponse(
        statuses=engine.get_all_image_statuses(),
        detection_stems=sorted(stems),
        unreadable=sorted(unreadable),
    )


class GenerationConfResponse(BaseModel):
    # The confidence the bucket's predictions were published at, or None for no published bucket
    # or a head with no confidence; read-only, for the filter-warning check.
    generation_conf: Optional[float]
    # The conf at or above which the bucket's own assessment admits a prediction without a
    # person's judgment: null with admission_reason naming why none applies.
    admission_conf: Optional[float]
    admission_reason: str


def _admission(bucket: Optional[Bucket]) -> tuple[Optional[float], str]:
    """The conf a review may accept ``bucket``'s predictions at on its assessment's authority
    (:func:`~tcip_mcp.delivery.admitted_conf`), or ``None`` and the sentence naming why; no
    published bucket admits none."""
    from tcip_mcp.delivery import admitted_conf

    if bucket is None:
        return None, "no published bucket holds these predictions, so no assessment admits any."
    return admitted_conf(store.open_root(), bucket)


@router.get("/generation_conf")
def get_generation_conf(pred_dir: str) -> GenerationConfResponse:
    """The prediction bucket's generation confidence and admission conf (:func:`_admission`)."""
    bucket = _published(str(allowed_path(pred_dir)))
    conf, reason = _admission(bucket)
    execution = bucket.execution if bucket is not None else None
    return GenerationConfResponse(
        generation_conf=execution.conf if execution is not None else None,
        admission_conf=conf, admission_reason=reason)


# ── Active-learning priority queue for review ───────────────────────────────

# This surfaces prioritize_review_queue's own ranking (feedback_tools.py), never reimplemented, as a browsable queue on a background thread (a forward pass per image can be slow), polled for the result.

# Its sibling door, triage_predictions, can auto-accept predictions as GT above a breeder-confirmed threshold, a different and more consequential capability deliberately left agent/operator-only for now.



@dataclass
class PriorityQueueJob:
    job_id: str
    # The project open when the job launched, which it runs for.
    project: str
    checkpoint_path: str
    images_dir: str
    dataset_root: str
    method: str = "combined"
    budget: int = 50
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None
    # [{image, score, reference_member?}], highest first; reference_member is present only
    # when the checkpoint's run was bound to a selection.
    queue: list[dict] = field(default_factory=list)
    total_candidates: int = 0
    reviewed_skipped: int = 0


def _pq_summary(job: PriorityQueueJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "error": job.error,
        "queue": job.queue, "total_candidates": job.total_candidates,
        "reviewed_skipped": job.reviewed_skipped,
    }


_pq_registry = jobstore.JobRegistry()
"""This queue's own live jobs (``jobstore.JobRegistry``)."""


def _pq_register(job: PriorityQueueJob) -> None:
    _pq_registry.register(job.job_id, job)


def _pq_get(job_id: str) -> Optional[PriorityQueueJob]:
    """A job by id, whichever project it runs for."""
    return _pq_registry.get(job_id)


def _pq_worker(job: PriorityQueueJob) -> None:
    try:
        job.status = "running"
        # The same MCP tool the agent calls, whose soft {"error": ...} answers map onto this job.
        from tcip_mcp.tools.feedback_tools import prioritize_review_queue

        result = prioritize_review_queue(
            Path(job.project),
            checkpoint_path=job.checkpoint_path,
            images_dir=job.images_dir,
            dataset_root=job.dataset_root,
            method=job.method,
            budget=job.budget,
        )
        if "error" in result:
            job.status = "failed"
            job.error = result["error"]
        else:
            job.status = "completed"
            job.queue = result["queue"]
            job.total_candidates = result["total_candidates"]
            job.reviewed_skipped = result["reviewed_skipped"]
    except Exception as exc:
        logger.exception("priority-queue job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)


class LaunchPriorityQueuePayload(BaseModel):
    dataset_root: str
    checkpoint_path: str
    images_dir: str
    method: str = "combined"
    budget: int = 50


@router.post("/queue/launch")
def launch_priority_queue(payload: LaunchPriorityQueuePayload) -> dict:
    # checkpoint_path is confined to the allowed roots, same as the Inference tab's own launch
    # route: a caller must not name a file outside them, registered checkpoint or not.
    dataset_root = allowed_path(payload.dataset_root)
    checkpoint_path = allowed_path(payload.checkpoint_path)
    images_dir = allowed_path(payload.images_dir)
    if not checkpoint_path.is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    # The dataset root, not a store path: the tool derives from it the one verdict store _get_engine opens.
    job = PriorityQueueJob(
        job_id=f"pq-{uuid.uuid4().hex[:8]}",
        project=str(store.open_root()),
        checkpoint_path=str(checkpoint_path),
        images_dir=str(images_dir),
        dataset_root=str(dataset_root),
        method=payload.method,
        budget=payload.budget,
    )
    _pq_register(job)
    threading.Thread(target=_pq_worker, args=(job,), daemon=True).start()
    return {"status": "launched", "job_id": job.job_id}


@router.get("/queue/{job_id}")
def get_priority_queue_job(job_id: str) -> dict:
    job = _pq_get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return _pq_summary(job)
