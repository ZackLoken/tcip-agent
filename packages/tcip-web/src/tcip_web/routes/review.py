"""Review routes: compute matches, walk detections, record actions, save GT.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance lives in memory per
project (keyed by project_root). Review state is persisted via the engine to per-image shards under
``<project_root>/.tcip/state/review/``.

Ground truth and predictions are each one JSON file per image holding every subject's annotations by
name (a prediction is an :class:`~tcip_annotation.state.Annotation` whose ``score`` is set); a class
is named by its ``subject``, never an integer id, so the recorded verdict carries the real subject
name and a resolved ``class_id``: the producing bucket's own recorded name->id map, read once at
record time (``_resolve_verdict_class_id``), never a registry re-derivation, so a class-aware
reference (``review_calibration.review_to_records``) can be built from these verdicts without
guessing.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation import (
    BBox,
    Point,
    Polygon,
    ReviewContext,
    ReviewDetection,
    ReviewEngine,
    compute_classified_trait_matches,
    compute_matches,
)
from tcip_annotation.json_io import annotation_from_payload, read_annotations
from tcip_annotation.state import Annotation
from tcip_mcp.dataset_layout import derive_status
from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
from tcip_mcp.utils.atomic_io import append_jsonl, read_json
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/review", tags=["review"])
logger = logging.getLogger(__name__)


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _current_user() -> str:
    """Reviewer fallback when the GUI request omits ``user``: env override else the OS login."""
    from tcip_web.identity import current_user

    return current_user()


def _get_engine(project_root: str) -> ReviewEngine:
    from tcip_mcp.prediction_buckets import review_state_dir_of

    key = str(Path(project_root).resolve())
    if key not in _engines:
        state_dir = review_state_dir_of(project_root)
        _engines[key] = ReviewEngine(state_dir=state_dir, current_user=_current_user())
    return _engines[key]


def _audit(project_root: str, tool: str, arguments: dict) -> None:
    """Append a GUI review mutation to ``<project_root>/.tcip/audit.jsonl`` (best-effort).

    Review verdicts + GT writes change tracked state, so (like @audited MCP tools and
    the annotate save path) they belong in the append-only log. Never fails the request.
    """
    if not project_root:
        return
    try:
        append_jsonl(
            os.path.join(project_root, ".tcip", "audit.jsonl"),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "source": "gui",
                "arguments": arguments,
                "status": "ok",
            },
        )
    except Exception:
        pass


def _resolve_producer_identity_for_dir(pred_dir: Optional[str]) -> Optional[dict]:
    """The producing model's identity for prediction bucket ``pred_dir``.

    Resolved from the bucket's own ``operating_point.json`` sidecar: ``checkpoint_sha256`` and
    ``experiment_id``, the same facts ``validate_reference`` already reads for its own scoping.
    ``None`` when there is no dir or no sidecar to read; callers store this as a plain fact on the
    verdict/image record rather than looking it up again at validation time.
    """
    if not pred_dir:
        return None
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(pred_dir)
    if sidecar is None:
        return None
    return {
        "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
        "experiment_id": sidecar.get("experiment_id"),
        "bucket_dir": str(Path(pred_dir)),
    }


def _resolve_producer_identity(pred_path: Optional[str]) -> Optional[dict]:
    """Same as :func:`_resolve_producer_identity_for_dir`, from a per-image prediction file path
    (``ActionPayload.pred_path``): the bucket dir is its parent, used only as the lookup key to
    find the sidecar, never as the identity itself."""
    if not pred_path:
        return None
    return _resolve_producer_identity_for_dir(str(Path(pred_path).parent))


def _resolve_verdict_class_id(pred_path: Optional[str], class_name: str) -> Optional[int]:
    """The 0-indexed class identity ``class_name`` resolves to under the producing bucket's own
    recorded name->id map: resolved at verdict-record time, from the same ``operating_point.json``
    ``id_map`` field ``phenology.resolve_positive_class_id`` reads for a prediction bucket, never a
    fresh registry re-derivation (the recorded map is what the bucket's predictions were actually
    decoded through; the registry could have changed since). ``None`` when there is no bucket, no
    recorded ``id_map`` on it, or ``class_name`` is not one of its keys (e.g. a GT annotation's raw
    ``subject`` on an attribute-scoped bucket, whose id_map is keyed by attribute values, not the
    subject name, since class-aware admission does not yet reach that case; see
    ``review_calibration.review_to_records``, which refuses rather than guesses when this is
    ``None``). Never defaults to 0: an unresolved identity is an honest fact, not a class.
    """
    if not pred_path:
        return None
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map

    id_map = bucket_id_map(Path(pred_path).parent)
    if id_map is None:
        return None
    cid = id_map.get(class_name)
    if cid is None:
        return None
    try:
        # Guards the same malformed-value case phenology.resolve_positive_class_id already guards
        # (a corrupt/hand-edited sidecar whose id_map value isn't actually numeric): a bad sidecar
        # must not 500 the breeder's accept/reject/edit click; it degrades to "unresolved" instead.
        return int(cid)
    except (TypeError, ValueError):
        return None


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    # Channel-aware: resolve_image_source folds a `.bandgroup` manifest (or a genuinely
    # multi-band raster) into the real frame image_dimensions measures, instead of a bare PIL
    # header read misreporting its axes.
    return image_dimensions(resolve_image_source(p.parent, p.stem))


def _guard_path(path: Optional[str]) -> None:
    """403 if a client-supplied label/dir path escapes the configured image roots.

    ``save_gt`` / ``backup_labels`` write to caller-provided paths, so an exposed
    deployment (``TCIP_IMAGE_ROOTS`` set) must confine them like image serving.
    """
    if not path:
        return
    try:
        assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _ensure_original_backup(label_path: Optional[str]) -> None:
    """Snapshot a label file to ``<dir>/.original/<name>`` before its first mutation, if no baseline
    exists yet: a per-file, O(1) safety net so a verdict never overwrites the pristine original
    without a copy, independent of (and closing any gap in) the client's dir-level backup. New GT
    files a verdict is creating have no original to preserve, so they're skipped. Best-effort."""
    if not label_path:
        return
    src = Path(label_path)
    if not src.is_file():
        return
    dst = src.parent / ".original" / src.name
    if dst.exists():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except OSError:
        pass


def _ann_dict(a: Annotation) -> dict:
    """Serialize an :class:`Annotation` for the canvas (pixel coords + attributes + provenance).

    ``rings`` (not ``points``) for a polygon: a prediction awaiting review can be a genuine
    multi-ring occlusion-split instance_seg output, and accepting it as-is must not silently keep
    only the first ring. The canvas itself still only ever *draws*/*edits* a single ring by hand
    (see ``edited_points`` below). ``point`` is the singular ``[x, y]`` of a placed prompt / keypoint,
    the same key the on-disk schema uses: a Point GT annotation on the frame is shown as itself
    rather than arriving geometry-less and being written back without its location.
    """
    out: dict = {"subject": a.subject, "attributes": dict(a.attributes)}
    geom = a.geometry
    if isinstance(geom, Polygon):
        out["rings"] = [[list(pt) for pt in ring] for ring in geom.rings]
    elif isinstance(geom, BBox):
        out["bbox"] = [geom.x1, geom.y1, geom.x2, geom.y2]
    elif isinstance(geom, Point):
        out["point"] = [geom.x, geom.y]
    if a.score is not None:
        out["score"] = a.score
    out["created_by"] = a.created_by
    out["created_at"] = a.created_at
    out["accepted_by"] = a.accepted_by
    out["accepted_at"] = a.accepted_at
    return out


def _check_classification_scope(subject: Optional[str], attribute: Optional[str]) -> None:
    """Reviewing a classified trait needs both facts: ``attribute`` alone can't say which GT
    instances it scopes. Raised before anything is read/mutated, not just at the matcher call, so a
    malformed request 400s instead of silently authoring GT under a ``None`` subject."""
    if attribute is not None and not subject:
        raise HTTPException(400, "attribute given for a classified-trait review, but no subject "
                                  "was provided to scope which GT instances it applies to")


def _compute_matches(
    gt: list, preds: list, *, iou_threshold: float, conf_threshold: float,
    subject: Optional[str], attribute: Optional[str],
) -> dict:
    """Dispatch to plain detection matching, or classified-trait matching when the caller names the
    (subject, attribute) axis under review. The one call site both ``/matches`` and ``/action`` use,
    so a verdict's freshly recomputed matches are always scoped identically to what produced it."""
    if attribute is None:
        return compute_matches(gt, preds, iou_threshold, conf_threshold)
    _check_classification_scope(subject, attribute)
    return compute_classified_trait_matches(
        gt, preds, subject=subject, attribute=attribute,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )


def _load_ctx(image_name: str, image_path: str, *, gt_path: Optional[str],
              pred_path: Optional[str]) -> ReviewContext:
    w, h = _image_dims(image_path)
    ctx = ReviewContext(img_name=image_name, img_width=w, img_height=h)
    if gt_path:
        ctx.gt = read_annotations(gt_path)
    if pred_path:
        ctx.preds = read_annotations(pred_path)
    return ctx


# ── Request/response schemas ──────────────────────────────────────────────


class MatchesRequest(BaseModel):
    project_root: str
    image_name: str
    image_path: str
    gt_path: Optional[str] = None      # the per-image ground-truth label file
    pred_path: Optional[str] = None    # the per-image prediction file
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    filter_type: str = "all"
    filter_class: str = "all"          # a class name (an annotation's subject) or "all"
    # Reviewing a classified trait rather than plain detection: `subject` names the object type GT
    # isolates (an enabling/trait subject already annotated), `attribute` the axis whose confirmed/
    # predicted value is under review. Both, or neither: a bare `attribute` can't scope which GT
    # instances it applies to. `None`/`None` (the default) is today's detection-only matching,
    # unchanged.
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
    reviewed_action: Optional[str] = None


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


def _matches_response(
    ctx: ReviewContext,
    matches: dict,
    engine: ReviewEngine,
    image_name: str,
    *,
    filter_type: str,
    filter_class: str,
) -> MatchesResponse:
    """Build the canvas payload (filtered + review-decorated detections, GT/pred annotations, status)
    from an already-computed match set. Shared by /matches and /action so both surfaces return the
    identical shape, letting a verdict return its fresh matches instead of forcing a second fetch."""
    dets = engine.build_detection_list(
        ctx, matches, filter_type=filter_type, filter_class=filter_class
    )
    out_dets: list[Detection] = []
    for d in dets:
        entry = engine.find_reviewed_entry(d, ctx)
        out_dets.append(Detection(
            det_type=d.det_type,
            class_name=d.class_name,
            conf=d.conf,
            iou=d.iou,
            gt_idx=d.gt_idx,
            pred_idx=d.pred_idx,
            bbox=d.bbox,
            reviewed=entry is not None,
            reviewed_action=entry.get("action") if entry else None,
        ))

    return MatchesResponse(
        img_width=ctx.img_width,
        img_height=ctx.img_height,
        n_tp=len(matches["tp"]),
        n_fp=len(matches["fp"]),
        n_fn=len(matches["fn"]),
        detections=out_dets,
        gt=[_ann_dict(a) for a in ctx.gt],
        preds=[_ann_dict(a) for a in ctx.preds],
        image_status=engine.get_image_review_status(image_name),
    )


@router.post("/matches")
def compute_image_matches(req: MatchesRequest) -> MatchesResponse:
    """Compute TP/FP/FN, decorate with review status, and return everything the canvas needs."""
    ctx = _load_ctx(req.image_name, req.image_path, gt_path=req.gt_path, pred_path=req.pred_path)
    engine = _get_engine(req.project_root)
    matches = _compute_matches(
        ctx.gt, ctx.preds, iou_threshold=req.iou_threshold, conf_threshold=req.conf_threshold,
        subject=req.subject, attribute=req.attribute,
    )
    return _matches_response(
        ctx, matches, engine, req.image_name,
        filter_type=req.filter_type, filter_class=req.filter_class,
    )


class ActionPayload(BaseModel):
    project_root: str
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
    action: str  # "accepted" | "rejected" | "edited" | "swept" (an explicit "checked this image
    # for missed objects, found none" attestation: no geometry, never mutates GT, see
    # _apply_gt_mutation)
    # GUI-set reviewer identity (bare name, e.g. "breeder"); stamped as accepted_by/created_by
    # ("user:<name>"). Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None
    # Edited shape committed from the Review canvas (only for action="edited"): a box, or a
    # polygon's points. Accept/Reject don't carry these: they act on the loaded pred/gt by index.
    edited_box: Optional[tuple[float, float, float, float]] = None
    edited_points: Optional[list[list[float]]] = None
    # Review thresholds so the route can decide (at the same op point as the GUI) whether
    # this verdict was the last one and the image should flip to 'completed'.
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    # Active review filters, so the fresh matches this route returns are scoped identically to
    # what /matches would have returned (the client installs them without a second fetch).
    filter_type: str = "all"
    filter_class: str = "all"
    # Same (subject, attribute) meaning as MatchesRequest: set together when this verdict is on a
    # classified trait rather than plain detection.
    subject: Optional[str] = None
    attribute: Optional[str] = None


def _apply_gt_mutation(
    ctx: ReviewContext, payload: "ActionPayload", reviewer: str, now_iso: str
) -> tuple[bool, Optional[int]]:
    """Author GT from a verdict; return ``(gt_changed, index the written annotation landed at in
    ctx.gt)``: the index is set only for edited/accepted writes. Accept an FP adds the prediction;
    accept a TP/FN keeps GT; reject a TP/FN deletes that GT; reject an FP is a no-op; edit writes
    the edited shape (replacing the matched GT, or adding it); ``action="swept"`` (an explicit
    "checked this image for missed objects, found none" attestation) matches none of the branches
    below and always no-ops, GT is never mutated by sweeping.

    Provenance (``reviewer`` = ``user:<name>``, ``now_iso`` = UTC): an accepted prediction carries
    its ``created_by``/``created_at`` into GT (origin travels) and gets ``accepted_by``/``accepted_at``
    with its ``score`` dropped (it is ground truth now); a reviewer-drawn edit is stamped
    ``created_by`` = reviewer.

    Reviewing a classified trait (``payload.attribute`` set): ``payload.class_name`` is the confirmed/
    predicted *value*, never the GT object identity, so an authored annotation keeps the real object
    type on ``subject`` (``payload.subject``) and stamps the value onto ``attributes[attribute]``,
    the same schema every other GT annotation for this subject already carries; plain detection review
    (``payload.attribute`` is ``None``) is unchanged, ``class_name`` is the object identity itself."""
    dt, act = payload.det_type, payload.action
    classifying = payload.attribute is not None

    def _author(base_subject: str, geometry, *, attributes: Optional[dict] = None, **fields) -> Annotation:
        if classifying:
            attrs = dict(attributes or {})
            attrs[payload.attribute] = base_subject
            return Annotation(subject=payload.subject, geometry=geometry, attributes=attrs, **fields)
        return Annotation(subject=base_subject, geometry=geometry, attributes=dict(attributes or {}),
                          **fields)

    if act == "edited":
        geom: BBox | Polygon | None = None
        if payload.edited_box is not None:
            geom = BBox(*payload.edited_box)
        elif payload.edited_points is not None:
            # The reviewer edits one contour by hand on the canvas: single-ring input.
            geom = Polygon(rings=[[(float(p[0]), float(p[1])) for p in payload.edited_points]])
        if geom is None:
            return False, None
        new = _author(payload.class_name, geom, created_by=reviewer, created_at=now_iso)
        if dt in ("tp", "fn") and payload.gt_idx is not None \
                and 0 <= payload.gt_idx < len(ctx.gt):
            ctx.gt[payload.gt_idx] = new
            return True, payload.gt_idx
        ctx.gt.append(new)
        return True, len(ctx.gt) - 1

    if act == "rejected" and dt in ("tp", "fn") and payload.gt_idx is not None \
            and 0 <= payload.gt_idx < len(ctx.gt):
        ctx.gt.pop(payload.gt_idx)
        return True, None

    if act == "accepted" and dt == "fp" and payload.pred_idx is not None \
            and 0 <= payload.pred_idx < len(ctx.preds):
        pred = ctx.preds[payload.pred_idx]
        if classifying:
            # `pred.subject` is the model's predicted value (an attribute-scoped detector's own
            # class space), never the real object type; accepting it as GT means confirming that
            # value for `payload.subject`'s real instance, not minting a new object of that name.
            accepted = _author(pred.subject, pred.geometry, attributes=pred.attributes,
                               created_by=pred.created_by, created_at=pred.created_at,
                               accepted_by=reviewer, accepted_at=now_iso)
        else:
            accepted = replace(pred, score=None, accepted_by=reviewer, accepted_at=now_iso)
        ctx.gt.append(accepted)
        return True, len(ctx.gt) - 1

    return False, None  # accept TP/FN and reject FP leave GT untouched


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision; auto-complete the image when done."""
    _check_classification_scope(payload.subject, payload.attribute)
    ctx = _load_ctx(payload.image_name, payload.image_path,
                    gt_path=payload.gt_path, pred_path=payload.pred_path)
    engine = _get_engine(payload.project_root)
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

    # Author GT on a copy so the guard can 400 before anything is recorded, and so the verdict
    # entry is recorded against the pristine ctx (its bbox lookups read gt_idx).
    work = replace(ctx, gt=list(ctx.gt))
    changed, landed_idx = _apply_gt_mutation(work, payload, reviewer, now_iso)
    if changed and not payload.gt_path:
        raise HTTPException(
            400, "this verdict writes ground truth, but no annotations path was provided")

    # An edited verdict rewrites the GT geometry, so key the entry to the post-edit geometry;
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == "edited" and changed and landed_idx is not None:
        norm_det = replace(det, gt_idx=landed_idx)
        norm_ctx = work
    producer_identity = _resolve_producer_identity(payload.pred_path)
    class_id = _resolve_verdict_class_id(payload.pred_path, payload.class_name)
    engine.record_detection_action(
        det, ctx, action=payload.action, norm_det=norm_det, norm_ctx=norm_ctx,
        producer_identity=producer_identity, conf_threshold=payload.conf_threshold,
        class_id=class_id,
    )

    # Write the single per-image GT file (keep_empty: an emptied GT stays an {"annotations": []}
    # record, not deleted). accept-TP/FN and reject-FP are no-ops.
    if changed and payload.gt_path:
        _guard_path(payload.gt_path)
        _ensure_original_backup(payload.gt_path)  # baseline this file before its first mutation
        engine.save_gt(work, path=payload.gt_path)

    # Annotation status to sync client-side (only when GT changed); an emptied GT reads as
    # "unannotated": a negative needs an explicit Complete, not just an empty file.
    annotation_status: Optional[str] = None
    if changed:
        annotation_status = "partial" if work.gt else "unannotated"

    # Promote to 'completed' once every detection at these thresholds is reviewed, the only path
    # by which a GUI review reaches 'completed'. Recompute against the (now-authored) GT.
    matches = _compute_matches(
        work.gt, ctx.preds,
        iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold,
        subject=payload.subject, attribute=payload.attribute,
    )
    engine.check_image_review_complete(payload.image_name, matches)
    _audit(payload.project_root, "gui_review_action", {
        "image_name": payload.image_name,
        "det_type": payload.det_type,
        "class_name": payload.class_name,
        "action": payload.action,
        "gt_changed": changed,
    })
    # Return the fresh matches this verdict just recomputed (gt_idx/pred_idx rebuilt against the
    # written GT), so the client installs them without a second /matches round-trip.
    fresh = _matches_response(
        work, matches, engine, payload.image_name,
        filter_type=payload.filter_type, filter_class=payload.filter_class,
    )
    return {
        "status": "ok",
        "image_status": engine.get_image_review_status(payload.image_name),
        "annotation_status": annotation_status,
        "matches": fresh,
    }


class MarkCompletePayload(BaseModel):
    project_root: str
    image_name: str
    gt_path: Optional[str] = None
    # The prediction bucket loaded for this image: a confirmed negative carries zero
    # verdict entries, so it has nowhere else to record which model it was reviewed against; this
    # stamps that producer-identity fact on the image-level record instead.
    pred_dir: Optional[str] = None
    completed: bool = True  # False reverses a manual mark (verdicts are kept)


@router.post("/mark_complete")
def mark_complete(payload: MarkCompletePayload) -> dict:
    """Mark (or unmark) an image fully reviewed; covers negatives / bulk-accept cases."""
    _guard_path(payload.gt_path)
    _guard_path(payload.pred_dir)
    engine = _get_engine(payload.project_root)
    if payload.completed:
        producer_identity = _resolve_producer_identity_for_dir(payload.pred_dir)
        # A zero-verdict Complete is FN-adjudication-covered only when it is a genuine negative:
        # the bucket held zero predictions for this image, so there was nothing to individually walk
        # and Complete is itself the confirming act. A bulk-accept of an image the bucket did
        # predict on, with no individual verdicts recorded, is not covered: the breeder never
        # actually looked, and treating it as covered would let exactly that padding dilute a real
        # reference's statistics. No prediction file for this stem reads as "nothing to check" ->
        # also covered.
        is_negative = True
        if payload.pred_dir:
            pred_file = Path(payload.pred_dir) / f"{Path(payload.image_name).stem}.json"
            is_negative = not _has_objects(pred_file)
        engine.mark_image_reviewed(payload.image_name, producer_identity=producer_identity,
                                   adjudication_covered=is_negative)
    else:
        engine.unmark_image_reviewed(payload.image_name)
    # Derive the annotation status from the GT file on disk: the client's matches snapshot can be
    # stale or null mid-navigation and once wrote negatives for annotated frames. A present file
    # with no annotations of any subject is an empty (negative) record.
    has_content = bool(payload.gt_path and os.path.isfile(payload.gt_path)
                       and read_annotations(payload.gt_path))
    annotation_status = derive_status(completed=payload.completed, has_content=has_content)
    _audit(payload.project_root, "gui_review_mark_complete", {
        "image_name": payload.image_name,
        "completed": payload.completed,
        "annotation_status": annotation_status,
    })
    return {
        "status": "ok",
        "image_status": engine.get_image_review_status(payload.image_name),
        "annotation_status": annotation_status,
    }


class BackupPayload(BaseModel):
    project_root: str
    label_dirs: list[str]


@router.post("/backup_labels")
def backup_labels(payload: BackupPayload) -> dict:
    """Top up ``<dir>/.original/``: capture any label file that has no baseline yet."""
    for d in payload.label_dirs:
        _guard_path(d)
    engine = _get_engine(payload.project_root)
    n = engine.backup_original_labels(*payload.label_dirs)
    return {"status": "ok", "files_backed_up": n}


class SaveGtPayload(BaseModel):
    project_root: str
    image_name: str
    image_path: str
    label_path: Optional[str] = None
    # [{subject, bbox?: [x1,y1,x2,y2], rings?: [[[x,y]...], ...], point?: [x,y], attributes?,
    #   created_by?, ...}]
    annotations: list[dict] = []
    user: Optional[str] = None    # GUI-set author; stamped as created_by unless the shape carries one


@router.post("/save_gt")
def save_gt(payload: SaveGtPayload) -> dict:
    """Persist edited GT (post-review modification) for a single image."""
    w, h = _image_dims(payload.image_path)
    _guard_path(payload.label_path)
    engine = _get_engine(payload.project_root)

    # The reviewer authors this committed GT; a shape that round-trips its own provenance keeps it.
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()

    ctx = ReviewContext(
        img_name=payload.image_name, img_width=w, img_height=h,
        gt=[annotation_from_payload(d, author=author, now=now_iso) for d in payload.annotations],
    )
    ok = engine.save_gt(ctx, path=payload.label_path)
    _audit(payload.project_root, "gui_review_save_gt", {
        "image_name": payload.image_name,
        "label_path": payload.label_path,
        "n_annotations": len(payload.annotations),
    })
    return {"status": "ok" if ok else "partial"}


# ── Promote a completed review into a validation reference ─────────────────


class ValidateReferenceRequest(BaseModel):
    project_root: str
    trait: str
    # The prediction bucket whose review is being promoted: the per-image prediction dir the
    # delivery gate reads an ``operating_point.json`` from.
    pred_dir: Optional[str] = None


class ValidateReferenceResponse(BaseModel):
    # True only when the review cleared the identical gate the backend uses (or the bucket was already
    # validated). A refusal is surfaced honestly here, never silently upgraded.
    validated: bool
    reference: Optional[str]  # a resolution.py validated_against value ("false" when unvalidated)
    reviewed_image_count: int
    conf: Optional[float]  # the derived count operating point (for transparency)
    reason: str  # plain-language, breeder-facing, always present
    buckets_stamped: list[str]


@router.post("/validate_reference")
def validate_reference(req: ValidateReferenceRequest) -> ValidateReferenceResponse:
    """Promote a completed review session into a validation reference for its (model, trait, date-set).

    Reconstructs the review verdicts into the COCO records ``resolve_operating_point`` consumes (the
    ``review_calibration`` adapter) and runs them through the identical disjoint-split + count-bias
    gate and conf-censoring guard the held-out-GT path uses: no shortcut to "validated". On success
    the bucket's ``operating_point.json`` is stamped ``VALIDATED_REVIEW_CONFIRMED`` (so the delivery
    gate reads it); on refusal an honest ``validated=false`` placeholder is written and the reason is
    returned. An already-validated bucket is never downgraded.
    """
    bucket_dirs = [req.pred_dir] if req.pred_dir else []
    for d in bucket_dirs:
        _guard_path(d)
    if not bucket_dirs:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No predictions are selected to validate. Choose a model with predictions for "
                   "this dataset, then try again.",
            buckets_stamped=[])

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems

    stems = bucket_stems(*bucket_dirs)
    engine = _get_engine(req.project_root)
    completed = {
        name: data
        for name, data in engine.raw_state.get("image", {}).items()
        if Path(name).stem in stems and data.get("img_status") == "completed"
    }
    n = len(completed)

    # Never downgrade: predictions already validated (e.g. against held-out GT) stay validated: a
    # review reference isn't needed there, and this action must not be able to lower them.
    sidecars = {d: (read_operating_point_sidecar(d) or {}) for d in bucket_dirs}
    if all(sc.get("validated") for sc in sidecars.values()):
        ref = next((((sc.get("operating_point") or {}).get("conf") or {}).get("validated_against")
                    for sc in sidecars.values()), None)
        return ValidateReferenceResponse(
            validated=True, reference=ref, reviewed_image_count=n, conf=None,
            reason="These predictions are already validated, so a review reference isn't needed here.",
            buckets_stamped=[])

    if n == 0:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No completed reviews yet for this model on this date. Review the predictions and "
                   "mark the images Reviewed, then try again.",
            buckets_stamped=[])

    from tcip_mcp.pipelines.feedback import (
        describe_review_validation,
        resolve_operating_point_from_review,
        review_conf_threshold,
        review_reference_hash,
        review_to_records,
    )
    from tcip_mcp.traits import TraitUnknownError

    review_state = {"image": completed}
    # Thread the producing run's experiment_id through so the calibration's train-disjointness
    # gate can check the reviewed images against that run's training split. Sourced from the
    # buckets' own operating_point.json sidecars (stamped by export_predictions), never asserted:
    # when multiple buckets disagree on which run produced them, pass None (mixed-provenance
    # shouldn't silently vouch for one run's disjointness) rather than raising, so this route keeps
    # working for a legitimate multi-bucket review call.
    bucket_exp_ids = {sc.get("experiment_id") for sc in sidecars.values() if sc.get("experiment_id")}
    review_experiment_id = next(iter(bucket_exp_ids)) if len(bucket_exp_ids) == 1 else None

    # Scope every verdict/negative record to the bucket(s) actually being validated, at the
    # deepest choke point (resolve_operating_point_from_review), not just here.
    bucket_identities = [
        {"checkpoint_sha256": sc.get("checkpoint_sha256"), "experiment_id": sc.get("experiment_id")}
        for sc in sidecars.values()
    ]
    # The review path's effective staging floor is max(generation_conf, review_conf_threshold):
    # the generation half read off the same sidecars already loaded above, the review half read
    # off the verdicts' own recorded conf_threshold (scoped identically to the bucket(s) above).
    # Either half unknown makes the combined floor None (fails closed).
    gen_confs = [
        v for sc in sidecars.values()
        if isinstance(v := ((sc.get("operating_point") or {}).get("conf") or {}).get("value"),
                     (int, float))
    ]
    generation_conf = max(float(v) for v in gen_confs) if gen_confs else None
    review_conf = review_conf_threshold(review_state, bucket_identities=bucket_identities,
                                        only_completed=True)
    staged_conf_floor = (
        max(generation_conf, review_conf)
        if generation_conf is not None and review_conf is not None
        else None
    )

    # Thread tile_size/tiled + their sources off the same sidecars already loaded above, so a
    # review-confirmed bundle honestly reports "derived"/"explicit" instead of always falling
    # back to "default" regardless of what the buckets actually carry. A single bucket's own
    # stamp is used; a mixed set of sources across buckets is not resolvable to one fact, so it
    # falls back to the honest default.
    from tcip_mcp.pipelines.resolution import tile_size_source_of

    tile_sizes = {((sc.get("operating_point") or {}).get("tile_size") or {}).get("value")
                  for sc in sidecars.values()}
    # From validated_against, not the bare source field, which a native-ratio edge shares with a
    # real persisted one: reading source alone would silently re-validate native-ratio on review.
    tile_size_valid_refs = {
        ((sc.get("operating_point") or {}).get("tile_size") or {}).get("validated_against")
        for sc in sidecars.values()}
    tiled_vals = {((sc.get("operating_point") or {}).get("tiled") or {}).get("value")
                 for sc in sidecars.values()}
    tiled_sources = {((sc.get("operating_point") or {}).get("tiled") or {}).get("source")
                     for sc in sidecars.values()}
    review_tile_size = next(iter(tile_sizes)) if len(tile_sizes) == 1 else None
    review_tile_size_valid_ref = (
        next(iter(tile_size_valid_refs)) if len(tile_size_valid_refs) == 1
        and review_tile_size is not None else None)
    review_tile_size_source = tile_size_source_of(
        review_tile_size_valid_ref, tile_size=review_tile_size)
    review_tiled = next(iter(tiled_vals)) if len(tiled_vals) == 1 else None
    review_tiled_source = (next(iter(tiled_sources)) if len(tiled_sources) == 1
                           and review_tiled is not None else "default")

    try:
        bundle = resolve_operating_point_from_review(
            review_state, req.trait, only_completed=True, experiment_id=review_experiment_id,
            bucket_identities=bucket_identities, staged_conf_floor=staged_conf_floor,
            tile_size=review_tile_size, tile_size_source=review_tile_size_source,
            tiled=review_tiled, tiled_source=review_tiled_source)
    except TraitUnknownError:
        raise HTTPException(
            400,
            f"a validation reference is not defined for trait {req.trait!r} yet. This action is "
            "available for traits the platform can calibrate a count operating point for.",
        ) from None
    except ValueError as exc:
        # A locked cal/holdout split refusing this call: a reviewed image was deleted/renamed
        # since the split locked, or the lock file itself is corrupt. Either way
        # this is an honest refusal, not a 500: surface it as such.
        raise HTTPException(400, str(exc)) from None

    result = describe_review_validation(bundle, reviewed_image_count=n)

    # Stamp each bucket's provenance sidecar (operating_point.json is not a label, so this never
    # touches the reviewed per-image predictions or the verdict-immutability guard).
    from tcip_mcp.pipelines.resolution import update_sidecar

    op_prov = bundle.to_provenance()["operating_point"]
    ref_hash = review_reference_hash(
        review_to_records(review_state, bucket_identities=bucket_identities))
    now_iso = datetime.now(timezone.utc).isoformat()

    def _promote(stored: dict) -> dict | None:
        """Merge this promotion into whatever the producing run left, inside the stamp's lock.

        The no-downgrade decision is made against the stored stamp, not the copy read before the
        lock: predictions validated some other way (held-out GT) stay validated, and a producer
        that stamped the bucket while this review was being reconciled is not overwritten.
        """
        if stored.get("validated"):
            return None
        merged = dict(stored)
        merged.update({
            "operating_point": op_prov,
            "validated": result["validated"],
            "validated_reference": result["reference"],
            "validation_source": "review_confirmed",
            "review_reference_hash": ref_hash,
            "review_image_count": n,
            "shippable_issues": bundle.shippable_issues(),
            "validated_at": now_iso,
        })
        merged.setdefault("produced_at", now_iso)
        return merged

    stamped: list[str] = []
    for d in bucket_dirs:
        if sidecars[d].get("validated"):
            continue  # a mixed set: leave an already-validated bucket untouched (no downgrade)
        Path(d).mkdir(parents=True, exist_ok=True)
        if update_sidecar(d, _promote):
            stamped.append(d)

    _audit(req.project_root, "gui_review_validate_reference", {
        "trait": req.trait,
        "validated": result["validated"],
        "reference": result["reference"],
        "reviewed_image_count": n,
        "buckets_stamped": stamped,
    })
    return ValidateReferenceResponse(
        validated=bool(result["validated"]),
        reference=result["reference"],
        reviewed_image_count=n,
        conf=result["conf"],
        reason=result["reason"],
        buckets_stamped=stamped,
    )


@router.get("/image_status")
def get_image_status(project_root: str, image_name: str) -> dict:
    engine = _get_engine(project_root)
    return {"status": engine.get_image_review_status(image_name)}


class ImageStatusesResponse(BaseModel):
    # image_name -> "not_started" | "started" | "completed"; images the engine has never
    # touched are absent (the client defaults them to "not_started").
    statuses: dict[str, str]
    # Stems (filename without extension) whose GT or prediction file holds at least one annotation,
    # i.e. the image has something to review. Images whose stem is absent contribute no TP/FP/FN,
    # so Review navigation skips them.
    detection_stems: list[str]


def _has_objects(path: Path) -> bool:
    """True if ``path`` is a per-image label JSON with a non-empty ``annotations`` list. An empty
    (confirmed-negative) or missing file has nothing to review."""
    try:
        data = read_json(path, default=None)
    except Exception:
        return False
    return isinstance(data, dict) and bool(data.get("annotations"))


def _stems_with_objects(*dirs: Optional[str]) -> set[str]:
    stems: set[str] = set()
    for d in dirs:
        if not d:
            continue
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("*.json"):
            if f.stem not in stems and _has_objects(f):
                stems.add(f.stem)
    return stems


@router.get("/image_statuses")
def image_statuses(
    project_root: str,
    gt_dir: Optional[str] = None,
    pred_dir: Optional[str] = None,
) -> ImageStatusesResponse:
    """Batch review status + detection presence for a whole (date): one call the Review tab makes
    on dataset entry to drive the image-level Reviewed/Unreviewed filter and to skip images with
    nothing to review. ``gt_dir``/``pred_dir`` are the per-image label dirs (annotations / a model's
    predictions on the date)."""
    for d in (gt_dir, pred_dir):
        _guard_path(d)
    engine = _get_engine(project_root)
    return ImageStatusesResponse(
        statuses=engine.get_all_image_statuses(),
        detection_stems=sorted(_stems_with_objects(gt_dir, pred_dir)),
    )


class GenerationConfResponse(BaseModel):
    # The bucket's own recorded generation confidence (the Conf floor predictions were exported
    # at), or None when the bucket has no sidecar / no recorded value. Read-only: this is the
    # same fact validate_reference reads to derive staged_conf_floor, exposed here without the
    # gate run or the sidecar stamp validate_reference performs, so the Review tab can warn as
    # soon as the breeder raises the filter instead of only after a review is complete.
    generation_conf: Optional[float]


@router.get("/generation_conf")
def get_generation_conf(pred_dir: str) -> GenerationConfResponse:
    """The prediction bucket's own generation confidence, for the Conf >= filter warning.

    Raising the review's own "Conf >=" filter above this value hides low-confidence detections
    from review; any verdict then recorded raises review_conf_threshold above it, which
    validate_reference's identical gate reads as conf_censored. This endpoint exposes the one
    fact needed to warn about that live, in the filter shelf, before a review is even complete.
    """
    _guard_path(pred_dir)
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(pred_dir) or {}
    conf = ((sidecar.get("operating_point") or {}).get("conf") or {}).get("value")
    return GenerationConfResponse(
        generation_conf=float(conf) if isinstance(conf, (int, float)) else None)


# ── Active-learning priority queue for review ───────────────────────────────
#
# prioritize_review_queue's own ranking (packages/tcip-mcp .../tools/feedback_tools.py) never
# reached the breeder-facing Review tab: the only path was the agent manually calling
# focus(tab='review', ...) once per ranked image, which doesn't scale. This surfaces the same
# tool (never a second implementation of its scoring/filtering) as a browsable queue: launch on a
# background thread (checkpoint loading + a forward pass per candidate image can be slow), poll
# for the result. Scoped to strategy="informativeness" only: the tool's other strategy,
# confidence_triage, can auto-accept predictions as GT above a breeder-confirmed threshold; that
# is a different, more consequential capability deliberately left agent-only for now.


@dataclass
class PriorityQueueJob:
    job_id: str
    checkpoint_path: str
    images_dir: str
    review_state_dir: str
    method: str = "combined"
    budget: int = 50
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None
    queue: list[dict] = field(default_factory=list)  # [{image, score}], highest first
    total_candidates: int = 0
    reviewed_skipped: int = 0
    thread: Optional[threading.Thread] = field(default=None, repr=False)


_pq_jobs: dict[str, PriorityQueueJob] = {}
_pq_lock = threading.Lock()


def _pq_summary(job: PriorityQueueJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "error": job.error,
        "queue": job.queue, "total_candidates": job.total_candidates,
        "reviewed_skipped": job.reviewed_skipped,
    }


def _pq_persist() -> None:
    from tcip_web import jobstore
    with _pq_lock:
        summaries = [_pq_summary(j) for j in _pq_jobs.values()]
    jobstore.persist("review_priority_jobs", summaries)


def _pq_register(job: PriorityQueueJob) -> None:
    from tcip_web import jobstore
    with _pq_lock:
        _pq_jobs[job.job_id] = job
        jobstore.evict_terminal(_pq_jobs)  # bound the registry (drop oldest terminal jobs)
    _pq_persist()


def _pq_get(job_id: str) -> Optional[PriorityQueueJob]:
    with _pq_lock:
        return _pq_jobs.get(job_id)


def _pq_worker(job: PriorityQueueJob) -> None:
    try:
        job.status = "running"
        _pq_persist()
        # The same MCP tool the agent calls: its scoring/filtering (build_predictor ->
        # require_composed_detector -> build_scorer -> score -> budget slice -> response shape) is
        # not re-derived here. It returns soft {"error": ...} dicts rather than raising, for every
        # failure mode (missing checkpoint, unknown scorer, non-composed detector, torch
        # unavailable), mapped onto this job's own status/error below rather than reimplemented.
        from tcip_mcp.tools.feedback_tools import prioritize_review_queue

        result = prioritize_review_queue(
            checkpoint_path=job.checkpoint_path,
            images_dir=job.images_dir,
            review_state_dir=job.review_state_dir,
            strategy="informativeness",
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
    finally:
        _pq_persist()


class LaunchPriorityQueuePayload(BaseModel):
    project_root: str
    checkpoint_path: str
    images_dir: str
    method: str = "combined"
    budget: int = 50


@router.post("/queue/launch")
def launch_priority_queue(payload: LaunchPriorityQueuePayload) -> dict:
    # checkpoint_path reaches torch.load via build_predictor (the same arbitrary-pickle sink the
    # Inference tab's own launch route confines): same guard, same treatment.
    for p in (payload.project_root, payload.checkpoint_path, payload.images_dir):
        _guard_path(p)
    if not Path(payload.checkpoint_path).is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not Path(payload.images_dir).is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    from tcip_mcp.prediction_buckets import review_state_dir_of

    # The same store _get_engine opens for this project: the queue skips images this engine
    # already holds verdicts for, so the two must not read different stores.
    review_state_dir = str(review_state_dir_of(payload.project_root))

    job = PriorityQueueJob(
        job_id=f"pq-{uuid.uuid4().hex[:8]}",
        checkpoint_path=payload.checkpoint_path,
        images_dir=payload.images_dir,
        review_state_dir=review_state_dir,
        method=payload.method,
        budget=payload.budget,
    )
    _pq_register(job)
    t = threading.Thread(target=_pq_worker, args=(job,), daemon=True)
    job.thread = t
    t.start()
    return {"status": "launched", "job_id": job.job_id}


@router.get("/queue/{job_id}")
def get_priority_queue_job(job_id: str) -> dict:
    job = _pq_get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return _pq_summary(job)
