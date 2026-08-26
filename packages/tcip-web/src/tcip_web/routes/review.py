"""Review routes: compute matches, walk detections, record actions, save GT.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance lives in memory per
dataset (keyed by dataset_root). Review state is persisted via the engine to per-image shards under
``<dataset_root>/.tcip/state/review/``, so a verdict travels with the images it was recorded on and
the immutability guard that counts verdicts reads the store the reviewer wrote.

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
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tcip_store
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from tcip_store import Version

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
from tcip_annotation.json_io import (
    UnreadableLabelDocument, annotation_from_payload, bbox_from_corners, check_box_extent,
    prediction_documents, read_annotations,
)
from tcip_annotation.review_engine import label_baseline_key
from tcip_annotation.state import Annotation
from tcip_annotation.verdicts import VerdictAction
from tcip_mcp.dataset_layout import annotations_hold_subject, derive_status
from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
from tcip_web.identity import resolve_user, user_id
from tcip_web.label_annotations_cache import cached_label_annotations
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/review", tags=["review"])
logger = logging.getLogger(__name__)


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _current_user() -> str:
    """Reviewer fallback when the GUI request omits ``user``: env override else the OS login."""
    from tcip_web.identity import current_user

    return current_user()


def _guarded(path: str) -> Path:
    """Confine a client-supplied path and hand back the resolved path every later read uses."""
    try:
        return assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _get_engine(dataset_root: str) -> ReviewEngine:
    """The review engine anchored on a client-supplied dataset root, confined first (403)."""
    from tcip_mcp.prediction_buckets import review_state_dir_of

    key = str(_guarded(dataset_root))
    if key not in _engines:
        state_dir = review_state_dir_of(key)
        _engines[key] = ReviewEngine(state_dir=state_dir, current_user=_current_user())
    return _engines[key]


def _audit(scope: str, tool: str, arguments: dict) -> None:
    """Record a GUI review mutation in the audit log ``scope`` names.

    Every mutation these routes make changes a record that travels with the dataset: the verdict
    store, the ground-truth labels and a prediction bucket's provenance stamp all live under the
    dataset root, so the dataset root the request states is the scope for all of them. The scope
    is confined before the append, so no audit line lands outside the allowed roots.
    """
    if not scope:
        return
    from tcip_mcp.audit import record_event

    record_event(tool, arguments, source="gui", scope=str(_guarded(scope)))


def _dataset_root_of_all(paths: Iterable[Optional[str]]) -> Optional[str]:
    """The one dataset root every path in ``paths`` belongs to, or None when that is not a single
    answer.

    A cross-check that a request naming one dataset is not pointing at another's files, never a
    source of the root itself: the request states that. ``None`` means the paths answer nothing to
    cross-check against (a bucket under no dataset root is legitimate work), or that they answer
    several things, which is the refusal a request accepting more than one prediction directory
    would need.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    roots = {str(root) for p in paths if p and (root := dataset_root_of(p)) is not None}
    return roots.pop() if len(roots) == 1 else None


def _prediction_digest(pred_dir: Optional[str], image_name: str) -> Optional[str]:
    """The content identity of ``image_name``'s prediction document in ``pred_dir``, as it is now.

    The platform's own :func:`~tcip_mcp.pipelines.resolution.dataset_hash` over that one stem, the
    hasher a bucket's whole-content digest is built from, so a verdict records the file the reviewer
    actually saw and the promotion can tell whether that file still says what it said. ``None`` when
    the image has no prediction document, or the review names no bucket at all: the confirmed-negative
    case, a value that is compared rather than a comparison that is skipped.
    """
    if not pred_dir:
        return None
    from tcip_mcp.pipelines.resolution import dataset_hash

    stem = Path(image_name).stem
    if not (Path(pred_dir) / f"{stem}.json").is_file():
        return None
    return dataset_hash(pred_dir, [stem])


def _recorded_prediction_digests(image_state: dict) -> set[Optional[str]]:
    """Every prediction-document identity recorded against one reviewed image at review time.

    The image-level producer fact a confirmed negative carries, plus the one on each verdict entry.
    A recorded identity carrying no digest reads as ``None``, the same value an image with no
    prediction document records, so an unrecorded identity is compared rather than waved through.
    """
    identities = [image_state.get("producer_identity")]
    identities += [d.get("producer_identity") for d in image_state.get("detections") or []]
    return {i.get("prediction_digest") for i in identities if isinstance(i, dict)}


def _resolve_producer_identity_for_dir(pred_dir: Optional[str], image_name: str) -> Optional[dict]:
    """The producing model's identity for ``image_name`` in prediction bucket ``pred_dir``.

    Resolved from the bucket's own ``operating_point.json`` sidecar: ``checkpoint_sha256`` and
    ``experiment_id``, the same facts ``validate_reference`` already reads for its own scoping,
    alongside the content identity of the one prediction document this review is being recorded
    against. ``None`` when there is no dir or no sidecar to read; callers store this as a plain fact
    on the verdict/image record rather than looking it up again at validation time.
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
        "prediction_digest": _prediction_digest(pred_dir, image_name),
    }


def _bucket_of_dir(pred_dir: Optional[str]) -> str:
    """The verdict store's key for the prediction bucket dir this request names, through the one
    spelling the immutability guard also uses. No dir is a review with no bucket at all."""
    from tcip_mcp.prediction_buckets import bucket_key_of

    return bucket_key_of(pred_dir)


def _bucket_of_file(pred_path: Optional[str]) -> str:
    """Same, from a per-image prediction file path: the bucket dir is its parent."""
    return _bucket_of_dir(str(Path(pred_path).parent) if pred_path else None)


def _resolve_producer_identity(pred_path: Optional[str]) -> Optional[dict]:
    """Same as :func:`_resolve_producer_identity_for_dir`, from a per-image prediction file path
    (``ActionPayload.pred_path``): the bucket dir is its parent, used only as the lookup key to
    find the sidecar, never as the identity itself."""
    if not pred_path:
        return None
    return _resolve_producer_identity_for_dir(str(Path(pred_path).parent), Path(pred_path).name)


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


def _guard_path(path: Optional[str]) -> Optional[str]:
    """Confine a client-supplied label/dir path and hand back its resolved spelling, or None.

    ``save_gt`` / ``backup_labels`` write to caller-provided paths, so every writer uses the
    path this returns, never the string the client sent. 403 on escape.
    """
    if not path:
        return None
    return str(_guarded(path))


def _ensure_original_backup(label_path: Optional[str]) -> None:
    """Capture one label file's pristine bytes before its first mutation, if none is held yet.

    The per-file, O(1) counterpart of :meth:`ReviewEngine.backup_original_labels`'s directory
    sweep: same baseline record, same create-only capture, so a verdict never overwrites a
    pristine original without a copy no matter which of the two ran first. New GT files a verdict
    is creating have no original to preserve, so they are skipped, and an already-held baseline is
    kept rather than replaced by this call's read of a file the platform may already have edited.
    """
    if not label_path:
        return
    src = Path(label_path)
    if not src.is_file():
        return
    try:
        tcip_store.put_blob(
            label_baseline_key(src.parent, src.stem), src.read_bytes(),
            expect=Version.ABSENT,
        )
    except tcip_store.VersionConflict:
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


def _read_annotations_or_400(path: str) -> list:
    """``read_annotations``, refused (400) naming the file when the document will not read: a
    review derived from a document nobody can read is a claim about nothing."""
    try:
        return read_annotations(path)
    except UnreadableLabelDocument as exc:
        raise HTTPException(400, str(exc)) from exc


def _load_ctx(image_name: str, image_path: str, *, gt_path: Optional[str],
              pred_path: Optional[str]) -> ReviewContext:
    w, h = _image_dims(image_path)
    ctx = ReviewContext(img_name=image_name, img_width=w, img_height=h)
    gt_path = _guard_path(gt_path)
    pred_path = _guard_path(pred_path)
    if gt_path:
        ctx.gt = _read_annotations_or_400(gt_path)
    if pred_path:
        ctx.preds = _read_annotations_or_400(pred_path)
    return ctx


# ── Request/response schemas ──────────────────────────────────────────────


class MatchesRequest(BaseModel):
    dataset_root: str
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


def _matches_response(
    ctx: ReviewContext,
    matches: dict,
    engine: ReviewEngine,
    image_name: str,
    *,
    bucket: str,
    filter_type: str,
    filter_class: str,
) -> MatchesResponse:
    """Build the canvas payload (filtered + review-decorated detections, GT/pred annotations, status)
    from an already-computed match set. Shared by /matches and /action so both surfaces return the
    identical shape, letting a verdict return its fresh matches instead of forcing a second fetch.

    ``n_reviewed``/``n_total`` come from :meth:`ReviewEngine.review_progress` over the unfiltered
    ``matches``, before ``filter_type``/``filter_class`` narrow ``detections`` to what is rendered:
    the status-bar wheel reports the whole image's progress regardless of the active filter.
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
            reviewed_action=entry.get("action") if entry else None,
        ))

    n_reviewed, n_total = engine.review_progress(bucket, ctx, all_dets)
    return MatchesResponse(
        img_width=ctx.img_width,
        img_height=ctx.img_height,
        n_tp=len(matches["tp"]),
        n_fp=len(matches["fp"]),
        n_fn=len(matches["fn"]),
        detections=out_dets,
        gt=[_ann_dict(a) for a in ctx.gt],
        preds=[_ann_dict(a) for a in ctx.preds],
        image_status=engine.get_image_review_status(bucket, image_name),
        n_reviewed=n_reviewed,
        n_total=n_total,
    )


@router.post("/matches")
def compute_image_matches(req: MatchesRequest) -> MatchesResponse:
    """Compute TP/FP/FN, decorate with review status, and return everything the canvas needs."""
    ctx = _load_ctx(req.image_name, req.image_path, gt_path=req.gt_path, pred_path=req.pred_path)
    engine = _get_engine(req.dataset_root)
    matches = _compute_matches(
        ctx.gt, ctx.preds, iou_threshold=req.iou_threshold, conf_threshold=req.conf_threshold,
        subject=req.subject, attribute=req.attribute,
    )
    return _matches_response(
        ctx, matches, engine, req.image_name, bucket=_bucket_of_file(req.pred_path),
        filter_type=req.filter_type, filter_class=req.filter_class,
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
            geom = bbox_from_corners(*payload.edited_box, where=f"editing {payload.class_name!r}")
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
        if isinstance(pred.geometry, BBox):
            check_box_extent(pred.geometry, where=f"accepting {payload.class_name!r}")
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
    gt_path = _guard_path(payload.gt_path)
    pred_path = _guard_path(payload.pred_path)
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

    # Author GT on a copy so the guard can 400 before anything is recorded, and so the verdict
    # entry is recorded against the pristine ctx (its bbox lookups read gt_idx).
    work = replace(ctx, gt=list(ctx.gt))
    try:
        changed, landed_idx = _apply_gt_mutation(work, payload, reviewer, now_iso)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if changed and not gt_path:
        raise HTTPException(
            400, "this verdict writes ground truth, but no annotations path was provided")

    # An edited verdict rewrites the GT geometry, so key the entry to the post-edit geometry;
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == "edited" and changed and landed_idx is not None:
        norm_det = replace(det, gt_idx=landed_idx)
        norm_ctx = work
    producer_identity = _resolve_producer_identity(pred_path)
    class_id = _resolve_verdict_class_id(pred_path, payload.class_name)
    bucket = _bucket_of_file(pred_path)
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
    # by which a GUI review reaches 'completed'. Recompute against the (now-authored) GT.
    matches = _compute_matches(
        work.gt, ctx.preds,
        iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold,
        subject=payload.subject, attribute=payload.attribute,
    )
    engine.check_image_review_complete(bucket, work, matches)
    _audit(payload.dataset_root, "gui_review_action", {
        "image_name": payload.image_name,
        "det_type": payload.det_type,
        "class_name": payload.class_name,
        "action": payload.action,
        "gt_changed": changed,
    })
    # Return the fresh matches this verdict just recomputed (gt_idx/pred_idx rebuilt against the
    # written GT), so the client installs them without a second /matches round-trip.
    fresh = _matches_response(
        work, matches, engine, payload.image_name, bucket=bucket,
        filter_type=payload.filter_type, filter_class=payload.filter_class,
    )
    return {
        "status": "ok",
        "image_status": engine.get_image_review_status(bucket, payload.image_name),
        "annotation_status": annotation_status,
        "matches": fresh,
    }


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
    pred_dir: Optional[str], image_name: str, subject: Optional[str]
) -> Optional[bool]:
    """Whether ``pred_dir``'s predictions for ``image_name`` hold nothing for ``subject``, the fact
    a zero-verdict Complete is confirming; ``None`` when the bucket cannot answer for ``subject``
    at all.

    No prediction bucket at all is unconditionally negative, there is nothing to check against. A
    subject-less Complete checks the whole file (the claim a subject-less Complete makes, about
    every subject). A named subject is checked against the bucket's own recorded name->id map
    (``phenology.bucket_id_map``, the same map ``_resolve_verdict_class_id`` reads for a verdict's
    class identity) only as an admission gate: membership proves the bucket assessed this subject
    at all (an attribute-scoped bucket's map is keyed by attribute values, not the object's subject
    name, so it admits no object subject). The comparison itself is by the decoded name
    (``read_annotations``' own ``subject`` field), never the id, since the bucket's own predictions
    were already decoded through that same map. A subject the map does not admit is ``None``: the
    caller omits the coverage entry rather than guessing one.
    """
    if not pred_dir:
        return True
    pred_file = Path(pred_dir) / f"{Path(image_name).stem}.json"
    if subject is None:
        return not _has_objects(pred_file)
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map

    id_map = bucket_id_map(Path(pred_dir))
    if id_map is None or subject not in id_map:
        return None
    return not any(a.subject == subject for a in read_annotations(str(pred_file)))


@router.post("/mark_complete")
def mark_complete(payload: MarkCompletePayload) -> dict:
    """Mark (or unmark) an image fully reviewed; covers negatives / bulk-accept cases.

    Adjudication coverage is recorded per subject: a map from subject name (or ``"*"`` for a
    subject-less Complete, a claim about every subject) to whether that zero-verdict completion
    was a genuine negative for it, so a later Complete under another subject on the same image
    adds its own entry rather than overwriting the first. A subject the bucket's own recorded
    class map cannot resolve writes no entry at all: the Complete and its status write still
    proceed, and the coverage reader fails closed on the missing entry at validation time.
    """
    gt_path = _guard_path(payload.gt_path)
    pred_dir = _guard_path(payload.pred_dir)
    engine = _get_engine(payload.dataset_root)
    bucket = _bucket_of_dir(pred_dir)
    is_negative: Optional[bool] = None
    if payload.completed:
        # Adjudication-covered only for a genuine negative: a bulk-accept with no individual
        # verdicts on an image the bucket did predict on is not covered.
        try:
            is_negative = _is_negative_for_subject(pred_dir, payload.image_name, payload.subject)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from None
    # The annotation status is derived from the GT file, scoped to the confirmed subject; the
    # read runs before either engine write, so an unreadable document persists nothing.
    annotations: list = []
    if payload.subject and gt_path:
        annotations = _read_annotations_or_400(gt_path)
    if payload.completed:
        producer_identity = _resolve_producer_identity_for_dir(pred_dir, payload.image_name)
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
    _audit(payload.dataset_root, "gui_review_mark_complete", {
        "image_name": payload.image_name,
        "completed": payload.completed,
        "subject": payload.subject,
        "annotation_status": annotation_status,
    })
    return {
        "status": "ok",
        "image_status": engine.get_image_review_status(bucket, payload.image_name),
        "annotation_status": annotation_status,
    }


class BackupPayload(BaseModel):
    dataset_root: str
    label_dirs: list[str]


@router.post("/backup_labels")
def backup_labels(payload: BackupPayload) -> dict:
    """Top up ``<dir>/.original/``: capture any label file that has no baseline yet."""
    label_dirs = [d for d in (_guard_path(d) for d in payload.label_dirs) if d]
    engine = _get_engine(payload.dataset_root)
    n = engine.backup_original_labels(*label_dirs)
    return {"status": "ok", "files_backed_up": n}


class SaveGtPayload(BaseModel):
    dataset_root: str
    image_name: str
    image_path: str
    # Non-empty: a save with nowhere to write can never succeed, so it is refused (422).
    label_path: str = Field(min_length=1)
    # [{subject, bbox?: [x1,y1,x2,y2], rings?: [[[x,y]...], ...], point?: [x,y], attributes?,
    #   created_by?, ...}]
    annotations: list[dict] = []
    user: Optional[str] = None    # GUI-set author; stamped as created_by unless the shape carries one


@router.post("/save_gt")
def save_gt(payload: SaveGtPayload) -> dict:
    """Persist edited GT (post-review modification) for a single image."""
    w, h = _image_dims(payload.image_path)
    label_path = _guard_path(payload.label_path)
    engine = _get_engine(payload.dataset_root)

    # The reviewer authors this committed GT; a shape that round-trips its own provenance keeps it.
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        gt = [annotation_from_payload(d, author=author, now=now_iso) for d in payload.annotations]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ctx = ReviewContext(img_name=payload.image_name, img_width=w, img_height=h, gt=gt)
    ok = engine.save_gt(ctx, path=label_path)
    # Ground truth travels with its dataset, so the edit is recorded beside the labels it changed.
    _audit(payload.dataset_root, "gui_review_save_gt", {
        "image_name": payload.image_name,
        "label_path": label_path,
        "n_annotations": len(payload.annotations),
    })
    return {"status": "ok" if ok else "partial"}


# ── Promote a completed review into a validation reference ─────────────────


class ValidateReferenceRequest(BaseModel):
    dataset_root: str
    trait: str
    # The prediction bucket whose review is being promoted: the per-image prediction dir the
    # delivery gate reads an ``operating_point.json`` from.
    pred_dir: Optional[str] = None
    # The object identity this reference validates. Required by the route; kept optional here so
    # an absent one earns the route's own named 400 rather than a generic pydantic error.
    subject: Optional[str] = None


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
    gate and conf-censoring guard the held-out-GT path uses: no shortcut to "validated". A passing
    gate is earned through ``open_validation``/``seal_validation``, which append the validation
    record and hand back the stamp carrying its pointer, so the bucket's ``operating_point.json`` can
    only claim ``VALIDATED_REVIEW_CONFIRMED`` with a record outside the bucket answering for it; on
    refusal an honest ``validated=false`` placeholder is written and the reason is returned.

    The promotion verifies before it decides. A bucket whose stamp claims validation that no record
    answers for is treated as unvalidated and is promotable over, and a review whose prediction
    documents are no longer the ones the reviewer saw earns nothing at all.
    """
    if not req.subject:
        raise HTTPException(
            400,
            "validate_reference requires the subject this reference validates; name one rather "
            "than leaving it unstated.",
        )
    pred_dir = _guard_path(req.pred_dir)
    bucket_dirs = [pred_dir] if pred_dir else []
    if not bucket_dirs:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=0, conf=None,
            reason="No predictions are selected to validate. Choose a model with predictions for "
                   "this dataset, then try again.",
            buckets_stamped=[])

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, verify_stamp_binding
    from tcip_mcp.prediction_buckets import bucket_stems

    # A bucket answering a different root than the stated one is another dataset's evidence.
    named_root = _dataset_root_of_all(bucket_dirs)
    if named_root is not None and Path(named_root).resolve() != Path(req.dataset_root).resolve():
        raise HTTPException(
            400,
            f"the predictions at {pred_dir} belong to dataset {named_root}, not to "
            f"{req.dataset_root}, the dataset this request names. Validate a bucket under its own "
            "dataset root, so the verdicts, the validation record and the stamp all hang off one "
            "dataset.")

    stems = bucket_stems(*bucket_dirs)
    engine = _get_engine(req.dataset_root)
    # The verdicts recorded against the bucket being promoted, so a stem that exists under two
    # buckets contributes only what was reviewed here.
    reviewed = {
        name: data
        for name, data in engine.image_states(_bucket_of_dir(pred_dir)).items()
        if data.get("img_status") == "completed"
    }
    completed = {name: data for name, data in reviewed.items() if Path(name).stem in stems}
    n = len(completed)

    # A claim no record answers for is an assertion: unvalidated, and promotable over.
    sidecars = {d: (read_operating_point_sidecar(d) or {}) for d in bucket_dirs}
    digest_memo: dict[str, str] = {}
    bindings = {d: verify_stamp_binding(sc, d, document="operating_point", digest_memo=digest_memo)
                for d, sc in sidecars.items()}
    if all(b.claimed and b.ok for b in bindings.values()):
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

    # A prediction document that changed, appeared or vanished since review is evidence for nothing.
    diverged = sorted(
        name for name, data in reviewed.items()
        if any(recorded != _prediction_digest(req.pred_dir, name)
               for recorded in _recorded_prediction_digests(data))
    )
    if diverged:
        return ValidateReferenceResponse(
            validated=False, reference=None, reviewed_image_count=n, conf=None,
            reason=f"The predictions for {', '.join(diverged)} are no longer the ones that were "
                   "reviewed: a prediction file has been added, replaced or removed in this bucket "
                   "since those verdicts were recorded. Re-running inference on a reviewed bucket "
                   "writes the next free variant of it instead (the same '@r2' redirect the "
                   "immutability guard makes), which keeps this review intact and can be reviewed "
                   "and validated on its own.",
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
    tile_size_derived_froms = {
        ((sc.get("operating_point") or {}).get("tile_size") or {}).get("derived_from")
        for sc in sidecars.values()}
    review_tile_size = next(iter(tile_sizes)) if len(tile_sizes) == 1 else None
    review_tile_size_valid_ref = (
        next(iter(tile_size_valid_refs)) if len(tile_size_valid_refs) == 1
        and review_tile_size is not None else None)
    review_tile_size_source = tile_size_source_of(
        review_tile_size_valid_ref, tile_size=review_tile_size)
    # The stamp's own derived_from text, carried forward unchanged: this route holds no predictor
    # to compose one from, only the record the producing run already wrote.
    review_tile_size_derived_from = (
        next(iter(tile_size_derived_froms)) if len(tile_size_derived_froms) == 1
        and review_tile_size is not None else None)
    review_tiled = next(iter(tiled_vals)) if len(tiled_vals) == 1 else None
    review_tiled_source = (next(iter(tiled_sources)) if len(tiled_sources) == 1
                           and review_tiled is not None else "default")

    # Refuse here, naming the bucket(s), rather than let the resolver's bare ValueError surface.
    if review_tile_size_source == "explicit" and review_tile_size_derived_from is None:
        per_bucket_derived_from = {
            d: ((sc.get("operating_point") or {}).get("tile_size") or {}).get("derived_from")
            for d, sc in sidecars.items()
        }
        raise HTTPException(
            400,
            "these predictions carry an explicit tile edge but disagree about, or omit, why it is "
            f"trusted ({per_bucket_derived_from}), so the review promotion cannot state one "
            "derivation for the validated claim. Validate the disagreeing bucket separately, or "
            "re-export the predictions from one run so their stamps agree.",
        )

    # One spelling of the evidence, shared by the description and open_validation's own resolver run.
    resolver_inputs = {
        "review_state": review_state,
        "only_completed": True,
        "bucket_identities": bucket_identities,
        "staged_conf_floor": staged_conf_floor,
        "tile_size": review_tile_size,
        "tile_size_source": review_tile_size_source,
        "tile_size_derived_from": review_tile_size_derived_from,
        "tiled": review_tiled,
        "tiled_source": review_tiled_source,
        # The root the verdict store was opened on, so the split lock travels with the verdicts.
        "scope_root": req.dataset_root,
        # True when the buckets named more than one producing run, false when none named one: both
        # collapse review_experiment_id to None above, but only the first is a real disagreement.
        "experiment_id_ambiguous": len(bucket_exp_ids) > 1,
        "subject": req.subject,
    }
    try:
        bundle = resolve_operating_point_from_review(
            trait_name=req.trait, experiment_id=review_experiment_id, **resolver_inputs)
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
    from tcip_mcp.pipelines.resolution import (
        claim_payload,
        open_validation,
        seal_validation,
        update_sidecar,
    )
    from tcip_mcp.prediction_buckets import review_state_dir_of

    op_prov = bundle.to_provenance()["operating_point"]
    ref_hash = review_reference_hash(
        review_to_records(review_state, bucket_identities=bucket_identities, subject=req.subject))
    now_iso = datetime.now(timezone.utc).isoformat()
    record_digests: dict[str, str] = {}
    stamped: list[str] = []

    try:
        draft = None
        if result["validated"]:
            shas = {sc.get("checkpoint_sha256") for sc in sidecars.values()
                    if sc.get("checkpoint_sha256")}
            draft = open_validation(
                document="operating_point",
                evidence={"resolver": "resolve_operating_point_from_review",
                          "inputs": resolver_inputs},
                trait=req.trait,
                checkpoint_sha256=next(iter(shas)) if len(shas) == 1 else None,
                producing_experiment_id=review_experiment_id,
                reference_inputs={
                    "dataset_root": req.dataset_root,
                    "scope_roots": {"verdicts": str(review_state_dir_of(req.dataset_root))},
                    "stated_values": {"review_reference_hash": ref_hash, "review_image_count": n},
                },
            )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    def _stamp_body(stored: dict) -> dict:
        """This promotion merged over whatever the producing run left in ``stored``.

        The trait is written only when a gate was cleared, so a bucket carries the trait its claim
        was earned for and an honest placeholder claims no scope at all.
        """
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
        if draft is not None:
            merged["trait"] = req.trait
        return merged

    def _promotion_of(pred_dir: str, earned: dict) -> Callable[[dict], Optional[dict]]:
        """The merge one bucket's stamp is promoted through, run inside that stamp's own lock."""

        def _promote(stored: dict) -> dict | None:
            """Merge this promotion into whatever the producing run left, inside the stamp's lock.

            The no-downgrade decision is made against the stored stamp, not the copy read before the
            lock: predictions whose validation a record answers for (held-out GT, an earlier review)
            stay as they are, and a producer that stamped the bucket while this review was being
            reconciled is not overwritten. ``earned`` is the body the record was sealed over, so the
            pointer is merged only while the stamp still makes the claim that record answers for; a
            claim that moved under the lock leaves the record inert rather than misnamed.
            """
            binding = verify_stamp_binding(stored, pred_dir, document="operating_point",
                                           digest_memo=digest_memo)
            if binding.claimed and binding.ok:
                return None
            merged = _stamp_body(stored)
            if draft is None:
                return merged
            if claim_payload(merged, document="operating_point") != claim_payload(
                    earned, document="operating_point"):
                return None
            merged["validated_by"] = earned["validated_by"]
            return merged

        return _promote

    try:
        for d in bucket_dirs:
            if bindings[d].claimed and bindings[d].ok:
                continue  # a mixed set: a bucket whose validation a record answers for is left alone
            Path(d).mkdir(parents=True, exist_ok=True)
            # Sealed outside the stamp's lock: no store write may open inside another's transaction.
            earned = _stamp_body(sidecars[d])
            if draft is not None:
                record_digests[d], earned = seal_validation(
                    draft, dataset_root=req.dataset_root, bucket_dirs=list(bucket_dirs),
                    stamp_body=earned)
            if update_sidecar(d, _promotion_of(d, earned)):
                stamped.append(d)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    # The sidecar this stamps sits in the prediction bucket, which travels with the dataset.
    _audit(req.dataset_root, "gui_review_validate_reference", {
        "trait": req.trait,
        "validated": result["validated"],
        "reference": result["reference"],
        "reviewed_image_count": n,
        "buckets_stamped": stamped,
        "record_digests": record_digests,
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
def get_image_status(dataset_root: str, image_name: str) -> dict:
    engine = _get_engine(dataset_root)
    # This route names no prediction bucket, so it answers across every bucket the image was
    # reviewed under rather than picking one.
    return {"status": engine.image_status_across_buckets(image_name)}


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
    """True if ``path`` holds at least one annotation record, read through the label memo shared
    with the classes and dataset routes. An empty (confirmed-negative) or missing file has
    nothing to review."""
    return bool(cached_label_annotations(path))


def _stems_with_objects(*dirs: Optional[str]) -> tuple[set[str], set[str]]:
    """Stems with >=1 annotation record across ``dirs``, and the absolute paths of documents that
    would not read (per file: one bad document costs its own stem, never the whole scan). A stem
    can appear in both sets at once, when one directory's document is unreadable and the other's
    holds objects for the same stem.

    Every directory's own document for a stem is opened, never skipped because an earlier
    directory already resolved that stem: a corrupt prediction document must surface as
    unreadable even when the ground truth already supplied an object for the same stem.
    """
    stems: set[str] = set()
    unreadable: set[str] = set()
    for d in dirs:
        if not d:
            continue
        for f in prediction_documents(d):
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
    """Batch review status + detection presence for a whole (date): one call the Review tab makes
    on dataset entry to drive the image-level Reviewed/Unreviewed filter and to skip images with
    nothing to review. ``gt_dir``/``pred_dir`` are the per-image label dirs (annotations / a model's
    predictions on the date)."""
    gt_dir = _guard_path(gt_dir)
    pred_dir = _guard_path(pred_dir)
    engine = _get_engine(dataset_root)
    stems, unreadable = _stems_with_objects(gt_dir, pred_dir)
    return ImageStatusesResponse(
        statuses=engine.get_all_image_statuses(),
        detection_stems=sorted(stems),
        unreadable=sorted(unreadable),
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
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(_guarded(pred_dir)) or {}
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
    dataset_root: str
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
            dataset_root=job.dataset_root,
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
    dataset_root: str
    checkpoint_path: str
    images_dir: str
    method: str = "combined"
    budget: int = 50


@router.post("/queue/launch")
def launch_priority_queue(payload: LaunchPriorityQueuePayload) -> dict:
    # checkpoint_path reaches torch.load via build_predictor (the same arbitrary-pickle sink the
    # Inference tab's own launch route confines): same guard, same treatment.
    dataset_root = _guarded(payload.dataset_root)
    checkpoint_path = _guarded(payload.checkpoint_path)
    images_dir = _guarded(payload.images_dir)
    if not checkpoint_path.is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    # The dataset root, not a store path: the tool derives from it the one verdict store _get_engine opens.
    job = PriorityQueueJob(
        job_id=f"pq-{uuid.uuid4().hex[:8]}",
        checkpoint_path=str(checkpoint_path),
        images_dir=str(images_dir),
        dataset_root=str(dataset_root),
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
