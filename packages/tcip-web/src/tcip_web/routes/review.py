"""Review routes: verdict/GT recording (compute matches, walk detections, record actions, save
GT) plus the image-status group (mark_complete, backup_labels, image_statuses,
generation_conf) and the priority queue. Promoting a review into a validation reference lives in
``routes/validation.py``, which shares this module's engine cache, audit writer and bucket-key
helpers rather than a second implementation of them.

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
from tcip_annotation.json_io import (
    UnreadableLabelDocument, bbox_from_corners, check_box_extent,
    prediction_documents, read_annotations,
)
from tcip_annotation.review_engine import capture_label_baseline
from tcip_annotation.state import Annotation
from tcip_annotation.verdicts import VerdictAction
from tcip_mcp.dataset_layout import annotations_hold_subject, derive_status
from tcip_mcp.pipelines.image_utils import (
    AmbiguousImageStem, image_dimensions, resolve_image_source,
)
from tcip_web import jobstore
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
    decoded through; the registry could have changed since). Under a classified scope
    ``class_name`` is the value the verdict confirmed, a genuine key of the bucket's value-keyed
    map; the case that remains unresolved is a bucket with no recorded ``id_map`` at all (a bare
    hand-split directory) or ``class_name`` naming a foreign, stale definition. ``None`` in either
    case; see ``review_calibration.review_to_records``, which refuses rather than guesses when this
    is ``None``. Never defaults to 0: an unresolved identity is an honest fact, not a class.
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
    # multi-band raster) into the frame image_dimensions measures, not a bare PIL header read.
    try:
        return image_dimensions(resolve_image_source(p.parent, p.stem))
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc


def _guard_path(path: Optional[str]) -> Optional[str]:
    """Confine a client-supplied label/dir path and hand back its resolved spelling, or None.

    ``/action`` / ``backup_labels`` write to caller-provided paths, so every writer uses the
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
    Called from ``/action``, the one route that rewrites a GT file.
    """
    if not label_path:
        return
    if not Path(label_path).is_file():
        return
    capture_label_baseline(label_path)


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


def _review_scope(
    pred_path: Optional[str], stated_subject: Optional[str], stated_attribute: Optional[str],
):
    """The ``(subject, attribute)`` axis this review reads under, resolved from the bucket the
    prediction file lies in, never the caller's statement alone.

    No prediction file: the caller's own statement governs (``_check_classification_scope`` still
    refuses a stated attribute with no subject). A bucket with no stamp: a stated attribute refuses
    (400, the bucket carries no scope of its own to classify along), else a detector review under
    the caller's own statement. A stamp that will not decode: 400 with the seam's own error. A
    stamp carrying neither key: 400 naming the conform script. A classified stamp: the bucket's own
    scope, whether or not the caller stated one; a stated pair that disagrees refuses. A detector
    stamp: a stated attribute refuses; otherwise a detector review under the bucket's own subject.
    """
    from tcip_mcp.pipelines.resolution import BucketScope, StampScopeUnstated, bucket_scope
    from tcip_store import StoreError

    _check_classification_scope(stated_subject, stated_attribute)
    if not pred_path:
        return BucketScope(subject=stated_subject, attribute=stated_attribute)
    bucket_dir = str(Path(pred_path).parent)
    try:
        scope = bucket_scope(bucket_dir)
    except StampScopeUnstated as exc:
        raise HTTPException(400, str(exc)) from exc
    except StoreError as exc:
        raise HTTPException(400, str(exc)) from exc
    if scope is None:
        if stated_attribute is not None:
            raise HTTPException(
                400, "this directory carries no stamp and no scope; a classified review reads "
                     "the bucket's own")
        return BucketScope(subject=stated_subject, attribute=None)
    if scope.classified:
        if stated_attribute is not None and (stated_subject, stated_attribute) != (
                scope.subject, scope.attribute):
            raise HTTPException(400, (
                f"this bucket's stamp records scope (subject={scope.subject!r}, "
                f"attribute={scope.attribute!r}), not the stated (subject={stated_subject!r}, "
                f"attribute={stated_attribute!r})"
            ))
        return scope
    if stated_attribute is not None:
        raise HTTPException(
            400, "this is a detector bucket, with no attribute to classify along; a classified "
                 "review needs a classified bucket")
    return scope


def _compute_matches(
    gt: list, preds: list, *, iou_threshold: float, conf_threshold: float,
    subject: Optional[str], attribute: Optional[str], vocabulary=None,
) -> dict:
    """Dispatch to plain detection matching, or classified-trait matching when the caller names the
    (subject, attribute) axis under review. The one call site both ``/matches`` and ``/action`` use,
    so a verdict's freshly recomputed matches are always scoped identically to what produced it."""
    if attribute is None:
        return compute_matches(gt, preds, iou_threshold, conf_threshold)
    _check_classification_scope(subject, attribute)
    return compute_classified_trait_matches(
        gt, preds, subject=subject, attribute=attribute, vocabulary=vocabulary or set(),
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
    scope=None,
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
        subject=scope.subject if scope is not None else None,
        attribute=scope.attribute if scope is not None else None,
    )


def _bucket_vocabulary(pred_path: Optional[str]) -> set:
    """The bucket's own recorded ``id_map`` keys, the vocabulary a classified match holds every
    prediction record to; empty for no prediction file or no recorded map."""
    if not pred_path:
        return set()
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map

    return set(bucket_id_map(Path(pred_path).parent) or {})


@router.post("/matches")
def compute_image_matches(req: MatchesRequest) -> MatchesResponse:
    """Compute TP/FP/FN, decorate with review status, and return everything the canvas needs."""
    ctx = _load_ctx(req.image_name, req.image_path, gt_path=req.gt_path, pred_path=req.pred_path)
    engine = _get_engine(req.dataset_root)
    scope = _review_scope(req.pred_path, req.subject, req.attribute)
    from tcip_annotation.json_io import ClassifiedRecordRefused

    try:
        matches = _compute_matches(
            ctx.gt, ctx.preds, iou_threshold=req.iou_threshold, conf_threshold=req.conf_threshold,
            subject=scope.subject, attribute=scope.attribute,
            vocabulary=_bucket_vocabulary(req.pred_path),
        )
    except ClassifiedRecordRefused as exc:
        raise HTTPException(400, str(exc)) from exc
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


def _is_reviewer_drawn_new_shape(payload: "ActionPayload") -> bool:
    """A reviewer drew a brand-new shape from scratch: no matched GT, no matched prediction, and
    not a sweep attestation (``ReviewTab.tsx``'s missed-object gesture, posted through ``/action``).

    Under a classified scope this is refused by :func:`_apply_gt_mutation`: the tab posts the
    missed object with ``class_name = dataset.subject`` (the object class), and authoring that as
    the attribute's value would fabricate a state nobody assessed.
    """
    return payload.gt_idx is None and payload.pred_idx is None and payload.action != "swept"


def _check_classified_value(class_name: str, vocabulary: set) -> None:
    """Refuse a value a classified bucket's own ``id_map`` does not declare."""
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
    vocabulary: set,
) -> tuple[bool, Optional[int]]:
    """Author GT from a verdict; return ``(gt_changed, index the written annotation landed at in
    ctx.gt)``: the index is set only for edited/accepted writes. ``action="swept"`` (an explicit
    "checked this image for missed objects, found none" attestation) matches none of the branches
    below and always no-ops, GT is never mutated by sweeping.

    ``scope`` is the resolved review scope (``_review_scope``), never ``payload.subject``/
    ``payload.attribute`` directly; ``vocabulary`` is the bucket's own recorded ``id_map`` keys
    (``_bucket_vocabulary``). Under a classified scope (``scope.attribute`` set) a verdict judges
    the *value* of an object a person already placed, checking a written ``payload.class_name``
    against ``vocabulary`` and an edited record's own subject against ``scope.subject``, refusing
    by name rather than trusting the client's string; under a detector review it judges the
    object's presence, as it always has.

    Accept on a false positive: a paired one (``payload.gt_idx`` set, its partner a ground-truth
    record of the subject whose value differs) replaces the confirmed value on the person's own
    record, keeping their geometry and authorship; an unpaired one appends a fresh GT record from
    the prediction, its origin traveling with it. Under a detector review, accept always appends
    with empty ``attributes``: ``reviewed`` is exactly the attribute values this review adjudicated,
    none under a detector review, which adjudicated presence and nothing about state. Reject on a
    false positive leaves ground truth untouched under either regime. Reject on a true positive or
    false negative under a classified scope refuses: removing the object is a detector-scope act.

    Edit authors the edited geometry onto the record it edits (a true positive/false negative, or a
    paired false positive) with the reviewer as author, keeping the record's other attribute values
    and dropping any sign-off; a stated ``gt_idx`` out of range refuses. An unpaired false positive
    edited into ground truth is a fresh record, no score and no sign-off. A reviewer-drawn new
    shape (:func:`_is_reviewer_drawn_new_shape`) refuses under a classified scope.
    """
    dt, act = payload.det_type, payload.action
    classifying = scope.attribute is not None

    if classifying and _is_reviewer_drawn_new_shape(payload):
        raise ValueError(
            "a reviewer-drawn new shape cannot be authored under a classified-trait review: it "
            "would fabricate a state nobody assessed. Add the object through the Annotate tab, "
            "then review its value here."
        )

    if act == "edited":
        geom: BBox | Polygon | None = None
        if payload.edited_box is not None:
            geom = bbox_from_corners(*payload.edited_box, where=f"editing {payload.class_name!r}")
        elif payload.edited_points is not None:
            # The reviewer edits one contour by hand on the canvas: single-ring input.
            geom = Polygon(rings=[[(float(p[0]), float(p[1])) for p in payload.edited_points]])
        if geom is None:
            return False, None
        if payload.gt_idx is not None:
            if not (0 <= payload.gt_idx < len(ctx.gt)):
                raise ValueError(
                    f"gt_idx {payload.gt_idx} is out of range for this image's {len(ctx.gt)} "
                    "ground-truth record(s): the record this edit targets no longer exists."
                )
            existing = ctx.gt[payload.gt_idx]
            if classifying:
                _check_target_subject(existing, scope)
                _check_classified_value(payload.class_name, vocabulary)
            attrs = dict(existing.attributes)
            if classifying:
                attrs[scope.attribute] = payload.class_name
            ctx.gt[payload.gt_idx] = replace(
                existing, geometry=geom, attributes=attrs, created_by=reviewer,
                created_at=now_iso, accepted_by=None, accepted_at=None)
            return True, payload.gt_idx
        # An unpaired false positive edited into ground truth: a fresh record.
        if classifying:
            _check_classified_value(payload.class_name, vocabulary)
        reviewed = {scope.attribute: payload.class_name} if classifying else {}
        new_subject = scope.subject if classifying else payload.class_name
        ctx.gt.append(Annotation(subject=new_subject, geometry=geom, attributes=reviewed,
                                 created_by=reviewer, created_at=now_iso))
        return True, len(ctx.gt) - 1

    if act == "rejected" and dt in ("tp", "fn"):
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

    if act == "accepted" and dt == "fp" and payload.pred_idx is not None \
            and 0 <= payload.pred_idx < len(ctx.preds):
        pred = ctx.preds[payload.pred_idx]
        if isinstance(pred.geometry, BBox):
            check_box_extent(pred.geometry, where=f"accepting {payload.class_name!r}")
        if classifying and payload.gt_idx is not None and 0 <= payload.gt_idx < len(ctx.gt):
            # A paired false positive: the person's object, a wrong value confirmed. Keep their
            # geometry and authorship; replace only the confirmed value.
            existing = ctx.gt[payload.gt_idx]
            _check_target_subject(existing, scope)
            _check_classified_value(payload.class_name, vocabulary)
            attrs = dict(existing.attributes)
            attrs[scope.attribute] = payload.class_name
            ctx.gt[payload.gt_idx] = replace(
                existing, attributes=attrs, accepted_by=reviewer, accepted_at=now_iso)
            return True, payload.gt_idx
        if classifying:
            _check_classified_value(payload.class_name, vocabulary)
            accepted = replace(pred, score=None, attributes={scope.attribute: payload.class_name},
                               accepted_by=reviewer, accepted_at=now_iso)
        else:
            accepted = replace(pred, score=None, attributes={},
                               accepted_by=reviewer, accepted_at=now_iso)
        ctx.gt.append(accepted)
        return True, len(ctx.gt) - 1

    return False, None  # accept TP/FN and reject FP leave GT untouched


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision; auto-complete the image when done."""
    from tcip_annotation.json_io import ClassifiedRecordRefused

    gt_path = _guard_path(payload.gt_path)
    pred_path = _guard_path(payload.pred_path)
    scope = _review_scope(pred_path, payload.subject, payload.attribute)
    vocabulary = _bucket_vocabulary(pred_path)
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
        changed, landed_idx = _apply_gt_mutation(
            work, payload, reviewer, now_iso, scope=scope, vocabulary=vocabulary)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if changed and not gt_path:
        raise HTTPException(
            400, "this verdict writes ground truth, but no annotations path was provided")

    # Recompute over the mutated in-memory documents before any write, so a refusal here leaves
    # neither the verdict log nor the GT file touched, never a write on disk behind a 400.
    bucket = _bucket_of_file(pred_path)
    try:
        matches = _compute_matches(
            work.gt, ctx.preds,
            iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold,
            subject=scope.subject, attribute=scope.attribute, vocabulary=vocabulary,
        )
    except ClassifiedRecordRefused as exc:
        raise HTTPException(400, str(exc)) from exc

    # An edited verdict rewrites the GT geometry, so key the entry to the post-edit geometry;
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == "edited" and changed and landed_idx is not None:
        norm_det = replace(det, gt_idx=landed_idx)
        norm_ctx = work
    producer_identity = _resolve_producer_identity(pred_path)
    class_id = _resolve_verdict_class_id(pred_path, payload.class_name)
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
        filter_type=payload.filter_type, filter_class=payload.filter_class, scope=scope,
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
    every subject). A named subject reads the bucket's own recorded scope
    (``resolution.bucket_scope``) first: a classified stamp admits exactly its own object class and
    answers ``None`` for any other name, so a value name (a key of the bucket's own value-keyed
    ``id_map``) can never be recorded as a negative subject. A bare directory or a detector stamp
    keeps the map-membership admission (``phenology.bucket_id_map``, the same map
    ``_resolve_verdict_class_id`` reads for a verdict's class identity): membership proves the
    bucket assessed this subject at all, and the comparison itself is by the decoded name
    (``cached_label_annotations``' own ``subject`` field), never the id, so the two branches read
    the file through this one memo and cannot disagree about its current content. A neither-key or
    undecodable stamp answers ``None`` too, caught here rather than left to the route (whose own
    catch covers only :class:`~tcip_annotation.json_io.UnreadableLabelDocument`): the Complete and
    its status write still proceed, with no coverage entry for a subject nothing here can resolve.
    """
    if not pred_dir:
        return True
    pred_file = Path(pred_dir) / f"{Path(image_name).stem}.json"
    if subject is None:
        return not _has_objects(pred_file)
    from tcip_mcp.pipelines.resolution import StampScopeUnstated, bucket_scope
    from tcip_store import StoreError

    try:
        scope = bucket_scope(Path(pred_dir))
    except (StampScopeUnstated, StoreError):
        return None
    if scope is not None and scope.classified:
        if subject != scope.subject:
            return None
        return not any(a.subject == subject for a in cached_label_annotations(pred_file))
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map

    id_map = bucket_id_map(Path(pred_dir))
    if id_map is None or subject not in id_map:
        return None
    return not any(a.subject == subject for a in cached_label_annotations(pred_file))


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

# prioritize_review_queue's own ranking (feedback_tools.py) never reached the breeder-facing Review tab, so this surfaces the same tool, never reimplemented, as a browsable queue on a background thread (a forward pass per image can be slow), polled for the result.

# Its sibling door, triage_predictions, can auto-accept predictions as GT above a breeder-confirmed threshold, a different and more consequential capability deliberately left agent/operator-only for now.


REVIEW_PRIORITY_REGISTRY = jobstore.REVIEW_PRIORITY_JOBS
"""The job registry this module persists its priority-queue jobs to."""


def _pq_current_root() -> str:
    from tcip_web import jobstore
    return jobstore.current_root()


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
    # [{image, score, calibration_member?}], highest first; calibration_member is present only
    # when the checkpoint's run was bound to a split manifest that could be read.
    queue: list[dict] = field(default_factory=list)
    total_candidates: int = 0
    reviewed_skipped: int = 0
    # Set when the run's split record, or its named manifest, could not be read: no entry above
    # carries calibration_member, and this names why, rather than a guess.
    marks_unresolved: Optional[str] = None
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    # The platform root this job launched under, resolved on the request thread.
    platform_root: str = field(default_factory=_pq_current_root)


def _pq_summary(job: PriorityQueueJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "error": job.error,
        "queue": job.queue, "total_candidates": job.total_candidates,
        "reviewed_skipped": job.reviewed_skipped, "platform_root": job.platform_root,
        "marks_unresolved": job.marks_unresolved,
    }


def _pq_from_summary(s: dict, root: str) -> PriorityQueueJob:
    return PriorityQueueJob(
        job_id=s["job_id"],
        checkpoint_path="",
        images_dir="",
        dataset_root="",
        status=jobstore.rehydrated_status(s),
        error=s.get("error"),
        queue=s.get("queue") or [],
        total_candidates=s.get("total_candidates", 0),
        reviewed_skipped=s.get("reviewed_skipped", 0),
        marks_unresolved=s.get("marks_unresolved"),
        platform_root=jobstore.require_platform_root(s, name=REVIEW_PRIORITY_REGISTRY, root=root),
    )


_pq_registry = jobstore.JobRegistry(
    REVIEW_PRIORITY_REGISTRY, to_summary=_pq_summary, from_summary=_pq_from_summary,
)
"""The dict-plus-lock live registry for this queue's own jobs (see ``jobstore.JobRegistry``),
the shared home inference.py's and tuning.py's own registries adopt too."""


def _pq_persist() -> None:
    _pq_registry.persist()


def _pq_register(job: PriorityQueueJob) -> None:
    _pq_registry.register(job.job_id, job, job_root=job.platform_root)


def _pq_get(job_id: str) -> Optional[PriorityQueueJob]:
    """A job by id, from any root this process holds: a repin to another project must not
    make an in-flight priority-queue job unreachable, the same contract inference and tuning
    hold for their own by-id lookups."""
    return _pq_registry.get(job_id)


def rehydrate_for_current_root() -> None:
    """Merge this root's persisted priority-queue jobs, not already live, into memory via
    :func:`_pq_from_summary`.

    Called at startup and again after this process repins to another root, the same
    treatment ``routes.inference``/``routes.tuning`` give their own registries. The worker
    thread behind a persisted non-terminal job is gone, so it is surfaced as ``interrupted``;
    its ranked queue is restored from what :func:`_pq_summary` persisted, so a completed job
    still answers its own ranked images after a restart or a repin. Bounds the dict
    afterwards the same way registering a job does, so adopting N roots without ever
    registering one here still keeps this process's memory bounded rather than growing by
    ``MAX_JOBS`` for every root adopted.
    """
    _pq_registry.rehydrate()


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
            method=job.method,
            budget=job.budget,
            project_path=job.platform_root,
        )
        if "error" in result:
            job.status = "failed"
            job.error = result["error"]
        else:
            job.status = "completed"
            job.queue = result["queue"]
            job.total_candidates = result["total_candidates"]
            job.reviewed_skipped = result["reviewed_skipped"]
            job.marks_unresolved = result.get("marks_unresolved")
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
    # checkpoint_path is confined to the allowed roots, same as the Inference tab's own launch
    # route: a caller must not name a file outside them, registered checkpoint or not.
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
