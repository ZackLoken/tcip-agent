"""Review routes: compute matches, walk detections, record actions, save GT.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance lives in memory per
project (keyed by project_root). Review state is persisted via the engine to per-image shards under
``<project_root>/.tcip/state/review/``.

Ground truth and predictions are each one JSON file per image holding every subject's annotations by
name (a prediction is an :class:`~tcip_annotation.state.Annotation` whose ``score`` is set); a class
is named by its ``subject``, never an integer id, so the recorded verdict carries the real subject
name with no registry lookup.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_annotation import (
    BBox,
    Polygon,
    ReviewContext,
    ReviewDetection,
    ReviewEngine,
    compute_matches,
)
from tcip_annotation.json_io import read_annotations
from tcip_annotation.state import Annotation
from tcip_annotation.utils import get_image_dimensions
from tcip_mcp.utils.atomic_io import append_jsonl, atomic_write_json, read_json
from tcip_web.identity import resolve_user, user_id
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/review", tags=["review"])


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _current_user() -> str:
    """Reviewer fallback when the GUI request omits ``user`` — env override else the OS login."""
    from tcip_web.identity import current_user

    return current_user()


def _get_engine(project_root: str) -> ReviewEngine:
    key = str(Path(project_root).resolve())
    if key not in _engines:
        state_dir = Path(project_root) / ".tcip" / "state"
        _engines[key] = ReviewEngine(state_dir=state_dir, current_user=_current_user())
    return _engines[key]


def _audit(project_root: str, tool: str, arguments: dict) -> None:
    """Append a GUI review mutation to ``<project_root>/.tcip/audit.jsonl`` (best-effort).

    Review verdicts + GT writes change tracked state, so — like @audited MCP tools and
    the annotate save path — they belong in the append-only log. Never fails the request.
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


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    return get_image_dimensions(str(p))  # header-only (w, h); never decodes pixels


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
    exists yet — a per-file, O(1) safety net so a verdict never overwrites the pristine original
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
    """Serialize an :class:`Annotation` for the canvas (pixel coords + attributes + provenance)."""
    out: dict = {"subject": a.subject, "attributes": dict(a.attributes)}
    geom = a.geometry
    if isinstance(geom, Polygon):
        out["points"] = [list(pt) for pt in geom.points]
    elif isinstance(geom, BBox):
        out["bbox"] = [geom.x1, geom.y1, geom.x2, geom.y2]
    if a.score is not None:
        out["score"] = a.score
    out["created_by"] = a.created_by
    out["created_at"] = a.created_at
    out["accepted_by"] = a.accepted_by
    out["accepted_at"] = a.accepted_at
    return out


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
    identical shape — letting a verdict return its fresh matches instead of forcing a second fetch."""
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
    matches = compute_matches(
        ctx.gt, ctx.preds, iou_threshold=req.iou_threshold, conf_threshold=req.conf_threshold
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
    action: str  # "accepted" | "rejected" | "edited"
    # GUI-set reviewer identity (bare name, e.g. "zack"); stamped as accepted_by/created_by
    # ("user:<name>"). Omitted by non-GUI callers -> backend falls back to the OS/env user.
    user: Optional[str] = None
    # Edited shape committed from the Review canvas (only for action="edited"): a box, or a
    # polygon's points. Accept/Reject don't carry these — they act on the loaded pred/gt by index.
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


def _apply_gt_mutation(
    ctx: ReviewContext, payload: "ActionPayload", reviewer: str, now_iso: str
) -> tuple[bool, Optional[int]]:
    """Author GT from a verdict; return ``(gt_changed, index the written annotation landed at in
    ctx.gt)`` — the index is set only for edited/accepted writes. Accept an FP adds the prediction;
    accept a TP/FN keeps GT; reject a TP/FN deletes that GT; reject an FP is a no-op; edit writes
    the edited shape (replacing the matched GT, or adding it).

    Provenance (``reviewer`` = ``user:<name>``, ``now_iso`` = UTC): an accepted prediction **carries**
    its ``created_by``/``created_at`` into GT (origin travels) and gets ``accepted_by``/``accepted_at``
    with its ``score`` dropped (it is ground truth now); a reviewer-drawn edit is stamped
    ``created_by`` = reviewer."""
    dt, act = payload.det_type, payload.action

    if act == "edited":
        geom: BBox | Polygon | None = None
        if payload.edited_box is not None:
            geom = BBox(*payload.edited_box)
        elif payload.edited_points is not None:
            geom = Polygon(points=[(float(p[0]), float(p[1])) for p in payload.edited_points])
        if geom is None:
            return False, None
        new = Annotation(subject=payload.class_name, geometry=geom,
                         created_by=reviewer, created_at=now_iso)
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
        ctx.gt.append(replace(pred, score=None, accepted_by=reviewer, accepted_at=now_iso))
        return True, len(ctx.gt) - 1

    return False, None  # accept TP/FN and reject FP leave GT untouched


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision; auto-complete the image when done."""
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

    # An edited verdict rewrites the GT geometry, so key the entry to the post-edit geometry —
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == "edited" and changed and landed_idx is not None:
        norm_det = replace(det, gt_idx=landed_idx)
        norm_ctx = work
    engine.record_detection_action(
        det, ctx, action=payload.action, norm_det=norm_det, norm_ctx=norm_ctx
    )

    # Write the single per-image GT file (keep_empty: an emptied GT stays an {"annotations": []}
    # record, not deleted). accept-TP/FN and reject-FP are no-ops.
    if changed and payload.gt_path:
        _guard_path(payload.gt_path)
        _ensure_original_backup(payload.gt_path)  # baseline this file before its first mutation
        engine.save_gt(work, path=payload.gt_path)

    # Annotation status to sync client-side (only when GT changed); an emptied GT reads as
    # "unannotated" — a negative needs an explicit Complete, not just an empty file.
    annotation_status: Optional[str] = None
    if changed:
        annotation_status = "partial" if work.gt else "unannotated"

    # Promote to 'completed' once every detection at these thresholds is reviewed — the only path
    # by which a GUI review reaches 'completed'. Recompute against the (now-authored) GT.
    matches = compute_matches(
        work.gt, ctx.preds,
        iou_threshold=payload.iou_threshold, conf_threshold=payload.conf_threshold,
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
    completed: bool = True  # False reverses a manual mark (verdicts are kept)


@router.post("/mark_complete")
def mark_complete(payload: MarkCompletePayload) -> dict:
    """Mark (or unmark) an image fully reviewed; covers negatives / bulk-accept cases."""
    _guard_path(payload.gt_path)
    engine = _get_engine(payload.project_root)
    if payload.completed:
        engine.mark_image_reviewed(payload.image_name)
    else:
        engine.unmark_image_reviewed(payload.image_name)
    # Derive the annotation status from the GT file on disk — the client's matches snapshot can be
    # stale or null mid-navigation and once wrote negatives for annotated frames. A present file
    # with no annotations of any subject is an empty (negative) record.
    has_content = bool(payload.gt_path and os.path.isfile(payload.gt_path)
                       and read_annotations(payload.gt_path))
    if payload.completed:
        annotation_status = "complete" if has_content else "negative"
    else:
        annotation_status = "partial" if has_content else "unannotated"
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
    """Top up ``<dir>/.original/`` — capture any label file that has no baseline yet."""
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
    # [{subject, bbox?: [x1,y1,x2,y2], points?: [[x,y]...], attributes?, created_by?, ...}]
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

    def _to_annotation(d: dict) -> Annotation:
        geom: BBox | Polygon | None = None
        if d.get("points"):
            geom = Polygon(points=[(float(p[0]), float(p[1])) for p in d["points"]])
        elif d.get("bbox") is not None:
            geom = BBox(*d["bbox"])
        round_tripped = bool(d.get("created_by"))
        return Annotation(
            subject=d["subject"],
            geometry=geom,
            attributes=dict(d.get("attributes") or {}),
            created_by=d.get("created_by") or author,
            created_at=d.get("created_at") or now_iso,
            # accepted_* only on round-tripped shapes — a new shape must not mint sign-off.
            accepted_by=d.get("accepted_by") if round_tripped else None,
            accepted_at=d.get("accepted_at") if round_tripped else None,
        )

    ctx = ReviewContext(
        img_name=payload.image_name, img_width=w, img_height=h,
        gt=[_to_annotation(d) for d in payload.annotations],
    )
    ok = engine.save_gt(ctx, path=payload.label_path)
    _audit(payload.project_root, "gui_review_save_gt", {
        "image_name": payload.image_name,
        "label_path": payload.label_path,
        "n_annotations": len(payload.annotations),
    })
    return {"status": "ok" if ok else "partial"}


# ── Promote a completed review into a validation reference (D17) ───────────


class ValidateReferenceRequest(BaseModel):
    project_root: str
    trait: str
    # The prediction bucket whose review is being promoted — the per-image prediction dir the
    # delivery gate reads an ``operating_point.json`` from.
    pred_dir: Optional[str] = None


class ValidateReferenceResponse(BaseModel):
    # True only when the review cleared the identical gate the backend uses (or the bucket was already
    # validated). A refusal is surfaced honestly here, never silently upgraded.
    validated: bool
    reference: Optional[str]  # "review_confirmed" | "validated_held_out" | "false" | None
    reviewed_image_count: int
    conf: Optional[float]  # the derived count operating point (for transparency)
    reason: str  # plain-language, breeder-facing — always present
    buckets_stamped: list[str]


@router.post("/validate_reference")
def validate_reference(req: ValidateReferenceRequest) -> ValidateReferenceResponse:
    """Promote a completed review session into a validation reference for its (model, trait, date-set).

    Reconstructs the review verdicts into the COCO records ``resolve_operating_point`` consumes (W1's
    ``review_calibration`` adapter) and runs them through the IDENTICAL disjoint-split + count-bias gate
    and conf-censoring guard the held-out-GT path uses — no shortcut to "validated". On success the
    bucket's ``operating_point.json`` is stamped ``review_confirmed`` (so the delivery gate can read it);
    on refusal an honest ``validated=false`` placeholder is written and the reason is returned. An
    already-validated bucket is never downgraded.
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

    # Never downgrade: predictions already validated (e.g. against held-out GT) stay validated — a
    # review reference isn't needed there, and this action must not be able to lower them.
    sidecars = {d: (read_operating_point_sidecar(d) or {}) for d in bucket_dirs}
    if all(sc.get("validated") for sc in sidecars.values()):
        ref = next((((sc.get("operating_point") or {}).get("conf") or {}).get("validated_vs_gt")
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
        review_reference_hash,
        review_to_records,
    )
    from tcip_mcp.traits import TraitUnknownError

    review_state = {"image": completed}
    # K1: thread the producing run's experiment_id through so the calibration's train-disjointness
    # gate can check the reviewed images against that run's training split. Sourced from the
    # buckets' own operating_point.json sidecars (stamped by export_predictions), never asserted —
    # when multiple buckets disagree on which run produced them, pass None (mixed-provenance
    # shouldn't silently vouch for one run's disjointness) rather than raising, so this route keeps
    # working for a legitimate multi-bucket review call.
    bucket_exp_ids = {sc.get("experiment_id") for sc in sidecars.values() if sc.get("experiment_id")}
    review_experiment_id = next(iter(bucket_exp_ids)) if len(bucket_exp_ids) == 1 else None
    try:
        bundle = resolve_operating_point_from_review(
            review_state, req.trait, only_completed=True, experiment_id=review_experiment_id)
    except TraitUnknownError:
        raise HTTPException(
            400,
            f"a validation reference is not defined for trait {req.trait!r} yet — this action is "
            "available for traits the platform can calibrate a count operating point for.",
        ) from None
    except ValueError as exc:
        # A locked cal/holdout split refusing this call (K1 finding 4): a reviewed image was
        # deleted/renamed since the split locked, or the lock file itself is corrupt. Either way
        # this is an honest refusal, not a 500 — surface it as such.
        raise HTTPException(400, str(exc)) from None

    result = describe_review_validation(bundle, reviewed_image_count=n)

    # Stamp each bucket's provenance sidecar (operating_point.json is not a label, so this never
    # touches the reviewed per-image predictions or the verdict-immutability guard).
    op_prov = bundle.to_provenance()["operating_point"]
    ref_hash = review_reference_hash(review_to_records(review_state))
    now_iso = datetime.now(timezone.utc).isoformat()
    stamped: list[str] = []
    for d in bucket_dirs:
        if sidecars[d].get("validated"):
            continue  # a mixed set: leave an already-validated bucket untouched (no downgrade)
        sidecar = dict(sidecars[d])
        sidecar.update({
            "operating_point": op_prov,
            "validated": result["validated"],
            "validated_reference": result["reference"],
            "validation_source": "review_confirmed",
            "review_reference_hash": ref_hash,
            "review_image_count": n,
            "shippable_issues": bundle.shippable_issues(),
            "validated_at": now_iso,
        })
        sidecar.setdefault("produced_at", now_iso)
        Path(d).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(d) / "operating_point.json", sidecar)
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
    # Stems (filename without extension) whose GT or prediction file holds at least one annotation
    # — i.e. the image has something to review. Images whose stem is absent contribute no TP/FP/FN,
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
    """Batch review status + detection presence for a whole (date) — one call the Review tab makes
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
