"""Reconstruct a calibration reference from human review verdicts (W1).

Turns per-image review verdicts (the same shards ``materialize.py`` reads) into the COCO record
shape ``resolve_operating_point`` consumes — so a breeder-confirmed sample of the model's own
outputs can validate the count operating point, not only dense held-out GT (the shared-reference
principle, CLAUDE.md). Per record:

  - ``gt`` = the boxes the breeder affirmed exist: accepted/edited matches, confirmed misses (FN),
    and false-positives the breeder promoted to real (accepted FP). Rejected boxes never enter gt.
  - ``dt`` = the model's own predictions carried with their recorded confidence, regardless of
    verdict — so the sweep re-derives TP/FP/FN by center-matching dt against the affirmed gt exactly
    as the GT path does.

The review-confirmed reference passes the IDENTICAL disjoint-split + count-bias gate the held-out-GT
path passes; ``resolve_operating_point`` stamps it ``review_confirmed`` (distinct from
``validated_held_out`` so provenance records which reference validated). The conf-censoring guard in
``resolve_operating_point`` still applies: verdicts whose predictions were staged above the display
floor are truncated and cannot stamp a validated claim — the reviewed predictions must have been
generated at a floored conf for the sweep to reach the low-conf tail (the G1 precondition).

Torch-free.
"""

from __future__ import annotations

import hashlib
import json

from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED, ResolvedBundle

_POSITIVE_ACTIONS = {"accepted", "edited"}


def _to_xywh(box_norm: list, img_w: float, img_h: float) -> list[float]:
    """Normalized center-form ``[cx, cy, w, h]`` -> top-left ``[x, y, w, h]`` scaled by image dims.

    With no image dimensions the unit square (1.0, 1.0) keeps every record on one consistent
    normalized scale — valid for the count sweep, whose tolerance is derived from the same records.
    """
    cx, cy, bw, bh = (float(v) for v in box_norm)
    return [(cx - bw / 2) * img_w, (cy - bh / 2) * img_h, bw * img_w, bh * img_h]


def review_to_records(
    review_state: dict,
    *,
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
) -> list[dict]:
    """Reconstruct per-image COCO records (gt=affirmed, dt=model predictions) from review verdicts.

    ``image_dims`` maps image name -> ``(width, height)`` to denormalize boxes to pixels (the
    faithful scale); omit it to keep records on the normalized unit square. ``only_completed``
    restricts to fully-reviewed images (a partially-reviewed image is not a confirmed reference).
    Each record carries ``image_id=img_name`` so the disjointness the gate checks is provable.
    """
    dims = image_dims or {}
    records: list[dict] = []
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        img_w, img_h = dims.get(img_name, (1.0, 1.0))
        gt: list[dict] = []
        dt: list[dict] = []
        for entry in img_data.get("detections", []):
            action = entry.get("action")
            cid = int(entry.get("class_id", 0))
            gt_norm = entry.get("gt_bbox_norm")
            pred_norm = entry.get("pred_bbox_norm")
            conf = entry.get("conf")
            # dt: the model's own prediction with its recorded score (any verdict that has one).
            if pred_norm and len(pred_norm) == 4 and conf is not None:
                dt.append({"category_id": cid + 1, "bbox": _to_xywh(pred_norm, img_w, img_h),
                           "score": float(conf)})
            # gt: boxes the breeder affirmed exist (accepted FP carries only a predicted box).
            if action in _POSITIVE_ACTIONS:
                box = gt_norm or pred_norm
                if box and len(box) == 4:
                    gt.append({"category_id": cid + 1, "bbox": _to_xywh(box, img_w, img_h),
                               "iscrowd": 0})
        records.append({"width": int(img_w), "height": int(img_h),
                        "image_id": img_name, "gt": gt, "dt": dt})
    return records


def review_reference_hash(records: list[dict]) -> str:
    """Content hash of the review-confirmed reference (image names + affirmed gt boxes).

    Scopes the derived conf to *this* reference so the firewall can flag it being inherited across a
    different one — the review analogue of ``resolution.dataset_hash`` over label bytes.
    """
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: str(r.get("image_id", ""))):
        h.update(str(rec.get("image_id", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(rec.get("gt", []), sort_keys=True).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def resolve_operating_point_from_review(
    review_state: dict,
    trait_name: str,
    *,
    image_dims: dict[str, tuple[int, int]] | None = None,
    only_completed: bool = True,
    tile_size: int | None = None,
    tiled: bool | None = None,
    cross_tile_nms: float | None = None,
    max_dets: int | None = None,
) -> ResolvedBundle:
    """Resolve the count operating point from review verdicts (the review-confirmation reference).

    Splits the reviewed images into a disjoint calibration/holdout half (by image, mirroring the
    GT calibration path) and hands both to ``resolve_operating_point`` with
    ``validated_reference='review_confirmed'`` — so the SAME disjoint + count-bias gate decides
    whether the conf is shippable, and the conf-censoring guard still fails a display-floored
    reference closed. Returns a bundle whose conf is stamped ``review_confirmed`` only if that gate
    passes, else ``false``.
    """
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    records = review_to_records(review_state, image_dims=image_dims, only_completed=only_completed)
    ref_hash = review_reference_hash(records)
    ordered = sorted(records, key=lambda r: str(r.get("image_id", "")))
    mid = max(1, len(ordered) // 2)
    cal_records, hold_records = ordered[:mid], ordered[mid:]
    return resolve_operating_point(
        trait_name, dataset_hash=ref_hash,
        calibration_records=cal_records or None,
        holdout_records=hold_records or None,
        tile_size=tile_size, tiled=tiled, cross_tile_nms=cross_tile_nms, max_dets=max_dets,
        validated_reference=VALIDATED_REVIEW_CONFIRMED,
    )
