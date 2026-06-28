"""Task-aware evaluation metrics + composite selection objective.

Single home for:
  * the pycocotools-backed detection / instance_seg metrics (mAP + operating-point
    TP/FP/FN), shared by training ``_validate``, ``run_test_evaluation`` and the
    agent/GUI tools ``evaluate_dataset`` / ``evaluate_detections`` — one source of
    truth, the canonical COCO mAP definition;
  * in-house scalar metrics for classification / ordinal / regression (the seam
    where pycocotools ``iou_type='segm'`` can later cover true instance seg);
  * the chestnut-burr composite selection objective (lower = better);
  * a task-agnostic two-pass ``evaluate()`` and a ``run_test_evaluation()``.

pycocotools is imported lazily inside the COCO functions. Every pycocotools call
is wrapped in ``redirect_stdout`` because ``createIndex``/``loadRes``/``summarize``
print to stdout, which would corrupt the MCP stdio transport.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {"loss": 0.45, "f1": 0.35, "map50": 0.20}


# ====================================================================
# Composite selection objective (ported verbatim from chestnut-burr)
# ====================================================================

def compute_composite_objective(
    val_loss: float, f1: float, map50: float, score_weights: dict | None = None
) -> float:
    """Lower-is-better selection/tuning score blending loss, F1 and mAP50.

    ``w["loss"]*loss + w["f1"]*(1-f1)*10 + w["map50"]*(1-map50)*10`` with ``1e6``
    sentinels for degenerate runs. The ``*10`` lifts the unit-interval quality
    terms to a typical loss magnitude.
    """
    w = score_weights or DEFAULT_SCORE_WEIGHTS
    vl = float(val_loss) if (val_loss is not None and math.isfinite(val_loss)) else float("inf")
    f1v = float(f1) if (f1 is not None and math.isfinite(f1)) else 0.0
    m50 = float(map50) if (map50 is not None and math.isfinite(map50)) else 0.0
    if vl <= 0:
        return 1e6
    if f1v < 0.01 and m50 < 0.01:
        return 1e6
    return w["loss"] * vl + w["f1"] * (1.0 - f1v) * 10 + w["map50"] * (1.0 - m50) * 10


# ====================================================================
# pycocotools detection / instance_seg metrics
# ====================================================================

def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def build_coco_image_record(width: int, height: int, gt: list[dict], dt: list[dict]) -> dict:
    """One per-image entry: ``{'width','height','gt':[ann...],'dt':[res...]}``."""
    return {"width": int(width), "height": int(height), "gt": list(gt), "dt": list(dt)}


def _counts_at_operating_point(coco_eval, iou_threshold: float, conf_threshold: float) -> dict:
    """Walk ``COCOeval.evalImgs`` to extract TP/FP/FN at a (conf, iou) point."""
    p = coco_eval.params
    iou_thrs = list(p.iouThrs)
    t = min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - iou_threshold))
    area_all = p.areaRng[0]

    tp = fp = total_gt = 0
    per_image: dict[int, dict] = {}
    for e in coco_eval.evalImgs:
        if e is None or e["aRng"] != area_all:
            continue
        img_id = e["image_id"]
        gt_ignore = np.asarray(e["gtIgnore"])
        n_gt = int((gt_ignore == 0).sum())
        total_gt += n_gt
        dt_scores = np.asarray(e["dtScores"])
        dt_matches = np.asarray(e["dtMatches"])
        dt_ignore = np.asarray(e["dtIgnore"])
        e_tp = e_fp = 0
        for d in range(dt_scores.shape[0] if dt_scores.size else 0):
            if dt_scores[d] < conf_threshold or dt_ignore[t, d]:
                continue
            if dt_matches[t, d] > 0:
                e_tp += 1
            else:
                e_fp += 1
        tp += e_tp
        fp += e_fp
        rec = per_image.setdefault(img_id, {"image_id": img_id, "tp": 0, "fp": 0, "gt": 0})
        rec["tp"] += e_tp
        rec["fp"] += e_fp
        rec["gt"] += n_gt

    per_image_counts = [
        {"image_id": r["image_id"], "tp": r["tp"], "fp": r["fp"], "fn": max(r["gt"] - r["tp"], 0)}
        for r in per_image.values()
    ]
    return {"tp": tp, "fp": fp, "fn": max(total_gt - tp, 0), "per_image_counts": per_image_counts}


def coco_detection_metrics(
    per_image: list[dict],
    *,
    iou_type: str = "bbox",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_dets: int = 100,
) -> dict:
    """Run ``COCOeval`` once over ``per_image`` records and return COCO metrics.

    Returns mAP (``map``/``map50``/``map75``) plus operating-point
    ``precision``/``recall``/``f1``/``tp``/``fp``/``fn`` and per-image counts.
    Short-circuits to all-zero metrics (no exception) for empty predictions
    (``loadRes([])`` raises ``IndexError``), empty GT (COCOeval ``stats == -1``),
    or a fully empty set.
    """
    images, annotations, results = [], [], []
    cat_ids: set[int] = set()
    ann_id = 1
    n_gt = n_pred = 0
    for img_id, rec in enumerate(per_image, start=1):
        images.append({"id": img_id, "width": int(rec.get("width", 0)), "height": int(rec.get("height", 0))})
        for ann in rec.get("gt", []):
            a = dict(ann)
            a["id"] = ann_id
            a["image_id"] = img_id
            a.setdefault("iscrowd", 0)
            if "area" not in a:
                bb = a["bbox"]
                a["area"] = float(bb[2] * bb[3])
            annotations.append(a)
            cat_ids.add(int(a["category_id"]))
            ann_id += 1
            n_gt += 1
        for res in rec.get("dt", []):
            r = dict(res)
            r["image_id"] = img_id
            results.append(r)
            cat_ids.add(int(r["category_id"]))
            n_pred += 1

    base = {
        "map": 0.0, "map50": 0.0, "map75": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "tp": 0, "fp": 0, "fn": n_gt,
        "n_images": len(per_image), "n_gt": n_gt, "n_pred": n_pred,
        "per_image_counts": [
            {"image_id": i + 1, "tp": 0, "fp": 0, "fn": len(rec.get("gt", []))}
            for i, rec in enumerate(per_image)
        ],
        "iou_type": iou_type, "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold, "max_dets": max_dets,
    }
    if n_pred == 0 or n_gt == 0:
        return base

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": c, "name": str(c)} for c in sorted(cat_ids)]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
        coco_gt.createIndex()
        try:
            coco_dt = coco_gt.loadRes(results)
        except IndexError:
            return base
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        coco_eval.params.maxDets = [1, 10, max_dets]
        coco_eval.params.imgIds = [im["id"] for im in images]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        counts = _counts_at_operating_point(coco_eval, iou_threshold, conf_threshold)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "map": max(float(stats[0]), 0.0),
        "map50": max(float(stats[1]), 0.0),
        "map75": max(float(stats[2]), 0.0),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_images": len(per_image), "n_gt": n_gt, "n_pred": n_pred,
        "per_image_counts": counts["per_image_counts"],
        "iou_type": iou_type, "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold, "max_dets": max_dets,
    }


# ---- converters -----------------------------------------------------

def records_from_detector(target: dict, output: dict, *, width: int, height: int) -> dict:
    """torchvision GT target + detector output -> one COCO per-image record."""
    gt = []
    gboxes = target.get("boxes")
    if gboxes is not None and len(gboxes):
        glabels = target["labels"].detach().cpu().tolist()
        for (x1, y1, x2, y2), c in zip(gboxes.detach().cpu().tolist(), glabels):
            gt.append({"category_id": int(c), "bbox": _xyxy_to_xywh(x1, y1, x2, y2),
                       "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
    dt = []
    pboxes = output.get("boxes")
    if pboxes is not None and len(pboxes):
        plabels = output["labels"].detach().cpu().tolist()
        pscores = output["scores"].detach().cpu().tolist()
        for (x1, y1, x2, y2), c, s in zip(pboxes.detach().cpu().tolist(), plabels, pscores):
            dt.append({"category_id": int(c), "bbox": _xyxy_to_xywh(x1, y1, x2, y2), "score": float(s)})
    return build_coco_image_record(width, height, gt, dt)


def _poly_flat(points) -> list[float]:
    return [float(c) for pt in points for c in (pt[0], pt[1])]


def records_from_annotation(gt_boxes, gt_polys, pred_boxes, pred_polys, *,
                            width: int, height: int, force_segm: bool = False):
    """BBox/Polygon GT + PredBBox/PredPolygon -> (iou_type, COCO per-image record).

    ``force_segm`` makes every box carry a rectangular ``segmentation`` so a whole
    dataset can be scored with ``iou_type='segm'`` even when some images are box-only.
    """
    use_segm = force_segm or bool(gt_polys or pred_polys)
    iou_type = "segm" if use_segm else "bbox"

    def _box_seg(x1, y1, x2, y2):
        return [[float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]]

    gt = []
    for b in gt_boxes:
        rec = {"category_id": int(b.class_id) + 1, "bbox": _xyxy_to_xywh(b.x1, b.y1, b.x2, b.y2),
               "area": float((b.x2 - b.x1) * (b.y2 - b.y1)), "iscrowd": 0}
        if use_segm:
            rec["segmentation"] = _box_seg(b.x1, b.y1, b.x2, b.y2)
        gt.append(rec)
    for p in gt_polys:
        xs = [pt[0] for pt in p.points]
        ys = [pt[1] for pt in p.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        gt.append({"category_id": int(p.class_id) + 1, "bbox": _xyxy_to_xywh(x1, y1, x2, y2),
                   "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0, "segmentation": [_poly_flat(p.points)]})

    dt = []
    for b in pred_boxes:
        rec = {"category_id": int(b.class_id) + 1, "bbox": _xyxy_to_xywh(b.x1, b.y1, b.x2, b.y2),
               "score": float(b.confidence)}
        if use_segm:
            rec["segmentation"] = _box_seg(b.x1, b.y1, b.x2, b.y2)
        dt.append(rec)
    for p in pred_polys:
        xs = [pt[0] for pt in p.points]
        ys = [pt[1] for pt in p.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        dt.append({"category_id": int(p.class_id) + 1, "bbox": _xyxy_to_xywh(x1, y1, x2, y2),
                   "score": float(p.confidence), "segmentation": [_poly_flat(p.points)]})

    return iou_type, build_coco_image_record(width, height, gt, dt)


# ====================================================================
# In-house scalar metrics (expansion seam — segm AP already covers
# true instance segmentation once a mask head exists)
# ====================================================================

def classification_metrics(pred_labels: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    pred = pred_labels.detach().cpu().long()
    gt = targets.detach().cpu().long()
    if gt.numel() == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    accuracy = (pred == gt).float().mean().item()
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return {"accuracy": accuracy, "f1": sum(f1s) / len(f1s) if f1s else 0.0}


def ordinal_metrics(pred_ranks: torch.Tensor, gt_ranks: torch.Tensor) -> dict:
    pred = pred_ranks.detach().cpu().float()
    gt = gt_ranks.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rank_acc": 0.0}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rank_acc": (pred.round() == gt.round()).float().mean().item(),
    }


def regression_metrics(pred_values: torch.Tensor, gt_values: torch.Tensor) -> dict:
    pred = pred_values.detach().cpu().float()
    gt = gt_values.detach().cpu().float()
    if gt.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0}
    return {
        "mae": (pred - gt).abs().mean().item(),
        "rmse": ((pred - gt) ** 2).mean().sqrt().item(),
    }


# ====================================================================
# Task-agnostic evaluate() — loss pass + prediction pass
# ====================================================================

@torch.no_grad()
def evaluate(
    model, loader, device, task: str, *,
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
) -> dict:
    """Compute per-task validation/test metrics. Returns BARE metric keys."""
    is_detection = task in ("detection", "instance_seg")
    eff_iou_type = iou_type or "bbox"  # training detector is box-only

    model.eval()
    total_loss = 0.0
    n_loss = 0
    per_image: list[dict] = []
    cls_p, cls_g, ord_p, ord_g, reg_p, reg_g = [], [], [], [], [], []
    detector = getattr(model, "detector", None)

    for batch in loader:
        if is_detection:
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            # Loss pass.
            if detector is not None:
                keep = [(im, t) for im, t in zip(images, targets)
                        if isinstance(t.get("boxes"), torch.Tensor) and t["boxes"].numel() > 0]
                if keep:
                    detector.train()
                    for m in detector.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.eval()
                    ld = detector([im for im, _ in keep], [t for _, t in keep])
                    total_loss += float(sum(ld.values()).item())
                    n_loss += 1
            else:
                model.training = True
                for head in getattr(model, "heads", []):
                    head.training = True
                ld = model(images, targets)
                total_loss += float(sum(ld.values()).item()) if isinstance(ld, dict) else float(ld)
                n_loss += 1
            # Prediction pass.
            model.eval()
            outputs = model(images)
            for img, t, out in zip(images, targets, outputs):
                h, w = int(img.shape[-2]), int(img.shape[-1])
                per_image.append(records_from_detector(t, out, width=w, height=h))
        else:
            images, targets = batch
            images = images.to(device)
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            # Loss pass (BN stays in eval via the top-level training flag trick).
            model.training = True
            ld = model(images, targets)
            total_loss += float(sum(ld.values()).item()) if isinstance(ld, dict) else float(ld)
            n_loss += 1
            # Prediction pass.
            model.eval()
            out = model(images)
            if task == "classification" and "head0_labels" in out:
                cls_p.append(out["head0_labels"].detach().cpu())
                cls_g.append(targets["labels"].detach().cpu())
            elif task == "ordinal" and "head0_ranks" in out:
                ord_p.append(out["head0_ranks"].detach().cpu())
                ord_g.append(targets["ranks"].detach().cpu())
            elif task == "regression" and "head0_values" in out:
                reg_p.append(out["head0_values"].detach().cpu())
                reg_g.append(targets["values"].detach().cpu())

    model.eval()
    loss = total_loss / max(n_loss, 1)
    result: dict = {"loss": round(loss, 6)}

    if is_detection:
        m = coco_detection_metrics(per_image, iou_type=eff_iou_type, iou_threshold=iou_threshold,
                                   conf_threshold=conf_threshold, max_dets=max_dets)
        result.update({
            "precision": round(m["precision"], 6), "recall": round(m["recall"], 6),
            "f1": round(m["f1"], 6), "map50": round(m["map50"], 6), "map": round(m["map"], 6),
        })
        result["objective"] = round(compute_composite_objective(loss, m["f1"], m["map50"], score_weights), 6)
    elif task == "classification" and cls_p:
        num_classes = getattr(model.heads[0], "num_classes", int(torch.cat(cls_g).max()) + 1)
        result.update({k: round(v, 6) for k, v in classification_metrics(torch.cat(cls_p), torch.cat(cls_g), num_classes).items()})
    elif task == "ordinal" and ord_p:
        result.update({k: round(v, 6) for k, v in ordinal_metrics(torch.cat(ord_p), torch.cat(ord_g)).items()})
    elif task == "regression" and reg_p:
        result.update({k: round(v, 6) for k, v in regression_metrics(torch.cat(reg_p), torch.cat(reg_g)).items()})

    return result


def run_test_evaluation(
    ckpt_path: str, loader, device, task: str, output_dir: str, *,
    conf_threshold: float = 0.25, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
) -> dict:
    """Load ``model_best.pt``, evaluate ``loader``, write ``test_results.json``."""
    from tcip_mcp.pipelines.composer import compose_model

    ckpt = torch.load(ckpt_path, map_location=device)
    model = compose_model(ckpt["model_spec"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loader, device, task, conf_threshold=conf_threshold,
                       iou_threshold=iou_threshold, iou_type=iou_type, max_dets=max_dets,
                       score_weights=score_weights)
    result = {
        **metrics,
        "model_path": str(ckpt_path), "task": task,
        "iou_type": iou_type or ("bbox" if task in ("detection", "instance_seg") else ""),
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold, "max_dets": max_dets,
    }
    out = Path(output_dir) / "test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = str(out)
    return result
