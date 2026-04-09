"""ReviewEngine — Review logic, detection matching, accept/reject.

GUI-free.  Manages review state persistence and detection cycling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon
from tcip_annotation.matching import compute_matches


@dataclass
class ReviewDetection:
    """A single detection under review."""

    det_type: str  # "tp", "fp", "fn"
    class_id: int
    conf: float | None = None
    iou: float | None = None
    gt_type: str | None = None
    gt_idx: int | None = None
    pred_type: str | None = None
    pred_idx: int | None = None
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class ReviewState:
    """Review state for a dataset, persisted to review_stats.json."""

    labels_backed_up: bool = False
    images: dict[str, dict] = field(default_factory=dict)


class ReviewEngine:
    """Review operations without any GUI dependency."""

    def __init__(self, state_dir: str | None = None):
        self._state_dir = state_dir
        self._review_state: dict = {}

    @property
    def state_path(self) -> str | None:
        if self._state_dir:
            return os.path.join(self._state_dir, "review_stats.json")
        return None

    def load_state(self) -> None:
        path = self.state_path
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._review_state = json.load(f)
            except Exception:
                self._review_state = {}
        else:
            self._review_state = {}

    def save_state(self) -> None:
        path = self.state_path
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._review_state, f, indent=2)
        except Exception:
            pass

    # ── Image-level status ────────────────────────────────────────────────

    def get_image_status(self, img_name: str) -> str:
        """Return 'completed', 'started', or 'not_started'."""
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return "not_started"
        return img_data.get("img_status", "not_started")

    def mark_image_reviewed(self, img_name: str) -> None:
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(img_name, {"img_status": "completed", "detections": []})
        img_data["img_status"] = "completed"
        self.save_state()

    # ── Matching ──────────────────────────────────────────────────────────

    def run_matching(
        self,
        gt_boxes: list[BBox],
        gt_polygons: list[Polygon],
        pred_boxes: list[PredBBox],
        pred_polygons: list[PredPolygon],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ) -> dict:
        """Run GT-vs-prediction matching.  Returns match dict."""
        return compute_matches(gt_boxes, gt_polygons, pred_boxes, pred_polygons, iou_threshold, conf_threshold)

    def build_detections(
        self,
        matches: dict,
        gt_boxes: list[BBox],
        gt_polygons: list[Polygon],
        pred_boxes: list[PredBBox],
        pred_polygons: list[PredPolygon],
        filter_type: str = "all",
        filter_class: str | int = "all",
    ) -> list[ReviewDetection]:
        """Build filtered detection list from matches."""
        dets: list[ReviewDetection] = []

        def _class_ok(cid: int) -> bool:
            return filter_class == "all" or cid == filter_class

        def _bbox_of_gt(gt_type: str, gt_idx: int) -> tuple[float, float, float, float]:
            if gt_type == "box" and 0 <= gt_idx < len(gt_boxes):
                b = gt_boxes[gt_idx]
                return (b.x1, b.y1, b.x2, b.y2)
            if gt_type == "polygon" and 0 <= gt_idx < len(gt_polygons):
                pts = gt_polygons[gt_idx].points
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
            return (0, 0, 0, 0)

        def _bbox_of_pred(p_type: str, p_idx: int) -> tuple[float, float, float, float]:
            if p_type == "box" and 0 <= p_idx < len(pred_boxes):
                b = pred_boxes[p_idx]
                return (b.x1, b.y1, b.x2, b.y2)
            if p_type == "polygon" and 0 <= p_idx < len(pred_polygons):
                pts = pred_polygons[p_idx].points
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
            return (0, 0, 0, 0)

        def _merge_bboxes(*bbs):
            valid = [b for b in bbs if b != (0, 0, 0, 0)]
            if not valid:
                return (0, 0, 0, 0)
            return (
                min(b[0] for b in valid),
                min(b[1] for b in valid),
                max(b[2] for b in valid),
                max(b[3] for b in valid),
            )

        if filter_type in ("all", "tp"):
            for tp in matches["tp"]:
                cid = tp["class_id"]
                if not _class_ok(cid):
                    continue
                bbox = _merge_bboxes(
                    _bbox_of_gt(tp["gt_type"], tp["gt_idx"]),
                    _bbox_of_pred(tp["pred_type"], tp["pred_idx"]),
                )
                dets.append(
                    ReviewDetection(
                        det_type="tp",
                        class_id=cid,
                        conf=tp["conf"],
                        iou=tp["iou"],
                        gt_type=tp["gt_type"],
                        gt_idx=tp["gt_idx"],
                        pred_type=tp["pred_type"],
                        pred_idx=tp["pred_idx"],
                        bbox=bbox,
                    )
                )

        if filter_type in ("all", "fp"):
            for fp in matches["fp"]:
                cid = fp["class_id"]
                if not _class_ok(cid):
                    continue
                bbox = _bbox_of_pred(fp["pred_type"], fp["pred_idx"])
                dets.append(
                    ReviewDetection(
                        det_type="fp",
                        class_id=cid,
                        conf=fp["conf"],
                        pred_type=fp["pred_type"],
                        pred_idx=fp["pred_idx"],
                        bbox=bbox,
                    )
                )

        if filter_type in ("all", "fn"):
            for fn in matches["fn"]:
                cid = fn["class_id"]
                if not _class_ok(cid):
                    continue
                bbox = _bbox_of_gt(fn["gt_type"], fn["gt_idx"])
                dets.append(
                    ReviewDetection(
                        det_type="fn",
                        class_id=cid,
                        gt_type=fn["gt_type"],
                        gt_idx=fn["gt_idx"],
                        bbox=bbox,
                    )
                )

        return dets

    # ── Recording actions ─────────────────────────────────────────────────

    def record_action(
        self,
        img_name: str,
        det: ReviewDetection,
        action: str,
        class_name: str = "",
        user: str = "",
    ) -> None:
        """Record a review action ('accepted', 'rejected', 'edited')."""
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(img_name, {"img_status": "started", "detections": []})

        entry = {
            "match_type": det.det_type.upper(),
            "det_status": "reviewed",
            "action": action,
            "reviewed_by": user,
            "class_id": det.class_id,
            "class_name": class_name,
            "iou": det.iou,
            "conf": det.conf,
        }
        img_data["detections"].append(entry)
        if img_data["img_status"] == "not_started":
            img_data["img_status"] = "started"
        self.save_state()

    def get_review_summary(self) -> dict:
        """Return aggregate review statistics."""
        per_image = self._review_state.get("image", {})
        total = len(per_image)
        completed = sum(1 for d in per_image.values() if d.get("img_status") == "completed")
        started = sum(1 for d in per_image.values() if d.get("img_status") == "started")
        total_dets = sum(len(d.get("detections", [])) for d in per_image.values())
        return {
            "total_images": total,
            "completed": completed,
            "started": started,
            "not_started": total - completed - started,
            "total_reviewed_detections": total_dets,
        }
