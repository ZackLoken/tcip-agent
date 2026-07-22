"""ReviewEngine — Review logic, detection walk-through, accept/reject.

GUI-free.  Ported from yolo-annotator (yololabeler.review.engine) with these
adaptations:

  * Operates on :mod:`tcip_annotation.state` dataclasses (``BBox``,
    ``Polygon``, ``PredBBox``, ``PredPolygon``).
  * Consumes the dict-based match format produced by
    :func:`tcip_annotation.matching.compute_matches` (a dict, not a tuple).
  * Per-image state (image dims, GT/pred lists) is supplied via
    :class:`ReviewContext` on each call, rather than embedded in a global
    AppState. This makes the engine safe to reuse across images and
    concurrent sessions.

The only state the engine holds between calls is the persisted review log — one JSON
shard per image under ``<state_dir>/review/`` (a verdict rewrites only its own image's
shard, not the whole cross-image log) — and a small spatial-hash cache for fast lookups.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tcip_annotation.json_io import write_detect as write_detect_labels
from tcip_annotation.json_io import write_segment as write_segment_labels
from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class ReviewDetection:
    """One walkable entry in the Review tab's sequential traversal.

    Combines match-type (TP/FP/FN), class, indices into the GT and pred
    lists, matching IoU / confidence (when applicable), and the image-coord
    bounding box used to auto-zoom the canvas.
    """

    det_type: str  # "tp" | "fp" | "fn"
    class_id: int
    conf: Optional[float]
    iou: Optional[float]
    gt_type: Optional[str]  # "box" | "polygon" | None
    gt_idx: Optional[int]
    pred_type: Optional[str]
    pred_idx: Optional[int]
    bbox: tuple[float, float, float, float]  # image-pixel coords


@dataclass
class ReviewContext:
    """Per-image context the engine needs for any spatial operation."""

    img_name: str
    img_width: int
    img_height: int
    gt_boxes: list[BBox] = field(default_factory=list)
    gt_polygons: list[Polygon] = field(default_factory=list)
    pred_boxes: list[PredBBox] = field(default_factory=list)
    pred_polygons: list[PredPolygon] = field(default_factory=list)


# ── Constants ─────────────────────────────────────────────────────────────

REVIEW_SHARD_DIRNAME = "review"
_LOOKUP_QUANT = 500
_LOOKUP_TOLERANCE = 0.002


# ── Engine ────────────────────────────────────────────────────────────────


class ReviewEngine:
    """Persist-and-compute engine for the Review tab.

    Parameters
    ----------
    state_dir : Path | str
        Directory holding the ``review/`` shard directory (one JSON file per
        reviewed image). Created if missing.
    class_names : dict[int, str], optional
        Class ID → human name mapping. Embedded in recorded entries for audit.
    current_user : str, optional
        Username recorded on every accept/reject/edit action.
    """

    def __init__(
        self,
        state_dir: Path | str,
        *,
        class_names: Optional[dict[int, str]] = None,
        current_user: str = "",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or {}
        self.current_user = current_user
        self._review_state: dict = {}
        self._reviewed_lookup: tuple[str, dict, dict] = ("", {}, {})
        self.load_review_state()

    # ── Persistence ───────────────────────────────────────────────────────

    @property
    def shard_dir(self) -> Path:
        return self.state_dir / REVIEW_SHARD_DIRNAME

    def _shard_path(self, img_name: str) -> Path:
        safe = img_name.replace("\\", "_").replace("/", "_")
        if safe != img_name:
            # Sanitizing separators can collide distinct keys ('a/b.jpg' vs 'a_b.jpg'); a short stable
            # hash of the true key keeps their shard files distinct. The key itself lives in the payload.
            safe = f"{safe}.{hashlib.sha1(img_name.encode('utf-8')).hexdigest()[:8]}"
        return self.shard_dir / f"{safe}.json"

    def load_review_state(self) -> None:
        shards = sorted(self.shard_dir.glob("*.json")) if self.shard_dir.is_dir() else []
        self._review_state = {"image": self._read_shards(shards)} if shards else {}
        self._invalidate_reviewed_lookup()

    def _read_shards(self, shards: list[Path]) -> dict:
        per_image: dict = {}
        for shard in shards:
            try:
                payload = json.loads(shard.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Could not load review shard %s", shard)
                continue
            # The true image key is stored inside the payload, so a sanitized/hash-suffixed filename
            # never mutates or merges keys on reload.
            img_name, state = payload.get("img_name"), payload.get("state")
            if img_name is None or state is None:
                img_name, state = shard.name[: -len(".json")], payload
            per_image[img_name] = state
        return per_image

    def _atomic_write_json(self, path: Path, obj: dict) -> None:
        """Serialize ``obj`` compactly and swap it into ``path`` atomically (temp file +
        ``os.replace``, fsync'd). A plain ``write_text`` can leave a half-written /
        truncated file if the process dies mid-write; the atomic swap means a reader
        always sees either the old or the new complete file, never a torn one.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _save_image(self, img_name: str) -> None:
        """Persist only ``img_name``'s shard — O(detections on that image), not O(all-reviewed)."""
        img_data = self._review_state.get("image", {}).get(img_name)
        if img_data is None:
            return
        try:
            # Store the true key alongside the state so reload reconstructs it from the payload, not
            # the (possibly sanitized) filename.
            self._atomic_write_json(self._shard_path(img_name), {"img_name": img_name, "state": img_data})
        except Exception:
            logger.exception("Could not save review shard for %s", img_name)

    def save_review_state(self) -> None:
        """Flush every image's shard. The per-verdict callers use :meth:`_save_image`
        instead, to touch only the image that changed; this is for migration / tests
        that want the whole in-memory state written out."""
        for img_name in list(self._review_state.get("image", {})):
            self._save_image(img_name)

    @property
    def raw_state(self) -> dict:
        """Expose the raw review_state dict. Mainly for tests / audit."""
        return self._review_state

    # ── Image-level status ────────────────────────────────────────────────

    def mark_image_reviewed(self, img_name: str) -> None:
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            img_name, {"img_status": "completed", "detections": []}
        )
        img_data["img_status"] = "completed"
        self._save_image(img_name)

    def unmark_image_reviewed(self, img_name: str) -> None:
        """Reverse a manual mark: back to started (verdicts kept) or not_started."""
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return
        img_data["img_status"] = "started" if img_data.get("detections") else "not_started"
        self._save_image(img_name)

    def is_image_reviewed(self, img_name: str) -> bool:
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return False
        return img_data.get("img_status") == "completed"

    def get_image_review_status(self, img_name: str) -> str:
        """Return ``"completed"``, ``"started"``, or ``"not_started"``."""
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return "not_started"
        return img_data.get("img_status", "not_started")

    def verdict_count_for_images(self, names: Iterable[str]) -> int:
        """Total recorded verdicts (accept/reject/edit detection entries) across ``names``,
        matched by image stem so a prediction bucket's ``<stem>.json`` files line up with the
        review log's image-name keys. Backs prediction-bucket immutability: a bucket whose
        images carry verdicts must not be silently overwritten by a re-run."""
        wanted = {Path(n).stem for n in names}
        if not wanted:
            return 0
        per_image = self._review_state.get("image", {})
        total = 0
        for img_name, data in per_image.items():
            if Path(img_name).stem in wanted:
                total += len(data.get("detections", []))
        return total

    def get_all_image_statuses(self) -> dict[str, str]:
        """Review status for every image the engine has state for (untouched images are
        absent — the caller defaults them to ``"not_started"``). Backs the image-level
        Reviewed/Unreviewed navigation filter, batch-fetched once per dataset."""
        per_image = self._review_state.get("image", {})
        return {
            name: data.get("img_status", "not_started") for name, data in per_image.items()
        }

    # ── Bounding-box helpers ──────────────────────────────────────────────

    def _bbox_of_gt(self, ctx: ReviewContext, gt_type: Optional[str], gt_idx: Optional[int]):
        if gt_type == "box" and gt_idx is not None and 0 <= gt_idx < len(ctx.gt_boxes):
            b = ctx.gt_boxes[gt_idx]
            return (b.x1, b.y1, b.x2, b.y2)
        if gt_type == "polygon" and gt_idx is not None and 0 <= gt_idx < len(ctx.gt_polygons):
            pts = ctx.gt_polygons[gt_idx].points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        return None

    def _bbox_of_pred(self, ctx: ReviewContext, pred_type: Optional[str], pred_idx: Optional[int]):
        if pred_type == "box" and pred_idx is not None and 0 <= pred_idx < len(ctx.pred_boxes):
            b = ctx.pred_boxes[pred_idx]
            return (b.x1, b.y1, b.x2, b.y2)
        if pred_type == "polygon" and pred_idx is not None and 0 <= pred_idx < len(ctx.pred_polygons):
            pts = ctx.pred_polygons[pred_idx].points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        return None

    def _detection_bbox(self, ctx: ReviewContext, gt_type, gt_idx, p_type, p_idx):
        """Return the image-coord bbox for a detection, covering GT and/or pred."""
        bboxes = []
        gt_b = self._bbox_of_gt(ctx, gt_type, gt_idx)
        if gt_b:
            bboxes.append(gt_b)
        p_b = self._bbox_of_pred(ctx, p_type, p_idx)
        if p_b:
            bboxes.append(p_b)
        if not bboxes:
            return (0.0, 0.0, float(ctx.img_width), float(ctx.img_height))
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    def _normalised_bbox(self, ctx: ReviewContext, which: str, det_like) -> Optional[list[float]]:
        """Return ``[cx, cy, w, h]`` normalised to image dimensions.

        ``det_like`` may be a :class:`ReviewDetection` or a dict with the
        same keys. ``which`` is ``"gt"`` or ``"pred"``.
        """
        img_w = max(ctx.img_width, 1)
        img_h = max(ctx.img_height, 1)

        if isinstance(det_like, ReviewDetection):
            gt_type = det_like.gt_type
            gt_idx = det_like.gt_idx
            p_type = det_like.pred_type
            p_idx = det_like.pred_idx
        else:
            gt_type = det_like.get("gt_type")
            gt_idx = det_like.get("gt_idx")
            p_type = det_like.get("pred_type")
            p_idx = det_like.get("pred_idx")

        if which == "gt":
            b = self._bbox_of_gt(ctx, gt_type, gt_idx)
        else:
            b = self._bbox_of_pred(ctx, p_type, p_idx)
        if b is None:
            return None
        x1, y1, x2, y2 = b
        return [
            round((x1 + x2) / 2 / img_w, 6),
            round((y1 + y2) / 2 / img_h, 6),
            round((x2 - x1) / img_w, 6),
            round((y2 - y1) / img_h, 6),
        ]

    # ── Reviewed-entry spatial hash ───────────────────────────────────────

    def _invalidate_reviewed_lookup(self) -> None:
        self._reviewed_lookup = ("", {}, {})

    def _build_reviewed_lookup(self, img_name: str) -> None:
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            self._reviewed_lookup = (img_name, {}, {})
            return
        reviewed_dets = img_data.get("detections", [])
        pred_map: dict = {}
        gt_map: dict = {}
        for entry in reviewed_dets:
            e_pred = entry.get("pred_bbox_norm")
            if e_pred:
                qk = (round(e_pred[0] * _LOOKUP_QUANT), round(e_pred[1] * _LOOKUP_QUANT))
                pred_map.setdefault(qk, []).append(entry)
            e_gt = entry.get("gt_bbox_norm")
            if e_gt:
                qk = (round(e_gt[0] * _LOOKUP_QUANT), round(e_gt[1] * _LOOKUP_QUANT))
                gt_map.setdefault(qk, []).append(entry)
        self._reviewed_lookup = (img_name, pred_map, gt_map)

    def find_reviewed_entry(self, det: ReviewDetection, ctx: ReviewContext) -> Optional[dict]:
        """Return the reviewed-entry dict for ``det`` on this image, if any."""
        if not ctx.img_name:
            return None
        if self._reviewed_lookup[0] != ctx.img_name:
            self._build_reviewed_lookup(ctx.img_name)
        _, pred_map, gt_map = self._reviewed_lookup

        if det.det_type in ("tp", "fp"):
            pred_bbox = self._normalised_bbox(ctx, "pred", det)
            if not pred_bbox:
                return None
            pcx, pcy = pred_bbox[0], pred_bbox[1]
            qx, qy = round(pcx * _LOOKUP_QUANT), round(pcy * _LOOKUP_QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in pred_map.get((qx + dx, qy + dy), ()):
                        e_pred = entry.get("pred_bbox_norm")
                        if (
                            e_pred
                            and abs(e_pred[0] - pcx) < _LOOKUP_TOLERANCE
                            and abs(e_pred[1] - pcy) < _LOOKUP_TOLERANCE
                        ):
                            return entry
        else:  # fn
            gt_bbox = self._normalised_bbox(ctx, "gt", det)
            if not gt_bbox:
                return None
            gcx, gcy = gt_bbox[0], gt_bbox[1]
            qx, qy = round(gcx * _LOOKUP_QUANT), round(gcy * _LOOKUP_QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in gt_map.get((qx + dx, qy + dy), ()):
                        e_gt = entry.get("gt_bbox_norm")
                        if (
                            e_gt
                            and abs(e_gt[0] - gcx) < _LOOKUP_TOLERANCE
                            and abs(e_gt[1] - gcy) < _LOOKUP_TOLERANCE
                        ):
                            return entry
        return None

    # ── Detection list build ──────────────────────────────────────────────

    def build_detection_list(
        self,
        ctx: ReviewContext,
        matches: dict,
        *,
        filter_type: str = "all",
        filter_class: int | str = "all",
    ) -> list[ReviewDetection]:
        """Produce the filtered, walkable list of detections for the Review UI.

        Review status is never hidden here: within an image every matching detection is
        walkable regardless of whether it has been reviewed (the Reviewed/Unreviewed filter
        is image-level navigation, not per-detection visibility). Reviewed/unreviewed state
        rides on each detection via :meth:`find_reviewed_entry` for the caller to decorate."""

        def _class_ok(cid: int) -> bool:
            return filter_class == "all" or cid == filter_class

        dets: list[ReviewDetection] = []

        if filter_type in ("all", "tp"):
            for m in matches.get("tp", []):
                cid = m["class_id"]
                if not _class_ok(cid):
                    continue
                det = ReviewDetection(
                    det_type="tp",
                    class_id=cid,
                    conf=m.get("conf"),
                    iou=m.get("iou"),
                    gt_type=m["gt_type"],
                    gt_idx=m["gt_idx"],
                    pred_type=m["pred_type"],
                    pred_idx=m["pred_idx"],
                    bbox=self._detection_bbox(
                        ctx, m["gt_type"], m["gt_idx"], m["pred_type"], m["pred_idx"]
                    ),
                )
                dets.append(det)

        if filter_type in ("all", "fp"):
            for m in matches.get("fp", []):
                cid = m["class_id"]
                if not _class_ok(cid):
                    continue
                det = ReviewDetection(
                    det_type="fp",
                    class_id=cid,
                    conf=m.get("conf"),
                    iou=None,
                    gt_type=None,
                    gt_idx=None,
                    pred_type=m["pred_type"],
                    pred_idx=m["pred_idx"],
                    bbox=self._detection_bbox(ctx, None, None, m["pred_type"], m["pred_idx"]),
                )
                dets.append(det)

        if filter_type in ("all", "fn"):
            for m in matches.get("fn", []):
                cid = m["class_id"]
                if not _class_ok(cid):
                    continue
                det = ReviewDetection(
                    det_type="fn",
                    class_id=cid,
                    conf=None,
                    iou=None,
                    gt_type=m["gt_type"],
                    gt_idx=m["gt_idx"],
                    pred_type=None,
                    pred_idx=None,
                    bbox=self._detection_bbox(ctx, m["gt_type"], m["gt_idx"], None, None),
                )
                dets.append(det)

        return dets

    # ── Record & check ────────────────────────────────────────────────────

    def record_detection_action(
        self,
        det: ReviewDetection,
        ctx: ReviewContext,
        action: str,
        *,
        norm_det: Optional[ReviewDetection] = None,
        norm_ctx: Optional[ReviewContext] = None,
    ) -> None:
        """Log an accept / reject / edit action for a detection.

        ``norm_det``/``norm_ctx`` override the geometry the entry is stored under. An
        edited verdict rewrites the GT bbox, so the entry must be keyed to the post-edit
        geometry (what the next reload's lookup sees) while any prior entry for this
        detection is still found via the pre-edit geometry of ``det``/``ctx``.
        """
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            ctx.img_name, {"img_status": "started", "detections": []}
        )

        nd = norm_det if norm_det is not None else det
        nc = norm_ctx if norm_ctx is not None else ctx
        class_name = self.class_names.get(det.class_id, f"class_{det.class_id}")
        entry = {
            "match_type": det.det_type.upper(),
            "det_status": "reviewed",
            "action": action,
            "reviewed_by": self.current_user,
            "class_id": det.class_id,
            "class_name": class_name,
            "gt_bbox_norm": self._normalised_bbox(nc, "gt", nd),
            "pred_bbox_norm": self._normalised_bbox(nc, "pred", nd),
            "iou": round(det.iou, 4) if det.iou is not None else None,
            "conf": round(det.conf, 4) if det.conf is not None else None,
        }

        existing = self.find_reviewed_entry(det, ctx)
        if existing:
            idx = img_data["detections"].index(existing)
            img_data["detections"][idx] = entry
        else:
            img_data["detections"].append(entry)

        self._invalidate_reviewed_lookup()

        if img_data.get("img_status") == "not_started":
            img_data["img_status"] = "started"

        self._save_image(ctx.img_name)

    def check_image_review_complete(self, img_name: str, matches: dict) -> bool:
        """If every detection on the image has been reviewed, mark it complete.

        Returns ``True`` if the image is now in the completed state.
        """
        total = (
            len(matches.get("tp", []))
            + len(matches.get("fp", []))
            + len(matches.get("fn", []))
        )
        if total == 0:
            return False

        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return False

        reviewed_count = len(img_data.get("detections", []))
        if reviewed_count >= total:
            img_data["img_status"] = "completed"
            self._save_image(img_name)
            return True
        return False

    # ── Label backup / save ───────────────────────────────────────────────

    def backup_original_labels(self, *label_dirs: Path | str) -> int:
        """Ensure every label file in each dir has a pristine copy in ``<dir>/.original/``.

        Per-file and idempotent: a file is captured the first time it is seen and never
        overwritten afterwards, so labels added after the first backup still get their
        baseline before the platform first mutates them. Returns the number of files
        newly captured by this call.
        """
        captured = 0
        for label_dir in label_dirs:
            d = Path(label_dir)
            if not d.is_dir():
                continue
            backup_dir = d / ".original"
            for src in d.iterdir():
                if not (src.is_file() and src.suffix == ".json"):
                    continue
                dst = backup_dir / src.name
                if dst.exists():
                    continue
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                captured += 1
        return captured

    def save_gt(self, ctx: ReviewContext, *, detect_path: Optional[str] = None, segment_path: Optional[str] = None) -> bool:
        """Write the current GT from ``ctx`` back to the canonical per-image label files.

        Returns ``True`` on full success, ``False`` if any write failed.
        """
        # keep_empty: an emptied GT keeps its record (objects: []) rather than deleting the file.
        # That record is not a negative until a human confirms the image Complete.
        ok = True
        if detect_path is not None:
            try:
                os.makedirs(os.path.dirname(detect_path) or ".", exist_ok=True)
                write_detect_labels(detect_path, ctx.gt_boxes, ctx.img_width, ctx.img_height,
                                    keep_empty=True)
            except OSError:
                logger.exception("Could not save detect labels to %s", detect_path)
                ok = False
        if segment_path is not None:
            try:
                os.makedirs(os.path.dirname(segment_path) or ".", exist_ok=True)
                write_segment_labels(segment_path, ctx.gt_polygons, ctx.img_width, ctx.img_height,
                                     keep_empty=True)
            except OSError:
                logger.exception("Could not save segment labels to %s", segment_path)
                ok = False
        return ok
