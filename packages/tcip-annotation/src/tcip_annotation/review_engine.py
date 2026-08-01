"""ReviewEngine — Review logic, detection walk-through, accept/reject.

GUI-free:

  * Operates on :class:`tcip_annotation.state.Annotation` records (a prediction is an annotation
    whose ``score`` is set); a class is named by its ``subject``, never an integer id.
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

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, Point, bbox_of

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class ReviewDetection:
    """One walkable entry in the Review tab's sequential traversal.

    Combines match-type (TP/FP/FN), class name, indices into the GT and pred
    lists, matching IoU / confidence (when applicable), and the image-coord
    bounding box used to auto-zoom the canvas.
    """

    det_type: str  # "tp" | "fp" | "fn"
    class_name: str
    conf: Optional[float]
    iou: Optional[float]
    gt_idx: Optional[int]
    pred_idx: Optional[int]
    bbox: tuple[float, float, float, float]  # image-pixel coords


@dataclass
class ReviewContext:
    """Per-image context the engine needs for any spatial operation.

    ``gt`` / ``preds`` are :class:`Annotation` lists indexed by the match dicts' ``gt_idx`` /
    ``pred_idx``; a prediction annotation carries a ``score``.
    """

    img_name: str
    img_width: int
    img_height: int
    gt: list[Annotation] = field(default_factory=list)
    preds: list[Annotation] = field(default_factory=list)


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
    current_user : str, optional
        Username recorded on every accept/reject/edit action.
    """

    def __init__(
        self,
        state_dir: Path | str,
        *,
        current_user: str = "",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
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

    def mark_image_reviewed(self, img_name: str, *, producer_identity: Optional[dict] = None,
                            adjudication_covered: Optional[bool] = None) -> None:
        """Mark ``img_name`` fully reviewed (e.g. a confirmed negative / bulk-accept).

        ``producer_identity`` (Fix G): the resolved producing-bucket fact (``checkpoint_sha256``/
        ``experiment_id``), a plain dict the CALLER resolves (this package never looks one up
        itself — see the module docstring). A confirmed negative carries zero verdict entries, so
        it has nowhere else to record which model it was reviewed against; this stamps that fact at
        the image level instead. ``None`` (the default) leaves any existing stamp untouched.

        ``adjudication_covered`` (Fix H, stage-6 review): whether this zero-verdict completion is a
        genuine negative the CALLER has already confirmed (the prediction bucket held zero
        detections for this image, so Complete is itself the confirming act) — never inferred by
        this package. A bulk-accept of an image the bucket DID predict on, completed with no
        individual verdicts, must pass ``False`` (or omit it): stamping every zero-verdict Complete
        as covered would let an unreviewed bulk-accept dilute a real reference's statistics, which is
        exactly the fail-open this distinction exists to close. ``None`` (the default) leaves any
        existing stamp untouched.
        """
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            img_name, {"img_status": "completed", "detections": []}
        )
        img_data["img_status"] = "completed"
        if producer_identity is not None:
            img_data["producer_identity"] = producer_identity
        if adjudication_covered is not None:
            img_data["adjudication_covered"] = adjudication_covered
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

    @staticmethod
    def _bbox_of_annotation(anns: list[Annotation], idx: Optional[int]):
        """The annotation's image-coord box, or ``None`` when it has none to read.

        A :class:`~tcip_annotation.state.Point` reads as ``None``, like a geometry-less label: a
        verdict's box is what the reviewer looked at and what the entry is keyed by, and a fabricated
        zero-area box at the point would key a real verdict to a shape nobody drew.
        """
        if idx is None or not (0 <= idx < len(anns)):
            return None
        geom = anns[idx].geometry
        if geom is None or isinstance(geom, Point):
            return None
        b = bbox_of(geom)
        return (b.x1, b.y1, b.x2, b.y2)

    def _bbox_of_gt(self, ctx: ReviewContext, gt_idx: Optional[int]):
        return self._bbox_of_annotation(ctx.gt, gt_idx)

    def _bbox_of_pred(self, ctx: ReviewContext, pred_idx: Optional[int]):
        return self._bbox_of_annotation(ctx.preds, pred_idx)

    def _detection_bbox(self, ctx: ReviewContext, gt_idx, p_idx):
        """Return the image-coord bbox for a detection, covering GT and/or pred."""
        bboxes = []
        gt_b = self._bbox_of_gt(ctx, gt_idx)
        if gt_b:
            bboxes.append(gt_b)
        p_b = self._bbox_of_pred(ctx, p_idx)
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
            gt_idx = det_like.gt_idx
            p_idx = det_like.pred_idx
        else:
            gt_idx = det_like.get("gt_idx")
            p_idx = det_like.get("pred_idx")

        if which == "gt":
            b = self._bbox_of_gt(ctx, gt_idx)
        else:
            b = self._bbox_of_pred(ctx, p_idx)
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
        filter_class: str = "all",
    ) -> list[ReviewDetection]:
        """Produce the filtered, walkable list of detections for the Review UI.

        ``filter_class`` is a class name (an annotation's ``subject``) or ``"all"``. Review status is
        never hidden here: within an image every matching detection is walkable regardless of whether
        it has been reviewed (the Reviewed/Unreviewed filter is image-level navigation, not
        per-detection visibility). Reviewed/unreviewed state rides on each detection via
        :meth:`find_reviewed_entry` for the caller to decorate."""

        def _class_ok(cname: str) -> bool:
            return filter_class == "all" or cname == filter_class

        dets: list[ReviewDetection] = []

        if filter_type in ("all", "tp"):
            for m in matches.get("tp", []):
                cname = m["class_name"]
                if not _class_ok(cname):
                    continue
                dets.append(ReviewDetection(
                    det_type="tp",
                    class_name=cname,
                    conf=m.get("conf"),
                    iou=m.get("iou"),
                    gt_idx=m["gt_idx"],
                    pred_idx=m["pred_idx"],
                    bbox=self._detection_bbox(ctx, m["gt_idx"], m["pred_idx"]),
                ))

        if filter_type in ("all", "fp"):
            for m in matches.get("fp", []):
                cname = m["class_name"]
                if not _class_ok(cname):
                    continue
                dets.append(ReviewDetection(
                    det_type="fp",
                    class_name=cname,
                    conf=m.get("conf"),
                    iou=None,
                    gt_idx=None,
                    pred_idx=m["pred_idx"],
                    bbox=self._detection_bbox(ctx, None, m["pred_idx"]),
                ))

        if filter_type in ("all", "fn"):
            for m in matches.get("fn", []):
                cname = m["class_name"]
                if not _class_ok(cname):
                    continue
                dets.append(ReviewDetection(
                    det_type="fn",
                    class_name=cname,
                    conf=None,
                    iou=None,
                    gt_idx=m["gt_idx"],
                    pred_idx=None,
                    bbox=self._detection_bbox(ctx, m["gt_idx"], None),
                ))

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
        producer_identity: Optional[dict] = None,
        conf_threshold: Optional[float] = None,
        class_id: Optional[int] = None,
    ) -> None:
        """Log an accept / reject / edit action for a detection.

        ``norm_det``/``norm_ctx`` override the geometry the entry is stored under. An
        edited verdict rewrites the GT bbox, so the entry must be keyed to the post-edit
        geometry (what the next reload's lookup sees) while any prior entry for this
        detection is still found via the pre-edit geometry of ``det``/``ctx``.

        ``producer_identity`` (Fix G): the resolved producing-bucket fact (``checkpoint_sha256``/
        ``experiment_id``, plus ``bucket_dir`` for human legibility) this verdict was recorded
        against — a plain dict the CALLER resolves (``tcip-web``, which can read a bucket's
        ``operating_point.json`` sidecar); this package stores it verbatim and never resolves one
        itself, keeping it free of any dependency on ``tcip-mcp``/``tcip-web``. Persisted on the
        verdict entry so a later validation pass can scope verdicts to the SAME producing model,
        not a directory-name comparison.

        ``conf_threshold`` (Fix D item 4): the review session's confidence-display threshold in
        effect when this verdict was recorded — persisted so the review-confirmed reference can
        reconstruct the effective floor the reviewed predictions were shown at.

        ``class_id`` (K25): the 0-indexed class identity ``det.class_name`` resolves to under the
        producing bucket's own recorded name->id map — the SAME shape as ``producer_identity``, a
        plain fact the CALLER resolves (``tcip-web``, via the bucket's ``operating_point.json``
        ``id_map``) and this package stores verbatim, never re-derives. ``None`` when the caller
        could not resolve one (no recorded map, or ``class_name`` isn't one of its keys) — an honest
        "unresolvable" fact, not a guessed default; a consumer building a class-aware reference from
        this verdict (``review_calibration.review_to_records``) must refuse rather than assume 0.

        On the FIRST verdict recorded for this image, stamps ``gt_preexisting = bool(ctx.gt)`` onto
        the image-level record (Fix H) — the pristine, pre-mutation GT the caller already holds at
        that point, a recorded fact (never inferred later) for whether this image had ground truth
        BEFORE the review session touched it.

        Every entry also stamps ``missed_object_attested`` (Fix H) — ``True`` only when BOTH
        ``det.gt_idx`` and ``det.pred_idx`` are ``None`` at the moment of THIS call: the exact
        call-site shape only the "mark missed object" tool produces (a verdict with no existing GT
        or prediction to key off of — see ``ReviewTab.tsx``'s ``recordMissedObject``). Recorded here,
        from the caller's own intent, rather than reconstructed later from the entry's persisted
        ``gt_bbox_norm``/``pred_bbox_norm`` shape — a rejected or accepted FN (an EXISTING,
        already-indexed GT box being corrected or confirmed) ends up with the identical
        ``pred_bbox_norm=None, gt_bbox_norm=<box>`` shape once written, so geometry alone cannot tell
        "a genuinely new missed object was attested" apart from "a pre-existing FN was adjudicated".
        """
        per_image = self._review_state.setdefault("image", {})
        is_first_verdict = ctx.img_name not in per_image
        img_data = per_image.setdefault(
            ctx.img_name, {"img_status": "started", "detections": []}
        )
        if is_first_verdict:
            img_data["gt_preexisting"] = bool(ctx.gt)

        nd = norm_det if norm_det is not None else det
        nc = norm_ctx if norm_ctx is not None else ctx
        entry = {
            "match_type": det.det_type.upper(),
            "det_status": "reviewed",
            "action": action,
            "reviewed_by": self.current_user,
            "class_name": det.class_name,
            "gt_bbox_norm": self._normalised_bbox(nc, "gt", nd),
            "pred_bbox_norm": self._normalised_bbox(nc, "pred", nd),
            "iou": round(det.iou, 4) if det.iou is not None else None,
            "conf": round(det.conf, 4) if det.conf is not None else None,
            "producer_identity": producer_identity,
            "conf_threshold": conf_threshold,
            "missed_object_attested": det.gt_idx is None and det.pred_idx is None,
            "class_id": class_id,
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

    def save_gt(self, ctx: ReviewContext, *, path: Optional[str] = None) -> bool:
        """Write the current GT from ``ctx`` back to the single per-image label file.

        The unified schema holds every subject's annotations (boxes and polygons alike) in one file
        per image, so a review save writes ``ctx.gt`` whole. Returns ``True`` on success, ``False``
        if the write failed.
        """
        if path is None:
            return True
        # keep_empty: an emptied GT keeps its record ({"annotations": []}) rather than deleting the
        # file. That record is not a negative until a human confirms the image Complete.
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            write_annotations(path, ctx.gt, ctx.img_width, ctx.img_height, keep_empty=True)
        except OSError:
            logger.exception("Could not save labels to %s", path)
            return False
        return True
