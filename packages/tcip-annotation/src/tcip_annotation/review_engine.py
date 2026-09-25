"""ReviewEngine: review logic, detection walk-through, accept/reject.

GUI-free:

  * Operates on :class:`tcip_annotation.state.Annotation` records (a prediction is an annotation
  whose ``score`` is set); a class is named by its ``subject``, never an integer id.
  * Consumes the dict-based match format produced by
  :func:`tcip_annotation.matching.compute_matches`.
  * Per-image state (image dims, GT/pred lists) is supplied via :class:`ReviewContext` on each
  call.

The only state the engine holds between calls is the persisted review log, one JSON shard per
(prediction bucket, image) under ``<state_dir>/review/``, and a small spatial-hash cache for fast
lookups.

A verdict is recorded against the prediction bucket the reviewer was looking at, so two dates of a
camera that reuses filenames keep separate verdicts. The bucket is a caller-supplied key this
package stores verbatim and never derives. ``NO_BUCKET`` (``"."``) is the key for a review carrying
no prediction bucket at all, which is a ground-truth-only review.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, Version, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation.json_io import LABEL_SUFFIX, write_annotations
from tcip_annotation.state import Annotation, bbox_of, box_derivable
from tcip_annotation.verdicts import (
    GT_BOX_KEY, PRED_BOX_KEY, VERDICT_ACTIONS, VerdictAction, decode_verdict,
)

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
_SHARD_SUFFIX = ".json"
_PATH_HOSTILE = '\\/:*?"<>|'
# A bucket key is a relative path of at least one segment, so "." names no bucket without
# being able to collide with one, and the store refuses an empty key field.
NO_BUCKET = "."


def _sanitized(key: str) -> str:
    """``key`` with path-hostile characters folded to ``_``, hash-suffixed when that changed it.

    Sanitizing can collide distinct keys ('a/b.jpg' vs 'a_b.jpg'); a short stable hash of the
    true key keeps their files distinct. The key itself lives in the payload.
    """
    safe = key
    for ch in _PATH_HOSTILE:
        safe = safe.replace(ch, "_")
    if safe != key:
        safe = f"{safe}.{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"
    return safe


def shard_filename(img_name: str) -> str:
    """The single filename an image key is stored under."""
    return f"{_sanitized(img_name)}{_SHARD_SUFFIX}"


def bucket_dirname(bucket: str) -> str:
    """The single directory name one bucket's shards are stored under: the bucket key, separators
    included, folded into one sanitized directory name. :data:`NO_BUCKET` names no directory, so a
    review carrying no prediction bucket keeps its shard directly under ``review/``.
    """
    return _sanitized(bucket)


@dataclass(frozen=True)
class _ShardLocator:
    """Places one (bucket, image) verdict shard, carrying the sanitizing the layout needs.

    The recoverable key from a path is the sanitized bucket directory and filename, which place
    that same file. A key that had to be sanitized is not recoverable from the path: the true keys
    are stored inside the payload.
    """

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> PurePosixPath:
        bucket, img_name = parts
        return PurePosixPath(
            REVIEW_SHARD_DIRNAME, bucket_dirname(bucket), shard_filename(img_name)
        )

    def parts_from(self, relative_path: PurePosixPath) -> tuple[str, ...] | None:
        segments = relative_path.parts
        if not 2 <= len(segments) <= 3 or segments[0] != REVIEW_SHARD_DIRNAME:
            return None
        if not segments[-1].endswith(_SHARD_SUFFIX):
            return None
        bucket = segments[1] if len(segments) == 3 else NO_BUCKET
        return (bucket, segments[-1][: -len(_SHARD_SUFFIX)])


def true_verdict_parts(entry: bytes) -> tuple[str, ...] | None:
    """The (bucket, image) a shard's own payload states, or None when it states neither.

    Bytes that will not decode state no key: that shard still enumerates under the parts its path
    spells, and reading it reports the corruption.
    """
    try:
        payload = RECORD_JSON.decode(entry)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    bucket, img_name = payload.get("bucket"), payload.get("img_name")
    if isinstance(bucket, str) and isinstance(img_name, str):
        return (bucket, img_name)
    return None


REVIEW_VERDICTS_STORE = "review_verdicts"
_SHARD_LOCATOR = _ShardLocator()
register_store(
    StoreDescriptor(
        name=REVIEW_VERDICTS_STORE,
        kind="record",
        key_fields=("bucket", "image"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=_SHARD_LOCATOR,
        true_parts_from_entry=true_verdict_parts,
    )
)


def review_verdict_key(state_dir: str | Path, bucket: str, img_name: str) -> Key:
    """One image's review verdicts under one prediction bucket.

    Keyed by the bucket as well as the image, so a filename a camera reuses across two dates holds
    one set of verdicts per bucket. ``cas``.
    """
    return Key(REVIEW_VERDICTS_STORE, str(state_dir), (bucket, img_name))


LABEL_BASELINES_STORE = "label_baselines"
BASELINE_DIRNAME = ".original"
register_store(
    StoreDescriptor(
        name=LABEL_BASELINES_STORE,
        kind="blob",
        key_fields=("stem",),
        frozen=True,
        locator=RootedFileLocator(prefix=(BASELINE_DIRNAME,), suffix=LABEL_SUFFIX),
    )
)


def label_baseline_key(label_dir: str | Path, stem: str) -> Key:
    """One label file's pristine copy, beside the directory the original lives in."""
    return Key(LABEL_BASELINES_STORE, str(Path(label_dir).absolute()), (str(stem),))


def capture_label_baseline(label_path: str | Path) -> bool:
    """Capture one label file's current bytes as its pristine baseline, if none is held yet.

    Create-only: a baseline already on record is left alone. Returns ``True`` if this call captured
    a new baseline, ``False`` if one was already held. The caller confirms ``label_path`` names an
    existing file first.
    """
    src = Path(label_path)
    try:
        tcip_store.put_blob(
            label_baseline_key(src.parent, src.stem), src.read_bytes(),
            expect=Version.ABSENT,
        )
    except tcip_store.VersionConflict:
        return False
    return True


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
        self._shard_versions: dict[tuple[str, str], Version] = {}
        self._reviewed_lookup: tuple[tuple[str, str], dict, dict] = ((NO_BUCKET, ""), {}, {})
        self.load_review_state()

    # ── Persistence ───────────────────────────────────────────────────────

    @property
    def shard_dir(self) -> Path:
        return self.state_dir / REVIEW_SHARD_DIRNAME

    def _shard_path(self, bucket: str, img_name: str) -> Path:
        relative = _SHARD_LOCATOR.relative_path(str(self.state_dir), (bucket, img_name))
        return Path(self.state_dir, *relative.parts)

    def _verdicts(self) -> dict:
        return self._review_state.get("verdicts", {})

    def load_review_state(self) -> None:
        verdicts: dict = {}
        self._shard_versions = {}
        for key in tcip_store.keys(REVIEW_VERDICTS_STORE, str(self.state_dir)):
            try:
                stored = tcip_store.read_versioned(key)
            except tcip_store.SchemaVersionRefused:
                # A newer writer's verdict, not corruption: propagate rather than lose it.
                raise
            except tcip_store.StoreError:
                logger.exception("Could not load review shard %s", key.parts[-1])
                continue
            payload = stored.value
            # The true bucket and image keys are stored inside the payload, so a sanitized/
            # hash-suffixed path never mutates or merges keys on reload.
            bucket, img_name = payload.get("bucket"), payload.get("img_name")
            state = payload.get("state")
            if bucket is None or img_name is None or state is None:
                bucket, img_name, state = key.parts[0], key.parts[1], payload
            verdicts[(bucket, img_name)] = state
            self._shard_versions[(bucket, img_name)] = stored.version
        self._review_state = {"verdicts": verdicts} if verdicts else {}
        self._invalidate_reviewed_lookup()

    def _save_image(self, bucket: str, img_name: str) -> None:
        """Persist only this (bucket, image)'s shard: O(detections on that image), not
        O(all-reviewed).

        Compare-and-set against the version this engine last saw; a conflict raises.
        """
        img_data = self._verdicts().get((bucket, img_name))
        if img_data is None:
            return
        # Store the true keys alongside the state so reload reconstructs them from the payload,
        # not the (possibly sanitized) path.
        self._shard_versions[(bucket, img_name)] = tcip_store.replace(
            review_verdict_key(self.state_dir, bucket, img_name),
            {"bucket": bucket, "img_name": img_name, "state": img_data},
            expect=self._shard_versions.get((bucket, img_name), Version.ABSENT),
        )

    def save_review_state(self) -> None:
        """Flush every shard."""
        for bucket, img_name in list(self._verdicts()):
            self._save_image(bucket, img_name)

    @property
    def raw_state(self) -> dict:
        """Expose the raw review_state dict, keyed ``(bucket, image)``. Mainly for tests / audit."""
        return self._review_state

    def reviewed_buckets(self) -> list[str]:
        """Every prediction bucket this store holds verdicts or completions for, sorted."""
        return sorted({bucket for bucket, _ in self._verdicts()})

    def image_states(self, bucket: str) -> dict[str, dict]:
        """One bucket's image-name to review-state map: the shape the verdict readers consume."""
        return {img: data for (b, img), data in self._verdicts().items() if b == bucket}

    # ── Image-level status ────────────────────────────────────────────────

    def mark_image_reviewed(self, bucket: str, img_name: str, *,
                            producer_identity: Optional[dict] = None,
                            adjudication_covered: Optional[dict[str, bool]] = None) -> None:
        """Mark ``img_name`` fully reviewed under ``bucket`` (e.g. a confirmed negative /
        bulk-accept).

        ``producer_identity``: the resolved producing-bucket fact
            (``checkpoint_sha256``/``experiment_id``), a plain dict the caller resolves, stamped at
            the image level. ``None`` (the default) leaves any existing stamp untouched.

        ``adjudication_covered``: a map from subject name (``"*"`` for a subject-less Complete, a
            claim about every subject) to whether this zero-verdict completion is a genuine
            negative the caller has already confirmed for it (the prediction bucket held zero
            detections resolving to that subject on this image). A bulk-accept of an image the
            bucket did predict on passes ``False`` for its subject. Merged into any existing map
            rather than replacing it. ``None`` (the default) leaves any existing map untouched.
        """
        verdicts = self._review_state.setdefault("verdicts", {})
        img_data = verdicts.setdefault(
            (bucket, img_name), {"img_status": "completed", "detections": []}
        )
        img_data["img_status"] = "completed"
        if producer_identity is not None:
            img_data["producer_identity"] = producer_identity
        if adjudication_covered is not None:
            existing = img_data.get("adjudication_covered")
            if existing is not None and not isinstance(existing, dict):
                raise ValueError(
                    f"adjudication_covered is {existing!r}, not the subject-keyed map a merge "
                    "expects; a bare boolean here is a shape no current writer produces."
                )
            merged = dict(existing) if existing is not None else {}
            merged.update(adjudication_covered)
            img_data["adjudication_covered"] = merged
        self._save_image(bucket, img_name)

    def unmark_image_reviewed(self, bucket: str, img_name: str) -> None:
        """Reverse a manual mark: back to started (verdicts kept) or not_started."""
        img_data = self._verdicts().get((bucket, img_name))
        if not img_data:
            return
        img_data["img_status"] = "started" if img_data.get("detections") else "not_started"
        self._save_image(bucket, img_name)

    def is_image_reviewed(self, bucket: str, img_name: str) -> bool:
        img_data = self._verdicts().get((bucket, img_name))
        if not img_data:
            return False
        return img_data.get("img_status") == "completed"

    def get_image_review_status(self, bucket: str, img_name: str) -> str:
        """Return ``"completed"``, ``"started"``, or ``"not_started"``."""
        img_data = self._verdicts().get((bucket, img_name))
        if not img_data:
            return "not_started"
        return img_data.get("img_status", "not_started")

    def verdict_count_for_images(self, bucket: str, names: Iterable[str]) -> int:
        """Total recorded verdicts (accept/reject/edit detection entries) on ``bucket`` across
        ``names``, matched by image stem so a prediction bucket's ``<stem>.json`` files line up
        with the review log's image-name keys. Scoped to the one bucket.
        """
        wanted = {Path(n).stem for n in names}
        if not wanted:
            return 0
        total = 0
        for (b, img_name), data in self._verdicts().items():
            if b == bucket and Path(img_name).stem in wanted:
                total += len(data.get("detections", []))
        return total

    def get_all_image_statuses(self) -> dict[str, str]:
        """Review status for every image the engine has state for, across every bucket (untouched
        images are absent, the caller defaults them to ``"not_started"``). An image reviewed under
        several buckets reports the furthest status any of them reached.
        """
        rank = {"not_started": 0, "started": 1, "completed": 2}
        statuses: dict[str, str] = {}
        for (_bucket, name), data in self._verdicts().items():
            status = data.get("img_status", "not_started")
            if rank.get(status, 0) >= rank.get(statuses.get(name, "not_started"), 0):
                statuses[name] = status
        return statuses

    # ── Bounding-box helpers ──────────────────────────────────────────────

    @staticmethod
    def _annotation_at(anns: list[Annotation], idx: Optional[int]) -> Optional[Annotation]:
        """The annotation ``idx`` names in ``anns``, or ``None`` when it names none: the one
        bounds check every read of a detection's record by index shares."""
        return anns[idx] if idx is not None and 0 <= idx < len(anns) else None

    @staticmethod
    def _box_of(record: Optional[Annotation]):
        """``record``'s image-coord box, or ``None`` when there is no record or no box to read. A
        :class:`~tcip_annotation.state.Point` reads as ``None``, like a geometry-less label.
        """
        geom = record.geometry if record is not None else None
        if not box_derivable(geom):
            return None
        b = bbox_of(geom)
        return (b.x1, b.y1, b.x2, b.y2)

    def _bbox_of_gt(self, ctx: ReviewContext, gt_idx: Optional[int]):
        return self._box_of(self._annotation_at(ctx.gt, gt_idx))

    def _bbox_of_pred(self, ctx: ReviewContext, pred_idx: Optional[int]):
        return self._box_of(self._annotation_at(ctx.preds, pred_idx))

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

    def _normalized_bbox(self, ctx: ReviewContext,
                         record: Optional[Annotation]) -> Optional[list[float]]:
        """``record``'s box as ``[cx, cy, w, h]`` normalized to the image, or ``None`` with no box."""
        img_w = max(ctx.img_width, 1)
        img_h = max(ctx.img_height, 1)
        b = self._box_of(record)
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
        self._reviewed_lookup = ((NO_BUCKET, ""), {}, {})

    def _build_reviewed_lookup(self, bucket: str, img_name: str) -> None:
        """Index this image's entries by the center of each box its decoded verdict carries,
        holding the center beside the entry so a lookup never reads the entry again."""
        img_data = self._verdicts().get((bucket, img_name)) or {}
        pred_map: dict = {}
        gt_map: dict = {}
        for entry in img_data.get("detections", []):
            verdict = decode_verdict(entry)
            for box, index in ((verdict.pred_box, pred_map), (verdict.gt_box, gt_map)):
                if box is not None:
                    qk = (round(box[0] * _LOOKUP_QUANT), round(box[1] * _LOOKUP_QUANT))
                    index.setdefault(qk, []).append((box[0], box[1], entry))
        self._reviewed_lookup = ((bucket, img_name), pred_map, gt_map)

    def find_reviewed_entry(
        self, bucket: str, det: ReviewDetection, ctx: ReviewContext
    ) -> Optional[dict]:
        """Return the reviewed-entry dict for ``det`` on this image under ``bucket``, if any: the
        entry whose predicted box (a TP or FP) or ground-truth box (an FN) shares its center."""
        if not ctx.img_name:
            return None
        if self._reviewed_lookup[0] != (bucket, ctx.img_name):
            self._build_reviewed_lookup(bucket, ctx.img_name)
        _, pred_map, gt_map = self._reviewed_lookup

        if det.det_type in ("tp", "fp"):
            box, index = self._normalized_bbox(ctx, self._annotation_at(ctx.preds, det.pred_idx)), pred_map
        else:  # fn
            box, index = self._normalized_bbox(ctx, self._annotation_at(ctx.gt, det.gt_idx)), gt_map
        if not box:
            return None
        cx, cy = box[0], box[1]
        qx, qy = round(cx * _LOOKUP_QUANT), round(cy * _LOOKUP_QUANT)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ex, ey, entry in index.get((qx + dx, qy + dy), ()):
                    if abs(ex - cx) < _LOOKUP_TOLERANCE and abs(ey - cy) < _LOOKUP_TOLERANCE:
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

        ``filter_class`` is a class name (an annotation's ``subject``) or ``"all"``. Within an
        image every matching detection is walkable regardless of whether it has been reviewed;
        reviewed/unreviewed state rides on each detection via :meth:`find_reviewed_entry`.
        """

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
                    gt_idx=m.get("gt_idx"),
                    pred_idx=m["pred_idx"],
                    bbox=self._detection_bbox(ctx, m.get("gt_idx"), m["pred_idx"]),
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
                    pred_idx=m.get("pred_idx"),
                    bbox=self._detection_bbox(ctx, m["gt_idx"], m.get("pred_idx")),
                ))

        return dets

    # ── Record & check ────────────────────────────────────────────────────

    def record_detection_action(
        self,
        bucket: str,
        det: ReviewDetection,
        ctx: ReviewContext,
        action: VerdictAction,
        *,
        norm_det: Optional[ReviewDetection] = None,
        norm_ctx: Optional[ReviewContext] = None,
        producer_identity: Optional[dict] = None,
        conf_threshold: Optional[float] = None,
        class_id: Optional[int] = None,
    ) -> None:
        """Log an accept / reject / edit / sweep action for a detection, against ``bucket``.

        ``action`` outside :data:`VERDICT_ACTIONS` refuses with a ``ValueError``. ``"swept"`` is an
        explicit "checked this image for missed objects, found none" attestation: it is recorded
        like any other verdict but never mutates ground truth.

        ``bucket``: the prediction bucket key the reviewer was looking at (:data:`NO_BUCKET` when
            the review carries no predictions), stored verbatim.

        ``norm_det``/``norm_ctx`` override the geometry the entry is stored under. An edited
        verdict rewrites the GT bbox, so the entry is keyed to the post-edit geometry while any
        prior entry for this detection is still found via the pre-edit geometry of ``det``/``ctx``.

        ``producer_identity``: the resolved producing-bucket fact
            (``checkpoint_sha256``/``experiment_id``, plus ``bucket_dir`` for human legibility)
            this verdict was recorded against, a plain dict the caller resolves and this package
            stores verbatim.

        ``conf_threshold``: the review session's confidence-display threshold in effect when this
            verdict was recorded.

        ``class_id``: the 0-indexed class identity ``det.class_name`` resolves to under the
            producing bucket's own recorded name->id map, a plain fact the caller resolves and this
            package stores verbatim. ``None`` when the caller could not resolve one (no recorded
            map, or ``class_name`` isn't one of its keys). Every entry also carries the crowd flag
            of the ground-truth record its box is read from.

        On the first verdict recorded for this image, stamps ``gt_preexisting = bool(ctx.gt)`` onto
        the image-level record: whether this image had ground truth before the review session
        touched it.

        Every entry also stamps ``missed_object_attested``: ``True`` only when both ``det.gt_idx``
        and ``det.pred_idx`` are ``None`` at the moment of this call, the shape the "mark missed
        object" action and the sweep attestation produce; a paired false positive/negative under a
        classified trait review carries an index on at least one side and never satisfies it.
        """
        if action not in VERDICT_ACTIONS:
            raise ValueError(
                f"unknown verdict action {action!r}; must be one of {VERDICT_ACTIONS}"
            )
        verdicts = self._review_state.setdefault("verdicts", {})
        is_first_verdict = (bucket, ctx.img_name) not in verdicts
        img_data = verdicts.setdefault(
            (bucket, ctx.img_name), {"img_status": "started", "detections": []}
        )
        if is_first_verdict:
            img_data["gt_preexisting"] = bool(ctx.gt)

        nd = norm_det if norm_det is not None else det
        nc = norm_ctx if norm_ctx is not None else ctx
        # The record the entry's ground-truth box is read from, read for its crowd flag too.
        gt_record = self._annotation_at(nc.gt, nd.gt_idx)
        entry = {
            "action": action,
            "reviewed_by": self.current_user,
            "class_name": det.class_name,
            GT_BOX_KEY: self._normalized_bbox(nc, gt_record),
            PRED_BOX_KEY: self._normalized_bbox(nc, self._annotation_at(nc.preds, nd.pred_idx)),
            "conf": round(det.conf, 4) if det.conf is not None else None,
            "producer_identity": producer_identity,
            "conf_threshold": conf_threshold,
            "missed_object_attested": det.gt_idx is None and det.pred_idx is None,
            "class_id": class_id,
            "iscrowd": gt_record is not None and gt_record.iscrowd,
        }

        existing = self.find_reviewed_entry(bucket, det, ctx)
        if existing:
            idx = img_data["detections"].index(existing)
            img_data["detections"][idx] = entry
        else:
            img_data["detections"].append(entry)

        self._invalidate_reviewed_lookup()

        if img_data.get("img_status") == "not_started":
            img_data["img_status"] = "started"

        self._save_image(bucket, ctx.img_name)

    def review_progress(
        self, bucket: str, ctx: ReviewContext, dets: list[ReviewDetection]
    ) -> tuple[int, int]:
        """``(reviewed, total)`` over ``dets`` (the caller's own :meth:`build_detection_list`
        result): how many distinct stored entries the detections in it find, by the lookup the
        per-detection ticks use (:meth:`find_reviewed_entry`), and how many detections there are in
        total.

        Reviewed means a stored entry exists for a detection still in this current match set. Two
        current detections whose centers alias to the same stored entry count as one reviewed of
        two. A coverage-only sweep attestation (carrying neither ``gt_bbox_norm`` nor
        ``pred_bbox_norm``) counts toward nothing here.
        """
        reviewed_entries = {
            id(entry) for d in dets if (entry := self.find_reviewed_entry(bucket, d, ctx)) is not None
        }
        return len(reviewed_entries), len(dets)

    def check_image_review_complete(
        self, bucket: str, ctx: ReviewContext, matches: dict
    ) -> bool:
        """If every current detection on the image has a stored verdict, mark it complete.

        Completion means every detection :meth:`review_progress` currently counts has an entry,
        never that stored entries outnumber that set: entries recorded for predictions a later,
        higher confidence threshold excludes do not complete an image early. Returns ``True`` if
        the image is in the completed state after this call.
        """
        dets = self.build_detection_list(ctx, matches)
        reviewed, total = self.review_progress(bucket, ctx, dets)
        if total == 0 or reviewed < total:
            return False

        img_data = self._verdicts().get((bucket, ctx.img_name))
        if not img_data:
            return False
        img_data["img_status"] = "completed"
        self._save_image(bucket, ctx.img_name)
        return True

    # ── Label backup / save ───────────────────────────────────────────────

    def backup_original_labels(self, *label_dirs: Path | str) -> int:
        """Ensure every label file in each dir has a pristine copy in ``<dir>/.original/``.

        Per-file, idempotent and create-only: a file is captured the first time it is seen and
        never overwritten afterwards. Returns the number of files newly captured by this call.
        """
        captured = 0
        for label_dir in label_dirs:
            d = Path(label_dir)
            if not d.is_dir():
                continue
            for src in d.iterdir():
                if not (src.is_file() and src.suffix == LABEL_SUFFIX):
                    continue
                if capture_label_baseline(src):
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
