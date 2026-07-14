import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Circle, Line, Rect, Text } from "react-konva";
import type Konva from "konva";

import { api, IMAGE_MAX_WIDTH } from "@/api/client";
import { classesApi } from "@/api/classes";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { ColorPickerModal } from "@/components/ColorPickerModal";
import { MAX_SCALE, MIN_SCALE } from "@/components/Canvas/zoom";
import { ReviewToolsDrawer } from "@/components/ReviewToolsDrawer";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePrefetchAdjacentImages } from "@/hooks/usePrefetchAdjacentImages";
import {
  applyEditDrag,
  clampShapeToImage,
  hitTestEdit,
  type EditDrag,
  type EditShape,
} from "@/lib/reviewEditGeometry";
import { useStore } from "@/store";
import type { Box, DatasetSelection, Detection, MatchesResponse, PredBox } from "@/store/types";

// Review symbology colours (color = outcome; line style = source). User-customisable and
// persisted, so a reviewer can retune TP/FP/FN/under-review to their imagery.
export interface ReviewColors {
  tp: string;
  fp: string;
  fn: string;
  active: string;
}
const DEFAULT_REVIEW_COLORS: ReviewColors = {
  tp: "#4CAF50", // matched
  fp: "#EF5350", // false positive
  fn: "#FFD54A", // missed (gold)
  active: "#00BFFF", // the detection under review — highlighter blue
};
const REVIEW_COLORS_KEY = "tcip.review.colors";
function loadReviewColors(): ReviewColors {
  try {
    const raw = localStorage.getItem(REVIEW_COLORS_KEY);
    if (raw) return { ...DEFAULT_REVIEW_COLORS, ...JSON.parse(raw) };
  } catch {
    /* disabled storage — fall back to defaults */
  }
  return DEFAULT_REVIEW_COLORS;
}
const COLOR_LABELS: { key: keyof ReviewColors; label: string; tag: string; dashed?: boolean }[] = [
  { key: "tp", label: "Matched (TP)", tag: "TP" },
  { key: "fp", label: "False positive (FP)", tag: "FP" },
  { key: "fn", label: "Missed (FN)", tag: "FN" },
  { key: "active", label: "Under review", tag: "active", dashed: true },
];
const MIN_BOX_SIDE = 3;
const HANDLE_HIT_PX = 10; // screen-px hit radius for edit handles

/** The shape Edit picks up: the matched GT for a TP/FN (what a save replaces), the
 *  prediction for an FP (what a save adds). Deep-copied so dragging never mutates matches. */
function seedEditShape(d: Detection, m: MatchesResponse): EditShape | null {
  if (d.det_type === "fp") {
    if (d.pred_type === "box" && d.pred_idx !== null && m.pred_boxes[d.pred_idx]) {
      const b = m.pred_boxes[d.pred_idx];
      return { kind: "box", box: [b.x1, b.y1, b.x2, b.y2] };
    }
    if (d.pred_type === "polygon" && d.pred_idx !== null && m.pred_polygons[d.pred_idx]) {
      const pts = m.pred_polygons[d.pred_idx].points;
      return { kind: "polygon", points: pts.map((p) => [p[0], p[1]]) };
    }
    return null;
  }
  if (d.gt_type === "box" && d.gt_idx !== null && m.gt_boxes[d.gt_idx]) {
    const b = m.gt_boxes[d.gt_idx];
    return { kind: "box", box: [b.x1, b.y1, b.x2, b.y2] };
  }
  if (d.gt_type === "polygon" && d.gt_idx !== null && m.gt_polygons[d.gt_idx]) {
    const pts = m.gt_polygons[d.gt_idx].points;
    return { kind: "polygon", points: pts.map((p) => [p[0], p[1]]) };
  }
  return null;
}

function currentImagePath(dataset: DatasetSelection): { path: string | null; name: string | null } {
  if (!dataset.dataset_root || !dataset.date) return { path: null, name: null };
  const name = dataset.image_list[dataset.current_image_index];
  if (!name) return { path: null, name: null };
  return { path: `${dataset.dataset_root}/images/${dataset.date}/${name}`, name };
}

function labelPaths(dataset: DatasetSelection, name: string | null) {
  if (!name) return { gt_detect: null, gt_segment: null, pred_detect: null, pred_segment: null };
  const stem = name.replace(/\.[^.]+$/, "");
  return {
    gt_detect: dataset.annotations_detect_dir
      ? `${dataset.annotations_detect_dir}/${stem}.txt`
      : null,
    gt_segment: dataset.annotations_segment_dir
      ? `${dataset.annotations_segment_dir}/${stem}.txt`
      : null,
    pred_detect: dataset.predictions_detect_dir
      ? `${dataset.predictions_detect_dir}/${stem}.txt`
      : null,
    pred_segment: dataset.predictions_segment_dir
      ? `${dataset.predictions_segment_dir}/${stem}.txt`
      : null,
  };
}

const TYPE_ORDER: ("tp" | "fp" | "fn")[] = ["tp", "fp", "fn"];

const IMAGE_STATUS_LABEL: Record<MatchesResponse["image_status"], string> = {
  not_started: "not started",
  started: "in progress",
  completed: "reviewed",
};
const IMAGE_STATUS_CLASS: Record<MatchesResponse["image_status"], string> = {
  not_started: "bg-tcip-border text-tcip-muted",
  started: "bg-tcip-fn/20 text-tcip-fn",
  completed: "bg-tcip-tp/20 text-tcip-tp",
};

export function ReviewTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  const gui = useStore((s) => s.gui);
  const setView = useStore((s) => s.setView);
  const matches = useStore((s) => s.review.matches);
  const setMatches = useStore((s) => s.setMatches);
  const setLoading = useStore((s) => s.setReviewLoading);
  const setDetectionIdx = useStore((s) => s.setReviewDetectionIdx);
  const markDetReviewed = useStore((s) => s.markDetectionReviewed);
  const setPredReference = useStore((s) => s.setPredReference);
  const className = useStore((s) => s.className);
  // Shared annotation status (nav filter, coloring, Complete lock) — synced when a verdict authors GT.
  const setStoreImageStatus = useStore((s) => s.setImageStatus);
  // Shared filtered navigation (same order as the arrow keys + TopBar Prev/Next).
  const nav = useImageNav();
  usePrefetchAdjacentImages();

  const detectionIdx = gui.review.detection_idx;
  const filters = gui.review;
  const { path: imgPath, name: imgName } = currentImagePath(dataset);
  const paths = useMemo(() => labelPaths(dataset, imgName), [dataset, imgName]);

  const [showGT, setShowGT] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [toolsOpen, setToolsOpen] = useState(false);
  // The filter shelf is collapsed by default and remembers the last state across sessions.
  const [filtersOpen, setFiltersOpen] = useState<boolean>(() => {
    try {
      return localStorage.getItem("tcip.review.filtersOpen") === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("tcip.review.filtersOpen", filtersOpen ? "1" : "0");
    } catch {
      /* private mode / disabled storage — the shelf just won't persist */
    }
  }, [filtersOpen]);
  const [counterDraft, setCounterDraft] = useState<string | null>(null);
  const counterRef = useRef<HTMLInputElement | null>(null);
  const [imageStatus, setImageStatus] = useState<MatchesResponse["image_status"]>("not_started");
  // A reviewed (completed) image is locked — no verdicts/edits until it's reopened.
  const reviewLocked = imageStatus === "completed";
  // User-tunable symbology colours (persisted); the legend swatches open a picker.
  const [reviewColors, setReviewColors] = useState<ReviewColors>(loadReviewColors);
  const [colorEditKey, setColorEditKey] = useState<keyof ReviewColors | null>(null);
  useEffect(() => {
    try {
      localStorage.setItem(REVIEW_COLORS_KEY, JSON.stringify(reviewColors));
    } catch {
      /* disabled storage — colours just won't persist */
    }
  }, [reviewColors]);
  const [edit, setEdit] = useState<EditShape | null>(null);
  const editDrag = useRef<EditDrag | null>(null);
  // One GT-mutating request at a time: key auto-repeat / double-clicks must not append
  // or delete twice, and no verdict may land while indices are stale mid-reload.
  const actionPending = useRef(false);

  async function reloadMatches(indexHint?: number, signal?: AbortSignal) {
    if (!dataset.project_root || !imgPath || !imgName) return;
    setLoading(true);
    try {
      const res = await api.review.matches(
        {
          project_root: dataset.project_root,
          image_name: imgName,
          image_path: imgPath,
          gt_detect_path: paths.gt_detect,
          gt_segment_path: paths.gt_segment,
          pred_detect_path: paths.pred_detect,
          pred_segment_path: paths.pred_segment,
          iou_threshold: filters.iou_threshold,
          conf_threshold: filters.conf_threshold,
          filter_type: filters.filter_type,
          filter_class: filters.filter_class,
          status_filter: filters.status_filter,
        },
        signal,
      );
      // Identity check: if the user navigated while this was in flight, installing the
      // response would put another image's matches under the current image.
      const now = useStore.getState().gui.dataset;
      if ((now.image_list[now.current_image_index] ?? null) !== imgName) return;
      setMatches(res);
      setImageStatus(res.image_status);
      // A pending `review_focus` index (the agent asked to center on detection N) wins for one
      // reload; otherwise honor an explicit hint, else jump to the first unreviewed detection.
      const focusIdx = useStore.getState().review.focusDetectionIdx;
      const effectiveHint = indexHint ?? focusIdx ?? undefined;
      if (focusIdx !== null && focusIdx !== undefined) useStore.getState().setReviewFocusIdx(null);
      if (effectiveHint === undefined) {
        const firstUnreviewed = res.detections.findIndex((d) => !d.reviewed);
        const target = firstUnreviewed >= 0 ? firstUnreviewed : 0;
        setDetectionIdx(target);
        zoomToDetection(res.detections[target]?.bbox);
      } else {
        const clamped = Math.max(0, Math.min(res.detections.length - 1, effectiveHint));
        setDetectionIdx(clamped);
        zoomToDetection(res.detections[clamped]?.bbox);
      }
    } catch (e) {
      // A superseded (aborted) request is expected during slider drags — ignore it.
      if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
      useStore
        .getState()
        .pushToast(`Could not load review matches: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  function zoomToDetection(bbox: [number, number, number, number] | undefined) {
    if (!bbox) return;
    const [x1, y1, x2, y2] = bbox;
    const dw = Math.max(1, x2 - x1);
    const dh = Math.max(1, y2 - y1);
    const wrapper = document.querySelector("[data-canvas-host]") as HTMLElement | null;
    const cw = wrapper?.clientWidth ?? 1200;
    const ch = wrapper?.clientHeight ?? 800;
    // Clamp to the wheel ladder's range — beyond MAX_SCALE the wheel goes dead/jumpy.
    const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, Math.min(cw / (dw * 3), ch / (dh * 3))));
    setView({
      scale,
      offset_x: cw / 2 - ((x1 + x2) / 2) * scale,
      offset_y: ch / 2 - ((y1 + y2) / 2) * scale,
    });
  }

  useEffect(() => {
    // Debounce so dragging the IoU/Conf sliders doesn't fire a /matches recompute per
    // tick, and abort the in-flight request so a slow earlier response can't clobber a
    // newer one (out-of-order responses previously won).
    const ac = new AbortController();
    const t = setTimeout(() => void reloadMatches(undefined, ac.signal), 180);
    return () => {
      clearTimeout(t);
      ac.abort();
    };
    // The four path STRINGS (not the `paths` object — mergeSnapshot rebuilds the
    // dataset object on every WS snapshot, which would spuriously re-fire this and
    // reset the detection index/zoom) so a backend-adopted change of prediction
    // dirs (e.g. the agent re-selects the dataset with a different model) refreshes
    // the matches instead of silently showing the previous model's TP/FP/FN.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    imgPath,
    paths.gt_detect,
    paths.gt_segment,
    paths.pred_detect,
    paths.pred_segment,
    filters.iou_threshold,
    filters.conf_threshold,
    filters.filter_type,
    filters.filter_class,
    filters.status_filter,
  ]);

  function stepImage(delta: number) {
    nav.stepImage(delta);
    setPredReference(null);
  }

  function stepDetection(delta: number) {
    if (!matches) return;
    const next = Math.max(0, Math.min(matches.detections.length - 1, detectionIdx + delta));
    setDetectionIdx(next);
    zoomToDetection(matches.detections[next]?.bbox);
  }

  /**
   * Find the next unreviewed detection. Mirrors yolo-annotator's
   * filter-exhaustion auto-switch: if the active type filter has no more
   * unreviewed detections, try the next type, and if nothing remains on
   * the image at all, advance to the next image.
   */
  function advanceToNextUnreviewed() {
    if (!matches) return;
    const dets = matches.detections;
    // Same filter, next unreviewed
    for (let i = 0; i < dets.length; i++) {
      const j = (detectionIdx + 1 + i) % dets.length;
      if (!dets[j].reviewed) {
        setDetectionIdx(j);
        zoomToDetection(dets[j].bbox);
        return;
      }
    }
    // Filter type exhausted — try other types under "all" filter context
    if (filters.filter_type !== "all") {
      const otherTypes = TYPE_ORDER.filter((t) => t !== filters.filter_type);
      for (const t of otherTypes) {
        // The current `dets` list reflects the current filter; we can use the
        // counts from matches (n_tp/n_fp/n_fn) only as a hint that there's
        // more to look at. The cleanest path is to relax the type filter to
        // "all" and reload; reloadMatches will jump to the first unreviewed.
        if (
          (t === "tp" && matches.n_tp > 0) ||
          (t === "fp" && matches.n_fp > 0) ||
          (t === "fn" && matches.n_fn > 0)
        ) {
          patchGui({ review: { ...filters, filter_type: "all" } });
          return; // reload will fire via the deps effect
        }
      }
    }
    // No more on this image — go to next image.
    stepImage(1);
  }

  const current = matches?.detections[detectionIdx] ?? null;

  async function recordAction(
    action: "accepted" | "rejected" | "edited",
    edited?: { box?: [number, number, number, number]; polygon?: number[][] },
  ): Promise<boolean> {
    if (actionPending.current) return false;
    if (reviewLocked) return false; // a completed/reviewed image is locked until reopened
    if (!current || !dataset.project_root || !imgPath || !imgName) return false;
    actionPending.current = true;
    try {
      // The .original snapshot must exist before the first GT write — awaited, and a
      // failure aborts the verdict rather than mutating labels with no pristine baseline.
      if (!(await ensureBackup())) return false;
      const res = await api.review.action({
        project_root: dataset.project_root,
        image_name: imgName,
        image_path: imgPath,
        gt_detect_path: paths.gt_detect,
        gt_segment_path: paths.gt_segment,
        pred_detect_path: paths.pred_detect,
        pred_segment_path: paths.pred_segment,
        det_type: current.det_type,
        class_id: current.class_id,
        conf: current.conf,
        iou: current.iou,
        gt_type: current.gt_type,
        gt_idx: current.gt_idx,
        pred_type: current.pred_type,
        pred_idx: current.pred_idx,
        bbox: current.bbox,
        action,
        edited_box: edited?.box ?? null,
        edited_polygon: edited?.polygon ?? null,
        iou_threshold: filters.iou_threshold,
        conf_threshold: filters.conf_threshold,
      });
      setImageStatus(res.image_status);
      markDetReviewed(detectionIdx, action);
      advanceToNextUnreviewed();
      if (res.annotation_status) {
        // GT changed: sync the status, then refresh so gt_idx/pred_idx are rebuilt from the
        // written files — awaited, so no verdict can land against the stale indices.
        setStoreImageStatus(imgName, res.annotation_status);
        await reloadMatches(useStore.getState().gui.review.detection_idx);
      }
      return true;
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not record review action: ${e instanceof Error ? e.message : String(e)}`);
      return false;
    } finally {
      actionPending.current = false;
    }
  }

  async function ensureBackup(): Promise<boolean> {
    if (!dataset.project_root) return false;
    const dirs = [dataset.annotations_detect_dir, dataset.annotations_segment_dir].filter(
      Boolean,
    ) as string[];
    if (!dirs.length) return true;
    try {
      await api.review.backupLabels(dataset.project_root, dirs);
      return true;
    } catch {
      useStore
        .getState()
        .pushToast("Could not back up original labels — nothing was written. Retry the action.");
      return false;
    }
  }

  async function markImageComplete(completed: boolean) {
    if (!dataset.project_root || !imgName) return;
    try {
      const res = await api.review.markComplete({
        project_root: dataset.project_root,
        image_name: imgName,
        gt_detect_path: paths.gt_detect,
        gt_segment_path: paths.gt_segment,
        completed,
      });
      setImageStatus(res.image_status); // local review badge
      // The annotation status comes from the server (GT files on disk), never from a
      // matches snapshot that can belong to the previous image mid-navigation.
      setStoreImageStatus(imgName, res.annotation_status);
      void classesApi
        .setImageStatus(dataset.project_root, imgName, res.annotation_status)
        .catch(() => {});
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not mark image reviewed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // ── In-place edit: pick the shape up on this canvas, adjust, save to GT ──

  function startEdit() {
    if (reviewLocked) return;
    if (!current || !matches) return;
    const seed = seedEditShape(current, matches);
    if (!seed) {
      useStore.getState().pushToast("This detection has no shape to adjust.");
      return;
    }
    setEdit(clampShapeToImage(seed, matches.img_width, matches.img_height));
  }

  function cancelEdit() {
    setEdit(null);
    editDrag.current = null;
  }

  async function commitEdit() {
    if (!edit) return;
    if (edit.kind === "box") {
      const [x1, y1, x2, y2] = edit.box;
      if (x2 - x1 < MIN_BOX_SIDE || y2 - y1 < MIN_BOX_SIDE) {
        useStore.getState().pushToast("Box too small to save — drag a corner to enlarge it.");
        return;
      }
      if (await recordAction("edited", { box: edit.box })) cancelEdit();
      return;
    }
    if (edit.points.length < 3) {
      useStore.getState().pushToast("Polygon needs at least 3 points to save.");
      return;
    }
    if (await recordAction("edited", { polygon: edit.points.map((p) => [p[0], p[1]]) }))
      cancelEdit();
  }

  // An edit belongs to one detection on one matches snapshot — leaving either discards it.
  useEffect(() => {
    setEdit(null);
    editDrag.current = null;
  }, [detectionIdx, imgName, matches]);

  function onEditDown(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    editDrag.current = null; // a miss (or a stale drag from an off-canvas release) grabs nothing
    if (!edit || ev.evt.button !== 0) return;
    const tol = HANDLE_HIT_PX / (useStore.getState().gui.view.scale || 1);
    editDrag.current = hitTestEdit(edit, x, y, tol);
  }

  function onEditMove(x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) {
    const drag = editDrag.current;
    if (!edit || !drag || !matches) return;
    // The button was released outside the canvas — Konva never delivered the mouseup.
    if (ev.evt.buttons === 0) {
      editDrag.current = null;
      return;
    }
    const r = applyEditDrag(edit, drag, x, y, matches.img_width, matches.img_height);
    editDrag.current = r.drag;
    if (r.shape !== edit) setEdit(r.shape);
  }

  function onEditUp() {
    editDrag.current = null;
  }

  useKeyboardShortcuts([
    {
      keys: "a",
      action: (e) => {
        if (!e.repeat) void recordAction("accepted");
      },
      when: () => !!current && !edit && !reviewLocked,
    },
    {
      keys: "r",
      action: (e) => {
        if (!e.repeat) void recordAction("rejected");
      },
      when: () => !!current && !edit && !reviewLocked,
    },
    { keys: "e", action: () => startEdit(), when: () => !!current && !edit },
    {
      keys: "enter",
      action: (e) => {
        // A focused button owns Enter (its native click already fired or will).
        if (!e.repeat && !(document.activeElement instanceof HTMLButtonElement)) void commitEdit();
      },
      when: () => !!edit,
    },
    { keys: "escape", action: () => cancelEdit(), when: () => !!edit },
    { keys: "arrowleft", action: () => stepDetection(-1), when: () => !edit },
    { keys: "arrowright", action: () => stepDetection(1), when: () => !edit },
    // Image flips ignore held-key auto-repeat — each one costs a full image render.
    {
      keys: "arrowup",
      action: (e) => {
        if (!e.repeat) stepImage(-1);
      },
      when: () => !edit,
    },
    {
      keys: "arrowdown",
      action: (e) => {
        if (!e.repeat) stepImage(1);
      },
      when: () => !edit,
    },
  ]);

  const imageUrl = imgPath ? api.images.url(imgPath, IMAGE_MAX_WIDTH) : null;
  const imgW = matches?.img_width ?? 0;
  const imgH = matches?.img_height ?? 0;

  // Verdicts author ground truth: accept an FP adds the prediction to GT; reject a detection that
  // has GT (TP/FN) deletes that GT box; accept a TP/FN keeps the existing GT. Edit adjusts the shape
  // first, then accept commits it.
  const acceptLabel = "Accept";
  const rejectLabel = "Reject";
  const acceptTitle =
    current?.det_type === "fp"
      ? "Add this prediction to ground truth (A)"
      : "Keep this ground-truth object (A)";
  const rejectTitle =
    current?.det_type === "fp"
      ? "Discard this prediction — ground truth unchanged (R)"
      : "Delete this ground-truth object (R)";

  return (
    <div className="flex-1 flex flex-col relative min-h-0">
      <div className="relative border-b border-tcip-border bg-tcip-panel">
        {/* Row 1 — filter shelf toggle + live summary + legend, then image / detection navigation */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-[11px]">
          <button
            className="tcip-btn"
            onClick={() => setFiltersOpen((v) => !v)}
            aria-expanded={filtersOpen}
            disabled={!!edit}
            title="Show or hide the review filters"
          >
            <span className={`inline-block transition-transform ${filtersOpen ? "rotate-90" : ""}`}>
              ▸
            </span>
            &nbsp;&nbsp;Filters
          </button>
          {/* Live summary — always shows every filter, so the shelf can stay collapsed. */}
          <span className="flex items-center gap-1.5 tabular-nums">
            <FilterChip>IoU ≥ {filters.iou_threshold.toFixed(2)}</FilterChip>
            <FilterChip>Conf ≥ {filters.conf_threshold.toFixed(2)}</FilterChip>
            <FilterChip>
              {filters.filter_type === "all" ? "All types" : filters.filter_type.toUpperCase()}
            </FilterChip>
            <FilterChip>
              {filters.status_filter === "all"
                ? "All status"
                : filters.status_filter === "reviewed"
                  ? "Reviewed"
                  : "Unreviewed"}
            </FilterChip>
            <FilterChip>
              {showGT && showPred
                ? "GT + Pred"
                : showGT
                  ? "GT only"
                  : showPred
                    ? "Pred only"
                    : "Hidden"}
            </FilterChip>
          </span>

          <span className="flex-1" />

          <span className={`tcip-badge ${IMAGE_STATUS_CLASS[imageStatus]}`}>
            {IMAGE_STATUS_LABEL[imageStatus]}
          </span>

          {/* Image navigation — same function + layout/order as the Annotate tab */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wide text-tcip-muted">
              Image
            </span>
            {imgName && (
              <span className="max-w-[150px] truncate font-mono text-tcip-fg" title={imgName}>
                {imgName}
              </span>
            )}
            <button
              className="tcip-btn"
              onClick={() => stepImage(-1)}
              disabled={!nav.canPrev || !!edit}
              aria-label="Previous image"
            >
              ◀
            </button>
            <input
              ref={counterRef}
              className="tcip-input w-10 text-center font-mono"
              value={counterDraft ?? (nav.position > 0 ? String(nav.position) : "")}
              onChange={(e) => setCounterDraft(e.target.value.replace(/[^0-9]/g, ""))}
              onFocus={() => setCounterDraft(String(nav.position || 1))}
              onBlur={() => setCounterDraft(null)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const num = parseInt(counterDraft ?? "", 10);
                  if (!Number.isNaN(num)) nav.jumpToPosition(num);
                  setCounterDraft(null);
                  counterRef.current?.blur();
                } else if (e.key === "Escape") {
                  setCounterDraft(null);
                  counterRef.current?.blur();
                }
              }}
            />
            <span className="tabular-nums text-tcip-muted">/ {nav.total}</span>
            <button
              className="tcip-btn"
              onClick={() => stepImage(1)}
              disabled={!nav.canNext || !!edit}
              aria-label="Next image"
            >
              ▶
            </button>
          </div>

          {/* Reviewed — same position as Annotate's Complete; a reversible confirm. */}
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={imageStatus === "completed"}
              onChange={(e) => void markImageComplete(e.target.checked)}
              disabled={!imgName || !!edit}
            />
            Reviewed
          </label>

          <button
            className="tcip-btn"
            onClick={() => setToolsOpen(true)}
            disabled={!!edit}
            title="Build training set / prioritize review queue"
          >
            ⚙&nbsp;&nbsp;Tools
          </button>
        </div>

        {/* Row 2 — the filter controls, collapsed by default and remembered across sessions */}
        {filtersOpen && (
          <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 border-t border-tcip-border text-[11px]">
            <span className="text-tcip-muted">IoU ≥</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={filters.iou_threshold * 100}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({
                  review: { ...filters, iou_threshold: Number(e.target.value) / 100 },
                })
              }
            />
            <span className="tabular-nums w-10">{filters.iou_threshold.toFixed(2)}</span>

            <span className="ml-3 text-tcip-muted">Conf ≥</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={filters.conf_threshold * 100}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({
                  review: { ...filters, conf_threshold: Number(e.target.value) / 100 },
                })
              }
            />
            <span className="tabular-nums w-10">{filters.conf_threshold.toFixed(2)}</span>

            <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
            <select
              className="tcip-select"
              value={filters.filter_type}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({ review: { ...filters, filter_type: e.target.value as never } })
              }
            >
              <option value="all">All</option>
              <option value="tp">TP</option>
              <option value="fp">FP</option>
              <option value="fn">FN</option>
            </select>
            <select
              className="tcip-select"
              value={filters.status_filter}
              disabled={!!edit}
              onChange={(e) =>
                patchGui({ review: { ...filters, status_filter: e.target.value as never } })
              }
            >
              <option value="all">All status</option>
              <option value="not_reviewed">Unreviewed</option>
              <option value="reviewed">Reviewed</option>
            </select>

            <span aria-hidden className="mx-2 h-4 w-px bg-tcip-border" />
            <span className="text-[10px] font-semibold uppercase tracking-wide text-tcip-muted">
              Visibility
            </span>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={showGT}
                onChange={(e) => setShowGT(e.target.checked)}
              />
              Ground truth
            </label>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={showPred}
                onChange={(e) => setShowPred(e.target.checked)}
              />
              Predictions
            </label>
          </div>
        )}
      </div>

      <div className="relative flex-1 flex flex-col min-h-0">
        <CanvasStage
          imageUrl={imageUrl}
          hiResImageUrl={imgPath ? api.images.hiResUrl(imgPath) : null}
          imgWidth={imgW}
          imgHeight={imgH}
          onPixelDown={edit ? onEditDown : undefined}
          onPixelMove={edit ? onEditMove : undefined}
          onPixelUp={edit ? onEditUp : undefined}
          overlay={edit ? <EditShapeOverlay edit={edit} color={reviewColors.active} /> : undefined}
        >
          {matches && (
            <ReviewOverlays
              matches={matches}
              focusedIdx={detectionIdx}
              showGT={showGT}
              showPred={showPred}
              classNameLookup={className}
              colors={reviewColors}
              suppressFocusedGt={!!edit && current?.det_type !== "fp"}
              suppressFocusedPred={!!edit && current?.det_type === "fp"}
            />
          )}
        </CanvasStage>
        {/* Screen-fixed detection-type badge — in image coords it was illegible at fit
            zoom and canvas-blanketing when zoomed to a detection. */}
        {current && (
          <span
            className="absolute top-2 right-3 tcip-badge border bg-tcip-panel/90 pointer-events-none font-bold"
            style={{
              color: reviewColors[current.det_type],
              borderColor: reviewColors[current.det_type],
            }}
          >
            {current.det_type.toUpperCase()}
            <span className="mx-1 text-tcip-border">|</span>
            <span className="font-normal text-tcip-muted">{edit ? "editing" : "reviewing"}</span>
          </span>
        )}

        <ReviewLegend colors={reviewColors} onEdit={setColorEditKey} />
      </div>

      {/* Empty-state card: tells the reviewer WHY there is nothing to step through —
          "no predictions configured" vs "filters exclude everything". Non-opaque and
          pointer-transparent so still-rendered GT overlays stay visible behind it. */}
      {matches && matches.detections.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="max-w-md rounded-lg border border-tcip-border bg-tcip-panel/90 px-5 py-4 text-center">
            <p className="text-sm font-semibold text-tcip-fg">No detections to review</p>
            <p className="mt-1 text-xs text-tcip-muted">
              {!dataset.predictions_detect_dir && !dataset.predictions_segment_dir
                ? "No predictions directory configured — run inference or select a model with predictions for this dataset."
                : `No detections on this image under the current filters (IoU ≥ ${filters.iou_threshold.toFixed(
                    2,
                  )}, Conf ≥ ${filters.conf_threshold.toFixed(2)}, type ${
                    filters.filter_type
                  }) — relax filters to see more.`}
            </p>
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-tcip-border bg-tcip-panel text-[11px]">
        <span className="text-tcip-muted">Detection</span>
        <button
          className="tcip-btn"
          onClick={() => stepDetection(-1)}
          disabled={!matches || matches.detections.length === 0 || detectionIdx <= 0 || !!edit}
          title="Previous detection (←)"
        >
          ◀
        </button>
        <span className="tabular-nums">
          {matches && matches.detections.length > 0
            ? `${detectionIdx + 1} / ${matches.detections.length}`
            : "0 / 0"}
        </span>
        <button
          className="tcip-btn"
          onClick={() => stepDetection(1)}
          disabled={
            !matches ||
            matches.detections.length === 0 ||
            detectionIdx >= matches.detections.length - 1 ||
            !!edit
          }
          title="Next detection (→)"
        >
          ▶
        </button>
        {current && (
          <>
            <span className="text-tcip-muted">
              {className(current.class_id)}
              {current.conf !== null && (
                <>
                  <span className="mx-1.5 text-tcip-border">|</span>conf {current.conf.toFixed(2)}
                </>
              )}
              {current.iou !== null && (
                <>
                  <span className="mx-1.5 text-tcip-border">|</span>IoU {current.iou.toFixed(2)}
                </>
              )}
            </span>
            {current.reviewed && (
              <span className="text-tcip-warn">({current.reviewed_action})</span>
            )}
          </>
        )}

        <span className="flex-1" />

        {current && !edit && (
          <>
            {reviewLocked && <span className="text-tcip-muted">Reviewed — uncheck to edit</span>}
            <button
              className="tcip-btn-primary"
              onClick={() => void recordAction("accepted")}
              disabled={reviewLocked}
              title={acceptTitle}
            >
              ✓&nbsp;&nbsp;{acceptLabel}
            </button>
            <button
              className="tcip-btn"
              onClick={startEdit}
              disabled={reviewLocked}
              title="Adjust this shape on the canvas (E)"
            >
              ✎&nbsp;&nbsp;Edit
            </button>
            <button
              className="tcip-btn-danger"
              onClick={() => void recordAction("rejected")}
              disabled={reviewLocked}
              title={rejectTitle}
            >
              ✕&nbsp;&nbsp;{rejectLabel}
            </button>
          </>
        )}
        {current && edit && (
          <>
            <span className="tcip-badge bg-transparent border border-tcip-pred text-tcip-pred">
              Editing
            </span>
            <button
              className="tcip-btn-primary"
              onClick={() => void commitEdit()}
              title={
                current.det_type === "fp"
                  ? "Write this shape to ground truth (Enter)"
                  : "Replace the ground-truth shape with this one (Enter)"
              }
            >
              ✓&nbsp;&nbsp;Save edit
            </button>
            <button
              className="tcip-btn"
              onClick={cancelEdit}
              title="Discard this adjustment — ground truth unchanged (Esc)"
            >
              Cancel
            </button>
          </>
        )}
      </div>

      <ReviewToolsDrawer open={toolsOpen} onClose={() => setToolsOpen(false)} />

      {colorEditKey && (
        <ColorPickerModal
          title={`${COLOR_LABELS.find((c) => c.key === colorEditKey)?.label ?? "Colour"}`}
          initialColor={reviewColors[colorEditKey]}
          onSubmit={(c) => {
            setReviewColors((prev) => ({ ...prev, [colorEditKey]: c }));
            setColorEditKey(null);
          }}
          onCancel={() => setColorEditKey(null)}
        />
      )}
    </div>
  );
}

function FilterChip({ children }: { children: ReactNode }) {
  return (
    <span className="rounded border border-tcip-border bg-tcip-bg px-1.5 py-0.5 text-tcip-muted">
      {children}
    </span>
  );
}

/** A legend row whose colour swatch is a button — click it to retune that symbology colour. */
function LegendRow({
  color,
  dashed,
  label,
  onEdit,
}: {
  color: string;
  dashed?: boolean;
  label: string;
  onEdit: () => void;
}) {
  return (
    <li className="flex items-center gap-2.5">
      <button
        type="button"
        onClick={onEdit}
        title="Click to change this colour"
        aria-label={`Change ${label} colour`}
        className="inline-block w-6 shrink-0 rounded-sm hover:opacity-70"
        style={{ borderTop: `2.5px ${dashed ? "dashed" : "solid"} ${color}` }}
      />
      <span className="text-tcip-fg">{label}</span>
    </li>
  );
}

/** Hover-triggered legend anchored lower-left of the canvas (same pattern as Annotate).
 *  Keyed to what the canvas draws: solid = outcome, dashed blue = the detection under review.
 *  Each swatch opens a colour picker so the symbology palette is user-tunable. */
function ReviewLegend({
  colors,
  onEdit,
}: {
  colors: ReviewColors;
  onEdit: (key: keyof ReviewColors) => void;
}) {
  return (
    <div className="group absolute bottom-3 left-3 z-20">
      <div className="pointer-events-none absolute bottom-full left-0 mb-2 w-max min-w-[10rem] translate-y-1 whitespace-nowrap rounded-md border border-tcip-border-hover bg-tcip-panel p-3 opacity-0 shadow-lg transition-all group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100">
        <h4 className="mb-1.5 text-[11px] font-semibold tracking-wide text-tcip-fg">
          Review Legend
        </h4>
        <p className="mb-2 text-[11px] text-tcip-muted">
          Solid = outcome&nbsp; | &nbsp;Dashed blue = under review
        </p>
        <ul className="space-y-1.5">
          {COLOR_LABELS.map((c) => (
            <LegendRow
              key={c.key}
              color={colors[c.key]}
              dashed={c.dashed}
              label={c.label}
              onEdit={() => onEdit(c.key)}
            />
          ))}
        </ul>
        <p className="mt-2 border-t border-tcip-border pt-1.5 text-[10px] text-tcip-muted">
          Click a swatch to recolour
        </p>
      </div>
      <button
        type="button"
        className="flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.4" />
          <path
            d="M8 7.2v3.4M8 5.2v.05"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
        Legend
      </button>
    </div>
  );
}

function EditShapeOverlay({ edit, color }: { edit: EditShape; color: string }) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const hs = 5 * lw; // handle half-size
  if (edit.kind === "box") {
    const [x1, y1, x2, y2] = edit.box;
    const corners: [number, number][] = [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ];
    return (
      <>
        <Rect
          x={x1}
          y={y1}
          width={x2 - x1}
          height={y2 - y1}
          stroke={color}
          strokeWidth={2.5 * lw}
          fill={`${color}14`}
        />
        {corners.map(([cx, cy], i) => (
          <Rect
            key={i}
            x={cx - hs}
            y={cy - hs}
            width={hs * 2}
            height={hs * 2}
            fill="#FFFFFF"
            stroke={color}
            strokeWidth={1.5 * lw}
          />
        ))}
      </>
    );
  }
  if (edit.points.length < 2) return null;
  return (
    <>
      <Line
        points={edit.points.flat()}
        closed
        stroke={color}
        strokeWidth={2.5 * lw}
        fill={`${color}14`}
      />
      {edit.points.map(([px, py], i) => (
        <Circle
          key={i}
          x={px}
          y={py}
          radius={4.5 * lw}
          fill="#FFFFFF"
          stroke={color}
          strokeWidth={1.5 * lw}
        />
      ))}
    </>
  );
}

interface OverlayProps {
  matches: MatchesResponse;
  focusedIdx: number;
  showGT: boolean;
  showPred: boolean;
  classNameLookup: (cid: number) => string;
  colors: ReviewColors;
  /** While editing, the picked-up shape is hidden here — it renders live in the edit overlay. */
  suppressFocusedGt?: boolean;
  suppressFocusedPred?: boolean;
}

function ReviewOverlays({
  matches,
  focusedIdx,
  showGT,
  showPred,
  classNameLookup,
  colors,
  suppressFocusedGt,
  suppressFocusedPred,
}: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const ACTIVE_COLOR = colors.active;

  // One annotation task at a time: an image can carry both detect (boxes) and segment
  // (polygons) labels, and drawing both is unreadable. Show the kind being reviewed —
  // driven by the predictions, falling back to the GT kind when there are no predictions.
  const reviewKind: "box" | "polygon" = (() => {
    if (matches.pred_boxes.length || matches.pred_polygons.length) {
      return matches.pred_polygons.length && !matches.pred_boxes.length ? "polygon" : "box";
    }
    return matches.gt_polygons.length && !matches.gt_boxes.length ? "polygon" : "box";
  })();

  const box = (idx: number | null) => (idx !== null ? matches.gt_boxes[idx] : undefined);
  const predBox = (idx: number | null) => (idx !== null ? matches.pred_boxes[idx] : undefined);
  const poly = (idx: number | null) => (idx !== null ? matches.gt_polygons[idx] : undefined);
  const predPoly = (idx: number | null) => (idx !== null ? matches.pred_polygons[idx] : undefined);

  // Non-active first, the active detection last so its blue overlay sits on top.
  const order = matches.detections
    .map((_, i) => i)
    .sort((a, b) => (a === focusedIdx ? 1 : 0) - (b === focusedIdx ? 1 : 0));

  return (
    <>
      {order.map((i) => {
        const d = matches.detections[i];
        // Skip detections of the other annotation kind.
        if ((d.gt_type ?? d.pred_type) !== reviewKind) return null;
        const active = i === focusedIdx;
        const outcome = colors[d.det_type];
        const weight = active ? 3 : 2;
        const nodes: ReactNode[] = [];

        if (d.det_type === "fp") {
          // FP = a prediction with no GT. Solid outcome red as context; the detection under
          // review turns dashed blue (see the review legend).
          if (showPred && !(active && suppressFocusedPred)) {
            const stroke = active ? ACTIVE_COLOR : outcome;
            const fill = `${stroke}26`;
            const b = predBox(d.pred_type === "box" ? d.pred_idx : null);
            const p = predPoly(d.pred_type === "polygon" ? d.pred_idx : null);
            if (b)
              nodes.push(
                <ReviewRect
                  key="fp"
                  box={b}
                  stroke={stroke}
                  lw={lw}
                  weight={weight}
                  dashed={active}
                  fill={fill}
                />,
              );
            else if (p)
              nodes.push(
                <ReviewLine
                  key="fp"
                  points={p.points}
                  stroke={stroke}
                  lw={lw}
                  weight={weight}
                  dashed={active}
                  fill={fill}
                />,
              );
          }
        } else {
          // TP / FN = ground truth, solid. Active FN turns blue; active TP keeps its green GT.
          if (showGT && d.gt_type && !(active && suppressFocusedGt)) {
            const activeFn = active && d.det_type === "fn";
            const stroke = activeFn ? ACTIVE_COLOR : outcome;
            // A faint blue wash on the shape under review reads through even where its dashed
            // line coincides with the solid GT below it.
            const fill = activeFn ? `${ACTIVE_COLOR}26` : d.reviewed ? `${outcome}26` : undefined;
            const b = box(d.gt_type === "box" ? d.gt_idx : null);
            const p = poly(d.gt_type === "polygon" ? d.gt_idx : null);
            if (b)
              nodes.push(
                <ReviewRect key="gt" box={b} stroke={stroke} lw={lw} weight={weight} fill={fill} />,
              );
            else if (p)
              nodes.push(
                <ReviewLine
                  key="gt"
                  points={p.points}
                  stroke={stroke}
                  lw={lw}
                  weight={weight}
                  fill={fill}
                />,
              );
          }
          // The TP under review also shows its prediction as a dashed-blue overlay (pred vs GT).
          if (active && d.det_type === "tp" && showPred && !suppressFocusedPred) {
            const b = predBox(d.pred_type === "box" ? d.pred_idx : null);
            const p = predPoly(d.pred_type === "polygon" ? d.pred_idx : null);
            if (b)
              nodes.push(
                <ReviewRect
                  key="tp-pred"
                  box={b}
                  stroke={ACTIVE_COLOR}
                  lw={lw}
                  weight={3}
                  dashed
                  fill={`${ACTIVE_COLOR}26`}
                />,
              );
            else if (p)
              nodes.push(
                <ReviewLine
                  key="tp-pred"
                  points={p.points}
                  stroke={ACTIVE_COLOR}
                  lw={lw}
                  weight={3}
                  dashed
                  fill={`${ACTIVE_COLOR}26`}
                />,
              );
          }
        }

        if (active) {
          nodes.push(
            <HaloLabel
              key="lbl"
              x={d.bbox[0]}
              y={d.bbox[1]}
              text={`${classNameLookup(d.class_id)}${d.conf !== null ? ` ${d.conf.toFixed(2)}` : ""}`}
              fill={ACTIVE_COLOR}
              size={11 * lw}
            />,
          );
        }

        return <Fragment key={`det-${i}`}>{nodes}</Fragment>;
      })}
    </>
  );
}

function ReviewRect({
  box,
  stroke,
  lw,
  weight,
  dashed,
  fill,
}: {
  box: Box | PredBox;
  stroke: string;
  lw: number;
  weight: number;
  dashed?: boolean;
  fill?: string;
}) {
  return (
    <Rect
      x={box.x1}
      y={box.y1}
      width={box.x2 - box.x1}
      height={box.y2 - box.y1}
      stroke={stroke}
      strokeWidth={weight * lw}
      dash={dashed ? [8 * lw, 4 * lw] : undefined}
      fill={fill}
    />
  );
}

function ReviewLine({
  points,
  stroke,
  lw,
  weight,
  dashed,
  fill,
}: {
  points: [number, number][];
  stroke: string;
  lw: number;
  weight: number;
  dashed?: boolean;
  fill?: string;
}) {
  if (points.length < 2) return null;
  return (
    <Line
      points={points.flat()}
      closed
      stroke={stroke}
      strokeWidth={weight * lw}
      dash={dashed ? [8 * lw, 4 * lw] : undefined}
      fill={fill}
    />
  );
}

function HaloLabel({
  x,
  y,
  text,
  fill,
  size,
}: {
  x: number;
  y: number;
  text: string;
  fill: string;
  size: number;
}) {
  return (
    <>
      <Text
        x={x + 2}
        y={y - size - 2}
        text={text}
        fill="#000000"
        fontSize={size}
        fontStyle="bold"
        shadowColor="#000000"
        shadowBlur={size * 0.2}
        shadowOpacity={0.9}
      />
      <Text x={x + 2} y={y - size - 2} text={text} fill={fill} fontSize={size} fontStyle="bold" />
    </>
  );
}
