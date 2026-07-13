import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Line, Rect, Text } from "react-konva";
import type Konva from "konva";

import { api, IMAGE_MAX_WIDTH } from "@/api/client";
import { classesApi } from "@/api/classes";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
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
import type {
  Box,
  DatasetSelection,
  Detection,
  MatchesResponse,
  PolygonShape,
  PredBox,
  PredPolygon,
} from "@/store/types";

const TAG_COLORS: Record<"tp" | "fp" | "fn", string> = {
  tp: "#4CAF50",
  fp: "#EF5350",
  fn: "#FFA726",
};

const EDIT_COLOR = "#00BFFF"; // tcip-pred — the shape currently picked up for adjustment
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
  const [imageStatus, setImageStatus] = useState<MatchesResponse["image_status"]>("not_started");
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
      when: () => !!current && !edit,
    },
    {
      keys: "r",
      action: (e) => {
        if (!e.repeat) void recordAction("rejected");
      },
      when: () => !!current && !edit,
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
  const acceptLabel = "Accept (A)";
  const rejectLabel = "Reject (R)";
  const acceptTitle =
    current?.det_type === "fp"
      ? "Add this prediction to ground truth (A)"
      : "Keep this ground-truth object (A)";
  const rejectTitle =
    current?.det_type === "fp"
      ? "Discard this prediction — ground truth unchanged (R)"
      : "Delete this ground-truth object (R)";

  return (
    <div className="flex-1 flex flex-col relative">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-tcip-border bg-tcip-panel text-[11px]">
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

        <label className="flex items-center gap-1 ml-3">
          <input type="checkbox" checked={showGT} onChange={(e) => setShowGT(e.target.checked)} />
          GT
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showPred}
            onChange={(e) => setShowPred(e.target.checked)}
          />
          Pred
        </label>

        <span className="flex-1" />

        {imgName && (
          <span className="text-tcip-fg font-medium truncate max-w-[12rem]" title={imgName}>
            {imgName}
          </span>
        )}
        <span className={`tcip-badge ${IMAGE_STATUS_CLASS[imageStatus]}`}>
          {IMAGE_STATUS_LABEL[imageStatus]}
        </span>
        {/* Same affordance as Annotate's Complete: a reversible checkbox. */}
        <label
          className="flex items-center gap-1"
          title="Mark this image fully reviewed — its annotation status is confirmed from the GT files; uncheck to reopen"
        >
          <input
            type="checkbox"
            checked={imageStatus === "completed"}
            onChange={(e) => void markImageComplete(e.target.checked)}
            disabled={!imgName || !!edit}
          />
          Reviewed
        </label>

        <button className="tcip-btn" onClick={() => stepImage(-1)} disabled={!!edit}>
          ◀&nbsp;&nbsp;Prev img
        </button>
        <span className="tabular-nums">
          {matches && matches.detections.length > 0
            ? `${detectionIdx + 1} / ${matches.detections.length}`
            : "0 / 0"}
        </span>
        <button className="tcip-btn" onClick={() => stepImage(1)} disabled={!!edit}>
          Next img&nbsp;&nbsp;▶
        </button>
        <button
          className="tcip-btn ml-2"
          onClick={() => setToolsOpen(true)}
          disabled={!!edit}
          title="Build training set / prioritize review queue"
        >
          ⚙&nbsp;&nbsp;Tools
        </button>
      </div>

      <div className="relative flex-1 flex flex-col">
        <CanvasStage
          imageUrl={imageUrl}
          imgWidth={imgW}
          imgHeight={imgH}
          onPixelDown={edit ? onEditDown : undefined}
          onPixelMove={edit ? onEditMove : undefined}
          onPixelUp={edit ? onEditUp : undefined}
          overlay={edit ? <EditShapeOverlay edit={edit} /> : undefined}
        >
          {matches && (
            <ReviewOverlays
              matches={matches}
              focusedIdx={detectionIdx}
              showGT={showGT}
              showPred={showPred}
              classNameLookup={className}
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
              color: TAG_COLORS[current.det_type],
              borderColor: TAG_COLORS[current.det_type],
            }}
          >
            {current.det_type.toUpperCase()}
          </span>
        )}
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
        <button
          className="tcip-btn"
          onClick={() => stepDetection(-1)}
          disabled={!matches || matches.detections.length === 0 || detectionIdx <= 0 || !!edit}
          title="Previous detection (←)"
        >
          ◀&nbsp;&nbsp;Prev
        </button>
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
          Next&nbsp;&nbsp;▶
        </button>
        {current && (
          <>
            <span
              className={`tcip-badge bg-transparent border ${
                current.det_type === "tp"
                  ? "border-tcip-tp text-tcip-tp"
                  : current.det_type === "fp"
                    ? "border-tcip-fp text-tcip-fp"
                    : "border-tcip-fn text-tcip-fn"
              }`}
            >
              {current.det_type.toUpperCase()}
            </span>
            <span className="text-tcip-muted">
              cid {current.class_id}: {className(current.class_id)}
              {current.conf !== null && ` · conf ${current.conf.toFixed(2)}`}
              {current.iou !== null && ` · iou ${current.iou.toFixed(2)}`}
            </span>
            {current.reviewed && (
              <span className="text-tcip-warn">({current.reviewed_action})</span>
            )}
          </>
        )}

        <span className="flex-1" />

        {current && !edit && (
          <>
            <span
              className="text-tcip-muted mr-1"
              title="Verdicts write ground truth — each button says what it does for this detection"
            >
              writes GT ·
            </span>
            <button
              className="tcip-btn-primary"
              onClick={() => void recordAction("accepted")}
              title={acceptTitle}
            >
              ✓&nbsp;&nbsp;{acceptLabel}
            </button>
            <button
              className="tcip-btn"
              onClick={startEdit}
              title="Adjust this shape on the canvas (E)"
            >
              ✎&nbsp;&nbsp;Edit (E)
            </button>
            <button
              className="tcip-btn-danger"
              onClick={() => void recordAction("rejected")}
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
            <span className="text-tcip-muted">
              {edit.kind === "box"
                ? "Drag a corner to resize · drag inside to move"
                : "Drag a point to reshape · drag inside to move"}
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
              ✓&nbsp;&nbsp;Save edit (Enter)
            </button>
            <button
              className="tcip-btn"
              onClick={cancelEdit}
              title="Discard this adjustment — ground truth unchanged (Esc)"
            >
              Cancel (Esc)
            </button>
          </>
        )}
      </div>

      <ReviewToolsDrawer open={toolsOpen} onClose={() => setToolsOpen(false)} />
    </div>
  );
}

function EditShapeOverlay({ edit }: { edit: EditShape }) {
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
          stroke={EDIT_COLOR}
          strokeWidth={2.5 * lw}
          fill="rgba(0, 191, 255, 0.08)"
        />
        {corners.map(([cx, cy], i) => (
          <Rect
            key={i}
            x={cx - hs}
            y={cy - hs}
            width={hs * 2}
            height={hs * 2}
            fill="#FFFFFF"
            stroke={EDIT_COLOR}
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
        stroke={EDIT_COLOR}
        strokeWidth={2.5 * lw}
        fill="rgba(0, 191, 255, 0.08)"
      />
      {edit.points.map(([px, py], i) => (
        <Circle
          key={i}
          x={px}
          y={py}
          radius={4.5 * lw}
          fill="#FFFFFF"
          stroke={EDIT_COLOR}
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
  suppressFocusedGt,
  suppressFocusedPred,
}: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const focused = matches.detections[focusedIdx];

  // Build a lookup for which (gt_type, gt_idx) and (pred_type, pred_idx)
  // belong to a reviewed detection — used to draw stippled / faded GT.
  const reviewedGtKeys = new Set<string>();
  matches.detections.forEach((d) => {
    if (d.reviewed && d.gt_type && d.gt_idx !== null) {
      reviewedGtKeys.add(`${d.gt_type}:${d.gt_idx}`);
    }
  });

  const focusedGt =
    focused && focused.gt_type && focused.gt_idx !== null
      ? `${focused.gt_type}:${focused.gt_idx}`
      : null;

  const tagColor = focused ? TAG_COLORS[focused.det_type] : null;

  return (
    <>
      {showGT &&
        matches.gt_boxes.map((b: Box, i: number) => {
          const key = `box:${i}`;
          const isFocused = focusedGt === key;
          if (isFocused && suppressFocusedGt) return null;
          const isReviewed = reviewedGtKeys.has(key);
          return (
            <GtBox
              key={`gt-b-${i}`}
              box={b}
              focused={isFocused}
              reviewed={isReviewed}
              lw={lw}
              label={`GT ${b.class_id}: ${classNameLookup(b.class_id)}`}
            />
          );
        })}
      {showGT &&
        matches.gt_polygons.map((p: PolygonShape, i: number) => {
          const key = `polygon:${i}`;
          const isFocused = focusedGt === key;
          if (isFocused && suppressFocusedGt) return null;
          const isReviewed = reviewedGtKeys.has(key);
          return (
            <GtPolygon
              key={`gt-p-${i}`}
              polygon={p}
              focused={isFocused}
              reviewed={isReviewed}
              lw={lw}
              label={`GT ${p.class_id}: ${classNameLookup(p.class_id)}`}
            />
          );
        })}

      {/* Focused prediction (only the active detection) */}
      {showPred &&
        !suppressFocusedPred &&
        focused &&
        tagColor &&
        focused.pred_type === "box" &&
        focused.pred_idx !== null &&
        matches.pred_boxes[focused.pred_idx] && (
          <FocusedPredBox
            b={matches.pred_boxes[focused.pred_idx]}
            color={tagColor}
            lw={lw}
            label={`Pred ${focused.class_id}: ${classNameLookup(focused.class_id)}${
              focused.conf !== null ? ` (${focused.conf.toFixed(2)})` : ""
            }`}
            isFp={focused.det_type === "fp"}
          />
        )}
      {showPred &&
        !suppressFocusedPred &&
        focused &&
        tagColor &&
        focused.pred_type === "polygon" &&
        focused.pred_idx !== null &&
        matches.pred_polygons[focused.pred_idx] && (
          <FocusedPredPoly
            p={matches.pred_polygons[focused.pred_idx]}
            color={tagColor}
            lw={lw}
            label={`Pred ${focused.class_id}: ${classNameLookup(focused.class_id)}${
              focused.conf !== null ? ` (${focused.conf.toFixed(2)})` : ""
            }`}
            isFp={focused.det_type === "fp"}
          />
        )}
    </>
  );
}

function GtBox({
  box,
  focused,
  reviewed,
  lw,
  label,
}: {
  box: Box;
  focused: boolean;
  reviewed: boolean;
  lw: number;
  label: string;
}) {
  const stroke = focused ? "#FFD700" : "#4CAF50";
  const fill = reviewed ? "rgba(76, 175, 80, 0.18)" : "";
  return (
    <>
      <Rect
        x={box.x1}
        y={box.y1}
        width={box.x2 - box.x1}
        height={box.y2 - box.y1}
        stroke={stroke}
        strokeWidth={(focused ? 3 : 2) * lw}
        fill={fill}
      />
      {focused && <HaloLabel x={box.x1} y={box.y1} text={label} fill={stroke} size={11 * lw} />}
    </>
  );
}

function GtPolygon({
  polygon,
  focused,
  reviewed,
  lw,
  label,
}: {
  polygon: PolygonShape;
  focused: boolean;
  reviewed: boolean;
  lw: number;
  label: string;
}) {
  const stroke = focused ? "#FFD700" : "#4CAF50";
  const fill = reviewed ? "rgba(76, 175, 80, 0.18)" : "";
  if (polygon.points.length < 2) return null;
  const [x0, y0] = polygon.points[0];
  return (
    <>
      <Line
        points={polygon.points.flat()}
        closed
        stroke={stroke}
        strokeWidth={(focused ? 3 : 2) * lw}
        fill={fill}
      />
      {focused && <HaloLabel x={x0} y={y0} text={label} fill={stroke} size={11 * lw} />}
    </>
  );
}

function FocusedPredBox({
  b,
  color,
  lw,
  label,
  isFp,
}: {
  b: PredBox;
  color: string;
  lw: number;
  label: string;
  isFp: boolean;
}) {
  return (
    <>
      <Rect
        x={b.x1}
        y={b.y1}
        width={b.x2 - b.x1}
        height={b.y2 - b.y1}
        stroke={color}
        strokeWidth={3 * lw}
        dash={[8 * lw, 4 * lw]}
        fill={isFp ? `${color}33` : ""}
      />
      <HaloLabel x={b.x1} y={b.y1 + (b.y2 - b.y1)} text={label} fill={color} size={11 * lw} />
    </>
  );
}

function FocusedPredPoly({
  p,
  color,
  lw,
  label,
  isFp,
}: {
  p: PredPolygon;
  color: string;
  lw: number;
  label: string;
  isFp: boolean;
}) {
  if (p.points.length < 2) return null;
  const [x0, y0] = p.points[p.points.length - 1];
  return (
    <>
      <Line
        points={p.points.flat()}
        closed
        stroke={color}
        strokeWidth={3 * lw}
        dash={[8 * lw, 4 * lw]}
        fill={isFp ? `${color}33` : ""}
      />
      <HaloLabel x={x0} y={y0} text={label} fill={color} size={11 * lw} />
    </>
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
