import { useEffect, useMemo, useState } from "react";
import { Line, Rect, Text } from "react-konva";

import { api, IMAGE_MAX_WIDTH } from "@/api/client";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { ReviewToolsDrawer } from "@/components/ReviewToolsDrawer";
import { useImageNav } from "@/hooks/useImageNav";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useStore } from "@/store";
import type {
  Box,
  DatasetSelection,
  Detection,
  MatchesResponse,
  PolygonShape,
  PredBox,
  PredPolygon,
  PredictionReference,
} from "@/store/types";

const TAG_COLORS: Record<"tp" | "fp" | "fn", string> = {
  tp: "#4CAF50",
  fp: "#EF5350",
  fn: "#FFA726",
};

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
  const setActiveTab = useStore((s) => s.setActiveTab);
  const setMode = useStore((s) => s.setMode);
  const setPredReference = useStore((s) => s.setPredReference);
  const className = useStore((s) => s.className);
  // Shared filtered navigation (same order as the arrow keys + TopBar Prev/Next).
  const nav = useImageNav();

  const detectionIdx = gui.review.detection_idx;
  const filters = gui.review;
  const { path: imgPath, name: imgName } = currentImagePath(dataset);
  const paths = useMemo(() => labelPaths(dataset, imgName), [dataset, imgName]);

  const [showGT, setShowGT] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [imageStatus, setImageStatus] = useState<MatchesResponse["image_status"]>("not_started");

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
      setMatches(res);
      setImageStatus(res.image_status);
      // jump to first unreviewed if no hint
      if (indexHint === undefined) {
        const firstUnreviewed = res.detections.findIndex((d) => !d.reviewed);
        const target = firstUnreviewed >= 0 ? firstUnreviewed : 0;
        setDetectionIdx(target);
        zoomToDetection(res.detections[target]?.bbox);
      } else {
        setDetectionIdx(Math.max(0, Math.min(res.detections.length - 1, indexHint)));
        zoomToDetection(res.detections[indexHint]?.bbox);
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
    const wrapper = document.querySelector(".flex-1.bg-tcip-canvas") as HTMLElement | null;
    const cw = wrapper?.clientWidth ?? 1200;
    const ch = wrapper?.clientHeight ?? 800;
    const scale = Math.max(0.05, Math.min(20, Math.min(cw / (dw * 3), ch / (dh * 3))));
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    imgPath,
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

  async function recordAction(action: "accepted" | "rejected" | "edited") {
    if (!current || !dataset.project_root || !imgPath || !imgName) return;
    try {
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
        iou_threshold: filters.iou_threshold,
        conf_threshold: filters.conf_threshold,
      });
      setImageStatus(res.image_status);
      markDetReviewed(detectionIdx, action);
      advanceToNextUnreviewed();
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not record review action: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function markImageComplete() {
    if (!dataset.project_root || !imgName) return;
    try {
      const res = await api.review.markComplete(dataset.project_root, imgName);
      setImageStatus(res.image_status);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Could not mark image reviewed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function editInAnnotate() {
    if (!current || !imgPath) return;
    // Back up the untouched GT once (idempotent per project) before the first edit, so a
    // mistaken save can be recovered from <dir>/.original/.
    if (dataset.project_root) {
      const dirs = [dataset.annotations_detect_dir, dataset.annotations_segment_dir].filter(
        Boolean,
      ) as string[];
      if (dirs.length) {
        void api.review.backupLabels(dataset.project_root, dirs).catch(() => {
          useStore.getState().pushToast("Could not back up original labels before editing.");
        });
      }
    }
    const predRef = buildPredReference(current, matches);
    setPredReference(predRef);
    setActiveTab("annotate");
    if (current.gt_type === "polygon" || current.pred_type === "polygon") setMode("polygon");
    else setMode("box");
    void api.annotate.openImage({
      image_path: imgPath,
      image_index: dataset.current_image_index,
      scale: gui.view.scale,
      offset_x: gui.view.offset_x,
      offset_y: gui.view.offset_y,
      mode: current.gt_type === "polygon" || current.pred_type === "polygon" ? "polygon" : "box",
      pred_reference: predRef,
    });
    // Mark editing pending so when the user returns to Review we offer a
    // confirm dialog.
    sessionStorage.setItem(
      "tcip.review_edit_pending",
      JSON.stringify({ image_name: imgName, detection_idx: detectionIdx }),
    );
  }

  // On Review-tab activation, if we just returned from an Edit flow, offer
  // a confirm dialog to record the edit.
  useEffect(() => {
    if (gui.active_tab !== "review") return;
    const raw = sessionStorage.getItem("tcip.review_edit_pending");
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw) as { image_name: string; detection_idx: number };
      if (parsed.image_name !== imgName) return;
      sessionStorage.removeItem("tcip.review_edit_pending");
      const ok = window.confirm(
        "Save this annotation edit to the GT dataset?\n\nOK to record as edited and advance.\nCancel to discard.",
      );
      if (ok && matches) {
        markDetReviewed(parsed.detection_idx, "edited");
        // Refresh matches so any GT changes are picked up
        void reloadMatches(parsed.detection_idx);
      }
    } catch {
      sessionStorage.removeItem("tcip.review_edit_pending");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gui.active_tab, imgName]);

  useKeyboardShortcuts([
    { keys: "a", action: () => void recordAction("accepted"), when: () => !!current },
    { keys: "r", action: () => void recordAction("rejected"), when: () => !!current },
    { keys: "e", action: () => editInAnnotate(), when: () => !!current },
    { keys: "arrowleft", action: () => stepDetection(-1) },
    { keys: "arrowright", action: () => stepDetection(1) },
    { keys: "arrowup", action: () => stepImage(-1) },
    { keys: "arrowdown", action: () => stepImage(1) },
  ]);

  const imageUrl = imgPath ? api.images.url(imgPath, IMAGE_MAX_WIDTH) : null;
  const imgW = matches?.img_width ?? 0;
  const imgH = matches?.img_height ?? 0;

  // Verdicts are recorded to the review log (for retraining/curation) — they do NOT
  // rewrite the GT label files. Only Edit (E) changes GT. Labels/tooltips reflect that
  // honestly; the old "Add to GT" / "Delete GT" wording implied a write that never happened.
  const acceptLabel = "Accept (A)";
  const rejectLabel = "Reject (R)";
  const acceptTitle =
    current?.det_type === "tp"
      ? "Record this match as correct (recorded for retraining, not written to GT)"
      : current?.det_type === "fp"
        ? "Record this prediction as a real object the GT was missing (recorded for retraining)"
        : "Record this ground-truth object as a real miss by the model (recorded for retraining)";
  const rejectTitle =
    current?.det_type === "tp"
      ? "Record this match as wrong (recorded for retraining, not written to GT)"
      : current?.det_type === "fp"
        ? "Record this prediction as a genuine false positive (recorded for retraining)"
        : "Record this ground-truth object as not a real object (recorded for retraining)";

  return (
    <div className="flex-1 flex flex-col relative">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-tcip-border bg-tcip-panel text-[11px]">
        <span>IoU ≥</span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={filters.iou_threshold * 100}
          onChange={(e) =>
            patchGui({
              review: { ...filters, iou_threshold: Number(e.target.value) / 100 },
            })
          }
        />
        <span className="tabular-nums w-10">{filters.iou_threshold.toFixed(2)}</span>

        <span className="ml-3">Conf ≥</span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={filters.conf_threshold * 100}
          onChange={(e) =>
            patchGui({
              review: { ...filters, conf_threshold: Number(e.target.value) / 100 },
            })
          }
        />
        <span className="tabular-nums w-10">{filters.conf_threshold.toFixed(2)}</span>

        <span className="mx-2 text-tcip-muted">|</span>
        <select
          className="tcip-select"
          value={filters.filter_type}
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

        <span className={`tcip-badge ${IMAGE_STATUS_CLASS[imageStatus]}`}>
          {IMAGE_STATUS_LABEL[imageStatus]}
        </span>
        <button
          className="tcip-btn"
          onClick={() => void markImageComplete()}
          disabled={imageStatus === "completed" || !imgName}
          title="Mark this image fully reviewed"
        >
          ✓&nbsp;&nbsp;Reviewed
        </button>

        <button className="tcip-btn" onClick={() => stepImage(-1)}>
          ◀&nbsp;&nbsp;Prev img
        </button>
        <span className="tabular-nums">
          {matches ? `${detectionIdx + 1} / ${matches.detections.length}` : "0 / 0"}
        </span>
        <button className="tcip-btn" onClick={() => stepImage(1)}>
          Next img&nbsp;&nbsp;▶
        </button>
        <button
          className="tcip-btn ml-2"
          onClick={() => setToolsOpen(true)}
          title="Build training set / prioritize review queue"
        >
          ⚙&nbsp;&nbsp;Tools
        </button>
      </div>

      <CanvasStage imageUrl={imageUrl} imgWidth={imgW} imgHeight={imgH}>
        {matches && (
          <ReviewOverlays
            matches={matches}
            focusedIdx={detectionIdx}
            showGT={showGT}
            showPred={showPred}
            classNameLookup={className}
          />
        )}
      </CanvasStage>

      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-tcip-border bg-tcip-panel text-[11px]">
        <button className="tcip-btn" onClick={() => stepDetection(-1)}>
          ◀&nbsp;&nbsp;Prev
        </button>
        <button className="tcip-btn" onClick={() => stepDetection(1)}>
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

        {current && (
          <>
            <span
              className="text-tcip-muted mr-1"
              title="Accept/Reject log a verdict for retraining. Edit changes the GT files."
            >
              logged for retraining ·
            </span>
            <button
              className="tcip-btn-primary"
              onClick={() => void recordAction("accepted")}
              title={acceptTitle}
            >
              ✓&nbsp;&nbsp;{acceptLabel}
            </button>
            <button className="tcip-btn" onClick={editInAnnotate} title="Edit GT for this image">
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
      </div>

      <ReviewToolsDrawer open={toolsOpen} onClose={() => setToolsOpen(false)} />
    </div>
  );
}

function buildPredReference(
  det: Detection,
  matches: MatchesResponse | null,
): PredictionReference | null {
  if (!matches) return null;
  if (det.pred_type === "box" && det.pred_idx !== null) {
    const b = matches.pred_boxes[det.pred_idx];
    if (!b) return null;
    return {
      type: "box",
      coords: [b.x1, b.y1, b.x2, b.y2],
      class_id: b.class_id,
      confidence: b.confidence,
    };
  }
  if (det.pred_type === "polygon" && det.pred_idx !== null) {
    const p = matches.pred_polygons[det.pred_idx];
    if (!p) return null;
    return {
      type: "polygon",
      coords: p.points as number[][],
      class_id: p.class_id,
      confidence: p.confidence,
    };
  }
  return null;
}

interface OverlayProps {
  matches: MatchesResponse;
  focusedIdx: number;
  showGT: boolean;
  showPred: boolean;
  classNameLookup: (cid: number) => string;
}

function ReviewOverlays({ matches, focusedIdx, showGT, showPred, classNameLookup }: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const focused = matches.detections[focusedIdx];
  const focusedColor = focused ? TAG_COLORS[focused.det_type] : "#E0E0E0";

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

      {/* Detection-type badge (top-right of screen, fixed canvas coords). */}
      {focused && (
        <DetTypeBadge
          color={focusedColor}
          label={focused.det_type.toUpperCase()}
          imgWidth={matches.img_width}
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

function DetTypeBadge({
  color,
  label,
  imgWidth,
}: {
  color: string;
  label: string;
  imgWidth: number;
}) {
  // Anchor the badge to image-coords near top-right (like yolo-annotator's
  // canvas overlay; sticks with the image as you pan/zoom).
  const x = Math.max(0, imgWidth - 80);
  const y = 4;
  return (
    <>
      <Rect
        x={x}
        y={y}
        width={70}
        height={28}
        fill="#1A1A1A"
        stroke="#444444"
        strokeWidth={1}
        cornerRadius={4}
      />
      <Text
        x={x}
        y={y + 6}
        width={70}
        align="center"
        text={`[${label}]`}
        fill={color}
        fontSize={14}
        fontStyle="bold"
      />
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
