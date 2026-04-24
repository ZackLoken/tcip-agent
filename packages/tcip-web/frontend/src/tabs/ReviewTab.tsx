import { useEffect, useMemo } from "react";
import { Line, Rect } from "react-konva";

import { api } from "@/api/client";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
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

const TAG_COLORS = { tp: "#4CAF50", fp: "#EF5350", fn: "#FFA726" } as const;

function currentImagePath(
  dataset: DatasetSelection,
): { path: string | null; name: string | null } {
  if (!dataset.dataset_root || !dataset.date) return { path: null, name: null };
  const name = dataset.image_list[dataset.current_image_index];
  if (!name) return { path: null, name: null };
  return {
    path: `${dataset.dataset_root}/images/${dataset.date}/${name}`,
    name,
  };
}

function labelPaths(dataset: DatasetSelection, name: string | null) {
  if (!name)
    return { gt_detect: null, gt_segment: null, pred_detect: null, pred_segment: null };
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

  const detectionIdx = gui.review.detection_idx;
  const filters = gui.review;
  const { path: imgPath, name: imgName } = currentImagePath(dataset);
  const paths = useMemo(() => labelPaths(dataset, imgName), [dataset, imgName]);

  async function reloadMatches(indexHint?: number) {
    if (!dataset.project_root || !imgPath || !imgName) return;
    setLoading(true);
    try {
      const res = await api.review.matches({
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
      });
      setMatches(res);
      setDetectionIdx(Math.max(0, Math.min(res.detections.length - 1, indexHint ?? 0)));
      // Ensure first-unreviewed is visible
      zoomToDetection(res.detections[indexHint ?? 0]?.bbox, res.img_width, res.img_height);
    } finally {
      setLoading(false);
    }
  }

  function zoomToDetection(
    bbox: [number, number, number, number] | undefined,
    imgW: number,
    imgH: number,
  ) {
    if (!bbox) return;
    const [x1, y1, x2, y2] = bbox;
    const dw = Math.max(1, x2 - x1);
    const dh = Math.max(1, y2 - y1);
    // Canvas-local dims — approximate from viewport; this is just the zoom
    const cw = (document.querySelector(".flex-1.bg-tcip-canvas") as HTMLElement)?.clientWidth ?? 1200;
    const ch = (document.querySelector(".flex-1.bg-tcip-canvas") as HTMLElement)?.clientHeight ?? 800;
    const scale = Math.max(0.05, Math.min(20, Math.min(cw / (dw * 3), ch / (dh * 3))));
    setView({
      scale,
      offset_x: cw / 2 - ((x1 + x2) / 2) * scale,
      offset_y: ch / 2 - ((y1 + y2) / 2) * scale,
    });
    void imgW;
    void imgH;
  }

  // Load matches when image or filters change
  useEffect(() => {
    void reloadMatches();
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
    if (!dataset.image_list.length) return;
    const next = Math.max(
      0,
      Math.min(dataset.image_list.length - 1, dataset.current_image_index + delta),
    );
    if (next === dataset.current_image_index) return;
    patchGui({ dataset: { ...dataset, current_image_index: next } });
    setPredReference(null);
  }

  function stepDetection(delta: number) {
    if (!matches) return;
    const next = Math.max(0, Math.min(matches.detections.length - 1, detectionIdx + delta));
    setDetectionIdx(next);
    zoomToDetection(
      matches.detections[next]?.bbox,
      matches.img_width,
      matches.img_height,
    );
  }

  const current = matches?.detections[detectionIdx] ?? null;

  async function recordAction(action: "accepted" | "rejected" | "edited") {
    if (!current || !dataset.project_root || !imgPath || !imgName) return;
    await api.review.action({
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
    });
    markDetReviewed(detectionIdx, action);
    // Advance to next unreviewed detection
    const dets = useStore.getState().review.matches?.detections ?? [];
    const start = detectionIdx + 1;
    let next = -1;
    for (let i = 0; i < dets.length; i++) {
      const j = (start + i) % dets.length;
      if (!dets[j].reviewed) {
        next = j;
        break;
      }
    }
    if (next >= 0) {
      setDetectionIdx(next);
      zoomToDetection(dets[next].bbox, matches!.img_width, matches!.img_height);
    }
  }

  function editInAnnotate() {
    if (!current || !imgPath) return;
    const predRef = buildPredReference(current, matches);
    setPredReference(predRef);
    setActiveTab("annotate");
    // Prefer polygon mode if polygons are in play
    if (current.gt_type === "polygon" || current.pred_type === "polygon") setMode("polygon");
    else setMode("box");
    // Notify backend so MCP audit sees the switch
    void api.annotate.openImage({
      image_path: imgPath,
      image_index: dataset.current_image_index,
      scale: gui.view.scale,
      offset_x: gui.view.offset_x,
      offset_y: gui.view.offset_y,
      mode: current.gt_type === "polygon" || current.pred_type === "polygon" ? "polygon" : "box",
      pred_reference: predRef,
    });
  }

  useKeyboardShortcuts([
    { keys: "a", action: () => recordAction("accepted"), when: () => !!current },
    { keys: "r", action: () => recordAction("rejected"), when: () => !!current },
    { keys: "e", action: () => editInAnnotate(), when: () => !!current },
    { keys: "arrowleft", action: () => stepDetection(-1) },
    { keys: "arrowright", action: () => stepDetection(1) },
    { keys: "arrowup", action: () => stepImage(-1) },
    { keys: "arrowdown", action: () => stepImage(1) },
  ]);

  const imageUrl = imgPath ? api.images.url(imgPath) : null;
  const imgW = matches?.img_width ?? 0;
  const imgH = matches?.img_height ?? 0;

  return (
    <div className="flex-1 flex flex-col">
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

        <span className="flex-1" />

        <button className="tcip-btn" onClick={() => stepImage(-1)}>◀ Prev img</button>
        <span className="tabular-nums">
          {matches ? `${detectionIdx + 1} / ${matches.detections.length}` : "0 / 0"}
        </span>
        <button className="tcip-btn" onClick={() => stepImage(1)}>Next img ▶</button>
      </div>

      <CanvasStage imageUrl={imageUrl} imgWidth={imgW} imgHeight={imgH}>
        {matches && <ReviewOverlays matches={matches} focusedIdx={detectionIdx} />}
      </CanvasStage>

      <div className="flex items-center gap-2 px-3 py-1.5 border-t border-tcip-border bg-tcip-panel text-[11px]">
        <button className="tcip-btn" onClick={() => stepDetection(-1)}>◀ Prev</button>
        <button className="tcip-btn" onClick={() => stepDetection(1)}>Next ▶</button>
        {current && (
          <>
            <span className={`tcip-badge bg-transparent border ${
              current.det_type === "tp" ? "border-tcip-tp text-tcip-tp" :
              current.det_type === "fp" ? "border-tcip-fp text-tcip-fp" :
              "border-tcip-fn text-tcip-fn"
            }`}>
              {current.det_type.toUpperCase()}
            </span>
            <span className="text-tcip-muted">
              cid {current.class_id}
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
            <button className="tcip-btn-primary" onClick={() => recordAction("accepted")}>
              ✓ Accept (A)
            </button>
            <button className="tcip-btn" onClick={editInAnnotate}>
              ✎ Edit (E)
            </button>
            <button className="tcip-btn-danger" onClick={() => recordAction("rejected")}>
              ✕ Reject (R)
            </button>
          </>
        )}
      </div>
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
}

function ReviewOverlays({ matches, focusedIdx }: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const focused = matches.detections[focusedIdx];
  const color: string = focused ? TAG_COLORS[focused.det_type] : "#E0E0E0";

  return (
    <>
      {matches.gt_boxes.map((b: Box, i: number) =>
        renderBox(b, i, false, lw, i === focused?.gt_idx && focused?.gt_type === "box"),
      )}
      {matches.gt_polygons.map((p: PolygonShape, i: number) =>
        renderPolygon(p, `g${i}`, lw, i === focused?.gt_idx && focused?.gt_type === "polygon"),
      )}
      {focused &&
        focused.pred_type === "box" &&
        focused.pred_idx !== null &&
        matches.pred_boxes[focused.pred_idx] &&
        renderPredBox(matches.pred_boxes[focused.pred_idx], "pred-focus", lw, color)}
      {focused &&
        focused.pred_type === "polygon" &&
        focused.pred_idx !== null &&
        matches.pred_polygons[focused.pred_idx] &&
        renderPredPoly(matches.pred_polygons[focused.pred_idx], "pp-focus", lw, color)}
    </>
  );
}

function renderBox(b: Box, i: number, _dashed: boolean, lw: number, focused: boolean) {
  return (
    <Rect
      key={`gt-${i}`}
      x={b.x1}
      y={b.y1}
      width={b.x2 - b.x1}
      height={b.y2 - b.y1}
      stroke={focused ? "#FFD700" : "#4CAF50"}
      strokeWidth={(focused ? 3 : 2) * lw}
    />
  );
}

function renderPolygon(
  p: PolygonShape,
  key: string,
  lw: number,
  focused: boolean,
) {
  return (
    <Line
      key={key}
      points={p.points.flat()}
      closed
      stroke={focused ? "#FFD700" : "#4CAF50"}
      strokeWidth={(focused ? 3 : 2) * lw}
    />
  );
}

function renderPredBox(b: PredBox, key: string, lw: number, color: string) {
  return (
    <Rect
      key={key}
      x={b.x1}
      y={b.y1}
      width={b.x2 - b.x1}
      height={b.y2 - b.y1}
      stroke={color}
      strokeWidth={3 * lw}
      dash={[8 * lw, 4 * lw]}
    />
  );
}

function renderPredPoly(p: PredPolygon, key: string, lw: number, color: string) {
  return (
    <Line
      key={key}
      points={p.points.flat()}
      closed
      stroke={color}
      strokeWidth={3 * lw}
      dash={[8 * lw, 4 * lw]}
    />
  );
}
