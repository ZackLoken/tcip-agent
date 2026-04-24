import { useEffect, useRef, useState } from "react";
import { Circle, Line, Rect } from "react-konva";
import Konva from "konva";

import { api } from "@/api/client";
import { CanvasStage } from "@/components/Canvas/CanvasStage";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useStore } from "@/store";
import type { Box, DatasetSelection, PredictionReference } from "@/store/types";

const CLASS_COLORS: Record<number, string> = {
  0: "#4CAF50",
  1: "#E6976B",
  2: "#00BFFF",
  3: "#FFD700",
};

function colorFor(cid: number): string {
  return CLASS_COLORS[cid] ?? "#E0E0E0";
}

function currentImagePath(dataset: DatasetSelection): string | null {
  if (!dataset.dataset_root || !dataset.date) return null;
  const name = dataset.image_list[dataset.current_image_index];
  if (!name) return null;
  return `${dataset.dataset_root}/images/${dataset.date}/${name}`;
}

export function AnnotateTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  const view = useStore((s) => s.gui.view);
  const mode = useStore((s) => s.gui.mode);
  const activeClass = useStore((s) => s.gui.active_class);
  const predRef = useStore((s) => s.gui.pred_reference);

  const canvas = useStore((s) => s.canvas);
  const loadLabels = useStore((s) => s.loadLabelsIntoCanvas);
  const addBox = useStore((s) => s.addBox);
  const deleteBox = useStore((s) => s.deleteBox);
  const deletePolygon = useStore((s) => s.deletePolygon);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const setCurrentPolygon = useStore((s) => s.setCurrentPolygon);
  const commitCurrentPolygon = useStore((s) => s.commitCurrentPolygon);
  const selectPolygon = useStore((s) => s.selectPolygon);
  const markClean = useStore((s) => s.markClean);
  const setPredReference = useStore((s) => s.setPredReference);

  const [drawing, setDrawing] = useState<Box | null>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);

  const imgPath = currentImagePath(dataset);

  // Load labels for current image
  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!imgPath) return;
      const name = dataset.image_list[dataset.current_image_index];
      const stem = name.replace(/\.[^.]+$/, "");
      const det = dataset.annotations_detect_dir
        ? `${dataset.annotations_detect_dir}/${stem}.txt`
        : null;
      const seg = dataset.annotations_segment_dir
        ? `${dataset.annotations_segment_dir}/${stem}.txt`
        : null;
      try {
        const labels = await api.annotate.load(imgPath, det, seg);
        if (!cancelled) loadLabels(labels);
      } catch {
        if (!cancelled)
          loadLabels({
            image_path: imgPath,
            img_width: 0,
            img_height: 0,
            boxes: [],
            polygons: [],
          });
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [imgPath, dataset, loadLabels]);

  async function save() {
    if (!imgPath) return;
    const name = dataset.image_list[dataset.current_image_index];
    const stem = name.replace(/\.[^.]+$/, "");
    const det = dataset.annotations_detect_dir
      ? `${dataset.annotations_detect_dir}/${stem}.txt`
      : null;
    const seg = dataset.annotations_segment_dir
      ? `${dataset.annotations_segment_dir}/${stem}.txt`
      : null;
    await api.annotate.save({
      image_path: imgPath,
      detect_path: det,
      segment_path: seg,
      boxes: canvas.boxes,
      polygons: canvas.polygons,
    });
    markClean();
  }

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

  useKeyboardShortcuts([
    { keys: "ctrl+z", action: () => undo() },
    { keys: "ctrl+shift+z", action: () => redo() },
    { keys: "ctrl+s", action: () => save() },
    { keys: "m", action: () => useStore.getState().setMode(mode === "box" ? "polygon" : "box") },
    { keys: "delete", action: () => {
        if (canvas.selectedPolygonIdx !== null) deletePolygon(canvas.selectedPolygonIdx);
    }},
    { keys: "escape", action: () => {
        setCurrentPolygon([]);
        setDrawing(null);
        selectPolygon(null);
    }},
    { keys: "arrowleft", action: () => stepImage(-1) },
    { keys: "arrowright", action: () => stepImage(1) },
    { keys: "enter", action: () => {
      if (mode === "polygon" && canvas.currentPolygon.length >= 3) commitCurrentPolygon();
    }},
  ]);

  const onDown = (ix: number, iy: number) => {
    if (mode === "box") {
      setDrawing({
        x1: ix,
        y1: iy,
        x2: ix,
        y2: iy,
        class_id: activeClass,
      });
    }
  };

  const onMove = (ix: number, iy: number) => {
    setCursor([ix, iy]);
    if (drawing) setDrawing({ ...drawing, x2: ix, y2: iy });
  };

  const onUp = (ix: number, iy: number) => {
    if (mode === "box" && drawing) {
      const box: Box = {
        x1: Math.min(drawing.x1, ix),
        y1: Math.min(drawing.y1, iy),
        x2: Math.max(drawing.x1, ix),
        y2: Math.max(drawing.y1, iy),
        class_id: activeClass,
      };
      if (box.x2 - box.x1 > 1 && box.y2 - box.y1 > 1) addBox(box);
      setDrawing(null);
    }
  };

  const onClick = (ix: number, iy: number, ev: Konva.KonvaEventObject<MouseEvent>) => {
    if (mode === "polygon" && !ev.evt.ctrlKey) {
      setCurrentPolygon([...canvas.currentPolygon, [ix, iy]]);
      return;
    }
    // Selection: click a polygon vertex area
    if (mode === "polygon") {
      // Ctrl+click on polygon = select
      const hit = canvas.polygons.findIndex((p) =>
        pointInPolygon([ix, iy], p.points),
      );
      selectPolygon(hit >= 0 ? hit : null);
    }
  };

  const scaleLineW = 1 / (view.scale || 1);
  const imageUrl = imgPath ? api.images.url(imgPath) : null;

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-tcip-border bg-tcip-panel text-[11px]">
        <button className="tcip-btn" onClick={() => stepImage(-1)} disabled={!dataset.image_list.length}>
          ◀ Prev
        </button>
        <button className="tcip-btn" onClick={() => stepImage(1)} disabled={!dataset.image_list.length}>
          Next ▶
        </button>
        <span className="mx-1 text-tcip-muted">|</span>
        <button className="tcip-btn" onClick={() => undo()} title="Ctrl+Z">
          ↶ Undo
        </button>
        <button className="tcip-btn" onClick={() => redo()} title="Ctrl+Shift+Z">
          ↷ Redo
        </button>
        {mode === "polygon" && (
          <button
            className="tcip-btn"
            onClick={() => commitCurrentPolygon()}
            disabled={canvas.currentPolygon.length < 3}
            title="Enter"
          >
            ✓ Close polygon
          </button>
        )}
        <span className="flex-1" />
        <span className="text-tcip-muted">
          Boxes: {canvas.boxes.length} · Polys: {canvas.polygons.length}
        </span>
        <button
          className={canvas.dirty ? "tcip-btn-primary" : "tcip-btn"}
          onClick={save}
          disabled={!imgPath}
          title="Ctrl+S"
        >
          {canvas.dirty ? "💾 Save" : "Saved"}
        </button>
      </div>

      <CanvasStage
        imageUrl={imageUrl}
        imgWidth={canvas.imgWidth}
        imgHeight={canvas.imgHeight}
        onStageRef={(s) => (stageRef.current = s)}
        onPixelDown={onDown}
        onPixelMove={onMove}
        onPixelUp={onUp}
        onPixelClick={onClick}
      >
        {/* Existing boxes */}
        {canvas.boxes.map((b, i) => (
          <Rect
            key={`box-${i}`}
            x={b.x1}
            y={b.y1}
            width={b.x2 - b.x1}
            height={b.y2 - b.y1}
            stroke={colorFor(b.class_id)}
            strokeWidth={2 * scaleLineW}
            onDblClick={() => deleteBox(i)}
          />
        ))}

        {/* Existing polygons */}
        {canvas.polygons.map((p, i) => {
          const flat = p.points.flat();
          const selected = canvas.selectedPolygonIdx === i;
          return (
            <Line
              key={`poly-${i}`}
              points={flat}
              closed
              stroke={selected ? "#FFD700" : colorFor(p.class_id)}
              strokeWidth={(selected ? 3 : 2) * scaleLineW}
              onDblClick={() => deletePolygon(i)}
            />
          );
        })}

        {/* In-progress polygon */}
        {mode === "polygon" && canvas.currentPolygon.length > 0 && (
          <Line
            points={canvas.currentPolygon.flat()}
            stroke={colorFor(activeClass)}
            strokeWidth={2 * scaleLineW}
            dash={[6 * scaleLineW, 4 * scaleLineW]}
          />
        )}
        {mode === "polygon" && canvas.currentPolygon.length > 0 && cursor && (
          <Line
            points={[
              ...canvas.currentPolygon[canvas.currentPolygon.length - 1],
              ...cursor,
            ]}
            stroke={colorFor(activeClass)}
            strokeWidth={scaleLineW}
            dash={[4 * scaleLineW, 4 * scaleLineW]}
          />
        )}
        {mode === "polygon" && canvas.currentPolygon.map(([x, y], i) => (
          <Circle
            key={`pv-${i}`}
            x={x}
            y={y}
            radius={3 * scaleLineW}
            fill={colorFor(activeClass)}
          />
        ))}

        {/* Draft box preview */}
        {drawing && (
          <Rect
            x={Math.min(drawing.x1, drawing.x2)}
            y={Math.min(drawing.y1, drawing.y2)}
            width={Math.abs(drawing.x2 - drawing.x1)}
            height={Math.abs(drawing.y2 - drawing.y1)}
            stroke={colorFor(activeClass)}
            strokeWidth={2 * scaleLineW}
            dash={[6 * scaleLineW, 4 * scaleLineW]}
          />
        )}

        {/* Prediction reference (dashed blue) */}
        {predRef && <PredReferenceOverlay pred={predRef} lineW={scaleLineW} />}
      </CanvasStage>
    </div>
  );
}

function PredReferenceOverlay({ pred, lineW }: { pred: PredictionReference; lineW: number }) {
  if (pred.type === "box") {
    const [x1, y1, x2, y2] = pred.coords as number[];
    return (
      <Rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        stroke="#00BFFF"
        strokeWidth={2 * lineW}
        dash={[8 * lineW, 4 * lineW]}
      />
    );
  }
  return (
    <Line
      points={(pred.coords as number[][]).flat()}
      closed
      stroke="#00BFFF"
      strokeWidth={2 * lineW}
      dash={[8 * lineW, 4 * lineW]}
    />
  );
}

function pointInPolygon(pt: [number, number], poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}
