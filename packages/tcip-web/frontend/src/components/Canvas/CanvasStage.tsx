/**
 * Shared Konva Stage wrapper with pan + zoom state managed in the store.
 * AnnotationCanvas and ReviewCanvas both render inside this host so the
 * view state (scale, offset_x, offset_y) stays in sync across tabs.
 */

import { useEffect, useRef, useState } from "react";
import { Image as KonvaImage, Layer, Stage } from "react-konva";
import Konva from "konva";

import { useStore } from "@/store";

export interface CanvasStageProps {
  imageUrl: string | null;
  imgWidth: number;
  imgHeight: number;
  children?: React.ReactNode;
  onStageRef?: (stage: Konva.Stage | null) => void;
  onPixelClick?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelMove?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelDown?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelUp?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelDoubleClick?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelContextMenu?: (x: number, y: number, ev: Konva.KonvaEventObject<PointerEvent>) => void;
}

// Discrete zoom levels mirror yolo-annotator (5% .. 1000%, 20 stops)
export const ZOOM_LEVELS = [
  0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67,
  0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0,
];

function nearestZoomIndex(scale: number): number {
  let bestIdx = 0;
  let bestDiff = Math.abs(ZOOM_LEVELS[0] - scale);
  for (let i = 1; i < ZOOM_LEVELS.length; i++) {
    const diff = Math.abs(ZOOM_LEVELS[i] - scale);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIdx = i;
    }
  }
  return bestIdx;
}

const MIN_SCALE = ZOOM_LEVELS[0];
const MAX_SCALE = ZOOM_LEVELS[ZOOM_LEVELS.length - 1];

export function CanvasStage(props: CanvasStageProps) {
  const wrapper = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  // Track wrapper size
  useEffect(() => {
    if (!wrapper.current) return;
    const ro = new ResizeObserver(() => {
      const r = wrapper.current!.getBoundingClientRect();
      setDims({ w: Math.max(1, Math.floor(r.width)), h: Math.max(1, Math.floor(r.height)) });
    });
    ro.observe(wrapper.current);
    return () => ro.disconnect();
  }, []);

  // Load image
  useEffect(() => {
    if (!props.imageUrl) {
      setImg(null);
      return;
    }
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => setImg(el);
    el.onerror = () => setImg(null);
    el.src = props.imageUrl;
    return () => {
      el.onload = null;
      el.onerror = null;
    };
  }, [props.imageUrl]);

  // Fit image to canvas the first time we know both dims + image size
  const didFit = useRef<string | null>(null);
  useEffect(() => {
    if (!img || !props.imgWidth || !props.imgHeight) return;
    const key = `${props.imageUrl}:${dims.w}x${dims.h}`;
    if (didFit.current === key) return;
    didFit.current = key;
    const sx = dims.w / props.imgWidth;
    const sy = dims.h / props.imgHeight;
    const scale = Math.min(sx, sy);
    setView({
      scale,
      offset_x: (dims.w - props.imgWidth * scale) / 2,
      offset_y: (dims.h - props.imgHeight * scale) / 2,
    });
  }, [img, props.imageUrl, props.imgWidth, props.imgHeight, dims, setView]);

  // Expose stage ref
  useEffect(() => {
    props.onStageRef?.(stageRef.current);
    return () => props.onStageRef?.(null);
  }, [props.onStageRef]);

  const toPixel = (sx: number, sy: number): [number, number] => {
    const s = view.scale || 1;
    return [(sx - view.offset_x) / s, (sy - view.offset_y) / s];
  };

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    if (!e.evt.ctrlKey) {
      // Plain scroll: pan vertically (Shift+scroll: pan horizontally)
      const delta = -e.evt.deltaY;
      if (e.evt.shiftKey) {
        setView({ ...view, offset_x: view.offset_x + delta });
      } else {
        setView({ ...view, offset_y: view.offset_y + delta });
      }
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const [ix, iy] = toPixel(pointer.x, pointer.y);
    // Snap to discrete zoom levels (yolo-annotator parity)
    const direction = e.evt.deltaY < 0 ? 1 : -1;
    const currentIdx = nearestZoomIndex(view.scale);
    const newIdx = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, currentIdx + direction));
    if (newIdx === currentIdx) return;
    const newScale = ZOOM_LEVELS[newIdx];
    setView({
      scale: Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale)),
      offset_x: pointer.x - ix * newScale,
      offset_y: pointer.y - iy * newScale,
    });
  };

  // Middle-click + drag = pan
  const panning = useRef<{ x: number; y: number } | null>(null);

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.evt.button === 1) {
      panning.current = { x: e.evt.clientX, y: e.evt.clientY };
      e.evt.preventDefault();
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelDown?.(ix, iy, e);
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (panning.current) {
      const dx = e.evt.clientX - panning.current.x;
      const dy = e.evt.clientY - panning.current.y;
      panning.current = { x: e.evt.clientX, y: e.evt.clientY };
      setView({ ...view, offset_x: view.offset_x + dx, offset_y: view.offset_y + dy });
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelMove?.(ix, iy, e);
  };

  const handleMouseUp = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (panning.current) {
      panning.current = null;
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelUp?.(ix, iy, e);
  };

  const handleClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelClick?.(ix, iy, e);
  };

  const handleDoubleClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelDoubleClick?.(ix, iy, e);
  };

  const handleContextMenu = (e: Konva.KonvaEventObject<PointerEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const p = stage.getPointerPosition();
    if (!p) return;
    const [ix, iy] = toPixel(p.x, p.y);
    props.onPixelContextMenu?.(ix, iy, e);
  };

  return (
    <div ref={wrapper} className="relative flex-1 bg-tcip-canvas overflow-hidden">
      <Stage
        ref={(s) => {
          stageRef.current = s;
        }}
        width={dims.w}
        height={dims.h}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        onDblClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
      >
        <Layer x={view.offset_x} y={view.offset_y} scaleX={view.scale} scaleY={view.scale}>
          {img ? (
            <KonvaImage image={img} x={0} y={0} width={props.imgWidth} height={props.imgHeight} />
          ) : null}
          {props.children}
        </Layer>
      </Stage>
    </div>
  );
}
