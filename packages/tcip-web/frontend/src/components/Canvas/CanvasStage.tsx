/**
 * Shared Konva Stage wrapper with pan + zoom state managed in the store.
 * AnnotationCanvas and ReviewCanvas both render inside this host so the
 * view state (scale, offset_x, offset_y) stays in sync across tabs.
 */

import { useEffect, useRef, useState } from "react";
import { Image as KonvaImage, Layer, Stage } from "react-konva";
import Konva from "konva";

import { MAX_SCALE, MIN_SCALE } from "@/components/Canvas/zoom";
import { useStore } from "@/store";

// Above this magnification the display-capped bitmap is visibly upscaled; fetch the
// full-native-resolution variant so pixel-level inspection isn't softened by the cap.
const HIRES_SCALE = 1.5;

export interface CanvasStageProps {
  imageUrl: string | null;
  imgWidth: number;
  imgHeight: number;
  /** Optional full-resolution variant, loaded lazily once zoomed past HIRES_SCALE. */
  hiResImageUrl?: string | null;
  /** Static shapes — rendered in the content layer (below the overlay). */
  children?: React.ReactNode;
  /** Cursor-following / transient shapes — rendered in a separate top layer so they
   *  can redraw on every mouse move without re-compositing the image or the shapes. */
  overlay?: React.ReactNode;
  onStageRef?: (stage: Konva.Stage | null) => void;
  onPixelClick?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelMove?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelDown?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelUp?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelDoubleClick?: (x: number, y: number, ev: Konva.KonvaEventObject<MouseEvent>) => void;
  onPixelContextMenu?: (x: number, y: number, ev: Konva.KonvaEventObject<PointerEvent>) => void;
}

export function CanvasStage(props: CanvasStageProps) {
  const wrapper = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  // Start at 0 so the one-shot fit waits for a REAL measurement (below) instead of fitting
  // to a placeholder size — a stale-dims fit left the image mis-scaled/off-screen on first open.
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [imgError, setImgError] = useState(false);
  const [hiResImg, setHiResImg] = useState<HTMLImageElement | null>(null);
  const hiResLoadedFor = useRef<string | null>(null);

  // Track wrapper size — measure synchronously on mount too, so the first fit uses the real
  // canvas size immediately rather than waiting a frame for the ResizeObserver.
  useEffect(() => {
    if (!wrapper.current) return;
    const measure = () => {
      const r = wrapper.current!.getBoundingClientRect();
      setDims({ w: Math.max(1, Math.floor(r.width)), h: Math.max(1, Math.floor(r.height)) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(wrapper.current);
    return () => ro.disconnect();
  }, []);

  // Load image
  useEffect(() => {
    if (!props.imageUrl) {
      setImg(null);
      setImgError(false);
      return;
    }
    // Drop the previous bitmap immediately — otherwise the new image's labels render
    // over the old photo for the seconds the fetch takes (misregistered symbology).
    setImg(null);
    setImgError(false);
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => {
      setImg(el);
      setImgError(false);
    };
    // Distinguish a failed load (missing / access-denied / server error) from an
    // empty canvas — otherwise overlays float on a blank stage with no explanation.
    el.onerror = () => {
      setImg(null);
      setImgError(true);
    };
    el.src = props.imageUrl;
    return () => {
      el.onload = null;
      el.onerror = null;
      el.src = ""; // cancel the abandoned download; rapid flips otherwise queue every skip
    };
  }, [props.imageUrl]);

  // A new source image invalidates any full-res variant held for the previous one.
  useEffect(() => {
    setHiResImg(null);
    hiResLoadedFor.current = null;
  }, [props.hiResImageUrl]);

  // Fetch the full-res variant once zoomed in far enough to see the difference. It swaps
  // in when ready (the capped bitmap covers the wait) and then stays — one big texture
  // per image, dropped on image change; loaded at most once per image.
  const wantHiRes = view.scale > HIRES_SCALE && !!props.hiResImageUrl;
  useEffect(() => {
    const url = props.hiResImageUrl;
    if (!wantHiRes || !url || hiResLoadedFor.current === url) return;
    let cancelled = false;
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => {
      if (cancelled) return;
      hiResLoadedFor.current = url; // guard set only on success, so a cancel can retry
      setHiResImg(el);
    };
    el.src = url;
    return () => {
      cancelled = true;
      el.onload = null;
      el.src = "";
    };
  }, [wantHiRes, props.hiResImageUrl]);

  // Fit the image to the canvas ONCE per image — not on every container resize. Refitting
  // on resize reset the user's zoom/pan, and when a reflow briefly reported a near-zero
  // height (e.g. the Review filter shelf expanding) it collapsed the image to sub-pixel
  // scale so it appeared to vanish. The key omits dims so a later resize can't re-fit;
  // it's keyed on image identity + native size so a genuine image change still fits.
  const didFit = useRef<string | null>(null);
  useEffect(() => {
    if (!img || !props.imgWidth || !props.imgHeight) return;
    const key = `${props.imageUrl}:${props.imgWidth}x${props.imgHeight}`;
    if (didFit.current === key) return;
    if (dims.w <= 1 || dims.h <= 1) return; // wait for a real measurement before fitting
    didFit.current = key;
    const scale = Math.min(dims.w / props.imgWidth, dims.h / props.imgHeight);
    setView({
      scale,
      offset_x: (dims.w - props.imgWidth * scale) / 2,
      offset_y: (dims.h - props.imgHeight * scale) / 2,
    });
  }, [img, props.imageUrl, props.imgWidth, props.imgHeight, dims, setView]);

  // Expose stage ref
  const { onStageRef } = props;
  useEffect(() => {
    onStageRef?.(stageRef.current);
    return () => onStageRef?.(null);
  }, [onStageRef]);

  // Handlers read the live store view (not the render-closure `view`): trackpads emit
  // event bursts faster than the 20MP canvas re-renders, and stale reads drop deltas.
  const liveView = () => useStore.getState().gui.view;

  const toPixel = (sx: number, sy: number): [number, number] => {
    const v = liveView();
    const s = v.scale || 1;
    return [(sx - v.offset_x) / s, (sy - v.offset_y) / s];
  };

  // Line-mode wheel deltas (legacy mice / some drivers) arrive in lines, not pixels.
  const normDelta = (d: number, mode: number) => (mode === 1 ? d * 16 : d);

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const v = liveView();
    const dx = normDelta(e.evt.deltaX, e.evt.deltaMode);
    const dy = normDelta(e.evt.deltaY, e.evt.deltaMode);
    if (!e.evt.ctrlKey) {
      if (e.evt.shiftKey) {
        // Shift+scroll: horizontal pan. Chromium pre-swaps the delta into deltaX.
        setView({ ...v, offset_x: v.offset_x - (dx || dy) });
      } else {
        // Plain scroll pans both axes — two-finger trackpad panning in any direction.
        setView({ ...v, offset_x: v.offset_x - dx, offset_y: v.offset_y - dy });
      }
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const s = v.scale || 1;
    const ix = (pointer.x - v.offset_x) / s;
    const iy = (pointer.y - v.offset_y) / s;
    // Ctrl+wheel (and precision-touchpad pinch, which browsers deliver as ctrl+wheel):
    // continuous magnitude-proportional zoom about the pointer. A touchpad pinch arrives as
    // many small deltas, a mouse notch as one large (~100) delta — so give the small deltas a
    // higher gain (pinch-zoom felt sluggish) while the mouse wheel keeps its tuned feel.
    const zoomGain = Math.abs(dy) < 40 ? 0.005 : 0.002;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, v.scale * Math.exp(-dy * zoomGain)));
    if (newScale === v.scale) return;
    setView({
      scale: newScale,
      offset_x: pointer.x - ix * newScale,
      offset_y: pointer.y - iy * newScale,
    });
  };

  // Middle-drag pans anywhere; space-hold turns left-drag into a pan (the touchpad answer).
  const panning = useRef<{ x: number; y: number; buttonsBit: number } | null>(null);
  const panMoved = useRef(false);
  const [spaceHeld, setSpaceHeld] = useState(false);

  useEffect(() => {
    const isTyping = () => {
      const el = document.activeElement as HTMLElement | null;
      return (
        !!el &&
        (el.tagName === "INPUT" ||
          el.tagName === "TEXTAREA" ||
          el.tagName === "SELECT" ||
          el.isContentEditable)
      );
    };
    const down = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat || isTyping()) return;
      e.preventDefault();
      setSpaceHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceHeld(false);
    };
    const reset = () => setSpaceHeld(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", reset);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", reset);
    };
  }, []);

  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.evt.button === 1 || (spaceHeld && e.evt.button === 0)) {
      // buttonsBit lets the move handler detect a release that happened off-canvas.
      panning.current = {
        x: e.evt.clientX,
        y: e.evt.clientY,
        buttonsBit: e.evt.button === 1 ? 4 : 1,
      };
      panMoved.current = false;
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
      if ((e.evt.buttons & panning.current.buttonsBit) === 0) {
        panning.current = null; // button released outside the canvas — end the pan
      } else {
        const dx = e.evt.clientX - panning.current.x;
        const dy = e.evt.clientY - panning.current.y;
        panning.current = { ...panning.current, x: e.evt.clientX, y: e.evt.clientY };
        if (dx || dy) panMoved.current = true;
        const v = liveView();
        setView({ ...v, offset_x: v.offset_x + dx, offset_y: v.offset_y + dy });
        return;
      }
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
    // A space-pan release must not place a vertex / start a shape under the cursor.
    if (spaceHeld || panMoved.current) {
      panMoved.current = false;
      return;
    }
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
    <div
      ref={wrapper}
      data-canvas-host
      className="relative flex-1 bg-tcip-canvas overflow-hidden"
      style={spaceHeld ? { cursor: "grab" } : undefined}
    >
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
        {/* Three layers share the pan/zoom transform but redraw independently. All are
            listening={false} — every interaction goes through the Stage-level pixel
            handlers (getPointerPosition + toPixel), never per-shape Konva hit-testing —
            so Konva skips building a hit graph for hundreds/thousands of shapes.
            Isolating the image means a cursor move (overlay only) never re-composites
            the 20MP bitmap. */}
        <Layer
          listening={false}
          x={view.offset_x}
          y={view.offset_y}
          scaleX={view.scale}
          scaleY={view.scale}
        >
          {img || hiResImg ? (
            <KonvaImage
              // Use the full-res variant only while zoomed in past the threshold; when zoomed
              // back out, revert to the capped bitmap (the huge texture drawn small could drop
              // out on some GPUs, making the image vanish while the overlays stayed).
              image={(wantHiRes && hiResImg ? hiResImg : img) ?? undefined}
              x={0}
              y={0}
              width={props.imgWidth}
              height={props.imgHeight}
            />
          ) : null}
        </Layer>
        <Layer
          listening={false}
          x={view.offset_x}
          y={view.offset_y}
          scaleX={view.scale}
          scaleY={view.scale}
        >
          {props.children}
        </Layer>
        {props.overlay != null && (
          <Layer
            listening={false}
            x={view.offset_x}
            y={view.offset_y}
            scaleX={view.scale}
            scaleY={view.scale}
          >
            {props.overlay}
          </Layer>
        )}
      </Stage>
      {imgError && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-4">
          <div className="max-w-sm rounded-md border border-tcip-fp/50 bg-tcip-panel/95 px-4 py-3 text-center text-[12px] text-tcip-fp">
            Could not load this image — it may be missing, or access was denied (check the path and,
            on a locked-down server, TCIP_IMAGE_ROOTS).
          </div>
        </div>
      )}
    </div>
  );
}
