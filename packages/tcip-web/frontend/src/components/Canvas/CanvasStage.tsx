/**
 * Shared Konva Stage wrapper with pan + zoom state managed in the store.
 * AnnotationCanvas and ReviewCanvas both render inside this host so the
 * view state (scale, offset_x, offset_y) stays in sync across tabs.
 */

import { useEffect, useRef, useState } from "react";
import { Image as KonvaImage, Layer, Stage } from "react-konva";
import Konva from "konva";

import { MAX_SCALE, MIN_SCALE } from "@/components/Canvas/zoom";
import { useOverviewBuild } from "@/hooks/useOverviewBuild";
import { loadImage, type LoadedImage } from "@/lib/imageLoader";
import { clampView } from "@/lib/viewGeometry";
import { useStore } from "@/store";
import type { ViewState } from "@/store/types";

/** One region serve drawn above the base image, positioned in image coords. */
export interface CanvasRegion {
  /** Stable identity (the coverage cell name); the bitmap cache is keyed on it. */
  key: string;
  url: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Called with the serve facts once this region's URL loads (served-at-native marking). */
  onLoaded?: (facts: LoadedImage) => void;
}

export interface CanvasStageProps {
  imageUrl: string | null;
  /** The image's own path, which the server needs to build the overviews a refused load names.
   *  Omitted, a refused load stays a plain load error. */
  imagePath?: string | null;
  imgWidth: number;
  imgHeight: number;
  /** Auto-fit the image to the canvas once per image (default true). The Review tab sets this
   *  false so its zoom-to-detection view is not overridden by the fit. */
  autoFit?: boolean;
  /** Cell-aligned region serves rendered above the base image while it is loading and after: a
   *  region unmounts when dropped from this list; one whose URL changes keeps its last-loaded
   *  bitmap on screen until the replacement finishes loading. */
  regions?: CanvasRegion[];
  /** The base serve's read facts: null while a load is in flight, then the loader's result. */
  onBaseFacts?: (facts: LoadedImage | null) => void;
  /** Static shapes, rendered in the content layer (below the overlay). */
  children?: React.ReactNode;
  /** Cursor-following / transient shapes, rendered in a separate top layer so they
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
  // Start at 0 so the one-shot fit waits for a real measurement (below) instead of fitting
  // to a placeholder size: a stale-dims fit left the image mis-scaled/off-screen on first open.
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const view = useStore((s) => s.gui.view);
  const setView = useStore((s) => s.setView);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [imgError, setImgError] = useState(false);
  const [baseError, setBaseError] = useState<string | null>(null);
  const [baseHeaderParseError, setBaseHeaderParseError] = useState<string | null>(null);
  const onBaseFactsRef = useRef(props.onBaseFacts);
  onBaseFactsRef.current = props.onBaseFacts;
  // A load the server refused for want of overviews is a wait, not a failure: the build runs and
  // the image is requested again once the pyramid exists.
  const overview = useOverviewBuild(props.imageUrl, props.imagePath ?? null, baseError);

  // Track wrapper size: measure synchronously on mount too, so the first fit uses the real
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

  // Load the base image through the shared loader so the serve facts (served size, stretch
  // provenance, refusal condition) arrive with the bitmap instead of needing a second request.
  useEffect(() => {
    if (!props.imageUrl) {
      setImg(null);
      setImgError(false);
      setBaseError(null);
      setBaseHeaderParseError(null);
      onBaseFactsRef.current?.(null);
      return;
    }
    // Drop the previous bitmap immediately, otherwise the new image's labels render
    // over the old photo for the seconds the fetch takes (misregistered symbology).
    setImg(null);
    setImgError(false);
    setBaseError(null);
    setBaseHeaderParseError(null);
    onBaseFactsRef.current?.(null);
    const ac = new AbortController();
    void loadImage(props.imageUrl, { signal: ac.signal }).then((res) => {
      if (res.aborted) return;
      setImg(res.ok ? res.image : null);
      // Distinguish a failed load (missing / access-denied / server error) from an
      // empty canvas, otherwise overlays float on a blank stage with no explanation.
      setImgError(!res.ok);
      setBaseError(res.imageError);
      setBaseHeaderParseError(res.headerParseError);
      onBaseFactsRef.current?.(res);
    });
    return () => ac.abort(); // cancel the abandoned download; rapid flips otherwise queue every skip
  }, [props.imageUrl, overview.reloadToken]);

  // Region bitmaps by key. A key dropped from the prop unmounts (and aborts an in-flight
  // load); a key whose URL changed keeps the last-loaded bitmap until the replacement lands.
  const [regionBitmaps, setRegionBitmaps] = useState<Map<string, HTMLImageElement>>(new Map());
  const regionLoadsRef = useRef<Map<string, { url: string; ac: AbortController }>>(new Map());
  const regionUrlsRef = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    const regions = props.regions ?? [];
    const keys = new Set(regions.map((r) => r.key));
    for (const [key, load] of regionLoadsRef.current) {
      if (!keys.has(key)) {
        load.ac.abort();
        regionLoadsRef.current.delete(key);
      }
    }
    setRegionBitmaps((prev) => {
      let next: Map<string, HTMLImageElement> | null = null;
      for (const key of prev.keys()) {
        if (!keys.has(key)) {
          next = next ?? new Map(prev);
          next.delete(key);
          regionUrlsRef.current.delete(key);
        }
      }
      return next ?? prev;
    });
    for (const region of regions) {
      if (regionUrlsRef.current.get(region.key) === region.url) continue;
      const inflight = regionLoadsRef.current.get(region.key);
      if (inflight?.url === region.url) continue;
      inflight?.ac.abort();
      const ac = new AbortController();
      regionLoadsRef.current.set(region.key, { url: region.url, ac });
      void loadImage(region.url, { signal: ac.signal }).then((res) => {
        if (res.aborted) return;
        if (regionLoadsRef.current.get(region.key)?.url !== region.url) return;
        regionLoadsRef.current.delete(region.key);
        region.onLoaded?.(res);
        if (res.ok && res.image) {
          regionUrlsRef.current.set(region.key, region.url);
          setRegionBitmaps((prev) => new Map(prev).set(region.key, res.image!));
        }
      });
    }
  }, [props.regions]);
  useEffect(() => {
    const loads = regionLoadsRef.current;
    return () => {
      for (const load of loads.values()) load.ac.abort();
      loads.clear();
    };
  }, []);

  // Fit the image to the canvas once per image, not on every container resize. Refitting
  // on resize reset the user's zoom/pan, and when a reflow briefly reported a near-zero
  // height (e.g. the Review filter shelf expanding) it collapsed the image to sub-pixel
  // scale so it appeared to vanish. The key omits dims so a later resize can't re-fit;
  // it's keyed on image identity + native size so a genuine image change still fits.
  const didFit = useRef<string | null>(null);
  useEffect(() => {
    if (props.autoFit === false) return; // consumer controls the view (e.g. Review zoom-to-detection)
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
  }, [img, props.imageUrl, props.imgWidth, props.imgHeight, props.autoFit, dims, setView]);

  // Expose stage ref
  const { onStageRef } = props;
  useEffect(() => {
    onStageRef?.(stageRef.current);
    return () => onStageRef?.(null);
  }, [onStageRef]);

  // Pan/zoom writes coalesce through one requestAnimationFrame per frame (mirrors
  // AnnotateTab's onMove): native wheel/mousemove bursts fire faster than 60/s, and an
  // un-batched setView re-rendered both tabs per event. The pending view holds the
  // accumulated target so a burst reads its own in-flight math, not the stale store,
  // keeping the zoom-about-pointer / pan deltas exact; only the commit is deferred.
  const pendingViewRef = useRef<ViewState | null>(null);
  const pendingBaseRef = useRef<ViewState | null>(null);
  const viewRafRef = useRef<number | null>(null);
  // Every interactive write (wheel pan, wheel zoom, drag pan) passes through here, so the pan
  // clamp lives at this one choke point; programmatic setView callers stay unclamped.
  const scheduleView = (raw: ViewState) => {
    const v = clampView(raw, dims, props.imgWidth, props.imgHeight);
    if (pendingViewRef.current == null) pendingBaseRef.current = useStore.getState().gui.view;
    pendingViewRef.current = v;
    if (viewRafRef.current != null) return;
    viewRafRef.current = requestAnimationFrame(() => {
      viewRafRef.current = null;
      const pending = pendingViewRef.current;
      const base = pendingBaseRef.current;
      pendingViewRef.current = null;
      pendingBaseRef.current = null;
      // An external setView (auto-fit, zoom-to-detection) landed mid-burst: it wins, the
      // queued math was computed against the pre-write view and would silently clobber it.
      if (pending && base === useStore.getState().gui.view) setView(pending);
    });
  };
  useEffect(() => {
    return () => {
      if (viewRafRef.current != null) cancelAnimationFrame(viewRafRef.current);
    };
  }, []);

  // Handlers read the live view: the pending in-flight target if a batched write is
  // queued, else the store. A stale read (trackpads burst faster than the canvas
  // re-renders) would drop accumulated deltas mid-burst.
  const liveView = () => pendingViewRef.current ?? useStore.getState().gui.view;

  const toPixel = (sx: number, sy: number): [number, number] => {
    const v = liveView();
    const s = v.scale || 1;
    return [(sx - v.offset_x) / s, (sy - v.offset_y) / s];
  };

  // Line-mode wheel deltas (some mice/drivers) arrive in lines, not pixels.
  const normDelta = (d: number, mode: number) => (mode === 1 ? d * 16 : d);

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const v = liveView();
    const dx = normDelta(e.evt.deltaX, e.evt.deltaMode);
    const dy = normDelta(e.evt.deltaY, e.evt.deltaMode);
    if (!e.evt.ctrlKey) {
      if (e.evt.shiftKey) {
        // Shift+scroll: horizontal pan. Chromium pre-swaps the delta into deltaX.
        scheduleView({ ...v, offset_x: v.offset_x - (dx || dy) });
      } else {
        // Plain scroll pans both axes: two-finger trackpad panning in any direction.
        scheduleView({ ...v, offset_x: v.offset_x - dx, offset_y: v.offset_y - dy });
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
    // many small deltas, a mouse notch as one large (~100) delta, so give the small deltas a
    // higher gain (pinch-zoom felt sluggish) while the mouse wheel keeps its tuned feel.
    const zoomGain = Math.abs(dy) < 40 ? 0.005 : 0.002;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, v.scale * Math.exp(-dy * zoomGain)));
    if (newScale === v.scale) return;
    scheduleView({
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
        panning.current = null; // button released outside the canvas: end the pan
      } else {
        const dx = e.evt.clientX - panning.current.x;
        const dy = e.evt.clientY - panning.current.y;
        panning.current = { ...panning.current, x: e.evt.clientX, y: e.evt.clientY };
        if (dx || dy) panMoved.current = true;
        const v = liveView();
        scheduleView({ ...v, offset_x: v.offset_x + dx, offset_y: v.offset_y + dy });
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
            listening={false}: every interaction goes through the Stage-level pixel
            handlers (getPointerPosition + toPixel), never per-shape Konva hit-testing,
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
          {img ? (
            <KonvaImage image={img} x={0} y={0} width={props.imgWidth} height={props.imgHeight} />
          ) : null}
          {/* Region serves ride the same transform above the base bitmap, so cold loads
              sharpen progressively while the base keeps rendering underneath. */}
          {(props.regions ?? []).map((region) => {
            const bitmap = regionBitmaps.get(region.key);
            return bitmap ? (
              <KonvaImage
                key={region.key}
                image={bitmap}
                x={region.x}
                y={region.y}
                width={region.width}
                height={region.height}
              />
            ) : null;
          })}
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
      {overview.building && (
        <div
          className="absolute inset-0 flex items-center justify-center pointer-events-none p-4"
          data-testid="overview-building"
        >
          <div className="max-w-sm rounded-md border border-tcip-border bg-tcip-panel/95 px-4 py-3 text-center text-[12px] text-tcip-fg">
            <p className="font-semibold">Preparing this image for display</p>
            <p className="mt-1 text-tcip-muted">
              It is larger than the server serves in one read, so a reduced-resolution copy is being
              built. This runs once per image; it opens normally afterwards.
            </p>
            <div className="mt-2 h-1 w-full overflow-hidden rounded bg-tcip-bg">
              <div
                className="h-full bg-tcip-accent"
                style={{ width: `${Math.round(overview.progress * 100)}%` }}
              />
            </div>
            <p className="mt-1 font-mono text-tcip-muted">{Math.round(overview.progress * 100)}%</p>
          </div>
        </div>
      )}
      {imgError && !overview.building && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-4">
          <div className="max-w-sm rounded-md border border-tcip-fp/50 bg-tcip-panel/95 px-4 py-3 text-center text-[12px] text-tcip-fp">
            {overview.error ? (
              <>Could not prepare this image for display: {overview.error}</>
            ) : baseHeaderParseError ? (
              <>Could not load this image: {baseHeaderParseError}</>
            ) : (
              <>
                Could not load this image: it may be missing, or its path is outside the location(s)
                this server is configured to serve images from. Check the path, or ask whoever set
                up this project where its images should live.
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
