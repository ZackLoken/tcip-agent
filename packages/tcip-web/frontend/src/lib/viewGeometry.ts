/**
 * Shared view math for the canvas: fit the view to a pixel rect and clamp pan offsets.
 * One implementation for every consumer (CanvasStage's interactive writes, Review's
 * zoom-to-detection, coverage cell navigation), so the clamping semantics cannot drift.
 */

import { MAX_SCALE, MIN_SCALE } from "@/components/Canvas/zoom";
import type { ViewState } from "@/store/types";

export interface HostSize {
  w: number;
  h: number;
}

/** A half-open image-pixel rect, the same convention the coverage grid's cells use. */
export interface PixelRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Clamp a view's offsets against the canvas. An image larger than the canvas on an axis may not
 * open a gap beyond either edge; an image smaller than the canvas stays fully on-canvas (free
 * positioning within, no forced centering). No-ops on a zero-size host or image. Returns the
 * same object when nothing changes, so callers can skip redundant writes.
 */
export function clampView(view: ViewState, host: HostSize, imgW: number, imgH: number): ViewState {
  if (host.w <= 0 || host.h <= 0 || imgW <= 0 || imgH <= 0) return view;
  const clampAxis = (offset: number, scaled: number, span: number) =>
    Math.min(Math.max(offset, Math.min(0, span - scaled)), Math.max(0, span - scaled));
  const offset_x = clampAxis(view.offset_x, imgW * view.scale, host.w);
  const offset_y = clampAxis(view.offset_y, imgH * view.scale, host.h);
  if (offset_x === view.offset_x && offset_y === view.offset_y) return view;
  return { ...view, offset_x, offset_y };
}

/**
 * The view that fits the whole `imgW` x `imgH` frame inside `host`, centered on both axes: one
 * implementation for the canvas's one-shot auto-fit and the Overview control, so a click on
 * Overview always lands on the exact view the canvas opens an image at.
 */
export function fitView(host: HostSize, imgW: number, imgH: number): ViewState {
  const scale = Math.min(host.w / imgW, host.h / imgH);
  return {
    scale,
    offset_x: (host.w - imgW * scale) / 2,
    offset_y: (host.h - imgH * scale) / 2,
  };
}

/**
 * The view that centers `rect`, padded by `padX`/`padY` (absolute pads in image pixels, per
 * axis), at the largest scale that fits the padded rect. The scale is clamped to the zoom
 * ladder's range and the offsets are pan-clamped against the image.
 */
export function zoomToRect(
  rect: PixelRect,
  args: { host: HostSize; imgW: number; imgH: number; padX: number; padY: number },
): ViewState {
  const w = Math.max(1, rect.x1 - rect.x0) + 2 * args.padX;
  const h = Math.max(1, rect.y1 - rect.y0) + 2 * args.padY;
  const scale = Math.max(
    MIN_SCALE,
    Math.min(MAX_SCALE, Math.min(args.host.w / w, args.host.h / h)),
  );
  const view: ViewState = {
    scale,
    offset_x: args.host.w / 2 - ((rect.x0 + rect.x1) / 2) * scale,
    offset_y: args.host.h / 2 - ((rect.y0 + rect.y1) / 2) * scale,
  };
  return clampView(view, args.host, args.imgW, args.imgH);
}
