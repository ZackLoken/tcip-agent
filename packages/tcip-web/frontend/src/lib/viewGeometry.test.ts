import { describe, expect, it } from "vitest";

import { clampView, fitView, zoomToRect } from "@/lib/viewGeometry";

describe("clampView", () => {
  it("pins an image larger than the canvas so no gap opens beyond either edge", () => {
    const host = { w: 400, h: 300 };
    // 1000x800 at scale 1 exceeds the host: offsets live in [host - scaled, 0].
    expect(clampView({ scale: 1, offset_x: 50, offset_y: 10 }, host, 1000, 800)).toEqual({
      scale: 1,
      offset_x: 0,
      offset_y: 0,
    });
    expect(clampView({ scale: 1, offset_x: -900, offset_y: -700 }, host, 1000, 800)).toEqual({
      scale: 1,
      offset_x: -600,
      offset_y: -500,
    });
  });

  it("keeps an image smaller than the canvas fully on-canvas without forcing centering", () => {
    const host = { w: 500, h: 400 };
    // 100x100 at scale 1 fits: offsets live in [0, host - scaled], free inside that range.
    expect(clampView({ scale: 1, offset_x: -50, offset_y: 450 }, host, 100, 100)).toEqual({
      scale: 1,
      offset_x: 0,
      offset_y: 300,
    });
    const free = { scale: 1, offset_x: 120, offset_y: 40 };
    expect(clampView(free, host, 100, 100)).toBe(free);
  });

  it("clamps per axis independently (wide image, short canvas)", () => {
    const host = { w: 400, h: 300 };
    // 1000x100: exceeds horizontally (clamp to [-600, 0]), fits vertically (clamp to [0, 200]).
    expect(clampView({ scale: 1, offset_x: 25, offset_y: -25 }, host, 1000, 100)).toEqual({
      scale: 1,
      offset_x: 0,
      offset_y: 0,
    });
  });

  it("no-ops on a zero-size host or image, returning the same object", () => {
    const view = { scale: 1, offset_x: 999, offset_y: -999 };
    expect(clampView(view, { w: 0, h: 300 }, 1000, 800)).toBe(view);
    expect(clampView(view, { w: 400, h: 0 }, 1000, 800)).toBe(view);
    expect(clampView(view, { w: 400, h: 300 }, 0, 800)).toBe(view);
    expect(clampView(view, { w: 400, h: 300 }, 1000, 0)).toBe(view);
  });

  it("returns the same object when nothing needs clamping", () => {
    const view = { scale: 1, offset_x: -100, offset_y: -50 };
    expect(clampView(view, { w: 400, h: 300 }, 1000, 800)).toBe(view);
  });
});

describe("fitView", () => {
  it("scales to the tighter axis and centers the image", () => {
    const view = fitView({ w: 400, h: 300 }, 1000, 500);
    expect(view.scale).toBe(0.4);
    expect(view.offset_x).toBe((400 - 1000 * 0.4) / 2);
    expect(view.offset_y).toBe((300 - 500 * 0.4) / 2);
  });

  it("matches CanvasStage's own auto-fit math for the same inputs", () => {
    const host = { w: 733, h: 517 };
    const scale = Math.min(host.w / 2000, host.h / 1200);
    const view = fitView(host, 2000, 1200);
    expect(view).toEqual({
      scale,
      offset_x: (host.w - 2000 * scale) / 2,
      offset_y: (host.h - 1200 * scale) / 2,
    });
  });
});

describe("zoomToRect", () => {
  it("fits the rect to the host and centers it", () => {
    const view = zoomToRect(
      { x0: 400, y0: 300, x1: 500, y1: 400 },
      { host: { w: 200, h: 200 }, imgW: 1000, imgH: 800, padX: 0, padY: 0 },
    );
    expect(view.scale).toBe(2);
    // Rect center (450, 350) lands at the host center (100, 100).
    expect(view.offset_x).toBe(100 - 450 * 2);
    expect(view.offset_y).toBe(100 - 350 * 2);
  });

  it("applies per-axis absolute pads in image pixels", () => {
    const view = zoomToRect(
      { x0: 100, y0: 100, x1: 200, y1: 150 },
      { host: { w: 240, h: 200 }, imgW: 2000, imgH: 2000, padX: 10, padY: 25 },
    );
    // Padded size is 120x100; the host fits it at exactly 2x.
    expect(view.scale).toBe(2);
  });

  it("clamps the fitted scale to the zoom ladder's range", () => {
    const tiny = zoomToRect(
      { x0: 0, y0: 0, x1: 1, y1: 1 },
      { host: { w: 1200, h: 800 }, imgW: 1000, imgH: 800, padX: 0, padY: 0 },
    );
    // A one-pixel rect would fit at 800x, held at the ladder's top stop of 1000%.
    expect(tiny.scale).toBe(10);
    const huge = zoomToRect(
      { x0: 0, y0: 0, x1: 100000, y1: 100000 },
      { host: { w: 100, h: 100 }, imgW: 100000, imgH: 100000, padX: 0, padY: 0 },
    );
    // The whole 100000px rect would fit at 0.001x, held at the bottom stop of 5%.
    expect(huge.scale).toBe(0.05);
  });

  it("pan-clamps the centered view against the image (a corner rect pins to the edge)", () => {
    const view = zoomToRect(
      { x0: 10, y0: 10, x1: 50, y1: 50 },
      { host: { w: 1200, h: 800 }, imgW: 1000, imgH: 800, padX: 40, padY: 40 },
    );
    // The scaled image exceeds the host on both axes, so offsets never open a gap.
    expect(view.offset_x).toBeLessThanOrEqual(0);
    expect(view.offset_y).toBeLessThanOrEqual(0);
  });
});
