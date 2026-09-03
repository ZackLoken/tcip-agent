import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import { useRegionServes } from "@/hooks/useRegionServes";
import type { GridCell } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";

const SERVING_LEFT: GridCell = { name: "S1", x0: 0, y0: 0, x1: 500, y1: 600 };
const SERVING_RIGHT: GridCell = { name: "S2", x0: 500, y0: 0, x1: 1000, y1: 600 };
const SMALL: GridCell = { name: "A1", x0: 0, y0: 0, x1: 500, y1: 600 };
const SPANNING: GridCell = { name: "BIG1", x0: 0, y0: 0, x1: 1000, y1: 600 };

const BASE_FACTS: LoadedImage = {
  ok: true,
  servedSize: { w: 500, h: 300 },
  servedSizeRaw: "500x300",
  statsSource: null,
  displayBounds: null,
  imageError: null,
  image: null,
  aborted: false,
  headerParseError: null,
};

function baseArgs() {
  return {
    imagePath: "C:/data/images/2026-01-01/mosaic.tif",
    imgW: 1000,
    imgH: 600,
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    servingCells: [SERVING_LEFT, SERVING_RIGHT],
    servingTileSize: 500,
    coverageCells: [SMALL, SPANNING],
    baseFacts: BASE_FACTS,
    composite: {},
  };
}

function loadedAtNative(w: number, h: number): LoadedImage {
  return { ...BASE_FACTS, servedSize: { w, h }, servedSizeRaw: `${w}x${h}` };
}

beforeEach(() => {
  const host = document.createElement("div");
  host.setAttribute("data-canvas-host", "");
  document.body.appendChild(host);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
    width: 1200,
    height: 800,
    top: 0,
    left: 0,
    right: 1200,
    bottom: 800,
    x: 0,
    y: 0,
    toJSON: () => "",
  } as DOMRect);
});

afterEach(() => {
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("useRegionServes served-at-native fold", () => {
  it("marks a coverage cell fully inside one serving cell once that cell is served at native", () => {
    const onCellServedAtNative = vi.fn();
    const { result } = renderHook(() => useRegionServes({ ...baseArgs(), onCellServedAtNative }));
    expect(result.current).toHaveLength(2);

    const left = result.current.find((r) => r.key === "S1");
    left?.onLoaded?.(loadedAtNative(500, 600));

    expect(onCellServedAtNative).toHaveBeenCalledWith("A1");
    expect(onCellServedAtNative).not.toHaveBeenCalledWith("BIG1");
  });

  it("marks a coverage cell spanning two serving cells only once both have been served at native", () => {
    const onCellServedAtNative = vi.fn();
    const { result } = renderHook(() => useRegionServes({ ...baseArgs(), onCellServedAtNative }));

    const left = result.current.find((r) => r.key === "S1");
    const right = result.current.find((r) => r.key === "S2");

    left?.onLoaded?.(loadedAtNative(500, 600));
    expect(onCellServedAtNative).not.toHaveBeenCalledWith("BIG1");

    right?.onLoaded?.(loadedAtNative(500, 600));
    expect(onCellServedAtNative).toHaveBeenCalledWith("BIG1");
  });

  it("marks nothing when the serve does not cover the cell at native resolution", () => {
    const onCellServedAtNative = vi.fn();
    const { result } = renderHook(() => useRegionServes({ ...baseArgs(), onCellServedAtNative }));

    result.current[0].onLoaded?.(loadedAtNative(400, 600));

    expect(onCellServedAtNative).not.toHaveBeenCalled();
  });
});
