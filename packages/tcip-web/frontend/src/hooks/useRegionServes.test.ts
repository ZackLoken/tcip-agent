import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import { useRegionServes } from "@/hooks/useRegionServes";
import type { GridCell } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";

const SERVING_CELL: GridCell = { name: "S1", x0: 0, y0: 0, x1: 1000, y1: 600 };
const A1: GridCell = { name: "A1", x0: 0, y0: 0, x1: 500, y1: 300 };
const B1: GridCell = { name: "B1", x0: 500, y0: 300, x1: 1000, y1: 600 };
const STRADDLING: GridCell = { name: "OUT1", x0: 900, y0: 500, x1: 1100, y1: 700 };

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
    servingCells: [SERVING_CELL],
    servingTileSize: 1000,
    coverageCells: [A1, B1, STRADDLING],
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
  it("marks only the coverage cells fully inside the served serving cell", () => {
    const onCellServedAtNative = vi.fn();
    const { result } = renderHook(() => useRegionServes({ ...baseArgs(), onCellServedAtNative }));
    expect(result.current).toHaveLength(1);
    expect(result.current[0].key).toBe("S1");

    result.current[0].onLoaded?.(loadedAtNative(1000, 600));

    expect(onCellServedAtNative).toHaveBeenCalledWith("A1");
    expect(onCellServedAtNative).toHaveBeenCalledWith("B1");
    expect(onCellServedAtNative).not.toHaveBeenCalledWith("OUT1");
    expect(onCellServedAtNative).toHaveBeenCalledTimes(2);
  });

  it("marks nothing when the serve does not cover the cell at native resolution", () => {
    const onCellServedAtNative = vi.fn();
    const { result } = renderHook(() => useRegionServes({ ...baseArgs(), onCellServedAtNative }));

    result.current[0].onLoaded?.(loadedAtNative(500, 300));

    expect(onCellServedAtNative).not.toHaveBeenCalled();
  });
});
