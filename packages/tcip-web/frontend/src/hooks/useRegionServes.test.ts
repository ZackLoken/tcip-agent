import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import type { ServingGrid } from "@/api/types.generated";
import { useRegionServes } from "@/hooks/useRegionServes";
import type { LoadedImage } from "@/lib/imageLoader";

const SERVING: ServingGrid = {
  tile_size: 500,
  cells: [
    { name: "S1", x0: 0, y0: 0, x1: 500, y1: 600 },
    { name: "S2", x0: 500, y0: 0, x1: 1000, y1: 600 },
  ],
};

const BASE_FACTS: LoadedImage = {
  ok: true,
  servedSize: { w: 500, h: 300 },
  imageError: null,
  image: null,
  aborted: false,
};

function baseArgs() {
  return {
    imagePath: "C:/data/images/2026-01-01/mosaic.tif",
    imgW: 1000,
    imgH: 600,
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    serving: SERVING,
    baseFacts: BASE_FACTS,
    composite: {},
  };
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

describe("useRegionServes", () => {
  it("serves each route-served cell in view past the base bitmap's resolution", () => {
    const { result } = renderHook(() => useRegionServes(baseArgs()));
    expect(result.current.map((r) => r.key).sort()).toEqual(["S1", "S2"]);
    expect(result.current.find((r) => r.key === "S2")).toMatchObject({ x: 500, width: 500 });
  });

  it("serves no region for a band composite or without a serving grid", () => {
    const composite = renderHook(() =>
      useRegionServes({ ...baseArgs(), composite: { bands: "1,2,3" } }),
    );
    expect(composite.result.current).toEqual([]);
    const none = renderHook(() => useRegionServes({ ...baseArgs(), serving: null }));
    expect(none.result.current).toEqual([]);
  });
});
