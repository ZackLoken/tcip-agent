import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useCoverageTracking } from "@/hooks/useCoverageTracking";
import type { GridCell, GridGeometry } from "@/lib/coverage";

const GRID: GridGeometry = {
  width: 300,
  height: 200,
  tile_size: 100,
  overlap: 0,
  cols: 3,
  rows: 2,
};
const CELLS: GridCell[] = [
  { name: "A1", x0: 0, y0: 0, x1: 100, y1: 100 },
  { name: "B1", x0: 100, y0: 0, x1: 200, y1: 100 },
  { name: "C1", x0: 200, y0: 0, x1: 300, y1: 100 },
  { name: "A2", x0: 0, y0: 100, x1: 100, y1: 200 },
  { name: "B2", x0: 100, y0: 100, x1: 200, y1: 200 },
  { name: "C2", x0: 200, y0: 100, x1: 300, y1: 200 },
];

function trackingArgs(subject: string | null) {
  return {
    imagePath: "C:/data/images/2026-01-01/mosaic.tif",
    datasetRoot: "C:/data",
    subject,
    date: "2026-01-01",
    grid: GRID,
    cells: CELLS,
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    imgW: 300,
    imgH: 200,
    viewing: {},
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useCoverageTracking subject gating", () => {
  it("no active subject: no hydration fetch, no accumulation, no POST", async () => {
    const get = vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    const push = vi.spyOn(api.coverage, "push").mockResolvedValue({ status: "ok" });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs(null)));

    result.current.noteAuthoringCommit();
    result.current.noteServedAtNative("A1");
    await new Promise((r) => setTimeout(r, 500));
    expect(get).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
    expect(result.current.swept.size).toBe(0);
  });

  it("with a subject, hydrates the stored record for the same (path, subject, date)", async () => {
    const get = vi
      .spyOn(api.coverage, "get")
      .mockResolvedValue({ grid: GRID, cells_swept: ["A1"], cells_served_at_native: [] });
    vi.spyOn(api.coverage, "push").mockResolvedValue({ status: "ok" });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip")));

    await waitFor(() =>
      expect(get).toHaveBeenCalledWith("C:/data/images/2026-01-01/mosaic.tif", "tip", "2026-01-01"),
    );
    await waitFor(() => expect(result.current.swept.has("A1")).toBe(true));
  });
});
