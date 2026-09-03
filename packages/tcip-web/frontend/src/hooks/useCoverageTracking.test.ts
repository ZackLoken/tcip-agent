import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import type { WorkingScale } from "@/api/types.generated";
import { useCoverageTracking } from "@/hooks/useCoverageTracking";
import type { GridCell, GridGeometry } from "@/lib/coverage";
import { resetCoverageOutbox } from "@/test/coverageOutbox";
import { useStore } from "@/store";

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
const NULL_VIEWING = {
  bands: null,
  stretch: null,
  stats_source: null,
  display_bounds: null,
  base_served_size: null,
};

function bar(value: number): WorkingScale {
  return { value, source: "s" };
}

function trackingArgs(subject: string | null, workingScale: WorkingScale | null = null) {
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
    viewing: NULL_VIEWING,
    workingScale,
  };
}

afterEach(() => {
  resetCoverageOutbox();
  vi.restoreAllMocks();
});

describe("useCoverageTracking subject gating", () => {
  it("no active subject: no hydration fetch, no accumulation, no POST", async () => {
    const get = vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    const push = vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs(null)));

    result.current.noteServedAtNative("A1");
    await new Promise((r) => setTimeout(r, 500));
    expect(get).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
    expect(result.current.swept.size).toBe(0);
  });

  it("with a subject, hydrates the stored record for the same (path, subject, date)", async () => {
    const get = vi.spyOn(api.coverage, "get").mockResolvedValue({
      grid: GRID,
      cells_seen_at_scale: { A1: 1 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip", bar(0.5))));

    await waitFor(() =>
      expect(get).toHaveBeenCalledWith("C:/data/images/2026-01-01/mosaic.tif", "tip", "2026-01-01"),
    );
    await waitFor(() => expect(result.current.swept.has("A1")).toBe(true));
  });

  it("a refusal from api.coverage.get surfaces as a toast naming it", async () => {
    vi.spyOn(api.coverage, "get").mockRejectedValue(
      new Error("plot.tif's stored view-coverage record does not validate"),
    );
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    renderHook(() => useCoverageTracking(trackingArgs("tip")));

    await waitFor(() =>
      expect(useStore.getState().toasts.some((t) => t.message.includes("does not validate"))).toBe(
        true,
      ),
    );
  });

  it("a missing record (the no-record answer) stays silent", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    const toastsBefore = useStore.getState().toasts.length;
    renderHook(() => useCoverageTracking(trackingArgs("tip")));

    await new Promise((r) => setTimeout(r, 50));
    expect(useStore.getState().toasts.length).toBe(toastsBefore);
  });
});

describe("useCoverageTracking sweep on another lattice", () => {
  const OTHER_GRID: GridGeometry = {
    width: 600,
    height: 400,
    tile_size: 200,
    overlap: 0,
    cols: 3,
    rows: 2,
  };

  it("states a record's seen-cell count and dims when its grid differs from the current one", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue({
      grid: OTHER_GRID,
      cells_seen_at_scale: { A1: 1, B1: 1 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip", bar(0.5))));

    await waitFor(() => expect(result.current.replaceRequired).not.toBeNull());
    expect(result.current.replaceRequired).toEqual({ cols: 3, rows: 2, cellsSeen: 2 });
    // Never hydrated onto the current lattice's swept set: a different lattice's cell names
    // cannot be trusted to mean the same cells here.
    expect(result.current.swept.size).toBe(0);
  });

  it("null when the record's grid matches the current one, the ordinary case", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue({
      grid: GRID,
      cells_seen_at_scale: { A1: 1 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip", bar(0.5))));

    await waitFor(() => expect(result.current.swept.has("A1")).toBe(true));
    expect(result.current.replaceRequired).toBeNull();
  });

  it("armReplace posts the next payload with replace: true", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue({
      grid: OTHER_GRID,
      cells_seen_at_scale: { A1: 1 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    const push = vi
      .spyOn(api.coverage, "push")
      .mockResolvedValue({ record: { cells_seen_at_scale: {} } });
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip", bar(0.5))));

    await waitFor(() => expect(result.current.replaceRequired).not.toBeNull());
    result.current.armReplace();
    await waitFor(() => expect(push).toHaveBeenCalled());
    expect(push.mock.calls[0][0].replace).toBe(true);
    await waitFor(() => expect(result.current.replaceRequired).toBeNull());
  });
});

describe("useCoverageTracking failed push", () => {
  it("a failed coverage push toasts through the tab's own toast path, not silently", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    vi.spyOn(api.coverage, "push").mockRejectedValue(new Error("connection refused"));
    const { result } = renderHook(() => useCoverageTracking(trackingArgs("tip")));

    result.current.noteServedAtNative("A1");
    await waitFor(() =>
      expect(
        useStore
          .getState()
          .toasts.some(
            (t) => t.message.includes("coverage") && t.message.includes("connection refused"),
          ),
      ).toBe(true),
    );
  });
});
