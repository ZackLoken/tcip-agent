import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import * as canvasSync from "@/lib/canvasSync";
import type { CoverageGridResponse } from "@/lib/coverage";

const SERVING = {
  width: 800,
  height: 600,
  tile_size: 800,
  overlap: 0,
  cols: 1,
  rows: 1,
  derivation: "cells sized to one full-resolution screenful",
  cells: [{ name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 }],
};

function response(overrides: Partial<CoverageGridResponse> = {}): CoverageGridResponse {
  return {
    grid: null,
    reason: null,
    fresh_derivation_differs: null,
    serving: SERVING,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useCoverageGrid pending", () => {
  it("is never pending with no image path: there is nothing to fetch yet", () => {
    const { result } = renderHook(() =>
      useCoverageGrid({ imagePath: null, subject: null, date: null, datasetRoot: null }),
    );
    expect(result.current.pending).toBe(false);
    expect(result.current.grid).toBeNull();
  });

  it("waits for a canvas-host measurement before fetching", async () => {
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue(null);
    const grid = vi.spyOn(api.coverage, "grid");
    renderHook(() =>
      useCoverageGrid({
        imagePath: "C:/data/images/2026-01-01/small.jpg",
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
      }),
    );
    await new Promise((r) => requestAnimationFrame(r));
    expect(grid).not.toHaveBeenCalled();
  });

  it("fetches once a measurement exists, single-cell rasters included", async () => {
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
    const grid = vi.spyOn(api.coverage, "grid").mockResolvedValue(
      response({
        grid: {
          width: 800,
          height: 600,
          tile_size: 800,
          overlap: 0,
          cols: 1,
          rows: 1,
          derivation: "one screenful at 1.5x zoom",
          cells: [{ name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 }],
        },
      }),
    );
    const { result } = renderHook(() =>
      useCoverageGrid({
        imagePath: "C:/data/images/2026-01-01/small.jpg",
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
      }),
    );
    expect(result.current.pending).toBe(true);
    await waitFor(() =>
      expect(grid).toHaveBeenCalledWith("C:/data/images/2026-01-01/small.jpg", {
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
        viewportW: 1000,
        viewportH: 800,
        rederive: false,
      }),
    );
    await waitFor(() => expect(result.current.grid).not.toBeNull());
    expect(result.current.cells).toEqual([{ name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 }]);
    expect(result.current.pending).toBe(false);
    expect(result.current.settled).toBe(true);
  });

  it("stays pending while a fetch is genuinely in flight", async () => {
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
    let resolveGrid: (v: CoverageGridResponse) => void = () => {};
    vi.spyOn(api.coverage, "grid").mockReturnValue(
      new Promise((resolve) => {
        resolveGrid = resolve;
      }),
    );
    const { result } = renderHook(() =>
      useCoverageGrid({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
      }),
    );
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    expect(result.current.pending).toBe(true);
    expect(result.current.settled).toBe(false);

    resolveGrid(response());
    await waitFor(() => expect(result.current.settled).toBe(true));
    expect(result.current.pending).toBe(false);
  });

  it("settles not-pending on a fetch failure too", async () => {
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
    vi.spyOn(api.coverage, "grid").mockRejectedValue(new Error("refused"));
    const { result } = renderHook(() =>
      useCoverageGrid({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
      }),
    );
    await waitFor(() => expect(result.current.error).toBe("refused"));
    expect(result.current.pending).toBe(false);
    expect(result.current.settled).toBe(true);
  });

  it("no set zoom answers grid null with the reason, and a non-empty serving", async () => {
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
    vi.spyOn(api.coverage, "grid").mockResolvedValue(
      response({ reason: "set the grid zoom to derive a coverage lattice for bush" }),
    );
    const { result } = renderHook(() =>
      useCoverageGrid({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        subject: "bush",
        date: "2026-01-01",
        datasetRoot: "C:/data",
      }),
    );
    await waitFor(() => expect(result.current.settled).toBe(true));
    expect(result.current.grid).toBeNull();
    expect(result.current.reason).toBe("set the grid zoom to derive a coverage lattice for bush");
    expect(result.current.serving).not.toBeNull();
    expect(result.current.servingCells.length).toBeGreaterThan(0);
  });
});
