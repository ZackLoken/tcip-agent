import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import type { CoverageGridResponse } from "@/lib/coverage";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useCoverageGrid pending", () => {
  it("is never pending with no image path: there is nothing to fetch yet", () => {
    const { result } = renderHook(() => useCoverageGrid(null));
    expect(result.current.pending).toBe(false);
    expect(result.current.grid).toBeNull();
  });

  it("fetches the grid for every raster, single-cell rasters included, not gated on served size", async () => {
    const grid = vi.spyOn(api.coverage, "grid").mockResolvedValue({
      width: 800,
      height: 600,
      tile_size: 800,
      overlap: 0,
      cols: 1,
      rows: 1,
      derivation: "cells sized to one full-resolution screenful",
      cells: [{ name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 }],
    });
    const { result } = renderHook(() => useCoverageGrid("C:/data/images/2026-01-01/small.jpg"));
    expect(result.current.pending).toBe(true);
    await waitFor(() => expect(grid).toHaveBeenCalledWith("C:/data/images/2026-01-01/small.jpg"));
    await waitFor(() => expect(result.current.grid).not.toBeNull());
    expect(result.current.cells).toEqual([{ name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 }]);
    expect(result.current.pending).toBe(false);
  });

  it("stays pending while a fetch is genuinely in flight", async () => {
    let resolveGrid: (v: CoverageGridResponse) => void = () => {};
    vi.spyOn(api.coverage, "grid").mockReturnValue(
      new Promise((resolve) => {
        resolveGrid = resolve;
      }),
    );
    const { result } = renderHook(() => useCoverageGrid("C:/data/images/2026-01-01/mosaic.tif"));
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    expect(result.current.pending).toBe(true);

    resolveGrid({
      width: 1000,
      height: 800,
      tile_size: 500,
      overlap: 0,
      cols: 2,
      rows: 2,
      derivation: "cells sized to one full-resolution screenful",
      cells: [{ name: "A1", x0: 0, y0: 0, x1: 500, y1: 400 }],
    });
    await waitFor(() => expect(result.current.grid).not.toBeNull());
    expect(result.current.pending).toBe(false);
  });

  it("settles not-pending on a fetch failure too", async () => {
    vi.spyOn(api.coverage, "grid").mockRejectedValue(new Error("refused"));
    const { result } = renderHook(() => useCoverageGrid("C:/data/images/2026-01-01/mosaic.tif"));
    await waitFor(() => expect(result.current.error).toBe("refused"));
    expect(result.current.pending).toBe(false);
  });
});
