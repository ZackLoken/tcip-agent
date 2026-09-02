import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useCoverageGrid } from "@/hooks/useCoverageGrid";
import type { CoverageGridResponse } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";

function baseFacts(overrides: Partial<LoadedImage> = {}): LoadedImage {
  return {
    ok: true,
    servedSize: { w: 500, h: 400 },
    servedSizeRaw: "500x400",
    statsSource: null,
    displayBounds: null,
    imageError: null,
    image: null,
    aborted: false,
    headerParseError: null,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useCoverageGrid pending", () => {
  it("is pending before the base serve has reported back, never a settled empty grid", () => {
    const { result } = renderHook(() =>
      useCoverageGrid("C:/data/images/2026-01-01/mosaic.tif", null, 1000, 800),
    );
    expect(result.current.pending).toBe(true);
    expect(result.current.grid).toBeNull();
  });

  it("settles not-pending once the base serve shows the raster fits inside one serve", () => {
    const grid = vi.spyOn(api.coverage, "grid");
    const { result } = renderHook(() =>
      useCoverageGrid(
        "C:/data/images/2026-01-01/mosaic.tif",
        baseFacts({ servedSize: { w: 1000, h: 800 } }),
        1000,
        800,
      ),
    );
    expect(result.current.pending).toBe(false);
    expect(grid).not.toHaveBeenCalled();
  });

  it("stays pending while a below-native fetch is genuinely in flight", async () => {
    let resolveGrid: (v: CoverageGridResponse) => void = () => {};
    vi.spyOn(api.coverage, "grid").mockReturnValue(
      new Promise((resolve) => {
        resolveGrid = resolve;
      }),
    );
    const { result } = renderHook(() =>
      useCoverageGrid("C:/data/images/2026-01-01/mosaic.tif", baseFacts(), 1000, 800),
    );
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    expect(result.current.pending).toBe(true);

    resolveGrid({
      width: 1000,
      height: 800,
      tile_size: 500,
      overlap: 0,
      cols: 2,
      rows: 2,
      derivation: "one display-bounded serve per cell",
      cells: [{ name: "A1", x0: 0, y0: 0, x1: 500, y1: 400 }],
    });
    await waitFor(() => expect(result.current.grid).not.toBeNull());
    expect(result.current.pending).toBe(false);
  });

  it("settles not-pending on a fetch failure too", async () => {
    vi.spyOn(api.coverage, "grid").mockRejectedValue(new Error("refused"));
    const { result } = renderHook(() =>
      useCoverageGrid("C:/data/images/2026-01-01/mosaic.tif", baseFacts(), 1000, 800),
    );
    await waitFor(() => expect(result.current.error).toBe("refused"));
    expect(result.current.pending).toBe(false);
  });
});
