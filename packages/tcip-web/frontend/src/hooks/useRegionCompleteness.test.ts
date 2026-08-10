import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useRegionCompleteness } from "@/hooks/useRegionCompleteness";
import type { CompletenessRecord } from "@/lib/coverage";
import { useStore } from "@/store";

const GRID = { width: 300, height: 200, tile_size: 100, overlap: 0, cols: 3, rows: 2 };

function record(
  subject: string,
  cells_complete: string[],
  stale_cells: string[] = [],
): CompletenessRecord {
  return {
    grid: GRID,
    cells_complete,
    attested_by: "user:z",
    attested_at: "t",
    stem: "mosaic",
    date: null,
    subject,
    stale_cells,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useRegionCompleteness", () => {
  it("no image path: no fetch, empty sets", async () => {
    const get = vi.spyOn(api.coverage, "completeness");
    const { result } = renderHook(() =>
      useRegionCompleteness({ imagePath: null, datasetRoot: null, subject: "bush" }),
    );
    expect(get).not.toHaveBeenCalled();
    expect(result.current.activeComplete.size).toBe(0);
    expect(result.current.otherComplete.size).toBe(0);
  });

  it("splits the active subject's cells from every other subject's", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: { bush: record("bush", ["A1"]), leaf: record("leaf", ["B2", "C1"]) },
    });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["A1"])));
    expect(result.current.otherComplete).toEqual(new Set(["B2", "C1"]));
  });

  it("a stale cell is excluded from both the active and the other-subject sets", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {
        bush: record("bush", ["A1", "B1"], ["A1"]),
        leaf: record("leaf", ["C1"], ["C1"]),
      },
    });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["B1"])));
    expect(result.current.otherComplete).toEqual(new Set());
  });

  it("no active subject: toggle is a no-op, no POST", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({ by_subject: {} });
    const post = vi.spyOn(api.coverage, "toggleCompleteness");
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: null,
      }),
    );
    result.current.toggle("A1", GRID);
    await new Promise((r) => setTimeout(r, 50));
    expect(post).not.toHaveBeenCalled();
  });

  it("toggle posts the cell for the active subject and refetches", async () => {
    const get = vi
      .spyOn(api.coverage, "completeness")
      .mockResolvedValueOnce({ by_subject: {} })
      .mockResolvedValueOnce({ by_subject: { bush: record("bush", ["A1"]) } });
    const post = vi
      .spyOn(api.coverage, "toggleCompleteness")
      .mockResolvedValue({ status: "ok", complete: true, cells_complete: ["A1"] });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
      }),
    );
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));

    result.current.toggle("A1", GRID);
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith({
        image_path: "C:/data/images/2026-01-01/mosaic.tif",
        dataset_root: "C:/data",
        subject: "bush",
        grid: GRID,
        cell: "A1",
        user: "",
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["A1"])));
  });

  it("a failed toggle surfaces an error toast, not a silent no-op", async () => {
    const get = vi.spyOn(api.coverage, "completeness").mockResolvedValue({ by_subject: {} });
    vi.spyOn(api.coverage, "toggleCompleteness").mockRejectedValue(new Error("refused"));
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
      }),
    );
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));

    result.current.toggle("A1", GRID);
    await waitFor(() =>
      expect(
        useStore
          .getState()
          .toasts.some((t) => t.message.includes("A1") && t.message.includes("refused")),
      ).toBe(true),
    );
    // A failure reloads too, the same as a success, rather than trusting a stale local guess.
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  });
});
