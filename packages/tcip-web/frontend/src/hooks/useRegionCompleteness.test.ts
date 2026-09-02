import { afterEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { useRegionCompleteness } from "@/hooks/useRegionCompleteness";
import type { CompletenessRecord, CompletenessResponse } from "@/lib/coverage";
import { useStore } from "@/store";

const GRID = { width: 300, height: 200, tile_size: 100, overlap: 0, cols: 3, rows: 2 };
const OTHER_GRID = { width: 300, height: 200, tile_size: 150, overlap: 0, cols: 2, rows: 2 };

function record(
  subject: string,
  cells_complete: string[],
  stale_cells: string[] = [],
  grid = GRID,
): CompletenessRecord {
  return {
    grid,
    cells_complete,
    attested_by: "user:z",
    attested_at: "t",
    stem: "mosaic",
    date: null,
    subject,
    stale_cells,
  };
}

function response(
  by_subject: Record<string, CompletenessRecord>,
  overrides: Partial<CompletenessResponse> = {},
): CompletenessResponse {
  return { by_subject, annotation_counts: {}, grid_error: null, ...overrides };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useRegionCompleteness", () => {
  it("no image path: no fetch, empty sets", () => {
    const get = vi.spyOn(api.coverage, "completeness");
    const { result } = renderHook(() =>
      useRegionCompleteness({ imagePath: null, datasetRoot: null, subject: "bush", grid: GRID }),
    );
    expect(get).not.toHaveBeenCalled();
    expect(result.current.activeComplete.size).toBe(0);
    expect(result.current.otherComplete.size).toBe(0);
  });

  it("splits the active subject's cells from every other subject's, both on the current grid", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1"]), leaf: record("leaf", ["B2", "C1"]) }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["A1"])));
    expect(result.current.otherComplete).toEqual(new Set(["B2", "C1"]));
  });

  it("a stale cell is excluded from the effective sets but named in activeStale", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({
        bush: record("bush", ["A1", "B1"], ["A1"]),
        leaf: record("leaf", ["C1"], ["C1"]),
      }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["B1"])));
    expect(result.current.activeStale).toEqual(new Set(["A1"]));
    expect(result.current.otherComplete).toEqual(new Set());
  });

  it("an attestation on another grid is never rendered on the current one, and its count is stated", async () => {
    // Guards the live defect: effectiveComplete() alone ignores which grid a record was
    // accumulated against; sameGrid() must gate rendering, the record's own dims still reported.
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1", "B1"], [], OTHER_GRID) }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.otherLattice).not.toBeNull());
    expect(result.current.activeComplete.size).toBe(0);
    expect(result.current.otherLattice).toEqual({
      count: 2,
      cols: OTHER_GRID.cols,
      rows: OTHER_GRID.rows,
    });
  });

  it("no active subject: write is a no-op, no POST", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(response({}));
    const post = vi.spyOn(api.coverage, "setCompleteness");
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: null,
        grid: GRID,
      }),
    );
    result.current.write("A1", GRID, true);
    await new Promise((r) => setTimeout(r, 50));
    expect(post).not.toHaveBeenCalled();
  });

  it("write posts the explicit direction for the active subject and refetches", async () => {
    const get = vi
      .spyOn(api.coverage, "completeness")
      .mockResolvedValueOnce(response({}))
      .mockResolvedValueOnce(response({ bush: record("bush", ["A1"]) }));
    const post = vi
      .spyOn(api.coverage, "setCompleteness")
      .mockResolvedValue({ status: "ok", complete: true, cells_complete: ["A1"] });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));

    result.current.write("A1", GRID, true);
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith({
        image_path: "C:/data/images/2026-01-01/mosaic.tif",
        dataset_root: "C:/data",
        subject: "bush",
        grid: GRID,
        cell: "A1",
        complete: true,
        user: "",
      }),
    );
    await waitFor(() => expect(result.current.activeComplete).toEqual(new Set(["A1"])));
  });

  it("write(cell, grid, false) unattests, and re-attesting a stale cell restamps it", async () => {
    vi.spyOn(api.coverage, "completeness")
      .mockResolvedValueOnce(response({ bush: record("bush", ["A1"], ["A1"]) }))
      .mockResolvedValueOnce(response({ bush: record("bush", ["A1"]) }));
    const post = vi
      .spyOn(api.coverage, "setCompleteness")
      .mockResolvedValue({ status: "ok", complete: true, cells_complete: ["A1"] });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.activeStale).toEqual(new Set(["A1"])));

    result.current.write("A1", GRID, true);
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith(expect.objectContaining({ cell: "A1", complete: true })),
    );
    await waitFor(() => expect(result.current.activeStale).toEqual(new Set()));
  });

  it("a failed write surfaces an error toast, not a silent no-op", async () => {
    const get = vi.spyOn(api.coverage, "completeness").mockResolvedValue(response({}));
    vi.spyOn(api.coverage, "setCompleteness").mockRejectedValue(new Error("refused"));
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));

    result.current.write("A1", GRID, true);
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

  it("a read failure surfaces as error, never silently read as nothing attested", async () => {
    vi.spyOn(api.coverage, "completeness").mockRejectedValue(new Error("network down"));
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.error).toBe("network down"));
    expect(result.current.activeComplete.size).toBe(0);
  });

  it("a grid_error on an otherwise successful read surfaces as countsError, records still served", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1"]) }, { grid_error: "band group incomplete" }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.countsError).toBe("band group incomplete"));
    expect(result.current.activeComplete).toEqual(new Set(["A1"]));
  });
});
