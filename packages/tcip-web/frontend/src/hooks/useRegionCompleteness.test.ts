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
  return {
    by_subject,
    annotation_counts: {},
    counts_grid: null,
    counts_error: null,
    working_scale: {},
    working_scale_error: null,
    working_scale_reason: {},
    ...overrides,
  };
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
    expect(result.current.otherCompleteBySubject).toEqual({ leaf: ["B2", "C1"] });
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
    result.current.write("A1", GRID, true, 0.5);
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
      .mockResolvedValue({ status: "ok", complete: true, cells_complete: ["A1"], replaced: null });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));

    result.current.write("A1", GRID, true, 0.5);
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith({
        image_path: "C:/data/images/2026-01-01/mosaic.tif",
        dataset_root: "C:/data",
        subject: "bush",
        grid: GRID,
        cell: "A1",
        complete: true,
        view_scale: 0.5,
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
      .mockResolvedValue({ status: "ok", complete: true, cells_complete: ["A1"], replaced: null });
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.activeStale).toEqual(new Set(["A1"])));

    result.current.write("A1", GRID, true, 0.5);
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

    result.current.write("A1", GRID, true, 0.5);
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

  it("a counts_error on an otherwise successful read surfaces as countsError, records still served", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1"]) }, { counts_error: "band group incomplete" }),
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

  it("otherLattice is null while the current grid is unknown, never a stated fact it cannot see", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1", "B1"], [], OTHER_GRID) }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: null,
      }),
    );
    await waitFor(() =>
      expect(api.coverage.completeness).toHaveBeenCalledWith(
        "C:/data/images/2026-01-01/mosaic.tif",
        "C:/data",
        "bush",
      ),
    );
    expect(result.current.otherLattice).toBeNull();
  });

  it("annotation counts render only when they were binned on the current grid", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({}, { annotation_counts: { bush: { A1: 3 } }, counts_grid: OTHER_GRID }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.countsError).not.toBeNull());
    expect(result.current.annotationCounts).toEqual({});
  });

  it("annotation counts render when the served counts grid matches the current grid", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({}, { annotation_counts: { bush: { A1: 3 } }, counts_grid: GRID }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.annotationCounts).toEqual({ A1: 3 }));
    expect(result.current.countsError).toBeNull();
  });

  it("exposes the active subject's served working-scale bar", async () => {
    const scaleBar = {
      value: 0.5,
      median_extent_native_px: 92,
      annotation_count: 2,
      judged_span_px: 46,
      source: "s",
    };
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({}, { working_scale: { bush: scaleBar } }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.workingScale).toEqual(scaleBar));
    expect(result.current.workingScaleReason).toBeNull();
  });

  it("states no active subject as the reason with no subject, rather than a silent null", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(response({}));
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: null,
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.workingScaleReason).toBe("no active subject"));
  });

  it("states the served working_scale_error as the reason when the label read failed", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({}, { working_scale: {}, working_scale_error: "plot.json: not valid JSON" }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() =>
      expect(result.current.workingScaleReason).toBe("plot.json: not valid JSON"),
    );
    expect(result.current.workingScale).toBeNull();
  });

  it("names the absent-annotation reason when the subject is served explicitly null", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({}, { working_scale: { bush: null } }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() =>
      expect(result.current.workingScaleReason).toBe("no saved box or polygon annotation of bush"),
    );
  });

  it("prefers the served per-subject reason over its own composed clause", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response(
        {},
        {
          working_scale: { bush: null },
          working_scale_reason: {
            bush:
              "no saved box or polygon annotation of bush on this image and no known " +
              "pixel size (it is not a raster)",
          },
        },
      ),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() =>
      expect(result.current.workingScaleReason).toBe(
        "no saved box or polygon annotation of bush on this image and no known pixel size " +
          "(it is not a raster)",
      ),
    );
  });

  it("exposes the active subject's cells_attested_view on the current grid, empty off it", async () => {
    const entry = {
      view_scale: 0.5,
      working_scale_bar_at_write: null,
      seen_on_record: { at_scale: null, grid_matched: false },
    };
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: { ...record("bush", ["A1"]), cells_attested_view: { A1: entry } } }),
    );
    const { result } = renderHook(() =>
      useRegionCompleteness({
        imagePath: "C:/data/images/2026-01-01/mosaic.tif",
        datasetRoot: "C:/data",
        subject: "bush",
        grid: GRID,
      }),
    );
    await waitFor(() => expect(result.current.activeCellsAttestedView).toEqual({ A1: entry }));
  });

  it("an absent cells_attested_view on an old-shape record reads as empty, not an error", async () => {
    vi.spyOn(api.coverage, "completeness").mockResolvedValue(
      response({ bush: record("bush", ["A1"]) }),
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
    expect(result.current.activeCellsAttestedView).toEqual({});
  });
});
