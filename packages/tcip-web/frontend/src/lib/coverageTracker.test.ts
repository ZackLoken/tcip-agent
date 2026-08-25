import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import type { GridCell, GridGeometry } from "@/lib/coverage";
import { CoverageTracker, type CoveragePayload } from "@/lib/coverageTracker";

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
const KEY = {
  imagePath: "C:/data/images/2026-01-01/mosaic.tif",
  datasetRoot: "C:/data",
  subject: "tip",
  date: "2026-01-01",
};
const FULL_VIEW = { x0: 0, y0: 0, x1: 300, y1: 200 };

let post: Mock<(body: CoveragePayload) => Promise<unknown>>;
let tracker: CoverageTracker;

beforeEach(() => {
  vi.useFakeTimers();
  post = vi.fn(() => Promise.resolve({}));
  tracker = new CoverageTracker(post);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("CoverageTracker sweep", () => {
  it("accumulates no sweep without an authoring commit (no bar, no sweep)", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport(FULL_VIEW, 1);
    expect(tracker.swept.size).toBe(0);
    vi.advanceTimersByTime(1000);
    expect(post).not.toHaveBeenCalled();
  });

  it("sweeps exactly the cells fully contained in the viewport at or above the bar", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteAuthoringScale(0.5);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 250, y1: 150 }, 0.5);
    expect(Array.from(tracker.swept).sort()).toEqual(["A1", "B1"]);
    // A pass below the bar sweeps nothing, whatever it contains.
    tracker.noteViewport(FULL_VIEW, 0.4);
    expect(tracker.swept.size).toBe(2);
  });

  it("a commit at a coarser scale lowers the bar and re-evaluates the current viewport", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport(FULL_VIEW, 0.4);
    tracker.noteAuthoringScale(0.5);
    expect(tracker.swept.size).toBe(0); // the on-screen pass was below the bar
    tracker.noteAuthoringScale(0.4);
    expect(tracker.swept.size).toBe(6); // the same viewport now meets the lowered bar
  });

  it("viewport moments before the first commit count once a bar exists", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 250, y1: 150 }, 1); // A1, B1 at full scale
    tracker.noteViewport({ x0: 180, y0: 0, x1: 310, y1: 160 }, 0.3); // C1 below the coming bar
    expect(tracker.swept.size).toBe(0);
    tracker.noteAuthoringScale(0.5);
    // The pre-commit pans at scale 1 qualify against the 0.5 bar; the 0.3 pass does not.
    expect(Array.from(tracker.swept).sort()).toEqual(["A1", "B1"]);
  });
});

describe("CoverageTracker sub-cell union sweep", () => {
  // 512px, divisions derived from SUB_CELL_TARGET_PX=128 -> exactly 4 sub-cells per side (128px
  // each): the whole cell never fits inside any of the 128px-wide viewports below.
  const BIG_CELL: GridCell = { name: "A1", x0: 0, y0: 0, x1: 512, y1: 512 };
  const OTHER_CELL: GridCell = { name: "B1", x0: 512, y0: 0, x1: 1024, y1: 512 };
  const BIG_GRID: GridGeometry = {
    width: 1024,
    height: 512,
    tile_size: 512,
    overlap: 0,
    cols: 2,
    rows: 1,
  };

  it("sweeps a cell that never once fits fully in the viewport, once its sub-cells' union does", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    tracker.noteAuthoringScale(1);
    // Each pass is one 128px sub-cell wide (< the cell's 512px) but full height, so it fully
    // contains one column of sub-cells at a time, never the whole cell.
    tracker.noteViewport({ x0: 0, y0: 0, x1: 128, y1: 512 }, 1);
    expect(tracker.swept.has("A1")).toBe(false);
    tracker.noteViewport({ x0: 128, y0: 0, x1: 256, y1: 512 }, 1);
    tracker.noteViewport({ x0: 256, y0: 0, x1: 384, y1: 512 }, 1);
    expect(tracker.swept.has("A1")).toBe(false); // union so far: x in [0, 384) only
    tracker.noteViewport({ x0: 384, y0: 0, x1: 512, y1: 512 }, 1); // reaches the far edge
    expect(tracker.swept.has("A1")).toBe(true);
  });

  it("never sweeps through a viewport narrower than one sub-cell, however densely it pans", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    tracker.noteAuthoringScale(1);
    // 100px passes are under the 128px sub-cell grain, so no sub-cell ever sits fully inside one.
    for (let x0 = 0; x0 < 512; x0 += 10) {
      tracker.noteViewport({ x0, y0: 0, x1: Math.min(x0 + 100, 512), y1: 512 }, 1);
    }
    expect(tracker.swept.has("A1")).toBe(false);
    // The same cell sweeps once the passes are as wide as the grain, so the tracker was live.
    for (let x0 = 0; x0 < 512; x0 += 128) {
      tracker.noteViewport({ x0, y0: 0, x1: x0 + 128, y1: 512 }, 1);
    }
    expect(tracker.swept.has("A1")).toBe(true);
  });

  it("does not sweep while the sub-cell union stays partial all session", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    tracker.noteAuthoringScale(1);
    // Always the same left-hand column: the rest of the cell's sub-cells are never seen.
    for (let i = 0; i < 5; i++) {
      tracker.noteViewport({ x0: 0, y0: 0, x1: 128, y1: 512 }, 1);
    }
    expect(tracker.swept.has("A1")).toBe(false);
  });

  it("a late-arriving bar credits sub-cell union progress recorded before it existed", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    // Panned across the whole cell at scale 1 before ever committing an annotation.
    tracker.noteViewport({ x0: 0, y0: 0, x1: 128, y1: 512 }, 1);
    tracker.noteViewport({ x0: 128, y0: 0, x1: 256, y1: 512 }, 1);
    tracker.noteViewport({ x0: 256, y0: 0, x1: 384, y1: 512 }, 1);
    tracker.noteViewport({ x0: 384, y0: 0, x1: 512, y1: 512 }, 1);
    expect(tracker.swept.has("A1")).toBe(false); // no bar yet
    tracker.noteAuthoringScale(0.5);
    expect(tracker.swept.has("A1")).toBe(true); // the pre-commit union already clears the bar
  });

  it("closes the large-raster-lattice gap: a viewport too narrow for the old fixed-32-division grain can now sweep", () => {
    // Real ValleyFarm large-raster cell edge: the old fixed 32-division grain gave ~469px
    // sub-cells (too big for this 300px viewport); the new ~127px per-cell derivation fits.
    const edge = 14996;
    const cell: GridCell = { name: "A1", x0: 0, y0: 0, x1: edge, y1: edge };
    const other: GridCell = { name: "B1", x0: edge, y0: 0, x1: edge * 2, y1: edge };
    const grid: GridGeometry = {
      width: edge * 2,
      height: edge,
      tile_size: edge,
      overlap: 0,
      cols: 2,
      rows: 1,
    };
    tracker.reset(KEY, grid, [cell, other]);
    tracker.noteAuthoringScale(1);
    const viewportWidth = 300; // comfortably below the old grain, above the new one
    const stride = 100; // dense overlap, so no sub-cell can straddle a gap between passes
    for (let x0 = 0; x0 < edge; x0 += stride) {
      tracker.noteViewport({ x0, y0: 0, x1: Math.min(x0 + viewportWidth, edge), y1: edge }, 1);
    }
    expect(tracker.swept.has("A1")).toBe(true);
  });
});

describe("CoverageTracker served-at-native", () => {
  it("records only names that belong to the lattice", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteServedAtNative("B1");
    tracker.noteServedAtNative("Z9");
    expect(Array.from(tracker.servedAtNative)).toEqual(["B1"]);
  });
});

describe("CoverageTracker posting", () => {
  it("debounces facts into one POST carrying both fact lists and the viewing context", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing({
      bands: "3,2,1",
      stretch: "percent_clip",
      stats_source: { read: "overview", seed: null, pixel_fraction: null, overview_scale: 0.5 },
      display_bounds: [[0, 255]],
      base_served_size: "150x100",
    });
    tracker.noteAuthoringScale(1);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 120, y1: 120 }, 1); // contains A1 only
    tracker.noteServedAtNative("B1");
    expect(post).not.toHaveBeenCalled();
    vi.advanceTimersByTime(400);
    expect(post).toHaveBeenCalledTimes(1);
    const body = post.mock.calls[0][0];
    expect(body.image_path).toBe(KEY.imagePath);
    expect(body.dataset_root).toBe(KEY.datasetRoot);
    expect(body.subject).toBe("tip");
    expect(body.date).toBe("2026-01-01");
    expect(body.grid).toEqual(GRID);
    expect(body.cells_swept).toEqual(["A1"]);
    expect(body.cells_served_at_native).toEqual(["B1"]);
    // The symbology travels with the cells, or the record cannot say what rendering was seen.
    // postNow always sets viewing; the payload type leaves it optional for other callers.
    const viewing = body.viewing!;
    expect(viewing.bands).toBe("3,2,1");
    expect(viewing.stretch).toBe("percent_clip");
    expect(viewing.stats_source).toEqual({
      read: "overview",
      seed: null,
      pixel_fraction: null,
      overview_scale: 0.5,
    });
    expect(viewing.display_bounds).toEqual([[0, 255]]);
    expect(viewing.base_served_size).toBe("150x100");
    expect(viewing.working_scale_bar).toEqual({
      value: 1,
      source: expect.stringContaining("annotation commits"),
    });
  });

  it("a switch to another identity flushes owed facts under the old key first, then clears", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteServedAtNative("B1");
    tracker.reset({ ...KEY, imagePath: "C:/data/images/2026-01-01/other.tif" }, GRID, CELLS);
    expect(post).toHaveBeenCalledTimes(1);
    expect(post.mock.calls[0][0].image_path).toBe(KEY.imagePath);
    expect(post.mock.calls[0][0].cells_served_at_native).toEqual(["B1"]);
    expect(tracker.servedAtNative.size).toBe(0);
    expect(tracker.swept.size).toBe(0);
  });

  it("stays inert with no identity (subject missing upstream resets to null)", () => {
    tracker.reset(null, null, []);
    tracker.noteAuthoringScale(1);
    tracker.noteViewport(FULL_VIEW, 1);
    tracker.noteServedAtNative("A1");
    vi.advanceTimersByTime(1000);
    expect(post).not.toHaveBeenCalled();
  });

  it("a single-cell grid gets no tracking (image inside the display bound)", () => {
    tracker.reset(KEY, { ...GRID, cols: 1, rows: 1 }, [CELLS[0]]);
    tracker.noteAuthoringScale(1);
    tracker.noteViewport(FULL_VIEW, 1);
    tracker.noteServedAtNative("A1");
    vi.advanceTimersByTime(1000);
    expect(post).not.toHaveBeenCalled();
  });
});

const NULL_VIEWING = {
  bands: null,
  stretch: null,
  stats_source: null,
  display_bounds: null,
  base_served_size: null,
  working_scale_bar: null,
};

describe("CoverageTracker hydration", () => {
  it("adopts a stored record's facts only when its grid matches", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: { ...GRID, tile_size: 50 },
      cells_swept: ["A1"],
      cells_served_at_native: ["B1"],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    expect(tracker.swept.size).toBe(0);
    tracker.hydrate({
      grid: GRID,
      cells_swept: ["A1"],
      cells_served_at_native: ["B1"],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    expect(Array.from(tracker.swept)).toEqual(["A1"]);
    expect(Array.from(tracker.servedAtNative)).toEqual(["B1"]);
  });
});

describe("CoverageTracker complete warning", () => {
  it("applies only when a bar exists and unswept cells remain", () => {
    tracker.reset(KEY, GRID, CELLS);
    expect(tracker.completeWarning()).toBeNull(); // no bar
    tracker.noteAuthoringScale(0.5);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 250, y1: 150 }, 0.5); // sweeps A1, B1
    expect(tracker.completeWarning()).toEqual({ unsweptCount: 4, total: 6, bar: 0.5 });
    tracker.noteViewport(FULL_VIEW, 0.5); // sweeps the rest
    expect(tracker.completeWarning()).toBeNull();
  });
});
