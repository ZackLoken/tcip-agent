import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import type { GridCell, GridGeometry } from "@/lib/coverage";
import { CoverageTracker, type CoveragePostBody } from "@/lib/coverageTracker";

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

let post: Mock<(body: CoveragePostBody) => Promise<unknown>>;
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
      stats_source: "overview",
      display_bounds: "0..255",
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
    expect(body.viewing.stats_source).toBe("overview");
    expect(body.viewing.display_bounds).toBe("0..255");
    expect(body.viewing.base_served_size).toBe("150x100");
    expect(body.viewing.working_scale_bar).toEqual({
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

describe("CoverageTracker hydration", () => {
  it("adopts a stored record's facts only when its grid matches", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: { ...GRID, tile_size: 50 },
      cells_swept: ["A1"],
      cells_served_at_native: ["B1"],
    });
    expect(tracker.swept.size).toBe(0);
    tracker.hydrate({ grid: GRID, cells_swept: ["A1"], cells_served_at_native: ["B1"] });
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
