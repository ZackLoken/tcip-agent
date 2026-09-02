import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import type { GridCell, GridGeometry, WorkingScaleBar } from "@/lib/coverage";
import { meetsBar } from "@/lib/coverage";
import {
  CoverageOutbox,
  CoverageTracker,
  coverageOutbox,
  type CoveragePayload,
  type CoveragePushResponse,
} from "@/lib/coverageTracker";
import { resetCoverageOutbox } from "@/test/coverageOutbox";

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
const NULL_VIEWING = {
  bands: null,
  stretch: null,
  stats_source: null,
  display_bounds: null,
  base_served_size: null,
};

function bar(value: number): WorkingScaleBar {
  return {
    value,
    median_extent_native_px: 46 / value,
    annotation_count: 1,
    judged_span_px: 46,
    source: "s",
  };
}

let post: Mock<(body: CoveragePayload) => Promise<CoveragePushResponse>>;
let tracker: CoverageTracker;

beforeEach(() => {
  vi.useFakeTimers();
  // Echoes back what was posted, the ordinary case with no other tab or session contributing.
  post = vi.fn((body: CoveragePayload) =>
    Promise.resolve({ record: { cells_seen_at_scale: body.cells_seen_at_scale ?? {} } }),
  );
  tracker = new CoverageTracker(post);
});

afterEach(() => {
  resetCoverageOutbox();
  vi.useRealTimers();
});

describe("CoverageTracker seen / swept", () => {
  it("accumulates seen cells even with no bar set (no bar means no swept)", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport(FULL_VIEW, 1);
    expect(tracker.seenAtScale.size).toBe(6);
    expect(tracker.swept.size).toBe(0);
    vi.advanceTimersByTime(1000);
    expect(post).toHaveBeenCalledTimes(1);
  });

  it("swept is derived: a bar meets some cells and not others by value", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 250, y1: 150 }, 0.5); // seen: A1, B1
    tracker.setWorkingScaleBar(bar(0.5));
    expect(Array.from(tracker.swept).sort()).toEqual(["A1", "B1"]);
    tracker.setWorkingScaleBar(bar(0.9));
    expect(tracker.swept.size).toBe(0);
  });

  it("meetsBar is the one comparison, tested at equality", () => {
    expect(meetsBar(0.5, bar(0.5))).toBe(true);
    expect(meetsBar(0.49, bar(0.5))).toBe(false);
    expect(meetsBar(null, bar(0.5))).toBe(false);
    expect(meetsBar(0.5, null)).toBe(false);
  });

  it("a lowered bar derives more cells swept; a raised bar un-derives one without losing its facts", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport(FULL_VIEW, 0.5);
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.swept.size).toBe(6);
    tracker.setWorkingScaleBar(bar(0.9));
    expect(tracker.swept.size).toBe(0);
    expect(tracker.seenAtScale.size).toBe(6); // the facts persist, only the derivation changed
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.swept.size).toBe(6);
  });

  it("a cell seen below the bar derives as swept once the bar lowers", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.3); // A1 only, at 0.3
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.swept.has("A1")).toBe(false);
    tracker.setWorkingScaleBar(bar(0.3));
    expect(tracker.swept.has("A1")).toBe(true);
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

  it("becomes seen once every sub-cell's union has been on screen, never at one whole pass", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 128, y1: 512 }, 1);
    expect(tracker.seenAtScale.has("A1")).toBe(false);
    tracker.noteViewport({ x0: 128, y0: 0, x1: 256, y1: 512 }, 1);
    tracker.noteViewport({ x0: 256, y0: 0, x1: 384, y1: 512 }, 1);
    expect(tracker.seenAtScale.has("A1")).toBe(false); // union so far: x in [0, 384) only
    tracker.noteViewport({ x0: 384, y0: 0, x1: 512, y1: 512 }, 1); // reaches the far edge
    expect(tracker.seenAtScale.get("A1")).toBe(1);
  });

  it("never becomes seen through a viewport narrower than one sub-cell, however densely it pans", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    for (let x0 = 0; x0 < 512; x0 += 10) {
      tracker.noteViewport({ x0, y0: 0, x1: Math.min(x0 + 100, 512), y1: 512 }, 1);
    }
    expect(tracker.seenAtScale.has("A1")).toBe(false);
    for (let x0 = 0; x0 < 512; x0 += 128) {
      tracker.noteViewport({ x0, y0: 0, x1: x0 + 128, y1: 512 }, 1);
    }
    expect(tracker.seenAtScale.has("A1")).toBe(true);
  });

  it("a later pass raising one sub-cell's max raises the cell's own recorded bound", () => {
    tracker.reset(KEY, BIG_GRID, [BIG_CELL, OTHER_CELL]);
    for (let x0 = 0; x0 < 512; x0 += 128) {
      tracker.noteViewport({ x0, y0: 0, x1: x0 + 128, y1: 512 }, 0.4);
    }
    expect(tracker.seenAtScale.get("A1")).toBe(0.4);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 128, y1: 512 }, 0.9);
    expect(tracker.seenAtScale.get("A1")).toBe(0.4); // the min over sub-cells is unchanged
    for (let x0 = 128; x0 < 512; x0 += 128) {
      tracker.noteViewport({ x0, y0: 0, x1: x0 + 128, y1: 512 }, 0.9);
    }
    expect(tracker.seenAtScale.get("A1")).toBe(0.9); // now every sub-cell's own max is 0.9
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
  it("debounces facts into one POST carrying both fact maps and the viewing context", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing({
      bands: "3,2,1",
      stretch: "percent_clip",
      stats_source: { read: "overview", seed: null, pixel_fraction: null, overview_scale: 0.5 },
      display_bounds: [[0, 255]],
      base_served_size: "150x100",
    });
    tracker.noteViewport({ x0: 0, y0: 0, x1: 120, y1: 120 }, 1); // contains A1 only
    tracker.noteServedAtNative("B1");
    expect(post).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1);
    const body = post.mock.calls[0][0];
    expect(body.image_path).toBe(KEY.imagePath);
    expect(body.dataset_root).toBe(KEY.datasetRoot);
    expect(body.subject).toBe("tip");
    expect(body.date).toBe("2026-01-01");
    expect(body.grid).toEqual(GRID);
    expect(body.cells_seen_at_scale).toEqual({ A1: 1 });
    expect(body.cells_served_at_native).toEqual(["B1"]);
    const viewing = body.viewing;
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
  });

  it("an acknowledged response moves a cell from pending to recorded", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 1);
    expect(tracker.pending.has("A1")).toBe(true);
    await vi.advanceTimersByTimeAsync(400);
    expect(tracker.pending.has("A1")).toBe(false);
  });

  it("a cell raised past its recorded value is pending again", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.5);
    await vi.advanceTimersByTimeAsync(400);
    expect(tracker.pending.has("A1")).toBe(false);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.9);
    expect(tracker.pending.has("A1")).toBe(true);
  });

  it("a response carrying more than the body credits cells another tab or session contributed", async () => {
    post.mockResolvedValueOnce({
      record: { cells_seen_at_scale: { A1: 1, C1: 0.6 } }, // C1 rode in from elsewhere
    });
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 1); // posts A1 alone
    await vi.advanceTimersByTimeAsync(400);
    expect(post.mock.calls[0][0].cells_seen_at_scale).toEqual({ A1: 1 });
    // C1 was never locally seen, but the response's own record credits it as seen and recorded.
    expect(tracker.seenAtScale.get("C1")).toBe(0.6);
    expect(tracker.pending.has("C1")).toBe(false);
  });

  it("a push is triggered by a newly seen cell, a newly served cell and a viewing change, not by an at_scale rise alone", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.5); // A1 newly seen
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1);

    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.9); // A1 rises, already seen
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1); // no push from the rise alone

    tracker.noteServedAtNative("B1"); // a newly served cell does trigger one
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(2);
    // The raised value rode along with this later push.
    expect(post.mock.calls[1][0].cells_seen_at_scale).toEqual({ A1: 0.9 });
  });

  it("a rise sets dirty without scheduling a push, so a flush before any later event still carries it", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.5); // A1 newly seen
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1);

    tracker.noteViewport({ x0: 0, y0: 0, x1: 100, y1: 100 }, 0.9); // A1 rises, already seen
    tracker.flush(); // a navigation right after the rise, with no other event in between
    expect(post).toHaveBeenCalledTimes(2);
    expect(post.mock.calls[1][0].cells_seen_at_scale).toEqual({ A1: 0.9 });
  });

  it("a push triggered by a viewing-only change with no cell delta still carries the facts", async () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    await vi.advanceTimersByTimeAsync(400);
    post.mockClear();
    tracker.setViewing({ ...NULL_VIEWING, stretch: "minmax" });
    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1);
  });

  it("a failed push reports itself and retries on the next debounce tick, never silently", async () => {
    const onPushError = vi.fn();
    const failingTracker = new CoverageTracker(post, { onPushError });
    post
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue({ record: { cells_seen_at_scale: {} } });
    failingTracker.reset(KEY, GRID, CELLS);
    failingTracker.setViewing(NULL_VIEWING);
    failingTracker.noteServedAtNative("B1");

    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(1);
    expect(onPushError).toHaveBeenCalledWith("network down");

    await vi.advanceTimersByTimeAsync(400);
    expect(post).toHaveBeenCalledTimes(2);
  });

  it("stays inert with no identity (subject missing upstream resets to null)", () => {
    tracker.reset(null, null, []);
    tracker.setWorkingScaleBar(bar(1));
    tracker.noteViewport(FULL_VIEW, 1);
    tracker.noteServedAtNative("A1");
    vi.advanceTimersByTime(1000);
    expect(post).not.toHaveBeenCalled();
  });

  it("a single-cell grid gets no tracking (image inside the display bound)", () => {
    tracker.reset(KEY, { ...GRID, cols: 1, rows: 1 }, [CELLS[0]]);
    tracker.noteViewport(FULL_VIEW, 1);
    tracker.noteServedAtNative("A1");
    vi.advanceTimersByTime(1000);
    expect(post).not.toHaveBeenCalled();
  });
});

describe("CoverageTracker synchronous flush", () => {
  it("a reset posts the outgoing image's owed facts immediately, under its own key, never deferred by the outbox's own delay", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteServedAtNative("A1");

    tracker.reset({ ...KEY, imagePath: "C:/data/images/2026-01-01/other.tif" }, GRID, CELLS);
    expect(post).toHaveBeenCalledTimes(1);
    expect(post.mock.calls[0][0].image_path).toBe(KEY.imagePath);
    expect(coverageOutbox.size).toBe(0); // it succeeded: never touched the outbox at all
  });

  it("reset clears viewing, so a push under a new identity can never carry the previous image's context", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteServedAtNative("A1");
    vi.advanceTimersByTime(400); // the first identity's own push lands, viewing intact
    post.mockClear();

    tracker.reset({ ...KEY, imagePath: "C:/data/images/2026-01-01/other.tif" }, GRID, CELLS);
    tracker.noteServedAtNative("B1"); // before setViewing runs again for the new identity
    vi.advanceTimersByTime(400);
    expect(post).not.toHaveBeenCalled(); // no viewing yet for this identity: nothing to post
  });
});

describe("CoverageTracker outbox", () => {
  it("a failed push at reset time posts once immediately, then is retried from the outbox under its own identity", async () => {
    post
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({ record: { cells_seen_at_scale: {} } });
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteServedAtNative("A1");

    tracker.reset(null, null, []); // flush() posts the outgoing image's facts at once
    expect(post).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(0); // let the rejection reach the outbox
    expect(coverageOutbox.size).toBeGreaterThan(0);
    await vi.advanceTimersByTimeAsync(5000); // the outbox's own first retry: succeeds
    expect(post).toHaveBeenCalledTimes(2);
    expect(coverageOutbox.size).toBe(0);
  });

  it("a 4xx answer drops the queued payload and the next one proceeds", async () => {
    const refusal = new StructuredRefusalError({ message: "unknown cell" }, 400, "unknown cell");
    post.mockRejectedValue(refusal);
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteServedAtNative("A1");
    tracker.reset(null, null, []);
    expect(post).toHaveBeenCalledTimes(1); // flush()'s own immediate attempt, refused
    await vi.advanceTimersByTimeAsync(5000); // the outbox's own retry, refused again: terminal
    expect(post).toHaveBeenCalledTimes(2);
    expect(coverageOutbox.size).toBe(0);
  });

  it("the audit-gap 500 is terminal by its own marker, not retried like an ordinary 5xx", async () => {
    const auditGap = new StructuredRefusalError(
      { error: "audit_entry_not_written", message: "disk full" },
      500,
      "disk full",
    );
    post.mockRejectedValue(auditGap);
    const onPushError = vi.fn();
    const withOutbox = new CoverageTracker(post, { onPushError });
    withOutbox.reset(KEY, GRID, CELLS);
    withOutbox.setViewing(NULL_VIEWING);
    withOutbox.noteServedAtNative("A1");
    withOutbox.reset(null, null, []);
    expect(post).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5000); // the outbox's own retry, refused again: terminal
    expect(post).toHaveBeenCalledTimes(2);
    expect(coverageOutbox.size).toBe(0);
    expect(onPushError).toHaveBeenCalledWith(
      expect.stringContaining("saved without its audit line"),
    );
  });

  it("a network failure stays queued and is retried, never dropped", async () => {
    post.mockRejectedValue(new TypeError("Failed to fetch"));
    tracker.reset(KEY, GRID, CELLS);
    tracker.setViewing(NULL_VIEWING);
    tracker.noteServedAtNative("A1");
    tracker.reset(null, null, []);
    await vi.advanceTimersByTimeAsync(0); // let flush()'s own failed attempt reach the outbox
    const owed = coverageOutbox.size;
    expect(owed).toBeGreaterThan(0);
    await vi.advanceTimersByTimeAsync(5000);
    expect(coverageOutbox.size).toBe(owed);
  });

  it("drain reschedules rather than stalling when no postFn is configured yet", async () => {
    const freshOutbox = new CoverageOutbox();
    freshOutbox.enqueue({
      image_path: KEY.imagePath,
      dataset_root: KEY.datasetRoot,
      subject: KEY.subject,
      date: KEY.date,
      grid: GRID,
      cells_served_at_native: [],
      cells_seen_at_scale: {},
      viewing: NULL_VIEWING,
    });
    await vi.advanceTimersByTimeAsync(5000); // the first attempt: no postFn, must reschedule
    expect(freshOutbox.size).toBe(1); // still queued, not silently stuck forever either

    freshOutbox.configure(post, () => {});
    await vi.advanceTimersByTimeAsync(5000); // the rescheduled attempt: now configured, succeeds
    expect(post).toHaveBeenCalledTimes(1);
    expect(freshOutbox.size).toBe(0);
  });
});

describe("CoverageTracker hydration", () => {
  it("adopts a stored record's facts only when its grid matches", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: { ...GRID, tile_size: 50 },
      cells_seen_at_scale: { A1: 1 },
      cells_served_at_native: ["B1"],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    expect(tracker.seenAtScale.size).toBe(0);
    tracker.hydrate({
      grid: GRID,
      cells_seen_at_scale: { A1: 0.7 },
      cells_served_at_native: ["B1"],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    expect(tracker.seenAtScale.get("A1")).toBe(0.7);
    expect(Array.from(tracker.servedAtNative)).toEqual(["B1"]);
    expect(tracker.pending.has("A1")).toBe(false); // hydrated = already acknowledged
  });

  it("a hydrated cell derives as swept under a bar at or below it, and not above it", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: GRID,
      cells_seen_at_scale: { A1: 0.5 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.swept.has("A1")).toBe(true);
    tracker.setWorkingScaleBar(bar(0.6));
    expect(tracker.swept.has("A1")).toBe(false);
  });
});

describe("CoverageTracker coarserCount", () => {
  it("counts recorded cells below the bar, never a locally-seen not-yet-saved cell", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: GRID,
      cells_seen_at_scale: { A1: 0.3 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.coarserCount).toBe(1); // A1 recorded at 0.3, below the 0.5 bar

    tracker.noteViewport({ x0: 100, y0: 0, x1: 200, y1: 100 }, 0.2); // B1 seen locally, pending
    expect(tracker.coarserCount).toBe(1); // B1 is not yet on record, so it never counts here
  });

  it("is zero once every recorded cell meets the bar", () => {
    tracker.reset(KEY, GRID, CELLS);
    tracker.hydrate({
      grid: GRID,
      cells_seen_at_scale: { A1: 0.5 },
      cells_served_at_native: [],
      viewing: NULL_VIEWING,
      updated_at: "2026-01-01T00:00:00+00:00",
    });
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.coarserCount).toBe(0);
  });
});

describe("CoverageTracker complete warning", () => {
  it("applies only when a bar exists and unmet cells remain", () => {
    tracker.reset(KEY, GRID, CELLS);
    expect(tracker.completeWarning()).toBeNull(); // no bar
    tracker.noteViewport({ x0: 0, y0: 0, x1: 250, y1: 150 }, 0.5); // seen: A1, B1
    tracker.setWorkingScaleBar(bar(0.5));
    expect(tracker.completeWarning()).toEqual({ unsweptCount: 4, total: 6, bar: 0.5 });
    tracker.noteViewport(FULL_VIEW, 0.5); // seen: the rest
    expect(tracker.completeWarning()).toBeNull();
  });
});
