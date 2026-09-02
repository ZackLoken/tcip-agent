import { describe, expect, it } from "vitest";

import {
  breederReadErrorReason,
  cellAt,
  cellsIntersecting,
  completeWarningMessage,
  currentCoverageCell,
  effectiveComplete,
  meetsBar,
  noWorkingScaleToast,
  planRegionFetches,
  replaceRequiredToastSentence,
  rectFullyInside,
  servedCellAtNative,
  stepUnsweptCell,
  subCellDivisionsFor,
  subdivideCell,
  type CompletenessRecord,
  type GridCell,
} from "@/lib/coverage";

// A 3x2 lattice of 100px cells over a 300x200 image, listed deliberately out of row-major
// order so ordering must come from the rects, not the list.
const CELLS: GridCell[] = [
  { name: "B2", x0: 100, y0: 100, x1: 200, y1: 200 },
  { name: "A1", x0: 0, y0: 0, x1: 100, y1: 100 },
  { name: "C1", x0: 200, y0: 0, x1: 300, y1: 100 },
  { name: "A2", x0: 0, y0: 100, x1: 100, y1: 200 },
  { name: "B1", x0: 100, y0: 0, x1: 200, y1: 100 },
  { name: "C2", x0: 200, y0: 100, x1: 300, y1: 200 },
];

describe("cell rect predicates", () => {
  it("intersecting uses open overlap, containment uses closed edges", () => {
    const viewport = { x0: 50, y0: 0, x1: 250, y1: 150 };
    expect(
      cellsIntersecting(CELLS, viewport)
        .map((c) => c.name)
        .sort(),
    ).toEqual(["A1", "A2", "B1", "B2", "C1", "C2"]);
    expect(
      rectFullyInside(
        CELLS.find((c) => c.name === "B1")!,
        viewport,
      ),
    ).toBe(true);
    expect(
      rectFullyInside(
        CELLS.find((c) => c.name === "A1")!,
        viewport,
      ),
    ).toBe(false);
    expect(
      rectFullyInside(
        CELLS.find((c) => c.name === "B2")!,
        viewport,
      ),
    ).toBe(false);
  });
});

describe("cellAt", () => {
  it("finds the cell a point falls in, the Map tool's click and the chrome's current-cell lookup", () => {
    expect(cellAt(CELLS, 150, 150)?.name).toBe("B2");
    expect(cellAt(CELLS, 0, 0)?.name).toBe("A1");
  });

  it("respects the half-open convention: a cell's own x1/y1 belongs to its neighbor", () => {
    expect(cellAt(CELLS, 100, 50)?.name).toBe("B1");
    expect(cellAt(CELLS, 99, 50)?.name).toBe("A1");
  });

  it("returns null outside every cell", () => {
    expect(cellAt(CELLS, 1000, 1000)).toBeNull();
    expect(cellAt([], 1, 1)).toBeNull();
  });
});

describe("currentCoverageCell", () => {
  it("names the clicked cell while any part of it is still in the viewport, even off-center", () => {
    // A corner-cell jump whose padded, edge-clamped view centers on the lattice's middle row:
    // the viewport's own centre falls in B1, but A1 (the clicked cell) still overlaps it.
    const a1 = CELLS.find((c) => c.name === "A1")!;
    const viewport = { x0: -50, y0: -20, x1: 250, y1: 180 };
    expect(
      cellAt(CELLS, (viewport.x0 + viewport.x1) / 2, (viewport.y0 + viewport.y1) / 2)?.name,
    ).toBe("B1");
    expect(currentCoverageCell(CELLS, viewport, a1)?.name).toBe("A1");
  });

  it("reverts to the viewport-centre cell once a pan or Overview leaves the clicked cell", () => {
    const a1 = CELLS.find((c) => c.name === "A1")!;
    const viewport = { x0: 150, y0: 100, x1: 300, y1: 200 }; // A1 no longer overlaps
    expect(currentCoverageCell(CELLS, viewport, a1)?.name).toBe("C2");
  });

  it("falls back to the viewport centre with no Map selection at all", () => {
    const viewport = { x0: 0, y0: 0, x1: 100, y1: 100 };
    expect(currentCoverageCell(CELLS, viewport, null)?.name).toBe("A1");
  });

  it("null with no viewport measured yet, whether or not a cell was clicked", () => {
    const a1 = CELLS.find((c) => c.name === "A1")!;
    expect(currentCoverageCell(CELLS, null, a1)).toBeNull();
  });
});

describe("subdivideCell", () => {
  it("tiles the cell exactly, gapless, at an evenly-divisible size", () => {
    const cell: GridCell = { name: "A1", x0: 0, y0: 0, x1: 100, y1: 100 };
    const subs = subdivideCell(cell, 4);
    expect(subs).toHaveLength(16);
    expect(subs[0]).toEqual({ x0: 0, y0: 0, x1: 25, y1: 25 });
    expect(subs[5]).toEqual({ x0: 25, y0: 25, x1: 50, y1: 50 }); // row 1, col 1
    expect(subs[15]).toEqual({ x0: 75, y0: 75, x1: 100, y1: 100 }); // last row/col reaches the edge
    // Every row spans the cell's full width with no gap or overlap between columns.
    for (let row = 0; row < 4; row++) {
      const rowSubs = subs.slice(row * 4, row * 4 + 4).sort((a, b) => a.x0 - b.x0);
      expect(rowSubs[0].x0).toBe(cell.x0);
      expect(rowSubs[3].x1).toBe(cell.x1);
      for (let i = 1; i < 4; i++) expect(rowSubs[i].x0).toBe(rowSubs[i - 1].x1);
    }
  });

  it("tiles a non-divisible size gaplessly, every sub-cell fully inside the parent", () => {
    const cell: GridCell = { name: "A1", x0: 0, y0: 0, x1: 10, y1: 10 };
    const subs = subdivideCell(cell, 3);
    expect(subs).toHaveLength(9);
    for (const s of subs) {
      expect(s.x0 >= cell.x0 && s.x1 <= cell.x1 && s.y0 >= cell.y0 && s.y1 <= cell.y1).toBe(true);
    }
    expect(subs[8]).toEqual({ x0: 6, y0: 6, x1: 10, y1: 10 }); // last sub-cell reaches the exact edge
  });

  it("distributes any remainder across the grid rather than concentrating it in one final cell", () => {
    // 100 / 32 = 3.125: floor-indexed boundaries put the extra pixel in several scattered
    // sub-cells (widths 3 or 4), not piled entirely into the last row/column.
    const cell: GridCell = { name: "A1", x0: 0, y0: 0, x1: 100, y1: 100 };
    const subs = subdivideCell(cell, 32);
    const rowWidths = subs.slice(0, 32).map((s) => s.x1 - s.x0);
    const oversized = rowWidths.filter((w) => w === 4).length;
    expect(oversized).toBeGreaterThan(1); // more than just the last slot
    expect(rowWidths[31]).toBe(4); // the last slot is still one of them, snapped to the exact edge
    expect(rowWidths.reduce((a, b) => a + b, 0)).toBe(100);
  });
});

describe("subCellDivisionsFor", () => {
  it("keeps sub-cell size roughly constant across very different cell sizes", () => {
    const small: GridCell = { name: "A1", x0: 0, y0: 0, x1: 4096, y1: 4096 }; // ordinary lattice
    const large: GridCell = { name: "A1", x0: 0, y0: 0, x1: 14996, y1: 8000 }; // large-raster lattice

    const smallDivisions = subCellDivisionsFor(small, 128);
    const largeDivisions = subCellDivisionsFor(large, 128);
    expect(smallDivisions).toBe(32); // 4096 / 128 exactly: the value this platform already shipped
    expect(largeDivisions).toBe(118); // ceil(14996 / 128), the long edge, not the short one

    const smallSubEdge = (small.x1 - small.x0) / smallDivisions;
    const largeSubEdge = (large.x1 - large.x0) / largeDivisions;
    expect(smallSubEdge).toBeLessThanOrEqual(128);
    expect(largeSubEdge).toBeLessThanOrEqual(128); // no longer ~469px: the bug this fixes
  });

  it("never returns fewer than 1 division, even for a cell smaller than the target", () => {
    const tiny: GridCell = { name: "A1", x0: 0, y0: 0, x1: 40, y1: 40 };
    expect(subCellDivisionsFor(tiny, 128)).toBe(1);
  });
});

describe("stepUnsweptCell", () => {
  const none = new Set<string>();
  const from = { x: 150, y: 50 }; // center of B1

  it("steps forward and backward in row-major order from the nearest cell", () => {
    expect(stepUnsweptCell(CELLS, none, from, 1)?.name).toBe("C1");
    expect(stepUnsweptCell(CELLS, none, from, -1)?.name).toBe("A1");
  });

  it("wraps at the lattice ends", () => {
    expect(stepUnsweptCell(CELLS, none, { x: 250, y: 150 }, 1)?.name).toBe("A1");
    expect(stepUnsweptCell(CELLS, none, { x: 50, y: 50 }, -1)?.name).toBe("C2");
  });

  it("skips swept cells", () => {
    expect(stepUnsweptCell(CELLS, new Set(["C1", "A2"]), from, 1)?.name).toBe("B2");
  });

  it("reports null when every cell is swept (the caller states that fact)", () => {
    expect(stepUnsweptCell(CELLS, new Set(CELLS.map((c) => c.name)), from, 1)).toBeNull();
  });
});

describe("planRegionFetches", () => {
  const host = { w: 200, h: 200 };

  it("returns nothing while the base bitmap already carries the on-screen resolution", () => {
    expect(
      planRegionFetches({
        cells: CELLS,
        viewport: { x0: 0, y0: 0, x1: 300, y1: 200 },
        scale: 0.1,
        baseScale: 0.2,
        host,
        tileSize: 100,
      }),
    ).toEqual([]);
  });

  it("requests each intersecting cell at the smallest power-of-two tier meeting the zoom", () => {
    const plan = planRegionFetches({
      cells: CELLS,
      viewport: { x0: 120, y0: 20, x1: 180, y1: 80 }, // inside B1
      scale: 0.3,
      baseScale: 0.05,
      host,
      tileSize: 100,
    });
    expect(plan).not.toBeNull();
    expect(plan!.map((p) => p.cell.name)).toEqual(["B1"]);
    // 0.3 needs the 1/2 tier (1/4 would undershoot): 100px cell at 1/2 is a 50px serve.
    expect(plan![0].maxWidth).toBe(50);
  });

  it("requests native (tier 1) once zoom passes native resolution", () => {
    const plan = planRegionFetches({
      cells: CELLS,
      viewport: { x0: 120, y0: 20, x1: 180, y1: 80 },
      scale: 2,
      baseScale: 0.05,
      host,
      tileSize: 100,
    });
    expect(plan![0].maxWidth).toBe(100);
  });

  it("does not fan out past the straddle-count cap", () => {
    // A viewport claiming more cells than a host this size can straddle at this scale.
    const plan = planRegionFetches({
      cells: CELLS,
      viewport: { x0: 0, y0: 0, x1: 300, y1: 200 },
      scale: 1,
      baseScale: 0.05,
      host: { w: 100, h: 100 },
      tileSize: 100,
    });
    expect(plan).toBeNull();
  });

  it("always admits the largest-intersection cell and defers the rest past the budget", () => {
    // A tiny host's budget is far below one native 100px cell: only the larger-intersection
    // cell serves now; the other waits for the viewport to move onto it.
    const plan = planRegionFetches({
      cells: CELLS,
      viewport: { x0: 110, y0: 20, x1: 280, y1: 80 }, // 90px of B1, 80px of C1
      scale: 1,
      baseScale: 0.05,
      host: { w: 10, h: 10 },
      tileSize: 100,
    });
    expect(plan!.map((p) => p.cell.name)).toEqual(["B1"]);
  });
});

describe("servedCellAtNative", () => {
  const cell = { name: "B1", x0: 100, y0: 0, x1: 200, y1: 100 };

  it("marks only a response whose served size equals the cell's native dims", () => {
    expect(servedCellAtNative(cell, { w: 100, h: 100 })).toBe(true);
    expect(servedCellAtNative(cell, { w: 50, h: 50 })).toBe(false);
    expect(servedCellAtNative(cell, null)).toBe(false);
  });
});

describe("effectiveComplete", () => {
  const GRID = { width: 300, height: 200, tile_size: 100, overlap: 0, cols: 3, rows: 2 };

  function record(overrides: Partial<CompletenessRecord> = {}): CompletenessRecord {
    return {
      grid: GRID,
      cells_complete: ["A1", "B2"],
      attested_by: "user:z",
      attested_at: "t",
      stem: "mosaic",
      date: null,
      subject: "bush",
      stale_cells: [],
      ...overrides,
    };
  }

  it("undefined record yields no complete cells", () => {
    expect(effectiveComplete(undefined)).toEqual(new Set());
  });

  it("with no stale cells, every attested cell is effectively complete", () => {
    expect(effectiveComplete(record())).toEqual(new Set(["A1", "B2"]));
  });

  it("a stale cell is excluded even though it is still in cells_complete", () => {
    expect(effectiveComplete(record({ stale_cells: ["A1"] }))).toEqual(new Set(["B2"]));
  });

  it("every cell stale yields an empty set", () => {
    expect(effectiveComplete(record({ stale_cells: ["A1", "B2"] }))).toEqual(new Set());
  });
});

describe("meetsBar", () => {
  const bar = {
    value: 0.5,
    median_extent_native_px: 92,
    annotation_count: 2,
    judged_span_px: 46,
    source: "s",
  };

  it("is exactly the equality boundary: a scale equal to the bar meets it", () => {
    expect(meetsBar(0.5, bar)).toBe(true);
  });

  it("a scale just below the bar does not meet it", () => {
    expect(meetsBar(0.499, bar)).toBe(false);
  });

  it("a scale above the bar meets it", () => {
    expect(meetsBar(0.9, bar)).toBe(true);
  });

  it("null on either side never meets it", () => {
    expect(meetsBar(null, bar)).toBe(false);
    expect(meetsBar(0.9, null)).toBe(false);
    expect(meetsBar(null, null)).toBe(false);
  });
});

describe("completeWarningMessage", () => {
  it("states cells and scale, no attention claim", () => {
    const msg = completeWarningMessage({ unsweptCount: 14, total: 35, bar: 0.35 });
    expect(msg).toBe(
      "Complete: 14 of 35 grid cells have not had every part on screen at 35.0% zoom or " +
        "closer, in any combination of views.",
    );
  });
});

describe("noWorkingScaleToast", () => {
  it("names the subject and the reason, both from the same read", () => {
    expect(noWorkingScaleToast("fruit", "no saved box or polygon annotation of fruit")).toBe(
      "Complete: no working scale for fruit on this image (no saved box or polygon " +
        "annotation of fruit), so coverage was not checked",
    );
  });

  it("states the reason alone with no active subject, never a literal null", () => {
    expect(noWorkingScaleToast(null, "no active subject")).toBe(
      "Complete: no active subject, so coverage was not checked",
    );
  });
});

describe("replaceRequiredToastSentence", () => {
  it("pluralizes by count", () => {
    expect(replaceRequiredToastSentence(1)).toContain("1 cell seen on a previous lattice");
    expect(replaceRequiredToastSentence(3)).toContain("3 cells seen on a previous lattice");
  });
});

describe("breederReadErrorReason", () => {
  it("strips the reader's dict dump after the colon and brace", () => {
    const raw = "record 0 carries no string subject: {'id': 1, 'image_id': 1, 'category_id': 0}";
    expect(breederReadErrorReason(raw)).toBe("record 0 carries no string subject");
  });

  it("passes a reason with no brace through unchanged", () => {
    expect(breederReadErrorReason("plot.json: not valid JSON")).toBe("plot.json: not valid JSON");
  });
});
