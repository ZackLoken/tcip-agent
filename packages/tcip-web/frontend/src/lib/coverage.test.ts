import { describe, expect, it } from "vitest";

import {
  cellContainedIn,
  cellsIntersecting,
  completeWarningMessage,
  effectiveComplete,
  indexCells,
  planRegionFetches,
  servedCellAtNative,
  stepUnsweptCell,
  subdivideCell,
  sweptFractionBlocks,
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
      cellContainedIn(
        CELLS.find((c) => c.name === "B1")!,
        viewport,
      ),
    ).toBe(true);
    expect(
      cellContainedIn(
        CELLS.find((c) => c.name === "A1")!,
        viewport,
      ),
    ).toBe(false);
    expect(
      cellContainedIn(
        CELLS.find((c) => c.name === "B2")!,
        viewport,
      ),
    ).toBe(false);
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

  it("absorbs the remainder into the last row/column on a non-divisible size", () => {
    const cell: GridCell = { name: "A1", x0: 0, y0: 0, x1: 10, y1: 10 };
    const subs = subdivideCell(cell, 3);
    expect(subs).toHaveLength(9);
    // Every sub-cell still lies fully inside the parent, and the last column/row reaches the edge.
    for (const s of subs) {
      expect(s.x0 >= cell.x0 && s.x1 <= cell.x1 && s.y0 >= cell.y0 && s.y1 <= cell.y1).toBe(true);
    }
    expect(subs[8]).toEqual({ x0: 6, y0: 6, x1: 10, y1: 10 });
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

describe("sweptFractionBlocks", () => {
  it("aggregates k x k cell blocks by swept fraction", () => {
    const swept = new Set(["0,0", "1,0", "0,1"]);
    const blocks = sweptFractionBlocks(4, 4, 2, (col, row) => swept.has(`${col},${row}`));
    expect(blocks.cols).toBe(2);
    expect(blocks.rows).toBe(2);
    expect(blocks.fractions[0]).toBe(3 / 4);
    expect(blocks.fractions[1]).toBe(0);
    expect(blocks.fractions[2]).toBe(0);
    expect(blocks.fractions[3]).toBe(0);
  });

  it("clips edge blocks to the lattice, keeping fractions over real cells only", () => {
    const blocks = sweptFractionBlocks(3, 2, 2, (col) => col === 2);
    // The right edge block covers a single column (2 cells), both swept.
    expect(blocks.cols).toBe(2);
    expect(blocks.rows).toBe(1);
    expect(blocks.fractions[1]).toBe(1);
    expect(blocks.fractions[0]).toBe(0);
  });
});

describe("indexCells", () => {
  it("indexes served cells by column/row from their own origins", () => {
    const index = indexCells(CELLS);
    expect(index.cols).toBe(3);
    expect(index.rows).toBe(2);
    expect(index.at(1, 0)?.name).toBe("B1");
    expect(index.at(2, 1)?.name).toBe("C2");
    expect(index.colX[1]).toEqual([100, 200]);
    expect(index.rowY[1]).toEqual([100, 200]);
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

describe("completeWarningMessage", () => {
  it("states cells and scale, no attention claim", () => {
    const msg = completeWarningMessage({ unsweptCount: 14, total: 35, bar: 0.35 });
    expect(msg).toBe(
      "Complete: 14 of 35 grid cells were never fully on screen at 35% zoom or closer this session.",
    );
  });
});
