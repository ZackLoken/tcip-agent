import { describe, expect, it } from "vitest";

import type { ServingCell } from "@/api/types.generated";
import { cellsIntersecting, planRegionFetches } from "@/lib/servingGrid";

// A 3x2 lattice of 100px cells over a 300x200 image, listed deliberately out of row-major
// order so ordering must come from the rects, not the list.
const CELLS: ServingCell[] = [
  { name: "B2", x0: 100, y0: 100, x1: 200, y1: 200 },
  { name: "A1", x0: 0, y0: 0, x1: 100, y1: 100 },
  { name: "C1", x0: 200, y0: 0, x1: 300, y1: 100 },
  { name: "A2", x0: 0, y0: 100, x1: 100, y1: 200 },
  { name: "B1", x0: 100, y0: 0, x1: 200, y1: 100 },
  { name: "C2", x0: 200, y0: 100, x1: 300, y1: 200 },
];

describe("cellsIntersecting", () => {
  it("uses open overlap: a cell touching the rect only on an edge is not in it", () => {
    const viewport = { x0: 50, y0: 0, x1: 250, y1: 150 };
    expect(
      cellsIntersecting(CELLS, viewport)
        .map((c) => c.name)
        .sort(),
    ).toEqual(["A1", "A2", "B1", "B2", "C1", "C2"]);
    expect(cellsIntersecting(CELLS, { x0: 0, y0: 0, x1: 100, y1: 100 }).map((c) => c.name)).toEqual(
      ["A1"],
    );
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
