/**
 * Pure helpers over the region-serving grid a raster's route serves (GET
 * /api/images/serving_grid); nothing here re-derives cells from the geometry.
 */

import type { ServingCell } from "@/api/types.generated";
import type { HostSize, PixelRect } from "@/lib/viewGeometry";

/** Whether two half-open pixel rects share any area (touching edges don't count). */
export function rectsOverlap(a: PixelRect, b: PixelRect): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

export function cellsIntersecting(cells: ServingCell[], rect: PixelRect): ServingCell[] {
  return cells.filter((c) => rectsOverlap(c, rect));
}

/** One planned cell-aligned region serve. */
export interface RegionFetch {
  cell: ServingCell;
  /** Requested output width for the cell, from the power-of-two tier ladder. */
  maxWidth: number;
  /** Output pixels this serve returns at the planned tier. */
  outputPixels: number;
}

/**
 * Plan the cell-aligned region fetches a viewport needs, resolution-tiered: each intersecting
 * cell is requested at the smallest power-of-two downscale of its native size that still meets
 * the on-screen resolution, so repeated zoom passes re-hit the same cached serves. Returns null
 * when the viewport straddles more cells than one of this size can at this scale (a runaway
 * trigger, not a working state). Otherwise the plan admits cells by descending viewport
 * intersection until an output-pixel budget of four times the host area is spent; the
 * largest-intersection cell is always admitted, since at the native tier a single cell can
 * exceed any small screen's budget and serving it is the whole point. Decoded-bitmap memory
 * thereby scales with the actual display, and cells the budget defers are served as the
 * viewport moves onto them.
 */
export function planRegionFetches(args: {
  cells: ServingCell[];
  viewport: PixelRect;
  /** Current view scale (screen px per native px). */
  scale: number;
  /** The base bitmap's served resolution (served width / native width). */
  baseScale: number;
  host: HostSize;
  tileSize: number;
}): RegionFetch[] | null {
  if (args.scale <= args.baseScale || args.tileSize <= 0) return [];
  const hits = cellsIntersecting(args.cells, args.viewport);
  if (hits.length === 0) return [];
  const tileScreen = Math.max(1, args.tileSize * args.scale);
  const maxCells =
    (Math.ceil(args.host.w / tileScreen) + 1) * (Math.ceil(args.host.h / tileScreen) + 1);
  if (hits.length > maxCells) return null;
  const needed = Math.min(1, args.scale);
  let tier = 1;
  while (tier / 2 >= needed) tier /= 2;
  const area = (cell: ServingCell) =>
    Math.max(0, Math.min(cell.x1, args.viewport.x1) - Math.max(cell.x0, args.viewport.x0)) *
    Math.max(0, Math.min(cell.y1, args.viewport.y1) - Math.max(cell.y0, args.viewport.y0));
  const ordered = [...hits].sort((a, b) => area(b) - area(a));
  const budget = 4 * args.host.w * args.host.h;
  const plan: RegionFetch[] = [];
  let total = 0;
  for (const cell of ordered) {
    const w = Math.max(1, Math.ceil((cell.x1 - cell.x0) * tier));
    const h = Math.max(1, Math.ceil((cell.y1 - cell.y0) * tier));
    const outputPixels = w * h;
    if (plan.length > 0 && total + outputPixels > budget) continue;
    plan.push({ cell, maxWidth: w, outputPixels });
    total += outputPixels;
  }
  return plan;
}
