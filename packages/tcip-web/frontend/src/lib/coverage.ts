/**
 * Pure helpers over the coverage lattice a raster's grid route serves. The lattice itself always
 * comes from GET /api/coverage/grid; nothing here re-derives cells from the geometry.
 */

import type { HostSize, PixelRect } from "@/lib/viewGeometry";

/** The serializable grid parameter tuple every coverage consumer echoes. */
export interface GridGeometry {
  width: number;
  height: number;
  tile_size: number;
  overlap: number;
  cols: number;
  rows: number;
}

/** One served cell: a name plus its half-open native-pixel rect, clipped to the image extent. */
export interface GridCell {
  name: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface CoverageGridResponse extends GridGeometry {
  cells: GridCell[];
}

/** One subject's region-completeness record, as GET /api/coverage/completeness returns it
 *  (per subject, in `by_subject`). `stale_cells` is recomputed server-side on every read: an
 *  attested cell whose annotation content has since been edited or deleted. */
export interface CompletenessRecord {
  grid: GridGeometry;
  cells_complete: string[];
  attested_by: string | null;
  attested_at: string | null;
  stem: string;
  date: string | null;
  subject: string;
  stale_cells: string[];
}

/** Cells genuinely complete right now: attested, minus any that have since gone stale. A stale
 *  attestation must never render as if it still held. */
export function effectiveComplete(record: CompletenessRecord | undefined): Set<string> {
  if (!record) return new Set();
  const stale = new Set(record.stale_cells);
  return new Set(record.cells_complete.filter((c) => !stale.has(c)));
}

/** One double-click: toggle one cell's completeness for a subject. POST /api/coverage/completeness. */
export interface CompletenessTogglePostBody {
  image_path: string;
  dataset_root: string | null;
  subject: string;
  grid: GridGeometry;
  cell: string;
  user?: string | null;
}

export function sameGrid(a: GridGeometry, b: GridGeometry): boolean {
  return (
    a.width === b.width &&
    a.height === b.height &&
    a.tile_size === b.tile_size &&
    a.overlap === b.overlap &&
    a.cols === b.cols &&
    a.rows === b.rows
  );
}

/** Whether two half-open pixel rects share any area (touching edges don't count). */
export function rectsOverlap(a: PixelRect, b: PixelRect): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

/** Whether `inner` sits entirely inside `outer` (both half-open pixel rects). */
export function rectFullyInside(inner: PixelRect, outer: PixelRect): boolean {
  return (
    inner.x0 >= outer.x0 && inner.y0 >= outer.y0 && inner.x1 <= outer.x1 && inner.y1 <= outer.y1
  );
}

export function cellsIntersecting(cells: GridCell[], rect: PixelRect): GridCell[] {
  return cells.filter((c) => rectsOverlap(c, rect));
}

/**
 * One cell divided into a `divisions` x `divisions` grid of near-equal, gapless sub-rects:
 * floor-indexed boundaries (`floor(size*i/divisions)`) distribute any remainder across the whole
 * row/column rather than concentrating it in one oversized final cell. The last row/column's far
 * edge is pinned to the cell's own `x1`/`y1` rather than the general formula, a defensive exact-
 * edge guarantee against float drift; for this platform's integer pixel cells the two already
 * agree. The coverage tracker's union-of-visibility sweep predicate needs this: "was the whole
 * cell ever on screen at once" is unsatisfiable on a raster whose cells are bigger than any
 * working viewport, but "was every one of its sub-cells, individually, fully on screen at some
 * point this session" is the same fact at a grain small enough to actually accumulate through
 * ordinary panning.
 */
export function subdivideCell(cell: GridCell, divisions: number): PixelRect[] {
  const w = cell.x1 - cell.x0;
  const h = cell.y1 - cell.y0;
  const subs: PixelRect[] = [];
  for (let row = 0; row < divisions; row++) {
    const y0 = cell.y0 + Math.floor((h * row) / divisions);
    const y1 = row === divisions - 1 ? cell.y1 : cell.y0 + Math.floor((h * (row + 1)) / divisions);
    for (let col = 0; col < divisions; col++) {
      const x0 = cell.x0 + Math.floor((w * col) / divisions);
      const x1 =
        col === divisions - 1 ? cell.x1 : cell.x0 + Math.floor((w * (col + 1)) / divisions);
      subs.push({ x0, y0, x1, y1 });
    }
  }
  return subs;
}

/**
 * How many divisions `subdivideCell` needs so `cell`'s sub-cells are no larger than `targetPx`
 * on their long edge: `ceil(long_edge / targetPx)`, at least 1. A fixed division count scales
 * sub-cell size with the cell itself, not with the viewport that has to contain one -- wrong on
 * any lattice whose cells vary in size (the ordinary display-derived lattice caps cells at
 * display_bounds.DISPLAY_MAX_EDGE=4096px; the large-raster lattice's cells run into the tens of
 * thousands of pixels). Deriving divisions from an absolute pixel target instead keeps sub-cell
 * size, and therefore whether a real viewport can ever fully contain one, consistent across
 * every lattice this platform serves.
 */
export function subCellDivisionsFor(cell: GridCell, targetPx: number): number {
  const longEdge = Math.max(cell.x1 - cell.x0, cell.y1 - cell.y0);
  return Math.max(1, Math.ceil(longEdge / targetPx));
}

/** One planned cell-aligned region serve. */
export interface RegionFetch {
  cell: GridCell;
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
  cells: GridCell[];
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
  const area = (cell: GridCell) =>
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

/**
 * Whether a region serve's response covered its cell at native resolution: the served output
 * size equals the cell's own dims. A response fact, never the request's intent.
 */
export function servedCellAtNative(
  cell: GridCell,
  servedSize: { w: number; h: number } | null,
): boolean {
  return !!servedSize && servedSize.w === cell.x1 - cell.x0 && servedSize.h === cell.y1 - cell.y0;
}

/** Cell indices in row-major order (top-to-bottom, left-to-right), from the cells' own rects. */
export function rowMajorOrder(cells: GridCell[]): number[] {
  return cells
    .map((_, i) => i)
    .sort((a, b) => cells[a].y0 - cells[b].y0 || cells[a].x0 - cells[b].x0);
}

/**
 * The next (delta=+1) or previous (delta=-1) unswept cell in row-major order, scanning from the
 * cell nearest `from` (image coords) and wrapping; null when every cell is swept.
 */
export function stepUnsweptCell(
  cells: GridCell[],
  swept: ReadonlySet<string>,
  from: { x: number; y: number },
  delta: 1 | -1,
): GridCell | null {
  if (cells.length === 0) return null;
  const order = rowMajorOrder(cells);
  let nearestPos = 0;
  let bestD = Infinity;
  order.forEach((cellIdx, pos) => {
    const c = cells[cellIdx];
    const d = Math.hypot((c.x0 + c.x1) / 2 - from.x, (c.y0 + c.y1) / 2 - from.y);
    if (d < bestD) {
      bestD = d;
      nearestPos = pos;
    }
  });
  for (let step = 1; step <= order.length; step++) {
    const pos = (nearestPos + delta * step + order.length * step) % order.length;
    const cell = cells[order[pos]];
    if (!swept.has(cell.name)) return cell;
  }
  return null;
}

/** Column/row lookup over served cells, indexed from the distinct cell origins (no re-derivation). */
export interface CellIndex {
  cols: number;
  rows: number;
  /** [x0, x1] extents per column and [y0, y1] per row, from the cells themselves. */
  colX: [number, number][];
  rowY: [number, number][];
  at(col: number, row: number): GridCell | undefined;
}

export function indexCells(cells: GridCell[]): CellIndex {
  const xs = Array.from(new Set(cells.map((c) => c.x0))).sort((a, b) => a - b);
  const ys = Array.from(new Set(cells.map((c) => c.y0))).sort((a, b) => a - b);
  const colOf = new Map(xs.map((x, i) => [x, i]));
  const rowOf = new Map(ys.map((y, i) => [y, i]));
  const byPos = new Map<number, GridCell>();
  const colX: [number, number][] = xs.map((x) => [x, x]);
  const rowY: [number, number][] = ys.map((y) => [y, y]);
  for (const c of cells) {
    const col = colOf.get(c.x0)!;
    const row = rowOf.get(c.y0)!;
    byPos.set(row * xs.length + col, c);
    colX[col] = [c.x0, Math.max(colX[col][1], c.x1)];
    rowY[row] = [c.y0, Math.max(rowY[row][1], c.y1)];
  }
  return {
    cols: xs.length,
    rows: ys.length,
    colX,
    rowY,
    at: (col, row) => byPos.get(row * xs.length + col),
  };
}

/**
 * Swept fractions over k x k blocks of the cell lattice, for display-time aggregation when
 * individual cells fall below the minimap's visibility floor. Block (bc, br) covers cell columns
 * [bc*k, (bc+1)*k) and rows likewise, clipped at the lattice edge; its fraction is swept cells
 * over cells in the block.
 */
export function sweptFractionBlocks(
  cols: number,
  rows: number,
  k: number,
  isSwept: (col: number, row: number) => boolean,
): { cols: number; rows: number; fractions: number[] } {
  const blockCols = Math.max(1, Math.ceil(cols / k));
  const blockRows = Math.max(1, Math.ceil(rows / k));
  const fractions: number[] = [];
  for (let by = 0; by < blockRows; by++) {
    for (let bx = 0; bx < blockCols; bx++) {
      let total = 0;
      let sweptCount = 0;
      for (let row = by * k; row < Math.min(rows, (by + 1) * k); row++) {
        for (let col = bx * k; col < Math.min(cols, (bx + 1) * k); col++) {
          total++;
          if (isSwept(col, row)) sweptCount++;
        }
      }
      fractions.push(total ? sweptCount / total : 0);
    }
  }
  return { cols: blockCols, rows: blockRows, fractions };
}

/**
 * The Complete warning's wording. States screen facts only, cells and scale, never a claim about
 * what the person looked at. Says nothing about the swept cells: a swept cell can have been
 * covered piecewise across several separate views (see subdivideCell), never claimed to have
 * sat on screen whole at once, so the wording must not imply that either.
 */
export function completeWarningMessage(w: {
  unsweptCount: number;
  total: number;
  bar: number;
}): string {
  const pct = Math.round(w.bar * 100);
  return `Complete: ${w.unsweptCount} of ${w.total} grid cells were never fully seen, in any combination of views, at ${pct}% zoom or closer this session.`;
}
