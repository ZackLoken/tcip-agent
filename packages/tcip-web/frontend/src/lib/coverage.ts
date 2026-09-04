/**
 * Pure helpers over the coverage lattice a raster's grid route serves. The lattice itself always
 * comes from GET /api/coverage/grid; nothing here re-derives cells from the geometry.
 */

import type { HostSize, PixelRect } from "@/lib/viewGeometry";

export type {
  CompletenessSetPayload as CompletenessSetPostBody,
  GridGeometry,
  GridZoomPayload,
  WorkingScale,
} from "@/api/types.generated";
import type { GridGeometry, WorkingScale } from "@/api/types.generated";

/** One served cell: a name plus its half-open native-pixel rect, clipped to the image extent. */
export interface GridCell {
  name: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** One rendered grid block (`get_grid`'s own `grid` and `serving` fields share this shape):
 *  geometry, the full cell list, and how the tile size was chosen -- "cells sized to one
 *  full-resolution screenful", "a chosen cell edge of <n> px" for an explicit tile_size, "one
 *  screenful at <zoom>x zoom" for the set-zoom lattice, or "the lattice this image's coverage
 *  was recorded on" for an already-worked image. Not part of GridGeometry; stripped back off
 *  (see useCoverageGrid) before a grid round-trips into a coverage or completeness payload,
 *  which forbids extra keys. */
export interface RenderedGrid extends GridGeometry {
  cells: GridCell[];
  derivation: string;
}

/** GET /api/coverage/grid's full response: the coverage lattice (`grid`, null with `reason`
 *  when none can be derived -- no set zoom, or a viewport not yet measured), plus the
 *  zoom-independent region-serving grid (`serving`, always present). `fresh_derivation_differs`
 *  is set only when `grid` came from an already-worked image's own recorded lattice and the
 *  subject's current zoom would derive a different tile size; null otherwise. */
export interface CoverageGridResponse {
  grid: RenderedGrid | null;
  reason: string | null;
  fresh_derivation_differs: boolean | null;
  serving: RenderedGrid;
}

/** One cell's attestation-time scale provenance, stamped by POST /api/coverage/completeness:
 *  the view scale the breeder pressed at (null for a non-GUI caller), the working scale (the
 *  subject's set zoom) in effect at write time (which can differ from the one read on screen if
 *  the zoom changed between the read and the press), and whether this image's own
 *  view-coverage record shows the cell already seen -- facts only, no verdict; `at_scale` is
 *  null and `grid_matched` false when no coverage record existed yet or its grid disagreed. */
export interface CellAttestedView {
  view_scale: number | null;
  working_scale_at_write: WorkingScale | null;
  seen_on_record: { at_scale: number | null; grid_matched: boolean };
}

/** One subject's region-completeness record, as GET /api/coverage/completeness returns it
 *  (per subject, in `by_subject`). `stale_cells` is recomputed server-side on every read: an
 *  attested cell whose annotation content has since been edited or deleted. `cells_attested_view`
 *  is always present; a record predating the key refuses on the backend rather than being
 *  served with it missing. */
export interface CompletenessRecord {
  grid: GridGeometry;
  cells_complete: string[];
  attested_by: string | null;
  attested_at: string | null;
  stem: string;
  date: string | null;
  subject: string;
  stale_cells: string[];
  cells_attested_view: Record<string, CellAttestedView>;
}

/** GET /api/coverage/completeness's full response: every subject's record, plus every subject's
 *  saved-annotation count per cell over the raster's current grid. `annotation_counts` sits
 *  beside `by_subject`, never inside it: a count belongs to the grid, not to any one attestation
 *  record, so it is served even for a subject with no record yet. `counts_grid` is the lattice
 *  the counts were binned against, served so a caller can refuse to render them against a
 *  different current grid the way it already does for a record on another lattice. The whole
 *  counts computation is best-effort beside `by_subject`, which needs no raster at all:
 *  `counts_error` is set (with `annotation_counts` empty and `counts_grid` null) when the grid
 *  could not be derived or the raster could not be read; `by_subject` still serves in that
 *  case. */
/** `working_scale` (subject -> WorkingScale or null) is read fresh from the subject's set grid
 *  zoom on every read, never derived from any annotation or echoed back from the browser;
 *  `working_scale_error` names a store-read failure, distinct from `counts_error` (a
 *  raster-read failure costs only the counts, never the working scale).
 *  `working_scale_reason` (subject -> clause) names why a null working scale has none, per
 *  subject, for every subject `working_scale` maps to null; absent for a subject whose zoom is
 *  set. */
export interface CompletenessResponse {
  by_subject: Record<string, CompletenessRecord>;
  annotation_counts: Record<string, Record<string, number>>;
  counts_grid: GridGeometry | null;
  counts_error: string | null;
  working_scale: Record<string, WorkingScale | null>;
  working_scale_error: string | null;
  working_scale_reason: Record<string, string>;
}

/** Cells genuinely complete right now: attested, minus any that have since gone stale. A stale
 *  attestation must never render as if it still held. */
export function effectiveComplete(record: CompletenessRecord | undefined): Set<string> {
  if (!record) return new Set();
  const stale = new Set(record.stale_cells);
  return new Set(record.cells_complete.filter((c) => !stale.has(c)));
}

/** The label reader's own reason, stripped of the record it dumps after a colon and a brace
 *  (`tcip_annotation`'s `UnreadableLabelDocument` messages end "...: {'id': 1, ...}"): a breeder
 *  reads the reader's own sentence, never a Python dict. Text with no brace passes through. */
export function breederReadErrorReason(raw: string): string {
  const idx = raw.indexOf("{");
  if (idx === -1) return raw;
  return raw.slice(0, idx).replace(/[:\s]+$/, "");
}

/** Whether a cell's recorded scale meets a subject's working scale: the one comparison the
 *  tracker, the overlay's derivation and the chrome's attested-view line all call, so "does this
 *  cell meet the working scale" is never answered twice. `null` on either side (no recorded
 *  scale, or no working scale to judge against) never meets it. */
export function meetsBar(atScale: number | null, bar: WorkingScale | null): boolean {
  return atScale !== null && bar !== null && atScale >= bar.zoom;
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

/** The cell containing image-coordinate point `(x, y)`, or null outside every cell: one lookup
 *  shared by the Map tool's click and the chrome's "current cell" (viewport center). */
export function cellAt(cells: GridCell[], x: number, y: number): GridCell | null {
  return cells.find((c) => x >= c.x0 && x < c.x1 && y >= c.y0 && y < c.y1) ?? null;
}

/** The cell the coverage chrome names: the cell a Map click just opened, while any part of it
 *  remains inside the viewport (a padded, edge-clamped jump can leave the viewport centered away
 *  from the clicked cell); otherwise the cell under the viewport center, the rule a pan or
 *  Overview reverts to once the clicked cell scrolls out of view. */
export function currentCoverageCell(
  cells: GridCell[],
  viewport: PixelRect | null,
  mapSelected: GridCell | null,
): GridCell | null {
  if (mapSelected && viewport && rectsOverlap(mapSelected, viewport)) return mapSelected;
  if (!viewport) return null;
  return cellAt(cells, (viewport.x0 + viewport.x1) / 2, (viewport.y0 + viewport.y1) / 2);
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
 * any lattice whose cells vary in size (the display-derived serving lattice caps cells at
 * display_bounds.DISPLAY_MAX_EDGE=4096px; the coverage lattice's cell edge grows with a low set
 * zoom and can run far larger). Deriving divisions from an absolute pixel target instead keeps
 * sub-cell size, and therefore whether a real viewport can ever fully contain one, consistent
 * across every lattice this platform serves.
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

/**
 * The Complete warning's wording. States screen facts only, cells and scale, never a claim about
 * what the person looked at, and never scoped to "this session": a cell's recorded scale can
 * come from an earlier session's hydrate as easily as this one's own viewport passes. Says
 * nothing about the swept cells: a swept cell can have been covered piecewise across several
 * separate views (see subdivideCell), never claimed to have sat on screen whole at once, so the
 * wording must not imply that either. `bar` is the unrounded value the comparison used, shown to
 * one decimal place as a percentage.
 */
export function completeWarningMessage(w: {
  unsweptCount: number;
  total: number;
  bar: number;
}): string {
  const pct = (w.bar * 100).toFixed(1);
  return `Complete: ${w.unsweptCount} of ${w.total} grid cells have not had every part on screen at ${pct}% zoom or closer, in any combination of views.`;
}

/** The Complete toggle's own no-bar toast: `subject` and `reason` come from the same read
 *  (`useRegionCompleteness`'s own `subject`/`workingScaleReason`), so the sentence can never name
 *  a different subject than the one the reason was computed for. `subject` null (no active
 *  subject) states the reason alone, since there is no working scale to name a subject for. */
export function noWorkingScaleToast(subject: string | null, reason: string): string {
  if (!subject) return `Complete: ${reason}, so coverage was not checked`;
  return `Complete: no working scale for ${subject} on this image (${reason}), so coverage was not checked`;
}

/** The sentence a Complete toast adds when the coverage tracker's replace hold stands: cells
 *  seen on a previous lattice have not been discarded, so this lattice's own sweeps are still
 *  unsaved. Stated beside whatever else the toggle already says, never in place of it. */
export function replaceRequiredToastSentence(cellsSeen: number): string {
  return (
    `${cellsSeen} cell${cellsSeen === 1 ? "" : "s"} seen on a previous lattice ` +
    "are not yet replaced, so coverage progress on this lattice is still unsaved."
  );
}
