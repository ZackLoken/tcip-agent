/**
 * Session accumulator for the per-image view-coverage record. It records two facts per cell:
 * cells_served_at_native (a region serve's response covered the cell at native resolution) and
 * cells_swept (every one of the cell's sub-cells, see subdivideCell, has at some point sat fully
 * inside the viewport at or above the working-scale bar -- a union-of-visibility fact, not "the
 * whole cell was on screen at once," which a raster whose cells exceed any working viewport can
 * never satisfy). The bar is the minimum view scale in force when annotations were committed on
 * this image and subject this session: the coarsest scale at which objects were demonstrably
 * judged. No authoring commits means no bar and no sweep accumulation.
 *
 * Facts are pushed with a trailing debounce; union-merging with the stored record is the
 * server's job. Tracking activates only for multi-cell grids (an image inside the display bound
 * derives a single trivially-covered cell and gets no tracking).
 */

import {
  rectFullyInside,
  rectsOverlap,
  sameGrid,
  subdivideCell,
  type GridCell,
  type GridGeometry,
} from "@/lib/coverage";
import type { PixelRect } from "@/lib/viewGeometry";

/**
 * Sub-cell grain for the union-of-visibility sweep predicate (see `subdivideCell`): a cell
 * sweeps once every one of its DIVISIONS x DIVISIONS sub-rects has, at some point, been fully on
 * screen at or above the working-scale bar. Provisional, a starting guess pending a real GUI
 * annotation session to check the grain against -- the same "ship a plain, documented,
 * revisit-later default" idiom as reference_grid.derive_large_raster_grid_tile_size's own
 * divisions=16.
 *
 * A fixed fraction of the cell scales with the cell, not with the viewport, so this does not
 * uniformly fix the predicate it replaces. On the ordinary display-derived lattice (cells capped
 * at display_bounds.DISPLAY_MAX_EDGE=4096px) a sub-cell is at most 128px, comfortably containable
 * at real annotation zoom. On the large-raster lattice (derive_large_raster_grid_tile_size, cells
 * long_edge/16 -- roughly 15000px at real ValleyFarm scale) a sub-cell is still roughly 469px,
 * which only fits inside a working viewport below about 1.9x zoom; an annotator working closer
 * than that on a large-raster project would still hit the unsatisfiable-predicate failure this
 * change exists to close, just moved rather than removed. Known, unresolved as of this writing.
 */
const SUB_CELL_DIVISIONS = 32;

export interface CoverageKeyParts {
  imagePath: string;
  datasetRoot: string | null;
  subject: string;
  date: string | null;
}

/** Display context the record carries so a reviewer can reconstruct what was on screen. */
export interface CoverageViewing {
  bands?: string;
  stretch?: string;
  stats_source?: string | null;
  display_bounds?: string | null;
  base_served_size?: string | null;
}

export interface CoveragePostBody {
  image_path: string;
  dataset_root: string | null;
  subject: string;
  date: string | null;
  grid: GridGeometry;
  cells_served_at_native: string[];
  cells_swept: string[];
  viewing: CoverageViewing & {
    working_scale_bar: { value: number; source: string } | null;
  };
}

/** The stored per-image record, as GET /api/coverage returns it (null when none exists yet). */
export interface CoverageRecord {
  grid: GridGeometry;
  cells_served_at_native?: string[];
  cells_swept?: string[];
}

const BAR_SOURCE =
  "minimum view scale at annotation commits on this image and subject this session";

/** The trailing-debounce cadence, matching the nav-index sync. */
const POST_DEBOUNCE_MS = 400;

export class CoverageTracker {
  private keyParts: CoverageKeyParts | null = null;
  private grid: GridGeometry | null = null;
  private cells: GridCell[] = [];
  private servedAtNativeSet = new Set<string>();
  private sweptSet = new Set<string>();
  // Each cell's fixed sub-cell partition (subdivideCell), computed once per grid.
  private subCells = new Map<string, PixelRect[]>();
  // Best containment scale seen per sub-cell (keyed "<cellName>#<index>"), recorded whether or
  // not a bar exists yet, so a bar arriving late still credits every earlier viewport moment.
  private subCellScale = new Map<string, number>();
  private barValue: number | null = null;
  private viewing: CoverageViewing = {};
  private dirty = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private postFn: (body: CoveragePostBody) => Promise<unknown>,
    private opts: { debounceMs?: number; onChange?: () => void } = {},
  ) {}

  /** Multi-cell grids only: single-cell grids (image within the display bound) get no tracking. */
  get active(): boolean {
    return this.keyParts !== null && this.grid !== null && this.cells.length > 1;
  }

  get swept(): ReadonlySet<string> {
    return this.sweptSet;
  }

  get servedAtNative(): ReadonlySet<string> {
    return this.servedAtNativeSet;
  }

  get bar(): number | null {
    return this.barValue;
  }

  /**
   * Point the tracker at a new (image, subject, date, dataset, grid) identity, flushing any
   * facts still owed for the previous one first. Pass nulls to deactivate.
   */
  reset(keyParts: CoverageKeyParts | null, grid: GridGeometry | null, cells: GridCell[]): void {
    this.flush();
    this.keyParts = keyParts;
    this.grid = grid;
    this.cells = cells;
    this.servedAtNativeSet = new Set();
    this.sweptSet = new Set();
    this.subCells = new Map(cells.map((c) => [c.name, subdivideCell(c, SUB_CELL_DIVISIONS)]));
    this.subCellScale = new Map();
    this.barValue = null;
    this.dirty = false;
    this.opts.onChange?.();
  }

  /** Adopt a stored record's cell facts, only when its grid matches the current one. */
  hydrate(record: CoverageRecord | null): void {
    if (!record || !this.grid || !record.grid || !sameGrid(record.grid, this.grid)) return;
    for (const name of record.cells_served_at_native ?? []) this.servedAtNativeSet.add(name);
    for (const name of record.cells_swept ?? []) this.sweptSet.add(name);
    this.opts.onChange?.();
  }

  /** An annotation was committed at `scale`; the bar is the minimum such scale this session. */
  noteAuthoringScale(scale: number): void {
    if (!this.active || scale <= 0) return;
    if (this.barValue !== null && scale >= this.barValue) return;
    this.barValue = scale;
    // A new or lowered bar can make already-recorded containment moments count.
    this.resweep();
    this.opts.onChange?.();
  }

  /**
   * The viewport now shows `rect` (image coords, clipped) at `scale`. A cell sweeps once every
   * one of its sub-cells has, across any number of viewport moments (not necessarily this one),
   * been fully inside the viewport at or above the bar: the union-of-visibility generalization
   * of "the whole cell was on screen at once," which a raster whose cells exceed any working
   * viewport can never satisfy directly.
   */
  noteViewport(rect: PixelRect, scale: number): void {
    if (!this.active) return;
    let changed = false;
    for (const cell of this.cells) {
      if (this.sweptSet.has(cell.name) || !rectsOverlap(cell, rect)) continue;
      const subs = this.subCells.get(cell.name);
      if (!subs) continue;
      for (let i = 0; i < subs.length; i++) {
        if (!rectFullyInside(subs[i], rect)) continue;
        const key = `${cell.name}#${i}`;
        const prev = this.subCellScale.get(key) ?? 0;
        if (scale > prev) this.subCellScale.set(key, scale);
      }
      if (this.allSubCellsAtBar(cell.name, subs.length)) {
        this.sweptSet.add(cell.name);
        changed = true;
      }
    }
    if (changed) {
      this.markDirty();
      this.opts.onChange?.();
    }
  }

  /** Every sub-cell of `cellName` has a recorded max scale at or above the bar (false, always,
   *  with no bar yet). */
  private allSubCellsAtBar(cellName: string, subCellCount: number): boolean {
    if (this.barValue === null) return false;
    for (let i = 0; i < subCellCount; i++) {
      if ((this.subCellScale.get(`${cellName}#${i}`) ?? 0) < this.barValue) return false;
    }
    return true;
  }

  /** A region serve's response covered this cell at its native dimensions. */
  noteServedAtNative(cellName: string): void {
    if (!this.active || this.servedAtNativeSet.has(cellName)) return;
    if (!this.cells.some((c) => c.name === cellName)) return;
    this.servedAtNativeSet.add(cellName);
    this.markDirty();
  }

  setViewing(viewing: CoverageViewing): void {
    this.viewing = viewing;
  }

  /** The facts behind the Complete warning, or null when the warning does not apply. */
  completeWarning(): { unsweptCount: number; total: number; bar: number } | null {
    if (!this.active || this.barValue === null) return null;
    const unsweptCount = this.cells.filter((c) => !this.sweptSet.has(c.name)).length;
    if (unsweptCount === 0) return null;
    return { unsweptCount, total: this.cells.length, bar: this.barValue };
  }

  /** Send any owed facts now (called before a reset and on unmount). */
  flush(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.postNow();
  }

  dispose(): void {
    this.flush();
  }

  /** Recompute swept membership from recorded sub-cell containment scales against the current
   *  bar. */
  private resweep(): void {
    if (this.barValue === null) return;
    let changed = false;
    for (const cell of this.cells) {
      if (this.sweptSet.has(cell.name)) continue;
      const subs = this.subCells.get(cell.name);
      if (!subs) continue;
      if (this.allSubCellsAtBar(cell.name, subs.length)) {
        this.sweptSet.add(cell.name);
        changed = true;
      }
    }
    if (changed) this.markDirty();
  }

  private markDirty(): void {
    this.dirty = true;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.postNow();
    }, this.opts.debounceMs ?? POST_DEBOUNCE_MS);
  }

  private postNow(): void {
    if (!this.dirty || !this.keyParts || !this.grid) return;
    this.dirty = false;
    const body: CoveragePostBody = {
      image_path: this.keyParts.imagePath,
      dataset_root: this.keyParts.datasetRoot,
      subject: this.keyParts.subject,
      date: this.keyParts.date,
      grid: this.grid,
      cells_served_at_native: Array.from(this.servedAtNativeSet).sort(),
      cells_swept: Array.from(this.sweptSet).sort(),
      viewing: {
        ...this.viewing,
        working_scale_bar:
          this.barValue === null ? null : { value: this.barValue, source: BAR_SOURCE },
      },
    };
    void this.postFn(body).catch(() => {
      this.dirty = true;
    });
  }
}
