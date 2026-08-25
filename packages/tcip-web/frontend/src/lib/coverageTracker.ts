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

import type { CoveragePayload, CoverageRecord, CoverageViewing } from "@/api/types.generated";
import {
  rectFullyInside,
  rectsOverlap,
  sameGrid,
  subCellDivisionsFor,
  subdivideCell,
  type GridCell,
  type GridGeometry,
} from "@/lib/coverage";
import type { PixelRect } from "@/lib/viewGeometry";

export type { CoveragePayload, CoverageRecord, CoverageViewing };

/** The viewing context the tracker accumulates before ``working_scale_bar`` joins it at post
 *  time (see ``postNow``): the caller supplies the rest, the tracker supplies the bar. */
export type CoverageViewingInput = Omit<CoverageViewing, "working_scale_bar">;

/**
 * Sub-cell grain for the union-of-visibility sweep predicate (see `subdivideCell`): a cell
 * sweeps once every one of its sub-rects has, at some point, been fully on screen at or above
 * the working-scale bar. Divisions are derived per cell (`subCellDivisionsFor`) from this target
 * pixel size, not a single fixed division count applied to every cell -- a fixed count scales
 * sub-cell size with the cell, not the viewport, so it stops working the moment a lattice's
 * cells get big (the large-raster lattice's cells run into the tens of thousands of pixels).
 * 128px is not a fresh guess: it is display_bounds.DISPLAY_MAX_EDGE=4096 divided by the sub-cell
 * count (32) the ordinary display-derived lattice already shipped with and nobody has flagged as
 * too coarse, now applied as an absolute size to every lattice instead of a division count that
 * only happened to produce it for one of them. Still provisional pending a real GUI annotation
 * session to check the target against, the same "ship a plain, documented, revisit-later
 * default" idiom as reference_grid.derive_large_raster_grid_tile_size's own divisions=16 -- but
 * no longer structurally guaranteed to fail on a large-raster lattice at real zoom the way a
 * fixed division count was.
 */
const SUB_CELL_TARGET_PX = 128;

export interface CoverageKeyParts {
  imagePath: string;
  datasetRoot: string | null;
  subject: string;
  date: string | null;
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
  private viewing: CoverageViewingInput = {
    stats_source: null,
    display_bounds: null,
    base_served_size: null,
  };
  private dirty = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private postFn: (body: CoveragePayload) => Promise<unknown>,
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
    this.subCells = new Map(
      cells.map((c) => [c.name, subdivideCell(c, subCellDivisionsFor(c, SUB_CELL_TARGET_PX))]),
    );
    this.subCellScale = new Map();
    this.barValue = null;
    this.dirty = false;
    this.opts.onChange?.();
  }

  /** Adopt a stored record's cell facts, only when its grid matches the current one. */
  hydrate(record: CoverageRecord | null): void {
    if (!record || !this.grid || !record.grid || !sameGrid(record.grid, this.grid)) return;
    for (const name of record.cells_served_at_native) this.servedAtNativeSet.add(name);
    for (const name of record.cells_swept) this.sweptSet.add(name);
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

  setViewing(viewing: CoverageViewingInput): void {
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
    const body: CoveragePayload = {
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
