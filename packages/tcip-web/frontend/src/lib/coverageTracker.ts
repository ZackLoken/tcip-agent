/**
 * Session accumulator for the per-image view-coverage record. It records two facts per cell:
 * cells_served_at_native (a region serve's response covered the cell at native resolution) and
 * cells_seen_at_scale (every one of the cell's sub-cells, see subdivideCell, has at some point
 * sat fully inside the viewport at some recorded scale -- a union-of-visibility bound, not "the
 * whole cell was on screen at once," which a raster whose cells exceed any working viewport can
 * never satisfy). Whether a seen cell counts as "swept" is a pure derivation against a subject's
 * working-scale bar, served by the completeness read and handed in through
 * `setWorkingScaleBar`, never accumulated from an authoring commit: a commit no longer moves any
 * bar, and an unsaved annotation never does either.
 *
 * Facts are pushed with a trailing debounce; merging with the stored record (by the greater
 * value per cell) is the server's job. A push that fails after the tracker has moved on (a
 * reset, a dispose, an unmount) is handed to the module-level outbox below rather than dropped:
 * the outbox is one per app, outside any tracker instance, and drains on its own retry timer
 * whether or not a tracker is currently active. Tracking activates only for multi-cell grids (an
 * image inside the display bound derives a single trivially-covered cell and gets no tracking).
 */

import type {
  CoveragePayload,
  CoverageRecord,
  CoverageViewing,
  WorkingScaleBar,
} from "@/api/types.generated";
import {
  meetsBar,
  rectFullyInside,
  rectsOverlap,
  sameGrid,
  subCellDivisionsFor,
  subdivideCell,
  type GridCell,
  type GridGeometry,
} from "@/lib/coverage";
import { StructuredRefusalError } from "@/api/http";
import type { PixelRect } from "@/lib/viewGeometry";

export type { CoveragePayload, CoverageRecord, CoverageViewing, WorkingScaleBar };
export { meetsBar };

/** The viewing context the tracker accumulates: the caller supplies it whole (the server derives
 *  the working-scale bar from the label file, never from anything echoed back here). */
export type CoverageViewingInput = CoverageViewing;

/** `post_coverage`'s own answer: the merged record beside its unchanged integer counts. Only the
 *  fields the tracker itself reads back are declared here (see `routes/coverage.py`'s
 *  `record_body`); a caller wanting the full response shapes its own type over `postFn`. */
export interface CoveragePushResponse {
  record: { cells_seen_at_scale: Record<string, number> };
}

/**
 * Sub-cell grain for the union-of-visibility sweep predicate (see `subdivideCell`): a cell
 * becomes "seen" once every one of its sub-rects has, at some point, been fully on screen.
 * Divisions are derived per cell (`subCellDivisionsFor`) from this target pixel size, not a
 * single fixed division count applied to every cell -- a fixed count scales sub-cell size with
 * the cell, not the viewport, so it stops working the moment a lattice's cells get big (the
 * large-raster lattice's cells run into the tens of thousands of pixels). 128px is a documented
 * default pending a real GUI annotation session to check the target against, the same
 * "ship a plain, documented, revisit-later default" idiom as
 * reference_grid.derive_large_raster_grid_tile_size's own divisions=16.
 */
const SUB_CELL_TARGET_PX = 128;

export interface CoverageKeyParts {
  imagePath: string;
  datasetRoot: string | null;
  subject: string;
  date: string | null;
}

/** The trailing-debounce cadence, matching the nav-index sync. */
const POST_DEBOUNCE_MS = 400;

function sameWorkingScaleBar(a: WorkingScaleBar | null, b: WorkingScaleBar | null): boolean {
  if (a === b) return true;
  if (a === null || b === null) return false;
  return (
    a.value === b.value &&
    a.median_extent_native_px === b.median_extent_native_px &&
    a.annotation_count === b.annotation_count &&
    a.judged_span_px === b.judged_span_px &&
    a.source === b.source
  );
}

/** How long a failed push stays queued before the outbox retries it. */
const OUTBOX_RETRY_MS = 5000;

/** The stable marker `post_coverage`'s 500 body carries when the record committed but its audit
 *  line could not be written (`routes/coverage.py`'s `AUDIT_ENTRY_NOT_WRITTEN`): a retry of the
 *  same payload merges to no change and writes no line, so this answer is terminal like a 4xx,
 *  never retried like an ordinary 5xx. */
const AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written";

/** The HTTP status a thrown push error carries, or null for one that carries none (a network
 *  failure, or a body with no structured `detail` at all): the single place that reads it, so
 *  the outbox's own terminality check and a live tracker's own retry stay one answer. */
function pushErrorStatus(err: unknown): number | null {
  return err instanceof StructuredRefusalError ? err.status : null;
}

function isAuditEntryNotWritten(err: unknown): boolean {
  return (
    err instanceof StructuredRefusalError &&
    err.status === 500 &&
    err.detail.error === AUDIT_ENTRY_NOT_WRITTEN
  );
}

/** Whether a failed push can never succeed by retrying the identical payload: any 4xx (an
 *  unknown cell, an unconformed stored record, a refused subject), or the audit-gap 500 by its
 *  own marker. A network failure or an ordinary 5xx is not terminal and stays queued. */
function isTerminalPushFailure(err: unknown): boolean {
  if (isAuditEntryNotWritten(err)) return true;
  const status = pushErrorStatus(err);
  return status !== null && status >= 400 && status < 500;
}

/** The detail text a dropped payload's toast names, and whether it names an audit gap (a
 *  distinct sentence from an ordinary refusal) rather than a plain refusal. */
function pushFailureDetail(err: unknown): { detail: string; auditGap: boolean } {
  return {
    detail: err instanceof Error ? err.message : String(err),
    auditGap: isAuditEntryNotWritten(err),
  };
}

/**
 * The module-level outbox for a coverage push a tracker could not deliver before moving on (a
 * reset, a dispose, an unmount): one per app, independent of any tracker instance, so a payload
 * survives the identity change that orphaned it and keeps retrying on its own timer. A network
 * failure or an ordinary 5xx stays at the queue's head and is retried on the next tick; a
 * terminal failure (`isTerminalPushFailure`) is dropped, so one refused payload never blocks the
 * facts behind it.
 */
export class CoverageOutbox {
  private queue: CoveragePayload[] = [];
  private postFn: ((body: CoveragePayload) => Promise<CoveragePushResponse>) | null = null;
  private onDropped: ((imagePath: string, detail: string, auditGap: boolean) => void) | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private draining = false;

  /** Every tracker shares one `postFn` in practice (the app's own `api.coverage.push`); the
   *  latest caller's is the one the outbox retries with. */
  configure(
    postFn: (body: CoveragePayload) => Promise<CoveragePushResponse>,
    onDropped: (imagePath: string, detail: string, auditGap: boolean) => void,
  ): void {
    this.postFn = postFn;
    this.onDropped = onDropped;
  }

  get size(): number {
    return this.queue.length;
  }

  enqueue(body: CoveragePayload): void {
    this.queue.push(body);
    this.schedule();
  }

  private schedule(): void {
    if (this.timer !== null || this.draining) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.drain();
    }, OUTBOX_RETRY_MS);
  }

  private async drain(): Promise<void> {
    if (this.draining || this.queue.length === 0) return;
    if (!this.postFn) {
      // Nothing configured yet to drain with: reschedule rather than stall the queue forever.
      this.schedule();
      return;
    }
    this.draining = true;
    while (this.queue.length > 0) {
      const body = this.queue[0];
      try {
        await this.postFn(body);
        this.queue.shift();
      } catch (err: unknown) {
        if (isTerminalPushFailure(err)) {
          this.queue.shift();
          const { detail, auditGap } = pushFailureDetail(err);
          this.onDropped?.(body.image_path, detail, auditGap);
          continue;
        }
        break;
      }
    }
    this.draining = false;
    if (this.queue.length > 0) this.schedule();
  }
}

export const coverageOutbox = new CoverageOutbox();

/** Test-only: drop every queued payload on `outbox` and any pending retry timer, so one test's
 *  leftover state never leaks into the next. A plain function, not a method the shipped
 *  singleton carries: `src/test/coverageOutbox.ts` is the one importer. */
export function resetCoverageOutboxForTests(outbox: CoverageOutbox): void {
  const internal = outbox as unknown as {
    queue: CoveragePayload[];
    timer: ReturnType<typeof setTimeout> | null;
    draining: boolean;
  };
  internal.queue = [];
  if (internal.timer !== null) clearTimeout(internal.timer);
  internal.timer = null;
  internal.draining = false;
}

/** Every tracker currently constructed and not yet disposed: one per open Annotate canvas (in
 *  practice, one). React never unmounts a component on `beforeunload`/`pagehide`, so a
 *  component's own cleanup (which would call `dispose`) never runs then either; App's unload
 *  guard and its `pagehide` flush both need to reach a live tracker some other way. */
const liveTrackers = new Set<CoverageTracker>();

/** Whether any live tracker currently owes the server a fact: App's `beforeunload` guard fires
 *  on this beside a non-empty `coverageOutbox`. */
export function anyTrackerDirty(): boolean {
  for (const t of liveTrackers) if (t.isDirty) return true;
  return false;
}

/** Flush every live tracker's owed facts (to the outbox on failure, same as `flush()` always
 *  does): App's `pagehide` handler, since a live tracker never gets its own unmount to call
 *  `dispose` from. */
export function flushAllTrackers(): void {
  for (const t of liveTrackers) t.flush();
}

export class CoverageTracker {
  private keyParts: CoverageKeyParts | null = null;
  private grid: GridGeometry | null = null;
  private cells: GridCell[] = [];
  private servedAtNativeSet = new Set<string>();
  // Each cell's fixed sub-cell partition (subdivideCell), computed once per grid.
  private subCells = new Map<string, PixelRect[]>();
  // Best containment scale ever seen per sub-cell ("<cellName>#<index>"), recorded whatever the
  // bar happens to be (there may be none yet, or none ever this session).
  private subCellScale = new Map<string, number>();
  // Every cell fully seen this session or hydrated from the stored record: cell name -> the
  // tightest bound (the minimum over its sub-cells' own maxima).
  private seenAtScaleMap = new Map<string, number>();
  // The last value the server acknowledged for each cell (from hydrate or a successful push):
  // a cell is pending while its current value exceeds this or it has none here at all.
  private recordedMap = new Map<string, number>();
  private bar: WorkingScaleBar | null = null;
  private viewing: CoverageViewingInput | null = null;
  private dirty = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private postFn: (body: CoveragePayload) => Promise<CoveragePushResponse>,
    private opts: {
      debounceMs?: number;
      onChange?: () => void;
      /** A push failed while the tracker was still live; the detail is the caller's to surface
       *  (a toast), never silent. A push that fails after the tracker moved on goes to the
       *  module outbox instead, which reports a terminal drop through the same channel. */
      onPushError?: (detail: string) => void;
    } = {},
  ) {
    coverageOutbox.configure(postFn, (imagePath, detail, auditGap) =>
      this.opts.onPushError?.(
        auditGap
          ? `coverage progress for ${imagePath} was saved without its audit line: ${detail}`
          : `coverage progress for ${imagePath} was refused and dropped: ${detail}`,
      ),
    );
    liveTrackers.add(this);
  }

  /** Whether this tracker currently owes the server a fact (dirty, or a debounce still pending):
   *  App's `beforeunload` guard reads this across every live tracker (`anyTrackerDirty`), since
   *  React never unmounts a component on unload and a tracker's own `dispose` never runs then. */
  get isDirty(): boolean {
    return this.dirty;
  }

  /** Multi-cell grids only: single-cell grids (image within the display bound) get no tracking. */
  get active(): boolean {
    return this.keyParts !== null && this.grid !== null && this.cells.length > 1;
  }

  /** Every cell fully seen this session or hydrated, with its tightest recorded scale. */
  get seenAtScale(): ReadonlyMap<string, number> {
    return this.seenAtScaleMap;
  }

  /** Cells whose recorded scale meets the current working-scale bar: the derived "swept" set,
   *  never stored. `meetsBar` is the one comparison. */
  get swept(): ReadonlySet<string> {
    const out = new Set<string>();
    for (const [name, atScale] of this.seenAtScaleMap) {
      if (meetsBar(atScale, this.bar)) out.add(name);
    }
    return out;
  }

  /** Cells the server has recorded (hydrated or acknowledged) whose recorded value falls below
   *  the current bar, or every recorded cell while there is no bar to judge against: the
   *  chrome's "coarser" remainder line. Counts `recordedMap` (what is actually on record), never
   *  `seenAtScale` (which includes this session's own not-yet-acknowledged facts): a locally
   *  seen, not-yet-saved cell is "not yet saved," never "on record ... coarser". */
  get coarserCount(): number {
    let count = 0;
    for (const atScale of this.recordedMap.values()) {
      if (!meetsBar(atScale, this.bar)) count++;
    }
    return count;
  }

  /** Cells whose current facts have not yet been acknowledged by the server: newly seen since
   *  the last hydrate/push response, or raised past the acknowledged value. */
  get pending(): ReadonlySet<string> {
    const out = new Set<string>();
    for (const [name, atScale] of this.seenAtScaleMap) {
      const recorded = this.recordedMap.get(name);
      if (recorded === undefined || atScale > recorded) out.add(name);
    }
    return out;
  }

  get servedAtNative(): ReadonlySet<string> {
    return this.servedAtNativeSet;
  }

  get workingScaleBar(): WorkingScaleBar | null {
    return this.bar;
  }

  /**
   * Point the tracker at a new (image, subject, date, dataset, grid) identity. Any fact still
   * owed for the previous identity is flushed first; a push that fails there is handed to the
   * module outbox (see `flush`) rather than dropped, so a lost push still leaves a trace.
   */
  reset(keyParts: CoverageKeyParts | null, grid: GridGeometry | null, cells: GridCell[]): void {
    this.flush();
    this.keyParts = keyParts;
    this.grid = grid;
    this.cells = cells;
    this.servedAtNativeSet = new Set();
    this.subCells = new Map(
      cells.map((c) => [c.name, subdivideCell(c, subCellDivisionsFor(c, SUB_CELL_TARGET_PX))]),
    );
    this.subCellScale = new Map();
    this.seenAtScaleMap = new Map();
    this.recordedMap = new Map();
    this.bar = null;
    this.viewing = null;
    this.dirty = false;
    this.opts.onChange?.();
  }

  /** Adopt a stored record's cell facts, only when its grid matches the current one: every
   *  recorded value seeds every one of that cell's sub-cells as a floor (so a lower local
   *  observation this session never reads as a regression) and both `seenAtScale` and
   *  `recorded` start from it, so a hydrated cell reads acknowledged, not pending. */
  hydrate(record: CoverageRecord | null): void {
    if (!record || !this.grid || !record.grid || !sameGrid(record.grid, this.grid)) return;
    for (const name of record.cells_served_at_native) this.servedAtNativeSet.add(name);
    for (const [name, atScale] of Object.entries(record.cells_seen_at_scale)) {
      this.seenAtScaleMap.set(name, Math.max(this.seenAtScaleMap.get(name) ?? -Infinity, atScale));
      this.recordedMap.set(name, this.seenAtScaleMap.get(name)!);
      const subs = this.subCells.get(name);
      if (!subs) continue;
      for (let i = 0; i < subs.length; i++) {
        const key = `${name}#${i}`;
        this.subCellScale.set(key, Math.max(this.subCellScale.get(key) ?? -Infinity, atScale));
      }
    }
    this.opts.onChange?.();
  }

  /** The bar this image/subject is judged against, served by the completeness read and never
   *  accumulated here: a change re-derives `swept`/`pending` on the next read, no recomputation
   *  needed since both are read-time derivations. A structurally-equal bar (same fields, a new
   *  object reference from a caller that recomputed rather than genuinely changed) is a no-op:
   *  onChange fires only for a real change, so a caller whose own bar reference is unstable
   *  across renders can never drive this into a render loop through it. */
  setWorkingScaleBar(bar: WorkingScaleBar | null): void {
    if (sameWorkingScaleBar(this.bar, bar)) return;
    this.bar = bar;
    this.opts.onChange?.();
  }

  /**
   * The viewport now shows `rect` (image coords, clipped) at `scale`. A cell becomes seen once
   * every one of its sub-cells has, across any number of viewport moments (not necessarily this
   * one), been fully inside the viewport: the union-of-visibility generalization of "the whole
   * cell was on screen at once," which a raster whose cells exceed any working viewport can
   * never satisfy directly. Its recorded value is the minimum over its sub-cells' own maxima,
   * the tightest bound the tracker knows; a later pass that raises that bound still counts,
   * whether or not the cell was already seen. A push is scheduled only when a cell newly
   * becomes seen, never by a rise on an already-seen cell alone: a rise sets `dirty` (so a
   * `flush()` before any later event still carries it, a navigation dropping it otherwise) but
   * schedules nothing itself, and rides with whatever push a later event (a newly seen cell, a
   * served cell, a viewing change, or flush) sends next.
   */
  noteViewport(rect: PixelRect, scale: number): void {
    if (!this.active) return;
    let changed = false;
    for (const cell of this.cells) {
      if (!rectsOverlap(cell, rect)) continue;
      const subs = this.subCells.get(cell.name);
      if (!subs) continue;
      for (let i = 0; i < subs.length; i++) {
        if (!rectFullyInside(subs[i], rect)) continue;
        const key = `${cell.name}#${i}`;
        const prev = this.subCellScale.get(key) ?? -Infinity;
        if (scale > prev) this.subCellScale.set(key, scale);
      }
      if (!subs.every((_sub, i) => this.subCellScale.has(`${cell.name}#${i}`))) continue;
      const atScale = Math.min(
        ...subs.map((_sub, i) => this.subCellScale.get(`${cell.name}#${i}`)!),
      );
      const prevSeen = this.seenAtScaleMap.get(cell.name);
      if (prevSeen === undefined || atScale > prevSeen) {
        this.seenAtScaleMap.set(cell.name, atScale);
        if (prevSeen === undefined) {
          this.markDirty();
        } else {
          this.dirty = true;
        }
        changed = true;
      }
    }
    if (changed) this.opts.onChange?.();
  }

  /** A region serve's response covered this cell at its native dimensions. */
  noteServedAtNative(cellName: string): void {
    if (!this.active || this.servedAtNativeSet.has(cellName)) return;
    if (!this.cells.some((c) => c.name === cellName)) return;
    this.servedAtNativeSet.add(cellName);
    this.markDirty();
  }

  setViewing(viewing: CoverageViewingInput): void {
    const changed = this.dirtyOnViewingChange(viewing);
    this.viewing = viewing;
    if (changed) this.markDirty();
  }

  private dirtyOnViewingChange(viewing: CoverageViewingInput): boolean {
    return this.viewing !== null && JSON.stringify(this.viewing) !== JSON.stringify(viewing);
  }

  /** The facts behind the Complete warning, or null when the warning does not apply: no bar to
   *  judge against, or every cell already meets it. */
  completeWarning(): { unsweptCount: number; total: number; bar: number } | null {
    if (!this.active || this.bar === null) return null;
    const bar = this.bar;
    const unsweptCount = this.cells.filter(
      (c) => !meetsBar(this.seenAtScaleMap.get(c.name) ?? null, bar),
    ).length;
    if (unsweptCount === 0) return null;
    return { unsweptCount, total: this.cells.length, bar: bar.value };
  }

  /** Send any owed facts now, at once (called before a reset and on unmount): the identity
   *  switching away must never defer its own outgoing image's push behind the outbox's own
   *  retry cadence, so this posts immediately under the outgoing image's own key and only hands
   *  the payload to the module outbox when that post itself fails -- the tracker is about to
   *  move on (or vanish) and can no longer retry it itself, but a fresh failure there starts the
   *  outbox's own retry timer from the failure, never from this call. */
  flush(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (!this.dirty || !this.keyParts || !this.grid || !this.viewing) {
      this.dirty = false;
      return;
    }
    const body = this.buildPayload(this.keyParts, this.grid, this.viewing);
    this.dirty = false;
    void this.postFn(body).then(
      () => {},
      () => coverageOutbox.enqueue(body),
    );
  }

  dispose(): void {
    this.flush();
    liveTrackers.delete(this);
  }

  private markDirty(): void {
    this.dirty = true;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.postNow();
    }, this.opts.debounceMs ?? POST_DEBOUNCE_MS);
  }

  private buildPayload(
    keyParts: CoverageKeyParts,
    grid: GridGeometry,
    viewing: CoverageViewingInput,
  ): CoveragePayload {
    const cellsSeenAtScale: Record<string, number> = {};
    for (const [name, atScale] of this.seenAtScaleMap) cellsSeenAtScale[name] = atScale;
    return {
      image_path: keyParts.imagePath,
      dataset_root: keyParts.datasetRoot,
      subject: keyParts.subject,
      date: keyParts.date,
      grid,
      cells_served_at_native: Array.from(this.servedAtNativeSet).sort(),
      cells_seen_at_scale: cellsSeenAtScale,
      viewing,
    };
  }

  private postNow(): void {
    if (!this.dirty || !this.keyParts || !this.grid || !this.viewing) return;
    this.dirty = false;
    const body = this.buildPayload(this.keyParts, this.grid, this.viewing);
    void this.postFn(body).then(
      (res) => this.adoptRecorded(res.record.cells_seen_at_scale),
      (err: unknown) => {
        this.dirty = true;
        this.opts.onPushError?.(err instanceof Error ? err.message : String(err));
        // A failed push must not go quiet: rescheduling here is the retry.
        this.markDirty();
      },
    );
  }

  /** Adopt `post_coverage`'s own merged answer into both `seenAtScale` and `recorded` (the same
   *  merge `hydrate` applies to a fresh read), never the posted body: the response is the
   *  server's authoritative merge, which can hold a cell another tab or session contributed that
   *  never rode in this payload at all, and can hold a value higher than this payload's own on a
   *  concurrent write. Reading the body instead would credit only what this tab itself sent. */
  private adoptRecorded(recorded: Record<string, number>): void {
    for (const [name, atScale] of Object.entries(recorded)) {
      this.seenAtScaleMap.set(name, Math.max(this.seenAtScaleMap.get(name) ?? -Infinity, atScale));
      this.recordedMap.set(name, Math.max(this.recordedMap.get(name) ?? -Infinity, atScale));
    }
    this.opts.onChange?.();
  }
}
