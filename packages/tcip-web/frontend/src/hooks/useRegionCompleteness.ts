/**
 * Wires the region-completeness store into the Annotate tab: fetches every subject's
 * attestation record and saved-annotation count for the open raster, exposes the active
 * subject's own complete/stale cells separate from every other subject's, and posts an explicit
 * attest/unattest/re-attest write. The store itself is the only source of truth, so every write
 * just refetches it.
 *
 * A record whose grid disagrees with the raster's current grid is never rendered on it (a grid
 * derivation can change between an attestation and now); its cell count and the record's own
 * grid dims are reported instead, so the breeder knows earlier work exists without it being
 * silently misread against the wrong lattice.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "@/api/client";
import {
  effectiveComplete,
  sameGrid,
  type CompletenessRecord,
  type GridGeometry,
} from "@/lib/coverage";
import { useStore } from "@/store";

export interface OtherLatticeAttestation {
  count: number;
  cols: number;
  rows: number;
}

export interface RegionCompleteness {
  /** Cells attested complete for the active subject on the current grid, stale ones excluded. */
  activeComplete: ReadonlySet<string>;
  /** Cells attested complete for the active subject on the current grid but now stale. */
  activeStale: ReadonlySet<string>;
  /** Cells attested complete for another subject on the current grid, stale ones excluded. */
  otherComplete: ReadonlySet<string>;
  /** The same cells as `otherComplete`, kept per owning subject so a state list can name which
   *  subject each attestation belongs to ("attested for leaf: B1"). */
  otherCompleteBySubject: Readonly<Record<string, readonly string[]>>;
  /** The active subject's attestation on a grid other than the current one, or null: also null
   *  while the current grid itself is unknown, since "on a previous lattice" is a comparison
   *  this hook cannot make without one. */
  otherLattice: OtherLatticeAttestation | null;
  /** The active subject's saved-annotation count per cell, from the current grid; empty when the
   *  counts were binned on a different lattice than the one now showing (see `countsError`). */
  annotationCounts: Record<string, number>;
  /** The completeness read's own failure, surfaced rather than read as "nothing attested". */
  error: string | null;
  /** Why `annotationCounts` is empty: the read's own `counts_error` (a raster the grid could not
   *  be derived for), or a stated lattice mismatch when the counts were served for a grid other
   *  than the one now showing; null while the counts are current. */
  countsError: string | null;
  /** Re-fetch the current image's record and counts, discarding nothing local: the store is the
   *  only source of truth, so a save that may have changed staleness or counts calls this. */
  reload: () => void;
  /** Attest, unattest or re-attest one cell in the direction ``complete`` states. */
  write: (cell: string, grid: GridGeometry, complete: boolean) => void;
}

export function useRegionCompleteness(args: {
  imagePath: string | null;
  datasetRoot: string | null;
  subject: string | null;
  grid: GridGeometry | null;
}): RegionCompleteness {
  const [bySubject, setBySubject] = useState<Record<string, CompletenessRecord>>({});
  const [countsBySubject, setCountsBySubject] = useState<Record<string, Record<string, number>>>(
    {},
  );
  const [countsGrid, setCountsGrid] = useState<GridGeometry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [readCountsError, setReadCountsError] = useState<string | null>(null);
  const { imagePath, datasetRoot, subject, grid } = args;

  const reload = useCallback(() => {
    if (!imagePath) {
      setBySubject({});
      setCountsBySubject({});
      setCountsGrid(null);
      setError(null);
      setReadCountsError(null);
      return;
    }
    void api.coverage.completeness(imagePath, datasetRoot).then(
      (res) => {
        setBySubject(res.by_subject);
        setCountsBySubject(res.annotation_counts);
        setCountsGrid(res.counts_grid);
        setError(null);
        setReadCountsError(res.counts_error);
      },
      (err: unknown) => {
        setBySubject({});
        setCountsBySubject({});
        setCountsGrid(null);
        setError(err instanceof Error ? err.message : String(err));
        setReadCountsError(null);
      },
    );
  }, [imagePath, datasetRoot]);

  useEffect(() => {
    reload();
  }, [reload]);

  const activeRecord = subject ? bySubject[subject] : undefined;
  const activeOnGrid = !!activeRecord && !!grid && sameGrid(activeRecord.grid, grid);

  const activeComplete = useMemo(
    () => (activeOnGrid ? effectiveComplete(activeRecord) : new Set<string>()),
    [activeRecord, activeOnGrid],
  );

  const activeStale = useMemo(
    () => (activeOnGrid && activeRecord ? new Set(activeRecord.stale_cells) : new Set<string>()),
    [activeRecord, activeOnGrid],
  );

  const otherLattice = useMemo<OtherLatticeAttestation | null>(() => {
    // With no current grid to compare against, "on a previous lattice" is a claim this hook
    // cannot make: stating it anyway would assert a fact the data does not carry.
    if (!activeRecord || !grid || activeOnGrid) return null;
    return {
      count: activeRecord.cells_complete.length,
      cols: activeRecord.grid.cols,
      rows: activeRecord.grid.rows,
    };
  }, [activeRecord, grid, activeOnGrid]);

  const otherCompleteBySubject = useMemo(() => {
    const out: Record<string, string[]> = {};
    for (const [subj, record] of Object.entries(bySubject)) {
      if (subj === subject || !grid || !sameGrid(record.grid, grid)) continue;
      const cells = Array.from(effectiveComplete(record)).sort();
      if (cells.length) out[subj] = cells;
    }
    return out;
  }, [bySubject, subject, grid]);

  const otherComplete = useMemo(() => {
    const out = new Set<string>();
    for (const cells of Object.values(otherCompleteBySubject)) {
      for (const cell of cells) out.add(cell);
    }
    return out;
  }, [otherCompleteBySubject]);

  const countsOnGrid = !!grid && !!countsGrid && sameGrid(countsGrid, grid);

  const annotationCounts = useMemo(
    () => (subject && countsOnGrid ? (countsBySubject[subject] ?? {}) : {}),
    [countsBySubject, subject, countsOnGrid],
  );

  const countsError = useMemo(() => {
    if (readCountsError) return readCountsError;
    if (grid && countsGrid && !countsOnGrid) {
      return "saved-annotation counts were binned on a lattice other than the one now showing";
    }
    return null;
  }, [readCountsError, grid, countsGrid, countsOnGrid]);

  const write = useCallback(
    (cell: string, writeGrid: GridGeometry, complete: boolean) => {
      if (!imagePath || !subject) return;
      void api.coverage
        .setCompleteness({
          image_path: imagePath,
          dataset_root: datasetRoot,
          subject,
          grid: writeGrid,
          cell,
          complete,
          user: useStore.getState().user,
        })
        .then(reload, (err: unknown) => {
          // A write can fail after partially committing, so reload() runs on error too, the
          // same as success, rather than trusting a "the store is untouched" assumption.
          const detail = err instanceof Error ? err.message : String(err);
          useStore
            .getState()
            .pushToast(`Could not update completeness for cell ${cell}: ${detail}`);
          reload();
        });
    },
    [imagePath, datasetRoot, subject, reload],
  );

  return {
    activeComplete,
    activeStale,
    otherComplete,
    otherCompleteBySubject,
    otherLattice,
    annotationCounts,
    error,
    countsError,
    reload,
    write,
  };
}
