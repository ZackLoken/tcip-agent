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
  /** The active subject's attestation on a grid other than the current one, or null. */
  otherLattice: OtherLatticeAttestation | null;
  /** The active subject's saved-annotation count per cell, from the current grid. */
  annotationCounts: Record<string, number>;
  /** The completeness read's own failure, surfaced rather than read as "nothing attested". */
  error: string | null;
  /** The read's grid_error: annotation_counts unavailable (an incomplete band group); by_subject
   *  still served. */
  countsError: string | null;
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
  const [error, setError] = useState<string | null>(null);
  const [countsError, setCountsError] = useState<string | null>(null);
  const { imagePath, datasetRoot, subject, grid } = args;

  const reload = useCallback(() => {
    if (!imagePath) {
      setBySubject({});
      setCountsBySubject({});
      setError(null);
      setCountsError(null);
      return;
    }
    void api.coverage.completeness(imagePath, datasetRoot).then(
      (res) => {
        setBySubject(res.by_subject);
        setCountsBySubject(res.annotation_counts);
        setError(null);
        setCountsError(res.grid_error);
      },
      (err: unknown) => {
        setBySubject({});
        setCountsBySubject({});
        setError(err instanceof Error ? err.message : String(err));
        setCountsError(null);
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
    if (!activeRecord || activeOnGrid) return null;
    return {
      count: activeRecord.cells_complete.length,
      cols: activeRecord.grid.cols,
      rows: activeRecord.grid.rows,
    };
  }, [activeRecord, activeOnGrid]);

  const otherComplete = useMemo(() => {
    const out = new Set<string>();
    for (const [subj, record] of Object.entries(bySubject)) {
      if (subj === subject || !grid || !sameGrid(record.grid, grid)) continue;
      for (const cell of effectiveComplete(record)) out.add(cell);
    }
    return out;
  }, [bySubject, subject, grid]);

  const annotationCounts = useMemo(
    () => (subject ? (countsBySubject[subject] ?? {}) : {}),
    [countsBySubject, subject],
  );

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
    otherLattice,
    annotationCounts,
    error,
    countsError,
    write,
  };
}
