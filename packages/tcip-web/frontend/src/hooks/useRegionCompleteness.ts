/**
 * Wires the region-completeness store into the Annotate tab: fetches every subject's
 * attestation record for the open raster, exposes the active subject's own complete cells
 * separate from every other subject's, and posts a double-click toggle. Unlike view coverage
 * (an advisory, session-accumulated sweep fact), an attestation is an explicit, immediate action
 * with no client-side session state to track: the store itself is the only source of truth, so
 * every toggle just refetches it.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "@/api/client";
import { effectiveComplete, type CompletenessRecord, type GridGeometry } from "@/lib/coverage";
import { useStore } from "@/store";

export interface RegionCompleteness {
  /** Cells attested complete for the active subject (stale attestations excluded). */
  activeComplete: ReadonlySet<string>;
  /** Cells attested complete for some other subject, not the active one (stale excluded). */
  otherComplete: ReadonlySet<string>;
  /** Toggle one cell's completeness for the active subject; a no-op with no active subject. */
  toggle: (cell: string, grid: GridGeometry) => void;
}

export function useRegionCompleteness(args: {
  imagePath: string | null;
  datasetRoot: string | null;
  subject: string | null;
}): RegionCompleteness {
  const [bySubject, setBySubject] = useState<Record<string, CompletenessRecord>>({});
  const { imagePath, datasetRoot, subject } = args;

  const reload = useCallback(() => {
    if (!imagePath) {
      setBySubject({});
      return;
    }
    void api.coverage.completeness(imagePath, datasetRoot).then(
      (res) => setBySubject(res.by_subject),
      () => {
        // No stored record readable: nothing attested yet for this raster.
        setBySubject({});
      },
    );
  }, [imagePath, datasetRoot]);

  useEffect(() => {
    reload();
  }, [reload]);

  const activeComplete = useMemo(
    () => effectiveComplete(subject ? bySubject[subject] : undefined),
    [bySubject, subject],
  );

  const otherComplete = useMemo(() => {
    const out = new Set<string>();
    for (const [subj, record] of Object.entries(bySubject)) {
      if (subj === subject) continue;
      for (const cell of effectiveComplete(record)) out.add(cell);
    }
    return out;
  }, [bySubject, subject]);

  const toggle = useCallback(
    (cell: string, grid: GridGeometry) => {
      if (!imagePath || !subject) return;
      void api.coverage
        .toggleCompleteness({
          image_path: imagePath,
          dataset_root: datasetRoot,
          subject,
          grid,
          cell,
          user: useStore.getState().user,
        })
        .then(reload, (err: unknown) => {
          // A write can fail after partially committing, so reload() runs on error too, the
          // same as success, rather than trusting a "the store is untouched" assumption.
          const detail = err instanceof Error ? err.message : String(err);
          useStore
            .getState()
            .pushToast(`Failed to update completeness for cell ${cell}: ${detail}`);
          reload();
        });
    },
    [imagePath, datasetRoot, subject, reload],
  );

  return { activeComplete, otherComplete, toggle };
}
