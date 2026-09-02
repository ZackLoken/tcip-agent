/**
 * Wires the CoverageTracker into the Annotate tab: resets on the (image, subject, date,
 * dataset, grid) identity, hydrates from the stored record, feeds it viewport passes and the
 * viewing context, and exposes the swept set for the coverage grid overlay plus the Complete
 * warning facts. No active subject means no accumulation and no POST. A stored record on a grid
 * other than the current one hydrates nothing (the tracker's own rule) but is stated in
 * `sweptOtherLattice` rather than silently vanishing; a failed push surfaces through the tab's
 * toast path and is retried by the tracker itself.
 *
 * Only the Annotate tab calls this hook, but the band selection its `viewing.bands`/`stretch`
 * carries is held per band-set signature across both tabs (`useBandSelection`), so a composite
 * chosen in the Review tab is the one this tab's coverage record carries too, once the two tabs
 * share the same band set.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/api/client";
import { completeWarningMessage, sameGrid, type GridCell, type GridGeometry } from "@/lib/coverage";
import { CoverageTracker, type CoverageViewingInput } from "@/lib/coverageTracker";
import { computeViewport, measureCanvasHost } from "@/lib/canvasSync";
import { useStore } from "@/store";
import type { ViewState } from "@/store/types";

/** A stored view-coverage record for this image/subject on a grid other than the one now
 *  showing: the sweep-history counterpart of region-completeness's `otherLattice`. */
export interface SweptOtherLattice {
  count: number;
  cols: number;
  rows: number;
}

export interface CoverageTracking {
  swept: ReadonlySet<string>;
  noteAuthoringCommit: () => void;
  noteServedAtNative: (cellName: string) => void;
  /** The Complete warning's wording, or null when the warning does not apply. */
  completeWarning: () => string | null;
  /** A sweep record on a grid other than the current one, or null: stated in the chrome rather
   *  than silently dropped the way an unmatched grid's hydration already is. */
  sweptOtherLattice: SweptOtherLattice | null;
}

export function useCoverageTracking(args: {
  imagePath: string | null;
  datasetRoot: string | null;
  subject: string | null;
  date: string | null;
  grid: GridGeometry | null;
  cells: GridCell[];
  view: ViewState;
  imgW: number;
  imgH: number;
  viewing: CoverageViewingInput;
}): CoverageTracking {
  // Bumped whenever the tracker's own facts change, purely to give the memo below a dependency
  // to recompute on: nothing reads the count itself.
  const [version, setVersion] = useState(0);
  const trackerRef = useRef<CoverageTracker | null>(null);
  if (trackerRef.current === null) {
    trackerRef.current = new CoverageTracker((body) => api.coverage.push(body), {
      onChange: () => setVersion((v) => v + 1),
      onPushError: (detail) =>
        useStore.getState().pushToast(`Could not save coverage progress: ${detail}`),
    });
  }
  const tracker = trackerRef.current;
  const [sweptOtherLattice, setSweptOtherLattice] = useState<SweptOtherLattice | null>(null);

  const { imagePath, datasetRoot, subject, date, grid, cells } = args;
  useEffect(() => {
    if (!imagePath || !subject || !grid || cells.length === 0) {
      tracker.reset(null, null, []);
      setSweptOtherLattice(null);
      return;
    }
    tracker.reset({ imagePath, datasetRoot, subject, date }, grid, cells);
    let cancelled = false;
    void api.coverage.get(imagePath, subject, date).then(
      (record) => {
        if (cancelled) return;
        tracker.hydrate(record);
        setSweptOtherLattice(
          record && record.grid && !sameGrid(record.grid, grid) && record.cells_swept.length > 0
            ? { count: record.cells_swept.length, cols: record.grid.cols, rows: record.grid.rows }
            : null,
        );
      },
      (err: unknown) => {
        // A missing record resolves as null above; a rejection here is a real refusal, so it
        // surfaces rather than accumulating silently as if nothing were stored.
        if (cancelled) return;
        setSweptOtherLattice(null);
        const detail = err instanceof Error ? err.message : String(err);
        useStore.getState().pushToast(`Could not read the stored coverage record: ${detail}`);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [tracker, imagePath, datasetRoot, subject, date, grid, cells]);

  const { view, imgW, imgH } = args;
  useEffect(() => {
    const host = measureCanvasHost();
    if (!host) return;
    const viewport = computeViewport(view, host, imgW, imgH);
    if (!viewport) return;
    tracker.noteViewport(
      {
        x0: viewport.x,
        y0: viewport.y,
        x1: viewport.x + viewport.w,
        y1: viewport.y + viewport.h,
      },
      view.scale,
    );
  }, [tracker, view, imgW, imgH]);

  const { viewing } = args;
  useEffect(() => {
    tracker.setViewing(viewing);
  }, [tracker, viewing]);

  useEffect(() => () => tracker.dispose(), [tracker]);

  return useMemo(
    () => ({
      swept: tracker.swept,
      noteAuthoringCommit: () => tracker.noteAuthoringScale(useStore.getState().gui.view.scale),
      noteServedAtNative: (cellName: string) => tracker.noteServedAtNative(cellName),
      completeWarning: () => {
        const facts = tracker.completeWarning();
        return facts ? completeWarningMessage(facts) : null;
      },
      sweptOtherLattice,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version forces the recompute a mutable Set's own mutation gives no other signal for.
    [tracker, version, sweptOtherLattice],
  );
}
