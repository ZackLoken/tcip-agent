/**
 * Wires the CoverageTracker into the Annotate tab: resets on the (image, subject, date,
 * dataset, grid) identity, hydrates from the stored record, feeds it viewport passes, the
 * viewing context and the subject's working-scale bar (derived server-side, never accumulated
 * from an authoring commit), and exposes the seen/swept/pending sets for the coverage grid
 * overlay plus the Complete warning facts. No active subject means no accumulation and no POST.
 * A stored record on a grid other than the current one hydrates nothing and sets the tracker's
 * own replace hold instead, read here as `replaceRequired` rather than silently vanishing; a
 * failed push surfaces through the tab's toast path while the tracker is still live, and is
 * handed to the module-level outbox (see `coverageTracker.ts`) once it moves on.
 *
 * Only the Annotate tab calls this hook, but the band selection its `viewing.bands`/`stretch`
 * carries is held per band-set signature across both tabs (`useBandSelection`), so a composite
 * chosen in the Review tab is the one this tab's coverage record carries too, once the two tabs
 * share the same band set.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/api/client";
import type { WorkingScaleBar } from "@/api/types.generated";
import { completeWarningMessage, type GridCell, type GridGeometry } from "@/lib/coverage";
import {
  CoverageTracker,
  type CoverageViewingInput,
  type ReplaceRequired,
} from "@/lib/coverageTracker";
import { computeViewport, measureCanvasHost } from "@/lib/canvasSync";
import { useStore } from "@/store";
import type { ViewState } from "@/store/types";

export interface CoverageTracking {
  /** Cells whose recorded scale meets the subject's working-scale bar: the derived "swept" set. */
  swept: ReadonlySet<string>;
  /** Cells fully seen locally whose facts have not yet been acknowledged by the server. */
  pending: ReadonlySet<string>;
  /** Cells recorded seen (this session or hydrated) but not meeting the current bar, or every
   *  recorded cell while there is no bar to judge them against: the chrome's "coarser" count. */
  coarserCount: number;
  noteServedAtNative: (cellName: string) => void;
  /** The Complete warning's wording, or null when the warning does not apply. */
  completeWarning: () => string | null;
  /** The tracker's own replace hold: a stored sweep record on a grid other than the current
   *  one, or null. Owned by the tracker (`hydrate`'s grid-mismatch disjunct and a push's own
   *  409), read here rather than re-derived. */
  replaceRequired: ReplaceRequired | null;
  /** Confirm discarding the previous lattice's sweeps and post immediately with `replace: true`. */
  armReplace: () => void;
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
  workingScale: WorkingScaleBar | null;
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

  const { imagePath, datasetRoot, subject, date, grid, cells } = args;
  useEffect(() => {
    if (!imagePath || !subject || !grid || cells.length === 0) {
      tracker.reset(null, null, []);
      return;
    }
    tracker.reset({ imagePath, datasetRoot, subject, date }, grid, cells);
    let cancelled = false;
    void api.coverage.get(imagePath, subject, date).then(
      (record) => {
        if (cancelled) return;
        tracker.hydrate(record);
      },
      (err: unknown) => {
        // A missing record resolves as null above; a rejection here is a real refusal, so it
        // surfaces rather than accumulating silently as if nothing were stored.
        if (cancelled) return;
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

  const { workingScale } = args;
  useEffect(() => {
    tracker.setWorkingScaleBar(workingScale);
  }, [tracker, workingScale]);

  useEffect(() => () => tracker.dispose(), [tracker]);

  return useMemo(
    () => ({
      swept: tracker.swept,
      pending: tracker.pending,
      coarserCount: tracker.coarserCount,
      noteServedAtNative: (cellName: string) => tracker.noteServedAtNative(cellName),
      completeWarning: () => {
        const facts = tracker.completeWarning();
        return facts ? completeWarningMessage(facts) : null;
      },
      replaceRequired: tracker.replaceRequired,
      armReplace: () => tracker.armReplace(),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version forces the recompute a mutable Set's own mutation gives no other signal for.
    [tracker, version],
  );
}
