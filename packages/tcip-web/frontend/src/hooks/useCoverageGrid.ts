/**
 * The coverage lattice for the open raster at a subject's set grid zoom, plus the
 * zoom-independent region-serving grid. Fetched once the canvas host has a real measurement
 * (never re-measured on a later resize: the viewport is captured once and reused for every
 * later fetch, including a re-derive); with no subject or no set zoom, `grid` is null and
 * `reason` names why. Cells always come from the route; nothing is derived client-side.
 */

import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import { measureCanvasHost } from "@/lib/canvasSync";
import type { GridCell, RenderedGrid } from "@/lib/coverage";

export interface CoverageGridState {
  grid: RenderedGrid | null;
  cells: GridCell[];
  /** How the tile size was chosen, or null while no grid is loaded. */
  derivation: string | null;
  /** Why `grid` is null: no subject, no set zoom, or the canvas host not yet measured. Null once
   *  a grid is loaded (or the fetch itself failed, see `error`). */
  reason: string | null;
  /** Set only when `grid` came from an already-worked image's own recorded lattice and the
   *  subject's current zoom would derive a different tile size; null otherwise. */
  freshDerivationDiffers: boolean | null;
  /** The zoom-independent region-serving grid: always present once the fetch answers. */
  serving: RenderedGrid | null;
  servingCells: GridCell[];
  /** The grid fetch's own refusal, or null; a failure must read as an error, never as "no
   *  grid needed here". */
  error: string | null;
  /** True once the fetch has answered (a grid, a reason, or an error), false while still
   *  unknown: the Map fallback and the chrome tell "not yet answered" from "no lattice". */
  settled: boolean;
  /** Refetch the grid at the currently open image/subject, keeping any already-worked lattice
   *  (``rederive`` stays false): a set-zoom write calls this, so a worked image's own lattice
   *  survives the change and ``freshDerivationDiffers`` is what offers a re-derive. */
  refetch: () => void;
  /** Ignore this image's already-worked lattice and derive fresh at the current zoom, for
   *  exactly the one fetch this triggers: consumed by that fetch alone, never sticky, and
   *  cleared outright by a change of image or subject. */
  rederiveLattice: () => void;
}

interface FetchedState {
  path: string;
  grid: RenderedGrid | null;
  cells: GridCell[];
  derivation: string | null;
  reason: string | null;
  freshDerivationDiffers: boolean | null;
  serving: RenderedGrid | null;
  servingCells: GridCell[];
  error: string | null;
  answered: boolean;
}

const EMPTY_FETCHED: FetchedState = {
  path: "",
  grid: null,
  cells: [],
  derivation: null,
  reason: null,
  freshDerivationDiffers: null,
  serving: null,
  servingCells: [],
  error: null,
  answered: false,
};

export function useCoverageGrid(args: {
  imagePath: string | null;
  subject: string | null;
  date: string | null;
  datasetRoot: string | null;
}): CoverageGridState {
  const { imagePath, subject, date, datasetRoot } = args;
  const [state, setState] = useState<FetchedState>(EMPTY_FETCHED);
  const fetchingKeyRef = useRef<string | null>(null);
  const viewportRef = useRef<{ w: number; h: number } | null>(null);
  const [refetchNonce, setRefetchNonce] = useState(0);
  // One-shot: true only for the fetch a rederiveLattice() press triggers, consumed the moment
  // that fetch fires, never sticky across later fetches or a plain refetch().
  const rederiveOnceRef = useRef(false);

  useEffect(() => {
    rederiveOnceRef.current = false;
  }, [imagePath, subject]);

  useEffect(() => {
    const path = imagePath;
    if (!path) return;
    let cancelled = false;
    let raf: number | null = null;

    const attempt = (): void => {
      if (cancelled) return;
      const host = viewportRef.current ?? measureCanvasHost();
      if (!host) {
        raf = requestAnimationFrame(attempt);
        return;
      }
      viewportRef.current = host;
      const key = `${path}|${subject ?? ""}|${date ?? ""}|${datasetRoot ?? ""}|${refetchNonce}`;
      if (fetchingKeyRef.current === key) return;
      fetchingKeyRef.current = key;
      const rederive = rederiveOnceRef.current;
      rederiveOnceRef.current = false;
      void api.coverage
        .grid(path, {
          subject,
          date,
          datasetRoot,
          viewportW: Math.round(host.w),
          viewportH: Math.round(host.h),
          rederive,
        })
        .then(
          (res) => {
            if (cancelled) return;
            const { cells: servingCells, derivation: servingDerivation, ...serving } = res.serving;
            setState({
              path,
              grid: res.grid,
              cells: res.grid?.cells ?? [],
              derivation: res.grid?.derivation ?? null,
              reason: res.reason,
              freshDerivationDiffers: res.fresh_derivation_differs,
              serving: { ...serving, derivation: servingDerivation, cells: servingCells },
              servingCells,
              error: null,
              answered: true,
            });
          },
          (e: unknown) => {
            if (cancelled) return;
            fetchingKeyRef.current = null;
            const detail = e instanceof Error ? e.message : String(e);
            setState({ ...EMPTY_FETCHED, path, error: detail, answered: true });
          },
        );
    };
    attempt();
    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, [imagePath, subject, date, datasetRoot, refetchNonce]);

  const current = state.path === imagePath && imagePath ? state : EMPTY_FETCHED;
  return {
    grid: current.grid,
    cells: current.cells,
    derivation: current.derivation,
    reason: current.reason,
    freshDerivationDiffers: current.freshDerivationDiffers,
    serving: current.serving,
    servingCells: current.servingCells,
    error: current.error,
    settled: current.answered,
    refetch: () => setRefetchNonce((n) => n + 1),
    rederiveLattice: () => {
      rederiveOnceRef.current = true;
      setRefetchNonce((n) => n + 1);
    },
  };
}
