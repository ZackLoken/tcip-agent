/**
 * The coverage lattice for the open raster, fetched for every raster once its path is known: an
 * ordinary image inside the display bound derives a trivial one-cell lattice, which still names
 * its own derivation and carries the one cell the chrome names and attests. Cells always come
 * from the route; nothing is derived client-side.
 */

import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import type { GridCell, GridGeometry } from "@/lib/coverage";

export interface CoverageGridState {
  grid: GridGeometry | null;
  cells: GridCell[];
  /** How the tile size was chosen, or null while no grid is loaded. */
  derivation: string | null;
  /** The grid fetch's own refusal, or null; a failure must read as an error, never as "no
   *  grid needed here". */
  error: string | null;
  /** True while the fetch for the open image has not yet settled (neither a grid nor an error
   *  has landed). Tells "still unknown" apart from "settled, and there is no multi-cell grid". */
  pending: boolean;
}

interface FetchedState {
  path: string;
  grid: GridGeometry | null;
  cells: GridCell[];
  derivation: string | null;
  error: string | null;
}

const EMPTY_FETCHED: FetchedState = {
  path: "",
  grid: null,
  cells: [],
  derivation: null,
  error: null,
};

export function useCoverageGrid(imagePath: string | null): CoverageGridState {
  const [state, setState] = useState<FetchedState>(EMPTY_FETCHED);
  const fetchingRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imagePath) return;
    if (fetchingRef.current === imagePath) return;
    fetchingRef.current = imagePath;
    let cancelled = false;
    void api.coverage.grid(imagePath).then(
      (res) => {
        if (cancelled) return;
        const { cells, derivation, ...grid } = res;
        setState({ path: imagePath, grid, cells, derivation, error: null });
      },
      (e: unknown) => {
        if (cancelled) return;
        if (fetchingRef.current === imagePath) fetchingRef.current = null;
        const detail = e instanceof Error ? e.message : String(e);
        setState({ path: imagePath, grid: null, cells: [], derivation: null, error: detail });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [imagePath]);

  const current = state.path === imagePath && imagePath ? state : EMPTY_FETCHED;
  const pending = !!imagePath && !current.grid && !current.error;
  return {
    grid: current.grid,
    cells: current.cells,
    derivation: current.derivation,
    error: current.error,
    pending,
  };
}
