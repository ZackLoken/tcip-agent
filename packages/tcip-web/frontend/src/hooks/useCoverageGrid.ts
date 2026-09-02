/**
 * The coverage lattice for the open raster, fetched once the base serve shows the raster is
 * larger than one display-bounded serve (Served-Size below the native dims). An ordinary image
 * inside the display bound never fetches a grid: its single serve is the whole image, so there
 * is nothing to track. Cells always come from the route; nothing is derived client-side.
 */

import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import type { GridCell, GridGeometry } from "@/lib/coverage";
import type { LoadedImage } from "@/lib/imageLoader";

export interface CoverageGridState {
  grid: GridGeometry | null;
  cells: GridCell[];
  /** How the tile size was chosen, or null while no grid is loaded. */
  derivation: string | null;
  /** The grid fetch's own refusal, or null; a failure must read as an error, never as "no
   *  grid needed here". */
  error: string | null;
  /** True while whether this image needs a grid at all is still unknown (the base serve has not
   *  reported back yet) or a needed fetch is genuinely in flight (the base serve already shows
   *  the raster below native, and neither a grid nor an error has landed yet). Tells "still
   *  unknown" apart from "settled, and there is no multi-cell grid". */
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

export function useCoverageGrid(
  imagePath: string | null,
  baseFacts: LoadedImage | null,
  imgW: number,
  imgH: number,
): CoverageGridState {
  const [state, setState] = useState<FetchedState>(EMPTY_FETCHED);
  const fetchingRef = useRef<string | null>(null);

  const served = baseFacts?.ok ? baseFacts.servedSize : null;
  const belowNative = !!served && imgW > 0 && imgH > 0 && (served.w < imgW || served.h < imgH);

  useEffect(() => {
    if (!imagePath || !belowNative) return;
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
  }, [imagePath, belowNative]);

  const current = state.path === imagePath && imagePath ? state : EMPTY_FETCHED;
  // Derived directly rather than from the effect above: whether a fetch is coming (or even
  // whether one is needed) is knowable a render earlier than that effect settles it.
  const pending = !!imagePath && (!baseFacts || (belowNative && !current.grid && !current.error));
  return {
    grid: current.grid,
    cells: current.cells,
    derivation: current.derivation,
    error: current.error,
    pending,
  };
}
