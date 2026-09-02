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
}

const EMPTY: CoverageGridState = { grid: null, cells: [], derivation: null, error: null };

export function useCoverageGrid(
  imagePath: string | null,
  baseFacts: LoadedImage | null,
  imgW: number,
  imgH: number,
): CoverageGridState {
  const [state, setState] = useState<{ path: string } & CoverageGridState>({ path: "", ...EMPTY });
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
        setState({ path: imagePath, ...EMPTY, error: detail });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [imagePath, belowNative]);

  return state.path === imagePath && imagePath ? state : EMPTY;
}
