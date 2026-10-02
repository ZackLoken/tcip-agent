/**
 * The region-serving grid over the open raster, fetched once per image from the route that
 * serves the raster; cells always come from the route, nothing is derived client-side.
 */

import { useEffect, useState } from "react";

import { api } from "@/api/client";
import type { ServingGrid } from "@/api/types.generated";

export function useServingGrid(imagePath: string | null): ServingGrid | null {
  const [state, setState] = useState<{ path: string; grid: ServingGrid } | null>(null);

  useEffect(() => {
    if (!imagePath) return;
    let canceled = false;
    void api.images.servingGrid(imagePath).then(
      (grid) => {
        if (!canceled) setState({ path: imagePath, grid });
      },
      () => {
        // No serving grid serves the base bitmap alone: region serving is an optimization.
      },
    );
    return () => {
      canceled = true;
    };
  }, [imagePath]);

  return state && state.path === imagePath ? state.grid : null;
}
