import { useEffect, useState } from "react";

import { api, type ImageBandsResponse } from "@/api/client";

/**
 * Band symbology for one image path, refetched whenever the path changes. Null while loading, for
 * a path-less call, or on a failed fetch — callers treat null as "nothing to show yet," never as a
 * signal to render band controls (only `band_count > 3` does that, never shown for a plain RGB
 * dataset).
 */
export function useImageBands(imagePath: string | null): ImageBandsResponse | null {
  const [result, setResult] = useState<ImageBandsResponse | null>(null);

  useEffect(() => {
    if (!imagePath) {
      setResult(null);
      return;
    }
    let cancelled = false;
    void api.images.bands(imagePath).then(
      (res) => {
        if (!cancelled) setResult(res);
      },
      () => {
        if (!cancelled) setResult(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [imagePath]);

  return result;
}
