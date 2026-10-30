import { useEffect, useState } from "react";

import { api } from "@/api/client";
import type { ImageBands } from "@/api/types.generated";

/**
 * Band symbology for one image path, refetched whenever the path changes. Null while loading, for
 * a path-less call, or on a failed fetch; callers treat null as "nothing to show yet," never as a
 * signal to render band controls (only an answer that is not `plain` does that).
 */
export function useImageBands(imagePath: string | null): ImageBands | null {
  const [result, setResult] = useState<ImageBands | null>(null);

  useEffect(() => {
    if (!imagePath) {
      setResult(null);
      return;
    }
    let canceled = false;
    void api.images.bands(imagePath).then(
      (res) => {
        if (!canceled) setResult(res);
      },
      () => {
        if (!canceled) setResult(null);
      },
    );
    return () => {
      canceled = true;
    };
  }, [imagePath]);

  return result;
}
