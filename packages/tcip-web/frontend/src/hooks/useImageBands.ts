import { useEffect, useState } from "react";

import { api } from "@/api/client";
import type { ImageBands } from "@/api/types.generated";

/**
 * Band symbology for one image path, refetched whenever the path changes. Null until that path's
 * own answer arrives (never another path's), for a path-less call, or on a failed fetch; callers
 * treat null as "nothing to show yet," never as a signal to render band controls (only an answer
 * that is not `plain` does that).
 */
export function useImageBands(imagePath: string | null): ImageBands | null {
  const [answer, setAnswer] = useState<{ path: string; bands: ImageBands | null } | null>(null);

  useEffect(() => {
    if (!imagePath) return;
    let canceled = false;
    void api.images.bands(imagePath).then(
      (bands) => {
        if (!canceled) setAnswer({ path: imagePath, bands });
      },
      () => {
        if (!canceled) setAnswer({ path: imagePath, bands: null });
      },
    );
    return () => {
      canceled = true;
    };
  }, [imagePath]);

  return answer !== null && answer.path === imagePath ? answer.bands : null;
}
