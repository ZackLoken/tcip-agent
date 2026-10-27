import { useEffect } from "react";

import { api } from "@/api/client";
import { useStore } from "@/store";

/**
 * Warm the next/previous images in the traversal once the current one has had a
 * head start. A cold image costs the server a multi-second render (first visit ever);
 * prefetching hides that behind the time spent reviewing the current frame, and also
 * populates the server's disk cache for later sessions.
 *
 * `displayPixels`, `bands` and `stretch` are the canvas' own request params (see
 * `compositeParams`): warming any other render of the image warms a cache entry the canvas will
 * never ask for.
 */
export function usePrefetchAdjacentImages(
  displayPixels: number,
  bands?: string,
  stretch?: string,
): void {
  const dataset = useStore((s) => s.gui.dataset);

  useEffect(() => {
    if (dataset === null) return;
    const { images, current_image_index: currentIndex } = dataset;
    // Forward-biased lookahead, deeper than one step so fast stepping lands on a warm frame.
    const targets = [1, 2, 3, -1].map((d) => images[currentIndex + d]).filter(Boolean);
    if (!targets.length) return;
    // Give the current image's own request a head start before warming neighbors.
    const t = setTimeout(() => {
      for (const image of targets) {
        const img = new Image();
        img.src = api.images.url(image.path, displayPixels, { bands, stretch });
      }
    }, 600);
    return () => clearTimeout(t);
  }, [dataset, displayPixels, bands, stretch]);
}
