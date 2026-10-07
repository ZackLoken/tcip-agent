import { useEffect } from "react";

import { api } from "@/api/client";
import { inImagesDir } from "@/lib/paths";
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
  const imagesDir = useStore((s) => s.gui.dataset.images_dir);
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);

  useEffect(() => {
    if (!imagesDir) return;
    // Forward-biased lookahead, deeper than one step so fast stepping lands on a warm frame.
    const targets = [1, 2, 3, -1].map((d) => imageList[currentIndex + d]).filter(Boolean);
    if (!targets.length) return;
    // Give the current image's own request a head start before warming neighbors.
    const t = setTimeout(() => {
      for (const name of targets) {
        const img = new Image();
        img.src = api.images.url(inImagesDir(imagesDir, name), displayPixels, { bands, stretch });
      }
    }, 600);
    return () => clearTimeout(t);
  }, [imagesDir, imageList, currentIndex, displayPixels, bands, stretch]);
}
