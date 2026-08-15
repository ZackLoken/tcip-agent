import { useEffect } from "react";

import { api } from "@/api/client";
import { useImageNav } from "@/hooks/useImageNav";
import { inImagesDir } from "@/lib/paths";
import { useStore } from "@/store";

/**
 * Warm the next/previous images in the filtered traversal once the current one has had a
 * head start. A cold image costs the server a multi-second render (first visit ever);
 * prefetching hides that behind the time spent reviewing the current frame, and also
 * populates the server's disk cache for later sessions.
 *
 * `bands`/`stretch` are the canvas' own request params (see `compositeParams`): warming any
 * other render of the image warms a cache entry the canvas will never ask for.
 */
export function usePrefetchAdjacentImages(bands?: string, stretch?: string): void {
  const imagesDir = useStore((s) => s.gui.dataset.images_dir);
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const nav = useImageNav();
  const { filteredIndices } = nav;

  useEffect(() => {
    if (!imagesDir) return;
    const pos = filteredIndices.indexOf(currentIndex);
    if (pos < 0) return;
    // Forward-biased lookahead: the user mostly steps forward, so warm the next few frames in the
    // direction of travel plus one back (for reviewing completed work). Deeper than ±1 so stepping
    // faster than a cold ~2 s server render still lands on an already-warm frame.
    const targets = [1, 2, 3, -1]
      .map((d) => filteredIndices[pos + d])
      .filter((i): i is number => typeof i === "number")
      .map((i) => imageList[i])
      .filter(Boolean);
    if (!targets.length) return;
    // Give the current image's own request a head start before warming neighbors.
    const t = setTimeout(() => {
      for (const name of targets) {
        const img = new Image();
        img.src = api.images.url(inImagesDir(imagesDir, name), { bands, stretch });
      }
    }, 600);
    return () => clearTimeout(t);
  }, [imagesDir, imageList, currentIndex, filteredIndices, bands, stretch]);
}
