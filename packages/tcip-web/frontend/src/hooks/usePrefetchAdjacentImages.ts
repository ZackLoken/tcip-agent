import { useEffect } from "react";

import { api, IMAGE_MAX_WIDTH } from "@/api/client";
import { useImageNav } from "@/hooks/useImageNav";
import { useStore } from "@/store";

/**
 * Warm the next/previous images in the filtered traversal once the current one has had a
 * head start. A cold image costs the server a multi-second render (first visit ever) —
 * prefetching hides that behind the time spent reviewing the current frame, and also
 * populates the server's disk cache for later sessions.
 */
export function usePrefetchAdjacentImages(): void {
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const date = useStore((s) => s.gui.dataset.date);
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const nav = useImageNav();
  const { filteredIndices } = nav;

  useEffect(() => {
    if (!datasetRoot || !date) return;
    const pos = filteredIndices.indexOf(currentIndex);
    if (pos < 0) return;
    const targets = [filteredIndices[pos + 1], filteredIndices[pos - 1]]
      .filter((i): i is number => typeof i === "number")
      .map((i) => imageList[i])
      .filter(Boolean);
    if (!targets.length) return;
    // Give the current image's own request a head start before warming neighbors.
    const t = setTimeout(() => {
      for (const name of targets) {
        const img = new Image();
        img.src = api.images.url(`${datasetRoot}/images/${date}/${name}`, IMAGE_MAX_WIDTH);
      }
    }, 600);
    return () => clearTimeout(t);
  }, [datasetRoot, date, imageList, currentIndex, filteredIndices]);
}
