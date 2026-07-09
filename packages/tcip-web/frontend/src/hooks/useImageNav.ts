/**
 * Single source of truth for image navigation. Arrow keys, the TopBar Prev/Next +
 * jump counter, and the Review tab's image nav all go through this so they share ONE
 * traversal order that honors the status filter — previously arrows walked the filtered
 * list while TopBar/jump walked raw indices with an unfiltered denominator (three
 * controls, three orders). The pure helpers are exported for unit testing.
 */

import { useCallback, useMemo } from "react";

import type { ImageStatus } from "@/api/classes";
import { useStore } from "@/store";

/** Indices of images that match the active status filter (all indices when "all"). */
export function computeFilteredIndices(
  imageList: string[],
  byImage: Record<string, ImageStatus>,
  activeFilter: "all" | ImageStatus,
): number[] {
  if (activeFilter === "all") return imageList.map((_, i) => i);
  return imageList
    .map((name, i) => (byImage[name] === activeFilter ? i : -1))
    .filter((i) => i >= 0);
}

/** The image index `delta` steps away within `indices`, or null if it wouldn't move.
 *  When the current index isn't in the filtered set (e.g. just after changing the
 *  filter), stepping enters the set at the nearest member in the direction of travel. */
export function stepTarget(indices: number[], currentIndex: number, delta: number): number | null {
  if (indices.length === 0) return null;
  const pos = indices.indexOf(currentIndex);
  let nextPos: number;
  if (pos >= 0) {
    nextPos = pos + delta;
  } else if (delta > 0) {
    const found = indices.findIndex((i) => i > currentIndex); // first member after current
    nextPos = found >= 0 ? found : indices.length - 1;
  } else {
    let last = -1; // last member before current
    for (let k = 0; k < indices.length; k++) if (indices[k] < currentIndex) last = k;
    nextPos = last >= 0 ? last : 0;
  }
  nextPos = Math.max(0, Math.min(indices.length - 1, nextPos));
  const next = indices[nextPos];
  return next === currentIndex ? null : next;
}

/** The image index at 1-based `oneBased` position within `indices` (clamped). */
export function jumpTarget(indices: number[], oneBased: number): number | null {
  if (indices.length === 0) return null;
  const clamped = Math.max(1, Math.min(indices.length, oneBased)) - 1;
  return indices[clamped];
}

export function useImageNav() {
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const byImage = useStore((s) => s.imageStatus.byImage);
  const activeFilter = useStore((s) => s.imageStatus.activeFilter);
  const patchGui = useStore((s) => s.patchGui);

  const filteredIndices = useMemo(
    () => computeFilteredIndices(imageList, byImage, activeFilter),
    [imageList, byImage, activeFilter],
  );

  const goTo = useCallback(
    (index: number | null) => {
      if (index === null || index === currentIndex) return;
      // Read the freshest dataset so a rapid sequence of navigations can't clobber
      // other dataset fields with a stale render closure.
      const dataset = useStore.getState().gui.dataset;
      patchGui({ dataset: { ...dataset, current_image_index: index } });
    },
    [currentIndex, patchGui],
  );

  const stepImage = useCallback(
    (delta: number) => goTo(stepTarget(filteredIndices, currentIndex, delta)),
    [filteredIndices, currentIndex, goTo],
  );

  const jumpToPosition = useCallback(
    (oneBased: number) => goTo(jumpTarget(filteredIndices, oneBased)),
    [filteredIndices, goTo],
  );

  const total = filteredIndices.length;
  // 1-based position of the current image within the filtered list; 0 if it isn't a
  // member (transient — e.g. right after changing the filter; stepping re-enters).
  const position = filteredIndices.indexOf(currentIndex) + 1;

  return {
    filteredIndices,
    total,
    position,
    stepImage,
    jumpToPosition,
    canPrev: total > 0 && position !== 1,
    canNext: total > 0 && position !== total,
  };
}
