/**
 * Single source of truth for image navigation: arrow keys and the toolbar's stepper all step
 * through the images the filter keeps, in the dataset's own order, through this hook.
 */

import { useCallback, useMemo } from "react";

import { api } from "@/api/client";
import { filteredIndices } from "@/lib/imageFilter";
import { useStore } from "@/store";

// Debounced: only the settled position persists; a dropped sync leaves the snapshot one image stale.
let navSyncTimer: ReturnType<typeof setTimeout> | null = null;
function syncNavIndex(index: number): void {
  if (navSyncTimer !== null) clearTimeout(navSyncTimer);
  navSyncTimer = setTimeout(() => {
    navSyncTimer = null;
    api.dataset.nav(index).catch(() => {});
  }, 400);
}

/** The image index `delta` steps from `current` along `indices`, the kept images' indices in
 *  list order: from an image the filter drops, the nearest kept one in that direction. Null when
 *  the step would not move. */
export function stepTarget(indices: number[], current: number, delta: number): number | null {
  if (indices.length === 0) return null;
  const at = indices.indexOf(current);
  let next: number | undefined;
  if (at >= 0) next = indices[Math.max(0, Math.min(indices.length - 1, at + delta))];
  else if (delta > 0) next = indices.find((i) => i > current);
  else next = indices.filter((i) => i < current).at(-1);
  return next === undefined || next === current ? null : next;
}

/** The image index at 1-based `oneBased` position among `indices` (clamped). */
export function jumpTarget(indices: number[], oneBased: number): number | null {
  if (indices.length === 0) return null;
  return indices[Math.max(1, Math.min(indices.length, oneBased)) - 1];
}

export function useImageNav() {
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const states = useStore((s) => s.annotateUi.imageStates);
  const filter = useStore((s) => s.annotateUi.imageFilter);
  const patchGui = useStore((s) => s.patchGui);
  const indices = useMemo(
    () => filteredIndices(imageList, states, filter),
    [imageList, states, filter],
  );

  const goTo = useCallback(
    (index: number | null) => {
      if (index === null || index === currentIndex) return;
      // Read the freshest dataset so a rapid sequence of navigations can't clobber
      // other dataset fields with a stale render closure.
      const state = useStore.getState();
      patchGui({ dataset: { ...state.gui.dataset, current_image_index: index } });
      syncNavIndex(index);
    },
    [currentIndex, patchGui],
  );

  const stepImage = useCallback(
    (delta: number) => goTo(stepTarget(indices, currentIndex, delta)),
    [indices, currentIndex, goTo],
  );

  const jumpToPosition = useCallback(
    (oneBased: number) => goTo(jumpTarget(indices, oneBased)),
    [indices, goTo],
  );

  const total = indices.length;
  // 0 when the image in view is one the filter drops.
  const position = indices.indexOf(currentIndex) + 1;

  return {
    total,
    position,
    stepImage,
    jumpToPosition,
    canPrev: stepTarget(indices, currentIndex, -1) !== null,
    canNext: stepTarget(indices, currentIndex, 1) !== null,
  };
}
