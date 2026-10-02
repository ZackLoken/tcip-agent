/**
 * Single source of truth for image navigation: arrow keys and the TopBar Prev/Next + jump counter
 * all step through the dataset's image list in its own order through this hook.
 */

import { useCallback } from "react";

import { api } from "@/api/client";
import { useStore } from "@/store";

// Debounced: only the settled position persists; a dropped sync leaves gui.json one image stale.
let navSyncTimer: ReturnType<typeof setTimeout> | null = null;
function syncNavIndex(index: number): void {
  if (navSyncTimer !== null) clearTimeout(navSyncTimer);
  navSyncTimer = setTimeout(() => {
    navSyncTimer = null;
    api.dataset.nav(index).catch(() => {});
  }, 400);
}

/** The image index `delta` steps from `current` in a list of `total`, clamped to its ends, or
 *  null when the step would not move. */
export function stepTarget(total: number, current: number, delta: number): number | null {
  if (total === 0) return null;
  const next = Math.max(0, Math.min(total - 1, current + delta));
  return next === current ? null : next;
}

/** The image index at 1-based `oneBased` position in a list of `total` (clamped). */
export function jumpTarget(total: number, oneBased: number): number | null {
  if (total === 0) return null;
  return Math.max(1, Math.min(total, oneBased)) - 1;
}

export function useImageNav() {
  const total = useStore((s) => s.gui.dataset.image_list.length);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const patchGui = useStore((s) => s.patchGui);

  // Leaving an image clears what was selected or focused on it, whichever control navigated.
  const goTo = useCallback(
    (index: number | null) => {
      if (index === null || index === currentIndex) return;
      // Read the freshest dataset so a rapid sequence of navigations can't clobber
      // other dataset fields with a stale render closure.
      const state = useStore.getState();
      patchGui({ dataset: { ...state.gui.dataset, current_image_index: index } });
      state.selectPolygon(null);
      state.setFocusedProposal(null);
      syncNavIndex(index);
    },
    [currentIndex, patchGui],
  );

  const stepImage = useCallback(
    (delta: number) => goTo(stepTarget(total, currentIndex, delta)),
    [total, currentIndex, goTo],
  );

  const jumpToPosition = useCallback(
    (oneBased: number) => goTo(jumpTarget(total, oneBased)),
    [total, goTo],
  );

  const position = total > 0 ? currentIndex + 1 : 0;

  return {
    total,
    position,
    stepImage,
    jumpToPosition,
    canPrev: total > 0 && position > 1,
    canNext: total > 0 && position < total,
  };
}
