/** Image navigation through the images the filter keeps, in the selection's own order. */

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

/** Navigation over the selection's images the filter keeps; with no selection, none to step. */
export function useImageNav() {
  const dataset = useStore((s) => s.gui.dataset);
  const states = useStore((s) => s.annotateUi.imageStates);
  const filter = useStore((s) => s.annotateUi.imageFilter);
  const patchGui = useStore((s) => s.patchGui);
  const images = dataset?.images;
  const indices = useMemo(
    () => (images === undefined ? [] : filteredIndices(images, states, filter)),
    [images, states, filter],
  );

  // The freshest selection is read at the step, so a rapid sequence of steps never lays a
  // stale render's selection over another field of it.
  const goTo = useCallback(
    (target: (current: number) => number | null) => {
      const held = useStore.getState().gui.dataset;
      if (held === null) return;
      const index = target(held.current_image_index);
      if (index === null || index === held.current_image_index) return;
      patchGui({ dataset: { ...held, current_image_index: index } });
      syncNavIndex(index);
    },
    [patchGui],
  );

  const stepImage = useCallback(
    (delta: number) => goTo((current) => stepTarget(indices, current, delta)),
    [indices, goTo],
  );

  const jumpToPosition = useCallback(
    (oneBased: number) => goTo(() => jumpTarget(indices, oneBased)),
    [indices, goTo],
  );

  const current = dataset?.current_image_index;
  const kept = (delta: number) =>
    current !== undefined && stepTarget(indices, current, delta) !== null;
  return {
    total: indices.length,
    // 0 when the image in view is one the filter drops, or no capture is selected.
    position: current === undefined ? 0 : indices.indexOf(current) + 1,
    stepImage,
    jumpToPosition,
    canPrev: kept(-1),
    canNext: kept(1),
  };
}
