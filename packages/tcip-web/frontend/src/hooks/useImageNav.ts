/** Image navigation through the images the filter keeps, in the order the filter walks them. */

import { useCallback, useMemo } from "react";

import { api } from "@/api/client";
import { filteredIndices, filterInEffect, rankedNames } from "@/lib/imageFilter";
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

/** The image index `delta` steps from `current` along `indices`, the kept images' indices in the
 *  order the filter walks them. From an image the filter drops, the step walks `indices` in its
 *  own direction (forward from the start, backward from the end) and lands on the first kept
 *  image lying that way from `current` in the selection. Null when the step would not move. */
export function stepTarget(indices: number[], current: number, delta: number): number | null {
  const at = indices.indexOf(current);
  const next =
    at >= 0
      ? indices[Math.max(0, Math.min(indices.length - 1, at + delta))]
      : delta > 0
        ? indices.find((i) => i > current)
        : indices.findLast((i) => i < current);
  return next === undefined || next === current ? null : next;
}

/** The image index at 1-based `oneBased` position among `indices` (clamped). */
export function jumpTarget(indices: number[], oneBased: number): number | null {
  if (indices.length === 0) return null;
  return indices[Math.max(1, Math.min(indices.length, oneBased)) - 1];
}

/** Navigation over the selection's images the filter in effect keeps; with no selection, none
 *  to step. `ranked` is the held review queue's image names when it ranked the selected capture,
 *  else null; `filter` the filter in effect (`filterInEffect`); `indices` the kept images in the
 *  order a step walks them. */
export function useImageNav() {
  const dataset = useStore((s) => s.gui.dataset);
  const states = useStore((s) => s.annotateUi.imageStates);
  const chosen = useStore((s) => s.annotateUi.imageFilter);
  const queue = useStore((s) => s.annotateUi.reviewQueue);
  const patchGui = useStore((s) => s.patchGui);
  const ranked = useMemo(() => rankedNames(queue, dataset), [queue, dataset]);
  const filter = filterInEffect(chosen, ranked);
  const images = dataset?.images;
  const indices = useMemo(
    () => (images === undefined ? [] : filteredIndices(images, states, filter, ranked ?? [])),
    [images, states, filter, ranked],
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
    ranked,
    filter,
    indices,
    total: indices.length,
    // 0 when the image in view is one the filter drops, or no capture is selected.
    position: current === undefined ? 0 : indices.indexOf(current) + 1,
    stepImage,
    jumpToPosition,
    canPrev: kept(-1),
    canNext: kept(1),
  };
}
