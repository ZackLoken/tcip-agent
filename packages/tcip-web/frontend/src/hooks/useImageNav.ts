/**
 * Single source of truth for image navigation. Arrow keys, the TopBar Prev/Next +
 * jump counter, and the Review tab's image nav all go through this so they share one
 * traversal order that honors the status filter: previously arrows walked the filtered
 * list while TopBar/jump walked raw indices with an unfiltered denominator (three
 * controls, three orders). The pure helpers are exported for unit testing.
 */

import { useCallback, useEffect, useMemo, useRef } from "react";

import { api } from "@/api/client";
import { useStore } from "@/store";

// Debounce the backend nav sync: rapid arrow-key traversal patches local state on every
// step but only persists the settled position (which view_gui_state reads). Fire-and-
// forget: a dropped sync just leaves gui.json one image stale until the next move.
let navSyncTimer: ReturnType<typeof setTimeout> | null = null;
function syncNavIndex(index: number): void {
  if (navSyncTimer !== null) clearTimeout(navSyncTimer);
  navSyncTimer = setTimeout(() => {
    navSyncTimer = null;
    api.dataset.nav(index).catch(() => {});
  }, 400);
}

/** Indices of images that match the active status filter (all indices when "all"), further
 *  narrowed by `isNavigable` when supplied (the Review tab skips images with zero detections).
 *  `order`, when supplied, replaces `imageList`'s own positional order as the traversal
 *  order (e.g. an active-learning priority ranking) while the same filter/isNavigable
 *  predicates still apply on top of it. Omitted, traversal is positional exactly as before. */
export function computeFilteredIndices(
  imageList: string[],
  byImage: Record<string, string>,
  activeFilter: string,
  isNavigable?: (name: string) => boolean,
  order?: number[],
): number[] {
  const candidates = order ?? imageList.map((_, i) => i);
  return candidates.filter((i) => {
    const name = imageList[i];
    if (name === undefined) return false;
    if (isNavigable && !isNavigable(name)) return false;
    if (activeFilter !== "all" && byImage[name] !== activeFilter) return false;
    return true;
  });
}

/** The image index `delta` steps away within `indices`, or null if it wouldn't move.
 *  When the current index isn't in the filtered set (e.g. just after changing the
 *  filter), stepping enters the set at the nearest member in the direction of travel.
 *  `wrap` (default false) carries a step past either end around to the other end instead of
 *  clamping there, for a control whose members are scattered and each equally worth reaching
 *  (e.g. the images still needing re-confirmation), rather than a linear traversal with real
 *  ends. */
export function stepTarget(
  indices: number[],
  currentIndex: number,
  delta: number,
  wrap = false,
): number | null {
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
  nextPos = wrap
    ? ((nextPos % indices.length) + indices.length) % indices.length
    : Math.max(0, Math.min(indices.length - 1, nextPos));
  const next = indices[nextPos];
  return next === currentIndex ? null : next;
}

/** The image index at 1-based `oneBased` position within `indices` (clamped). */
export function jumpTarget(indices: number[], oneBased: number): number | null {
  if (indices.length === 0) return null;
  const clamped = Math.max(1, Math.min(indices.length, oneBased)) - 1;
  return indices[clamped];
}

/** Per-tab overrides. The Annotate tab (default) filters by annotation status from the store;
 *  the Review tab passes its own image-level review-status map + filter and an `isNavigable`
 *  predicate that skips images with nothing to review. `order` is an explicit traversal
 *  order (indices into `image_list`, e.g. an active-learning priority ranking) in place of
 *  positional order. */
export interface ImageNavOptions {
  byImage?: Record<string, string>;
  activeFilter?: string;
  isNavigable?: (name: string) => boolean;
  order?: number[];
  /** Step past either end of the filtered set around to the other end (see stepTarget). */
  wrap?: boolean;
}

export function useImageNav(options?: ImageNavOptions) {
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const currentIndex = useStore((s) => s.gui.dataset.current_image_index);
  const storeByImage = useStore((s) => s.imageStatus.byImage);
  const storeFilter = useStore((s) => s.imageStatus.activeFilter);
  const patchGui = useStore((s) => s.patchGui);

  const byImage = options?.byImage ?? storeByImage;
  const activeFilter: string = options?.activeFilter ?? storeFilter;
  const isNavigable = options?.isNavigable;
  const order = options?.order;
  const wrap = options?.wrap ?? false;

  const filteredIndices = useMemo(
    () => computeFilteredIndices(imageList, byImage, activeFilter, isNavigable, order),
    [imageList, byImage, activeFilter, isNavigable, order],
  );

  const goTo = useCallback(
    (index: number | null) => {
      if (index === null || index === currentIndex) return;
      // Read the freshest dataset so a rapid sequence of navigations can't clobber
      // other dataset fields with a stale render closure.
      const dataset = useStore.getState().gui.dataset;
      patchGui({ dataset: { ...dataset, current_image_index: index } });
      syncNavIndex(index);
    },
    [currentIndex, patchGui],
  );

  const stepImage = useCallback(
    (delta: number) => goTo(stepTarget(filteredIndices, currentIndex, delta, wrap)),
    [filteredIndices, currentIndex, goTo, wrap],
  );

  const jumpToPosition = useCallback(
    (oneBased: number) => goTo(jumpTarget(filteredIndices, oneBased)),
    [filteredIndices, goTo],
  );

  // When the status filter changes and the current image falls outside the new set, enter the
  // set at its first member. Without this the old image stays on the canvas and the index counter
  // reads empty (current isn't a member -> position 0) until the user manually steps. Gated on a
  // real filter change so reviewing an image (which shifts byImage) doesn't auto-advance.
  const prevFilterRef = useRef(activeFilter);
  useEffect(() => {
    const filterChanged = prevFilterRef.current !== activeFilter;
    prevFilterRef.current = activeFilter;
    if (!filterChanged || filteredIndices.length === 0) return;
    if (filteredIndices.includes(currentIndex)) return;
    goTo(filteredIndices[0]);
  }, [activeFilter, filteredIndices, currentIndex, goTo]);

  const total = filteredIndices.length;
  // 1-based position of the current image within the filtered list; 0 if it isn't a
  // member (transient, e.g. right after changing the filter; stepping re-enters).
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
