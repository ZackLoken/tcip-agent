/**
 * The page's one telling of the backend which image is in view: the held selection's index,
 * posted through the nav route once the selection or its index has stayed put for the echo's
 * wait, or at once when asked.
 */

import { api } from "@/api/client";
import { useStore } from "@/store";
import { datasetIdentityChanged } from "@/store/slices/gui";
import type { DatasetSelection } from "@/store/types";

// A dropped post leaves the backend at the last index it received, however far that is.
const ECHO_WAIT_MS = 400;

let pending: ReturnType<typeof setTimeout> | null = null;

function clearPending(): void {
  if (pending !== null) clearTimeout(pending);
  pending = null;
}

/** Post the index of the selection the store holds now, clearing any pending post; nothing when
 *  no selection is held. A failed post is discarded. */
export function echoImageNow(): void {
  clearPending();
  const held = useStore.getState().gui.dataset;
  if (held !== null) api.dataset.nav(held.current_image_index).catch(() => {});
}

/** Whether `next` is a held selection other than `prev` by the store's rule for telling two
 *  apart, or another index of the same one. */
function moved(prev: DatasetSelection | null, next: DatasetSelection | null): boolean {
  if (next === null) return false;
  return (
    prev === null ||
    datasetIdentityChanged(prev, next) ||
    next.current_image_index !== prev.current_image_index
  );
}

/** Watch the store: each move of the held selection (`moved`) arms one post of the store's index
 *  at the end of the wait, clearing the post an earlier move armed. Returns the stop, which ends
 *  the watch and clears a pending post. */
export function startImageEcho(): () => void {
  const unsubscribe = useStore.subscribe((s, prev) => {
    if (!moved(prev.gui.dataset, s.gui.dataset)) return;
    clearPending();
    pending = setTimeout(echoImageNow, ECHO_WAIT_MS);
  });
  return () => {
    unsubscribe();
    clearPending();
  };
}
