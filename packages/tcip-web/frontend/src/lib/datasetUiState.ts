/**
 * Per-(project, date, trait/model) UI state, so switching dates/projects and returning lands you
 * back where you were. Stored in sessionStorage — deliberately SESSION-scoped: a fresh app load
 * starts clean (matching the backend's fresh-open-at-image-0, which avoids resurrecting a stale
 * cross-session position), while in-session switches resume where you were. Position + review
 * filters live in one blob; Review's GT/Pred visibility (ReviewTab-local) is under its own key.
 *
 * Zoom/pan is deliberately not persisted here: both tabs actively drive the view (Annotate
 * auto-fits each image, Review auto-zooms to each detection), so a restored view would be
 * immediately overridden — restoring it would fight the tab, not help.
 */

import type { ImageStatus } from "@/api/classes";
import type { DatasetSelection, ReviewFilters } from "@/store/types";

export interface DatasetUiState {
  index: number;
  review: ReviewFilters;
  statusFilter: "all" | ImageStatus;
}

const UI_PREFIX = "tcip.dsui.";
const VIS_PREFIX = "tcip.dsvis.";

/** Stable key for a dataset selection: project + date + trait + model, so distinct views don't share. */
export function datasetKey(d: DatasetSelection): string | null {
  if (!d.project_root || !d.date) return null;
  const trait = d.annotation_type ?? "";
  const predDir = d.predictions_detect_dir ?? d.predictions_segment_dir ?? "";
  const model = predDir.split(/[/\\]predictions[/\\]/)[1]?.split(/[/\\]/)[0] ?? "";
  return `${d.project_root} ${d.date} ${trait} ${model}`;
}

export function saveDatasetUi(key: string, state: DatasetUiState): void {
  try {
    sessionStorage.setItem(UI_PREFIX + key, JSON.stringify(state));
  } catch {
    /* private mode / disabled storage — just won't restore */
  }
}

export function loadDatasetUi(key: string): DatasetUiState | null {
  try {
    const raw = sessionStorage.getItem(UI_PREFIX + key);
    return raw ? (JSON.parse(raw) as DatasetUiState) : null;
  } catch {
    return null;
  }
}

export interface DatasetVisibility {
  showGT: boolean;
  showPred: boolean;
}

export function saveDatasetVisibility(key: string, vis: DatasetVisibility): void {
  try {
    sessionStorage.setItem(VIS_PREFIX + key, JSON.stringify(vis));
  } catch {
    /* ignore */
  }
}

export function loadDatasetVisibility(key: string): DatasetVisibility | null {
  try {
    const raw = sessionStorage.getItem(VIS_PREFIX + key);
    return raw ? (JSON.parse(raw) as DatasetVisibility) : null;
  } catch {
    return null;
  }
}
