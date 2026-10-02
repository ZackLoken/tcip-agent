/**
 * Per-(project, dataset, date, subject/bucket) UI state in sessionStorage, so switching and
 * returning within a session lands where you were. Each project's last tab is in localStorage.
 */

import { TAB_NAMES } from "@/api/types.generated";
import type { DatasetSelection, OpenProject, TabName } from "@/store/types";

export interface DatasetUiState {
  index: number;
}

const UI_PREFIX = "tcip.dsui.";

/** Stable key for a dataset selection: project + dataset root + date + subject + bucket, so
 *  distinct views don't share. */
export function datasetKey(project: OpenProject | null, d: DatasetSelection): string | null {
  if (!project || !d.dataset_root || !d.date) return null;
  return JSON.stringify([project.id, d.dataset_root, d.date, d.subject, d.predictions_dir]);
}

export function saveDatasetUi(key: string, state: DatasetUiState): void {
  sessionStorage.setItem(UI_PREFIX + key, JSON.stringify(state));
}

/** Whether ``value`` is an object holding every key of ``shape``, each of the same type. */
function holdsShape(value: unknown, shape: object): boolean {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return Object.entries(shape).every(([k, v]) => typeof record[k] === typeof v);
}

/** The UI state saved under ``key``, or null when none is; a saved value that does not parse as
 *  the whole shape ``saveDatasetUi`` writes throws, naming it. */
export function loadDatasetUi(key: string): DatasetUiState | null {
  const raw = sessionStorage.getItem(UI_PREFIX + key);
  if (raw === null) return null;
  const parsed: unknown = JSON.parse(raw);
  if (!holdsShape(parsed, { index: 0 })) {
    throw new Error(
      `the saved view state for this dataset is not the shape it is saved in: ${raw}`,
    );
  }
  return parsed as DatasetUiState;
}

// Last-used tab per project: localStorage, cross-session on purpose (unlike the blobs above);
// no record means a first-ever open, which lands on Annotate.
const TAB_PREFIX = "tcip.lasttab.";

export function recordLastTab(projectId: string, tab: TabName): void {
  localStorage.setItem(TAB_PREFIX + projectId, tab);
}

/** The project's last tab, or null when none is recorded; a recorded value naming no tab
 *  throws. */
export function loadLastTab(projectId: string | null): TabName | null {
  if (!projectId) return null;
  const raw = localStorage.getItem(TAB_PREFIX + projectId);
  if (raw === null) return null;
  if (!(TAB_NAMES as readonly string[]).includes(raw)) {
    throw new Error(`the last tab recorded for this project names no tab: ${raw}`);
  }
  return raw as TabName;
}
