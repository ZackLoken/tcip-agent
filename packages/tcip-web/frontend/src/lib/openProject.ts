/**
 * Opening a workspace project = pointing the GUI at it (project root = dataset root) via
 * /dataset/select. Shared by the ProjectPicker (human clicks Open) and App's agent→GUI
 * channel (the agent calls set_active_project → "app" panel event → open it here), so the
 * two paths can't drift.
 */

import { api, type ProjectSummary } from "@/api/client";
import { recordRecentProject } from "@/lib/recentProjects";
import { useStore } from "@/store";
import type { DatasetSelection } from "@/store/types";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/** The date to open a project on by default: its most recent ISO capture date. */
export function defaultDate(dates: string[]): string {
  const iso = dates.filter((d) => ISO_DATE.test(d));
  if (iso.length) return iso[iso.length - 1];
  return dates[dates.length - 1] ?? "";
}

export async function openWorkspaceProject(
  p: ProjectSummary,
  date: string,
  annotationType: string | null,
  modelName: string | null,
): Promise<DatasetSelection> {
  // Snapshot the outgoing dataset's UI state before the select's broadcast can move it, then
  // adopt the new selection with its saved position/filters restored. Every open path (picker,
  // recent-projects fast-track, agent) funnels through here, so restore is defined once.
  useStore.getState().saveCurrentDatasetUi();
  const res = await api.dataset.select({
    project_root: p.path,
    dataset_root: p.path,
    annotation_type: annotationType || null,
    date: date || null,
    model_name: modelName || null,
  });
  recordRecentProject(p.name, p.path);
  useStore.getState().applyRestoredDataset(res.selection);
  return res.selection;
}

/** The most-recent date that actually has a labelled trait, or null if none do. */
function newestLabelledDate(p: ProjectSummary): string | null {
  const labelled = p.dates.filter((d) => (p.traits_by_date[d] ?? []).length > 0);
  return labelled.length ? defaultDate(labelled) : null;
}

/** Look a project up by name and open it on sensible defaults; null if it's gone. */
export async function openProjectByName(name: string): Promise<DatasetSelection | null> {
  const { projects } = await api.projects.list();
  const p = projects.find((x) => x.name === name);
  if (!p) return null;
  // Open on the most-recent date that actually has labels (not merely the newest date), and
  // scope trait/model to that date's per-date availability — otherwise an agent that just
  // ingested a still-unlabelled newer date would jump the human past their annotations onto a
  // blank canvas (there's no date selector inside the Annotate tab to recover). Falls back to
  // the newest date only when nothing is labelled yet (a genuinely empty project).
  const date = newestLabelledDate(p) ?? defaultDate(p.dates);
  const trait = (p.traits_by_date[date] ?? [])[0] ?? null;
  const model = (p.models_by_date[date] ?? [])[0] ?? null;
  return openWorkspaceProject(p, date, trait, model);
}
