/**
 * Opening a workspace project = pointing the GUI at it (project root = dataset root) via
 * /dataset/select. Shared by the ProjectPicker (human clicks Open) and App's agent→GUI
 * channel (the agent calls set_active_project → "app" panel event → open it here), so the
 * two paths can't drift.
 */

import { api, type ProjectSummary } from "@/api/client";
import { toastLabelProblem } from "@/lib/labelProblemToast";
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
  subject: string | null,
  modelName: string | null,
): Promise<DatasetSelection> {
  // Snapshot the outgoing dataset's UI state before the select's broadcast can move it, then
  // adopt the new selection with its saved position/filters restored. Every open path (picker,
  // recent-projects fast-track, agent) funnels through here, so restore is defined once.
  useStore.getState().saveCurrentDatasetUi();
  const res = await api.dataset.select({
    project_root: p.path,
    dataset_root: p.path,
    subject: subject || null,
    date: date || null,
    model_name: modelName || null,
  });
  recordRecentProject(p.name, p.path);
  useStore.getState().applyRestoredDataset(res.selection, res.generation);
  toastLabelProblem(res.label_problem);
  return res.selection;
}

/** Open a project like `openWorkspaceProject`, and also write the active-project marker: a
 *  human-initiated open should be what the GUI/ritual find active next time. */
export async function adoptWorkspaceProject(
  p: ProjectSummary,
  date: string,
  subject: string | null,
  modelName: string | null,
): Promise<DatasetSelection> {
  const selection = await openWorkspaceProject(p, date, subject, modelName);
  // Fire-and-forget: a rejected write is a toast, never a failed or delayed open.
  void api.projects.setActive(p.name).catch((e) => {
    useStore
      .getState()
      .pushToast(
        `Opened ${p.name}, but could not set it as the active project: ` +
          (e instanceof Error ? e.message : String(e)),
      );
  });
  return selection;
}

/** The most-recent date that actually has a labelled subject, or null if none do. */
function newestLabelledDate(p: ProjectSummary): string | null {
  const labelled = p.dates.filter((d) => (p.subjects_by_date[d] ?? []).length > 0);
  return labelled.length ? defaultDate(labelled) : null;
}

/** The project (by name) plus the default date/subject/model to open it on; null if the name
 *  is not in the workspace. Prefers the newest date that actually has labels. */
async function resolveDefaultOpen(name: string): Promise<{
  p: ProjectSummary;
  date: string;
  subject: string | null;
  model: string | null;
} | null> {
  const { projects } = await api.projects.list();
  const p = projects.find((x) => x.name === name);
  if (!p) return null;
  // Prefers a labelled date: an agent ingesting a still-unlabelled newer date would
  // otherwise land the human on a blank canvas with no date selector to recover.
  const date = newestLabelledDate(p) ?? defaultDate(p.dates);
  const subject = (p.subjects_by_date[date] ?? [])[0] ?? null;
  const model = (p.models_by_date[date] ?? [])[0] ?? null;
  return { p, date, subject, model };
}

/** Look a project up by name and open it on sensible defaults; null if it's gone. Writes no
 *  marker (the agent's own `active_project_changed` event uses this). */
export async function openProjectByName(name: string): Promise<DatasetSelection | null> {
  const resolved = await resolveDefaultOpen(name);
  if (!resolved) return null;
  return openWorkspaceProject(resolved.p, resolved.date, resolved.subject, resolved.model);
}

/** Same as `openProjectByName`, but adopts (writes the marker): for a human-initiated open,
 *  e.g. the breadcrumb's recent-projects list. */
export async function adoptProjectByName(name: string): Promise<DatasetSelection | null> {
  const resolved = await resolveDefaultOpen(name);
  if (!resolved) return null;
  return adoptWorkspaceProject(resolved.p, resolved.date, resolved.subject, resolved.model);
}
