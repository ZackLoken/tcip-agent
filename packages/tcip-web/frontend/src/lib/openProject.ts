/**
 * Opening a workspace project = pointing the GUI at it (project root = dataset root) via
 * /dataset/select. Shared by the ProjectPicker (human clicks Open) and App's agent→GUI
 * channel (the agent calls set_active_project → "app" panel event → open it here), so the
 * two paths can't drift.
 */

import { api, type ProjectSummary } from "@/api/client";
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
  const res = await api.dataset.select({
    project_root: p.path,
    dataset_root: p.path,
    annotation_type: annotationType || null,
    date: date || null,
    model_name: modelName || null,
  });
  return res.selection;
}

/** Look a project up by name and open it on sensible defaults; null if it's gone. */
export async function openProjectByName(name: string): Promise<DatasetSelection | null> {
  const { projects } = await api.projects.list();
  const p = projects.find((x) => x.name === name);
  if (!p) return null;
  return openWorkspaceProject(p, defaultDate(p.dates), p.traits[0] ?? null, p.models[0] ?? null);
}
