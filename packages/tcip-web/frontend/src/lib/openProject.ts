/**
 * Opening a workspace project: making it the backend's open project, then pointing the GUI at a
 * dataset inside it (the project's own tree) via /dataset/select.
 */

import { api, type ProjectSummary } from "@/api/client";
import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
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

/** Open ``p`` (a listed project whose record reads, so it carries an id) on a capture, subject
 *  and bucket name (one ``bucketsForDate`` serves); the backend points the workspace's
 *  last-opened pointer at it. A project holding no capture (``date`` empty) opens with no
 *  dataset selected. */
export async function openWorkspaceProject(
  p: ProjectSummary & { id: string },
  date: string,
  subject: string | null,
  bucket: string | null,
): Promise<DatasetSelection> {
  // Snapshot the outgoing dataset's UI state before the open's broadcast can move it; the
  // restore for the new selection is defined once, below.
  useStore.getState().saveCurrentDatasetUi();
  const opened = await api.projects.open(p.id);
  const res = date
    ? await api.dataset.select({
        dataset_root: opened.path,
        subject: subject || null,
        date,
        bucket: bucket || null,
      })
    : { selection: GUI_STATE_DEFAULTS.dataset, label_problem: null };
  try {
    recordRecentProject(opened.id);
  } catch (e) {
    useStore
      .getState()
      .pushToast(
        `Opened, but the recent projects list was not updated: ${e instanceof Error ? e.message : String(e)}`,
      );
  }
  useStore.getState().applyRestoredDataset(res.selection, { id: opened.id, path: opened.path });
  toastLabelProblem(res.label_problem);
  return res.selection;
}

/** The subjects with labels on date ``d``; empty when nothing is labeled there, so a selector
 *  never offers a choice that would open a blank canvas. */
export const subjectsForDate = (p: ProjectSummary, d: string): string[] =>
  p.subjects_by_date[d] ?? [];

/** The names of the buckets published over date ``d``; empty when none is. */
export const bucketsForDate = (p: ProjectSummary, d: string): string[] =>
  p.buckets_by_date[d] ?? [];

/** The most-recent date that actually has a labeled subject, or null if none do. */
function newestLabeledDate(p: ProjectSummary): string | null {
  const labeled = p.dates.filter((d) => subjectsForDate(p, d).length > 0);
  return labeled.length ? defaultDate(labeled) : null;
}

/** Open the listed project with ``id`` on sensible defaults, preferring the newest date that
 *  actually has labels; null when the workspace lists no project with it. */
export async function openProjectById(id: string): Promise<DatasetSelection | null> {
  const { projects } = await api.projects.list();
  const p = projects.find((x) => x.id === id);
  if (!p || p.id === null) return null;
  // Prefers a labeled date: an agent ingesting a still-unlabeled newer date would
  // otherwise land the human on a blank canvas with no date selector to recover.
  const date = newestLabeledDate(p) ?? defaultDate(p.dates);
  const subject = subjectsForDate(p, date)[0] ?? null;
  const bucket = bucketsForDate(p, date)[0] ?? null;
  return openWorkspaceProject({ ...p, id: p.id }, date, subject, bucket);
}
