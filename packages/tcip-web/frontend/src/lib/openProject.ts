/**
 * Opening a workspace project: making it the backend's open project, then pointing the GUI at a
 * dataset inside it (the project's own tree) via /dataset/select.
 */

import { api, type ProjectSummary } from "@/api/client";
import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
import { toastLabelProblem } from "@/lib/labelProblemToast";
import { recordRecentProject } from "@/lib/recentProjects";
import { settleVisits } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import type { OpenHold } from "@/store/slices/projectOpen";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/** The date to open a project on by default: its most recent ISO capture date. */
export function defaultDate(dates: string[]): string {
  const iso = dates.filter((d) => ISO_DATE.test(d));
  if (iso.length) return iso[iso.length - 1];
  return dates[dates.length - 1] ?? "";
}

/** A listed project whose record reads, so it carries an id and a display name. */
export type OpenableProject = ProjectSummary & { id: string; display_name: string };

export const openable = (p: ProjectSummary): p is OpenableProject =>
  p.id !== null && p.display_name !== null;

let lastRequestId = 0;

/** The held open when it is the one with ``requestId``, else null: nothing has superseded it. */
export function heldOpen(requestId: number | null): OpenHold | null {
  const { opening } = useStore.getState();
  return opening !== null && opening.requestId === requestId ? opening : null;
}

/** Take the one open hold for ``projectId`` (null while the project is not yet known), before
 *  any step of the open that waits; null when an open already holds it. */
export function takeOpenHold(projectId: string | null): number | null {
  const state = useStore.getState();
  if (state.opening !== null) return null;
  const requestId = ++lastRequestId;
  state.patchOpenStatus({
    opening: { requestId, projectId, accepted: null },
    openError: null,
  });
  return requestId;
}

/** Whether the initial open held as ``requestId`` still stands for ``listedId``, the project the
 *  listing names as open: no snapshot was accepted since the hold was taken and none is open
 *  now, or the latest accepted state is that project. An accepted state of no project open
 *  cancels it. */
export function initialOpenWanted(requestId: number, listedId: string): boolean {
  const hold = heldOpen(requestId);
  if (!hold) return false;
  const seen = hold.accepted
    ? hold.accepted.projectId
    : (useStore.getState().openProject?.id ?? null);
  return seen === listedId || (hold.accepted === null && seen === null);
}

/** Give up the hold ``requestId`` took, unless it was already released or superseded. */
export function releaseOpenHold(requestId: number): void {
  if (heldOpen(requestId)) useStore.getState().patchOpenStatus({ opening: null });
}

/** The open itself, as the committed person, once this page's visits are settled
 *  (:func:`settleVisits`, whose rejection leaves the open unsent). It stops, with nothing sent
 *  or applied after that point, as soon as its request is no longer the held one: before asking
 *  the backend to open, before selecting a dataset and again before adopting. */
async function performOpen(
  requestId: number,
  p: ProjectSummary & { id: string },
  date: string,
  subject: string,
  bucket: string,
): Promise<void> {
  // Snapshot the outgoing dataset's UI state before the open's broadcast can move it; the
  // restore for the new selection is defined once, below.
  useStore.getState().saveCurrentDatasetUi();
  await settleVisits();
  if (!heldOpen(requestId)) return;
  const opened = await api.projects.open({ id: p.id, user: useStore.getState().user });
  if (!heldOpen(requestId)) return;
  const res = date
    ? await api.dataset.select({
        dataset_root: opened.path,
        subject: subject || null,
        date,
        bucket: bucket || null,
      })
    : { selection: GUI_STATE_DEFAULTS.dataset, label_problem: null };
  if (!heldOpen(requestId)) return;
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
}

/** The one transition owner for opening ``p`` (a listed project whose record reads, so it
 *  carries an id) on a capture, subject and bucket name (one ``bucketsForDate`` serves), as the
 *  committed person; the backend points the workspace's last-opened pointer at it. A project
 *  holding no capture (``date`` empty) opens with no dataset selected, and an open already in
 *  flight leaves this one unstarted. The open holds a request id in the store, taken by this call
 *  or, as ``held``, by a caller that took it before its own lookup; a second choice of project or
 *  a snapshot adopting another project releases it, and a result from a request no longer held,
 *  success or failure, is dropped. A held failure is recorded for ``p`` and toasted so a tab that
 *  replaced the picker does not hide it. An image visit the settlement stopped and no departure
 *  closed resumes counting when the open ends, whatever its outcome. */
export async function startOpen(
  p: ProjectSummary & { id: string },
  date: string,
  subject: string,
  bucket: string,
  held?: number,
): Promise<void> {
  const hold = heldOpen(held ?? takeOpenHold(p.id));
  if (!hold) return;
  const { requestId } = hold;
  useStore.getState().patchOpenStatus({ opening: { ...hold, projectId: p.id } });
  try {
    await performOpen(requestId, p, date, subject, bucket);
  } catch (e) {
    const now = useStore.getState();
    if (!heldOpen(requestId)) return;
    now.patchOpenStatus({ openError: { projectId: p.id, message: String(e) } });
    now.pushToast(String(e));
  } finally {
    releaseOpenHold(requestId);
    useStore.getState().resumeSessionInterval();
  }
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

/** Start opening the listed project ``p`` on sensible defaults, preferring the newest date that
 *  actually has labels. */
export function openWithDefaults(p: ProjectSummary & { id: string }): Promise<void> {
  // Prefers a labeled date: an agent ingesting a still-unlabeled newer date would
  // otherwise land the human on a blank canvas with no date selector to recover.
  const date = newestLabeledDate(p) ?? defaultDate(p.dates);
  const subject = subjectsForDate(p, date)[0] ?? "";
  const bucket = bucketsForDate(p, date)[0] ?? "";
  return startOpen(p, date, subject, bucket);
}
