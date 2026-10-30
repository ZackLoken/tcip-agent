/**
 * Opening a workspace project: making it the backend's open project, then pointing the GUI at a
 * dataset inside it (the project's own tree) via /dataset/select.
 */

import { api, type ProjectSummary } from "@/api/client";
import { messageOf } from "@/api/http";
import type { Capture, DatasetTree, ProjectRecord } from "@/api/types.generated";
import { toastLabelProblem } from "@/lib/labelProblemToast";
import { recordRecentProject } from "@/lib/recentProjects";
import { settleVisits } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import type { OpenHold } from "@/store/slices/projectOpen";

/** The capture to open a project on by default: the last of `captures` a date names, else its
 *  last capture; null for none. */
export function defaultDate(captures: Capture[]): string | null {
  return (captures.filter((c) => c.date !== null).at(-1) ?? captures.at(-1))?.name ?? null;
}

/** A listed project whose record reads. */
export type OpenableProject = ProjectSummary & { record: ProjectRecord };

export const openable = (p: ProjectSummary): p is OpenableProject => p.record !== null;

let lastRequestId = 0;

/** The held open when it is the one with ``requestId``, else null: nothing has superseded it. */
export function heldOpen(requestId: number | null): OpenHold | null {
  const { opening } = useStore.getState();
  return opening !== null && opening.requestId === requestId ? opening : null;
}

/** Take the one open hold for ``projectId`` before any step of the open that waits; null when
 *  an open already holds it. */
export function takeOpenHold(projectId: string): number | null {
  const state = useStore.getState();
  if (state.opening !== null) return null;
  const requestId = ++lastRequestId;
  state.patchOpenStatus({
    opening: { requestId, projectId },
    openError: null,
  });
  return requestId;
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
  p: OpenableProject,
  date: string | null,
  subject: string | null,
  bucket: string | null,
): Promise<void> {
  // Snapshot the outgoing dataset's UI state before the open's broadcast can move it; the
  // restore for the new selection is defined once, below.
  useStore.getState().saveCurrentDatasetUi();
  await settleVisits();
  if (!heldOpen(requestId)) return;
  const { project } = await api.projects.open({
    id: p.record.id,
    user: useStore.getState().user,
  });
  if (!heldOpen(requestId)) return;
  const res =
    date !== null
      ? await api.dataset.select({ dataset_root: project.path, subject, date, bucket })
      : { selection: null, label_problem: null };
  if (!heldOpen(requestId)) return;
  try {
    recordRecentProject(project.id);
  } catch (e) {
    useStore
      .getState()
      .pushToast(`Opened, but the recent projects list was not updated: ${messageOf(e)}`);
  }
  useStore.getState().applyRestoredDataset(res.selection, project);
  toastLabelProblem(res.label_problem);
}

/** The one transition owner for opening ``p`` (a listed project whose record reads) on a capture, subject and bucket name (one ``bucketsForDate`` serves), as the
 *  committed person. A project
 *  holding no capture (``date`` null) opens with no dataset selected, and an open already in
 *  flight leaves this one unstarted. The open holds a request id in the store; a second choice
 *  of project or a snapshot adopting another project releases it, and a result from a request
 *  no longer held, success or failure, is dropped. A held failure is recorded for ``p`` and
 *  toasted so a tab that replaced the picker does not hide it. An image visit the settlement
 *  stopped and no departure closed resumes counting when the open ends, whatever its outcome. */
export async function startOpen(
  p: OpenableProject,
  date: string | null,
  subject: string | null,
  bucket: string | null,
): Promise<void> {
  const hold = heldOpen(takeOpenHold(p.record.id));
  if (!hold) return;
  const { requestId } = hold;
  try {
    await performOpen(requestId, p, date, subject, bucket);
  } catch (e) {
    const now = useStore.getState();
    if (!heldOpen(requestId)) return;
    now.patchOpenStatus({ openError: { projectId: p.record.id, message: messageOf(e) } });
    now.pushToast(messageOf(e));
  } finally {
    releaseOpenHold(requestId);
    useStore.getState().resumeSessionInterval();
  }
}

/** The subjects with labels on date ``d``; empty when nothing is labeled there or no date is
 *  chosen, so a selector never offers a choice that would open a blank canvas. */
export const subjectsForDate = (tree: DatasetTree, d: string | null): string[] =>
  (d !== null && tree.subjects_by_date[d]) || [];

/** The names of the buckets published over date ``d``; empty when none is or no date is chosen. */
export const bucketsForDate = (tree: DatasetTree, d: string | null): string[] =>
  (d !== null && tree.buckets_by_date[d]) || [];

/** The subject and bucket to open ``tree`` on at date ``d``: ``kept`` when that date labels it,
 *  else the date's first subject (null ``kept`` keeps none), and the date's first bucket, since a
 *  bucket is published under one date; each null when the date offers none. */
export function choiceForDate(
  tree: DatasetTree,
  d: string | null,
  kept: string | null,
): { subject: string | null; bucket: string | null } {
  const subjects = subjectsForDate(tree, d);
  return {
    subject: subjects.find((s) => s === kept) ?? subjects[0] ?? null,
    bucket: bucketsForDate(tree, d)[0] ?? null,
  };
}

/** The most-recent date that actually has a labeled subject, or null if none do. */
function newestLabeledDate(tree: DatasetTree): string | null {
  return defaultDate(tree.captures.filter((c) => subjectsForDate(tree, c.name).length > 0));
}

/** Start opening the listed project ``p`` on sensible defaults, preferring the newest date that
 *  actually has labels. */
export function openWithDefaults(p: OpenableProject): Promise<void> {
  // Prefers a labeled date: an agent ingesting a still-unlabeled newer date would
  // otherwise land the human on a blank canvas with no date selector to recover.
  const { tree } = p;
  const date = newestLabeledDate(tree) ?? defaultDate(tree.captures);
  const { subject, bucket } = choiceForDate(tree, date, null);
  return startOpen(p, date, subject, bucket);
}
