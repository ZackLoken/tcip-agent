/**
 * Drive the Annotate tab to a dataset, subject, date, bucket, image, mode and proposal.
 *
 * `wanted` is asked first and again once a requested selection resolves; while it answers true,
 * a dataset root differing from the selection in root, subject, date or bucket is selected
 * through `/dataset/select` and adopted, a label problem it names toasted, then each stated image
 * index (persisted), mode, active subject and focused proposal applied and the Annotate tab
 * shown. Answers `adopted` when the focus was applied, `declined-unselected` when `wanted`
 * answered false before any selection was requested, and `declined-selected` when it answered
 * false after the selection resolved, the backend now holding that selection; a decline adopts
 * and applies nothing.
 */

import { api } from "@/api/client";
import type { AnnotateFocus } from "@/api/types.generated";
import { toastLabelProblem } from "@/lib/labelProblemToast";
import { useStore } from "@/store";

export type AnnotateFocusOutcome = "adopted" | "declined-unselected" | "declined-selected";

export async function applyAnnotateFocus(
  d: AnnotateFocus,
  wanted: () => boolean,
): Promise<AnnotateFocusOutcome> {
  if (!wanted()) return "declined-unselected";
  const cur = useStore.getState().gui.dataset;
  const needsSwitch =
    cur === null ||
    d.dataset_root !== cur.dataset_root ||
    d.subject !== cur.subject ||
    d.date !== cur.date ||
    d.bucket !== cur.bucket;
  const project = useStore.getState().openProject;
  if (needsSwitch && project) {
    const res = await api.dataset.select({
      dataset_root: d.dataset_root,
      subject: d.subject,
      date: d.date,
      bucket: d.bucket,
    });
    if (!wanted()) return "declined-selected";
    useStore.getState().applyRestoredDataset(res.selection, project);
    toastLabelProblem(res.label_problem);
  }
  // Apply the view controls after any dataset switch has resolved, so a same-identity
  // snapshot (which keeps index/mode local) can't overwrite them.
  const store = useStore.getState();
  const ds = store.gui.dataset;
  if (ds) {
    store.patchGui({ dataset: { ...ds, current_image_index: d.image_index } });
    // Persist like user navigation does, so view_gui_state sees the focused frame.
    void api.dataset.nav(d.image_index).catch(() => {});
  }
  store.setMode(d.mode);
  // The canvas renders only shapes of the active subject, so a frame labeled for another
  // subject would show a blank canvas even in the right mode.
  store.setActiveSubject(d.subject);
  store.setFocus(d.proposal === null ? null : { kind: "proposal", index: d.proposal });
  store.setActiveTab("annotate");
  return "adopted";
}
