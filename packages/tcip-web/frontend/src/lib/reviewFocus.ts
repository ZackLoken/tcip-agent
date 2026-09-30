/**
 * Drive the Review tab to a model's predictions on a specific frame/detection in response to
 * the agent's `review_focus` event. The Review analog of `applyAnnotateFocus`: uses local store
 * setters, never the passive state snapshot (which keeps `current_image_index` / review filters
 * browser-local so a re-broadcast can't yank the user mid-review). A `review_focus` event is a
 * deliberate command, so applying it locally is correct and preserves that invariant.
 */

import { api } from "@/api/client";
import { toastLabelProblem } from "@/lib/labelProblemToast";
import { useStore } from "@/store";

export interface ReviewFocusData {
  dataset_root?: string;
  subject?: string | null;
  date?: string | null;
  model_name?: string | null;
  image_index?: number;
  detection_idx?: number;
  filter_type?: "all" | "tp" | "fp" | "fn";
  iou_threshold?: number;
  conf_threshold?: number;
}

export async function applyReviewFocus(d: ReviewFocusData): Promise<void> {
  const cur = useStore.getState().gui.dataset;
  const project = useStore.getState().openProject;
  const needsSwitch =
    !!d.dataset_root &&
    (d.dataset_root !== cur.dataset_root ||
      (d.subject ?? null) !== cur.subject ||
      (d.date ?? null) !== cur.date ||
      (!!d.model_name && d.model_name !== cur.model_name));
  if (needsSwitch && project) {
    const res = await api.dataset.select({
      dataset_root: d.dataset_root!,
      subject: d.subject ?? null,
      date: d.date ?? null,
      model_name: d.model_name ?? null,
    });
    useStore.getState().applyRestoredDataset(res.selection, project);
    toastLabelProblem(res.label_problem);
  }

  // Apply view + filter controls after any dataset switch resolved, so a same-identity snapshot
  // (which keeps index/filters local) can't overwrite them.
  const store = useStore.getState();
  if (typeof d.image_index === "number") {
    const ds = store.gui.dataset;
    store.patchGui({ dataset: { ...ds, current_image_index: d.image_index } });
    // Persist like user navigation does, so view_gui_state sees the focused frame.
    void api.dataset.nav(d.image_index).catch(() => {});
  }
  const review = { ...store.gui.review };
  if (d.filter_type) review.filter_type = d.filter_type;
  if (typeof d.iou_threshold === "number") review.iou_threshold = d.iou_threshold;
  if (typeof d.conf_threshold === "number") review.conf_threshold = d.conf_threshold;
  useStore.getState().patchGui({ review });
  // The detection index goes through a one-shot the reload effect honors once: writing
  // gui.review.detection_idx directly would be clobbered by the reload's "jump to first
  // unreviewed" when the frame/filters change (which this focus always causes).
  if (typeof d.detection_idx === "number") useStore.getState().setReviewFocusIdx(d.detection_idx);
  // Re-focusing the already-open image leaves paths unchanged; force a refetch past the effect's identical-path skip.
  useStore.getState().bumpReviewRefetch();
  useStore.getState().setActiveTab("review");
}
