/**
 * Drive the Annotate tab to a specific (subject, date, image, mode) in response to the agent's
 * `annotate_focus` event. Uses local store setters (the exact path the Review→Edit button
 * uses), never the state snapshot: `mergeSnapshot` deliberately keeps `mode` and
 * `current_image_index` browser-local so a passive re-broadcast can't yank the user mid-edit.
 * An `annotate_focus` event is a deliberate command, so applying it locally is correct and
 * safe, and preserves that invariant.
 */

import { api } from "@/api/client";
import { useStore } from "@/store";
import type { Mode } from "@/store/types";

export interface AnnotateFocusData {
  project_root?: string;
  dataset_root?: string;
  subject?: string | null;
  date?: string | null;
  image_index?: number;
  mode?: Mode;
  active_subject?: string | null;
}

export async function applyAnnotateFocus(d: AnnotateFocusData): Promise<void> {
  const cur = useStore.getState().gui.dataset;
  const needsSwitch =
    !!d.dataset_root &&
    (d.dataset_root !== cur.dataset_root ||
      (d.subject ?? null) !== cur.subject ||
      (d.date ?? null) !== cur.date);
  if (needsSwitch) {
    const res = await api.dataset.select({
      project_root: d.project_root ?? d.dataset_root!,
      dataset_root: d.dataset_root!,
      subject: d.subject ?? null,
      date: d.date ?? null,
      model_name: null,
    });
    useStore.getState().patchGui({ dataset: res.selection });
  }
  // Apply the view controls after any dataset switch has resolved, so a same-identity
  // snapshot (which keeps index/mode local) can't overwrite them.
  const store = useStore.getState();
  if (typeof d.image_index === "number") {
    const ds = store.gui.dataset;
    store.patchGui({ dataset: { ...ds, current_image_index: d.image_index } });
    // Persist like user navigation does, so view_gui_state sees the focused frame.
    void api.dataset.nav(d.image_index).catch(() => {});
  }
  if (d.mode) store.setMode(d.mode);
  // The canvas renders only shapes of the active subject, so set it to the subject present on the
  // focused frame, otherwise a frame labelled for another subject shows a blank canvas even in
  // the right mode.
  if (d.active_subject) store.setActiveSubject(d.active_subject);
  store.setActiveTab("annotate");
}
