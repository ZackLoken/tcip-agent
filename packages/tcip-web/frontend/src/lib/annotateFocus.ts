/**
 * Drive the Annotate tab to a specific (trait, date, image, mode) in response to the agent's
 * `annotate_focus` event. Uses LOCAL store setters (the exact path the Review→Edit button
 * uses), never the state snapshot: `mergeSnapshot` deliberately keeps `mode` and
 * `current_image_index` browser-local so a passive re-broadcast can't yank the user mid-edit.
 * An `annotate_focus` event is a deliberate command, so applying it locally is correct and
 * safe — and preserves that invariant.
 */

import { api } from "@/api/client";
import { useStore } from "@/store";

export interface AnnotateFocusData {
  project_root?: string;
  dataset_root?: string;
  trait?: string | null;
  date?: string | null;
  image_index?: number;
  mode?: "box" | "polygon";
  active_class?: number;
}

export async function applyAnnotateFocus(d: AnnotateFocusData): Promise<void> {
  const cur = useStore.getState().gui.dataset;
  const needsSwitch =
    !!d.dataset_root &&
    (d.dataset_root !== cur.dataset_root ||
      (d.trait ?? null) !== cur.annotation_type ||
      (d.date ?? null) !== cur.date);
  if (needsSwitch) {
    const res = await api.dataset.select({
      project_root: d.project_root ?? d.dataset_root!,
      dataset_root: d.dataset_root!,
      annotation_type: d.trait ?? null,
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
    // Persist like user navigation does, so get_active_context sees the focused frame.
    void api.dataset.nav(d.image_index).catch(() => {});
  }
  if (d.mode) store.setMode(d.mode);
  // The canvas renders ONLY shapes of the active class, so set it to the class present on the
  // focused frame — otherwise a frame labelled with a non-zero class shows a blank canvas even
  // in the right mode (activeClass defaults to 0).
  if (typeof d.active_class === "number") store.setActiveClass(d.active_class);
  store.setActiveTab("annotate");
}
