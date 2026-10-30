import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { AnnotateFocus } from "@/api/types.generated";
import { applyAnnotateFocus } from "@/lib/annotateFocus";
import { useStore } from "@/store";
import { PANEL_EVENTS } from "@/test/panelEvents";
import {
  DS_BUCKET,
  FOCUSED_BUCKET_SELECT_ANSWER,
  FOCUSED_SELECT_ANSWER,
  NAV_ANSWER,
  openTestProject as seedDataset,
  RAW_SELECTION,
  selected as selection,
  stateEnvelope,
  SUBJECT_SELECT_ANSWER,
  TEST_PROJECT as PROJECT,
  TROUBLED_SELECT_ANSWER,
} from "@/test/store";

vi.mock("@/api/client", () => ({
  api: {
    dataset: {
      select: vi.fn(),
      nav: vi.fn(),
    },
  },
}));

/** The produced focus: the first image of the produced capture of photographs, in box mode, no
 *  bucket and no proposal. */
const FOCUS = PANEL_EVENTS.annotate_focus.event;

/** The subject the produced focus names. */
const FOCUSED = FOCUS.subject;

/** The subject the produced selection made for a subject names, another than {@link FOCUSED}. */
const CURRENT = SUBJECT_SELECT_ANSWER.selection.subject;

/** The first bucket published over the produced capture, the one the produced selection for
 *  {@link FOCUSED} with a bucket chose. */
const BUCKET = DS_BUCKET;

/** {@link FOCUS} asking for the image, mode, bucket or proposal `change` names: the agent's
 *  request varied by the test, its capture the produced one. */
const focus = (
  change: Partial<Pick<AnnotateFocus, "image_index" | "mode" | "bucket" | "proposal">>,
): AnnotateFocus => ({ ...FOCUS, ...change });

beforeEach(() => {
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.dataset.nav).mockResolvedValue(NAV_ANSWER);
});
afterEach(() => vi.clearAllMocks());

describe("applyAnnotateFocus", () => {
  it("switches the dataset, then applies mode + index + tab locally", async () => {
    seedDataset(selection({ subject: CURRENT }));
    vi.mocked(api.dataset.select).mockResolvedValue(FOCUSED_SELECT_ANSWER);

    const outcome = await applyAnnotateFocus(
      focus({ image_index: 4, mode: "polygon" }),
      () => true,
    );

    expect(outcome).toBe("adopted");

    // Switched the dataset (identity differed).
    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: RAW_SELECTION.dataset_root,
      subject: FOCUSED,
      date: RAW_SELECTION.date,
    });
    // The local view controls win over the backend's index=0 reset.
    const g = useStore.getState().gui;
    expect(g.dataset?.current_image_index).toBe(4);
    expect(g.mode).toBe("polygon");
    expect(g.active_tab).toBe("annotate");
  });

  it("keeps the focus index even if the /select WS snapshot (index 0) arrives afterward", async () => {
    seedDataset(selection({ subject: CURRENT }));
    const newIdentity = FOCUSED_SELECT_ANSWER.selection;
    vi.mocked(api.dataset.select).mockResolvedValue(FOCUSED_SELECT_ANSWER);

    const outcome = await applyAnnotateFocus(
      focus({ image_index: 4, mode: "polygon" }),
      () => true,
    );
    expect(outcome).toBe("adopted");

    // Emulate the backend's /select broadcast landing after the local setters: same identity
    // now, so mergeSnapshot must keep the local (focus) index, not reset to 0.
    useStore.getState().mergeSnapshot(
      stateEnvelope({
        state: { ...useStore.getState().gui, dataset: { ...newIdentity, current_image_index: 0 } },
        version: 999,
        project: PROJECT,
      }),
    );
    expect(useStore.getState().gui.dataset?.current_image_index).toBe(4);
  });

  it("does not re-select when the dataset identity already matches, but still applies view", async () => {
    seedDataset(selection({ subject: FOCUSED }));

    const outcome = await applyAnnotateFocus(
      focus({ image_index: 3, mode: "polygon" }),
      () => true,
    );

    expect(outcome).toBe("adopted");
    expect(api.dataset.select).not.toHaveBeenCalled();
    const g = useStore.getState().gui;
    expect(g.dataset?.current_image_index).toBe(3);
    expect(g.mode).toBe("polygon");
    expect(g.active_tab).toBe("annotate");
  });

  it("selects the named proposal bucket and focuses the named proposal", async () => {
    expect(FOCUSED_BUCKET_SELECT_ANSWER.selection.bucket).toBe(BUCKET);
    seedDataset(selection({ subject: FOCUSED }));
    vi.mocked(api.dataset.select).mockResolvedValue(FOCUSED_BUCKET_SELECT_ANSWER);

    const outcome = await applyAnnotateFocus(
      focus({ bucket: BUCKET, proposal: { bucket: BUCKET, index: 3 } }),
      () => true,
    );

    expect(outcome).toBe("adopted");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      bucket: BUCKET,
    });
    expect(useStore.getState().canvas.focus).toEqual({
      kind: "proposal",
      bucket: BUCKET,
      index: 3,
    });
  });

  it("toasts a label_problem the selection carries", async () => {
    const { label_problem } = TROUBLED_SELECT_ANSWER;
    seedDataset(selection({ subject: CURRENT }));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    vi.mocked(api.dataset.select).mockResolvedValue(TROUBLED_SELECT_ANSWER);

    const outcome = await applyAnnotateFocus(
      PANEL_EVENTS.annotate_focus_troubled.event,
      () => true,
    );

    expect(outcome).toBe("adopted");
    expect(pushToast).toHaveBeenCalledWith(label_problem);
  });

  it("adopts nothing and answers declined-selected when the focus is no longer wanted once the selection resolves", async () => {
    seedDataset(selection({ subject: CURRENT, current_image_index: 4 }));
    vi.mocked(api.dataset.select).mockResolvedValue(FOCUSED_BUCKET_SELECT_ANSWER);
    let asked = 0;

    const outcome = await applyAnnotateFocus(
      focus({ bucket: BUCKET, image_index: 2 }),
      () => asked++ === 0,
    );

    expect(outcome).toBe("declined-selected");
    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    const g = useStore.getState().gui;
    expect(g.dataset).toMatchObject({
      subject: CURRENT,
      bucket: null,
      current_image_index: 4,
    });
    expect(api.dataset.nav).not.toHaveBeenCalled();
  });
});
