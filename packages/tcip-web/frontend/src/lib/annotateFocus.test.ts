import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { AnnotateFocus } from "@/api/types.generated";
import { applyAnnotateFocus } from "@/lib/annotateFocus";
import { useStore } from "@/store";
import {
  openTestProject as seedDataset,
  RAW_SELECTION,
  selectAnswer,
  selected as selection,
  type SelectionChange,
  stateEnvelope,
  TEST_PROJECT as PROJECT,
} from "@/test/store";

vi.mock("@/api/client", () => ({
  api: {
    dataset: {
      select: vi.fn(),
      nav: vi.fn((current_image_index: number) => Promise.resolve({ current_image_index })),
    },
  },
}));

// The backend always answers a fresh selection at its first image.
const selected = (change: SelectionChange, labelProblem: string | null = null) =>
  selectAnswer(selection(change), labelProblem);

/** A focus on the first image of the produced capture of photographs, of `bush` in box mode, no
 *  bucket and no proposal, `change` laid over. */
const focus = (change: Partial<AnnotateFocus>): AnnotateFocus => ({
  dataset_root: RAW_SELECTION.dataset_root,
  subject: "bush",
  date: RAW_SELECTION.date,
  bucket: null,
  image_index: 0,
  mode: "box",
  proposal: null,
  ...change,
});

beforeEach(() => {
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
});
afterEach(() => vi.clearAllMocks());

describe("applyAnnotateFocus", () => {
  it("switches the dataset, then applies mode + index + tab locally", async () => {
    seedDataset(selection({ subject: "subject_a" }));
    vi.mocked(api.dataset.select).mockResolvedValue(selected({ subject: "bush" }));

    const outcome = await applyAnnotateFocus(
      focus({ image_index: 4, mode: "polygon" }),
      () => true,
    );

    expect(outcome).toBe("adopted");

    // Switched the dataset (identity differed).
    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: RAW_SELECTION.dataset_root,
      subject: "bush",
      date: RAW_SELECTION.date,
    });
    // The local view controls win over the backend's index=0 reset.
    const g = useStore.getState().gui;
    expect(g.dataset?.current_image_index).toBe(4);
    expect(g.mode).toBe("polygon");
    expect(g.active_tab).toBe("annotate");
  });

  it("keeps the focus index even if the /select WS snapshot (index 0) arrives afterward", async () => {
    seedDataset(selection({ subject: "subject_a" }));
    const answer = selected({ subject: "bush" });
    const newIdentity = answer.selection;
    vi.mocked(api.dataset.select).mockResolvedValue(answer);

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
    seedDataset(selection({ subject: "bush" }));

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
    seedDataset(selection({ subject: "bush" }));
    vi.mocked(api.dataset.select).mockResolvedValue(
      selected({ subject: "bush", bucket: "m1/2026-01-01" }),
    );

    const outcome = await applyAnnotateFocus(
      focus({ bucket: "m1/2026-01-01", proposal: 3 }),
      () => true,
    );

    expect(outcome).toBe("adopted");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      bucket: "m1/2026-01-01",
    });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "proposal", index: 3 });
  });

  it("toasts a label_problem the selection carries", async () => {
    seedDataset(selection({ subject: "subject_b" }));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    vi.mocked(api.dataset.select).mockResolvedValue(
      selected(
        { subject: "subject_a" },
        "label_documents['2026-02-11', 'IMG_0000'] under /ws/proj: is a list",
      ),
    );

    const outcome = await applyAnnotateFocus(focus({ subject: "subject_a" }), () => true);

    expect(outcome).toBe("adopted");
    expect(pushToast).toHaveBeenCalledWith(
      "label_documents['2026-02-11', 'IMG_0000'] under /ws/proj: is a list",
    );
  });

  it("adopts nothing and answers declined-selected when the focus is no longer wanted once the selection resolves", async () => {
    seedDataset(selection({ subject: "subject_a", current_image_index: 4 }));
    vi.mocked(api.dataset.select).mockResolvedValue(
      selected({ subject: "bush", bucket: "m1/2026-01-01" }),
    );
    let asked = 0;

    const outcome = await applyAnnotateFocus(
      focus({ bucket: "m1/2026-01-01", image_index: 2 }),
      () => asked++ === 0,
    );

    expect(outcome).toBe("declined-selected");
    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    const g = useStore.getState().gui;
    expect(g.dataset).toMatchObject({
      subject: "subject_a",
      bucket: null,
      current_image_index: 4,
    });
    expect(api.dataset.nav).not.toHaveBeenCalled();
  });
});
