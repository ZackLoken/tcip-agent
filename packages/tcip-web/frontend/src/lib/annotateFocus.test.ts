import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { applyAnnotateFocus } from "@/lib/annotateFocus";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: { dataset: { select: vi.fn(), nav: vi.fn(async () => ({ status: "ok" })) } },
}));

import { api } from "@/api/client";

const PROJECT = { id: "a1b2c3d4e5f6", path: "/ws/proj" };

function seedDataset(partial: Record<string, unknown>) {
  useStore.getState().patchGui({
    dataset: {
      ...useStore.getState().gui.dataset,
      ...partial,
    },
  });
  useStore.setState({ openProject: PROJECT });
}

function selection(over: Record<string, unknown>) {
  return {
    dataset_root: "/ws/proj",
    image_list: [],
    current_image_index: 0, // backend always resets to 0
    images_dir: null,
    bucket: null,
    ...over,
  };
}

beforeEach(() => {
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
});
afterEach(() => vi.clearAllMocks());

describe("applyAnnotateFocus", () => {
  it("switches the dataset, then applies mode + index + tab locally", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "subject_a", date: "2026-02-11" });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: selection({ subject: "bush", date: "2026-03-02" }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      image_index: 47,
      mode: "polygon",
    });

    // Switched the dataset (identity differed).
    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
    });
    // The local view controls win over the backend's index=0 reset.
    const g = useStore.getState().gui;
    expect(g.dataset.current_image_index).toBe(47);
    expect(g.mode).toBe("polygon");
    expect(g.active_tab).toBe("annotate");
  });

  it("keeps the focus index even if the /select WS snapshot (index 0) arrives afterward", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "subject_a", date: "2026-02-11" });
    const newIdentity = selection({ subject: "bush", date: "2026-03-02" });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      selection: newIdentity as any,
    });

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      image_index: 47,
      mode: "polygon",
    });

    // Emulate the backend's /select broadcast landing after the local setters: same identity
    // now, so mergeSnapshot must keep the local (focus) index, not reset to 0.
    useStore.getState().mergeSnapshot(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      { dataset: { ...newIdentity, current_image_index: 0 } } as any,
      999,
      PROJECT,
      null,
    );
    expect(useStore.getState().gui.dataset.current_image_index).toBe(47);
  });

  it("does not re-select when the dataset identity already matches, but still applies view", async () => {
    seedDataset({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      current_image_index: 0,
    });

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      image_index: 12,
      mode: "polygon",
    });

    expect(api.dataset.select).not.toHaveBeenCalled();
    const g = useStore.getState().gui;
    expect(g.dataset.current_image_index).toBe(12);
    expect(g.mode).toBe("polygon");
    expect(g.active_tab).toBe("annotate");
  });

  it("selects the named proposal bucket and focuses the named proposal", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "bush", date: "2026-03-02" });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: selection({ subject: "bush", date: "2026-03-02", bucket: "m1/2026-03-02" }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      bucket: "m1/2026-03-02",
      proposal: 3,
    });

    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      bucket: "m1/2026-03-02",
    });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "proposal", index: 3 });
  });

  it("toasts a label_problem the selection carries", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "subject_a", date: "2026-01-01" });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: selection({ subject: "subject_a", date: "2026-02-11" }),
      label_problem: "label_documents['2026-02-11', 'IMG_0000'] under /ws/proj: is a list",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "subject_a",
      date: "2026-02-11",
    });

    expect(pushToast).toHaveBeenCalledWith(
      "label_documents['2026-02-11', 'IMG_0000'] under /ws/proj: is a list",
    );
  });
});
