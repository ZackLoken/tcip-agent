import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { applyAnnotateFocus } from "@/lib/annotateFocus";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: { dataset: { select: vi.fn(), nav: vi.fn(async () => ({ status: "ok" })) } },
}));

import { api } from "@/api/client";

function seedDataset(partial: Record<string, unknown>) {
  useStore.getState().patchGui({
    dataset: {
      ...useStore.getState().gui.dataset,
      ...partial,
    },
  });
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
      selection: {
        project_root: "/ws/proj",
        dataset_root: "/ws/proj",
        subject: "bush",
        date: "2026-03-02",
        image_list: [],
        current_image_index: 0, // backend always resets to 0
        annotations_dir: "/ws/proj/annotations/2026-03-02",
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      project_root: "/ws/proj",
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
    const newIdentity = {
      project_root: "/ws/proj",
      dataset_root: "/ws/proj",
      subject: "bush",
      date: "2026-03-02",
      image_list: [],
      current_image_index: 0,
      annotations_dir: "/ws/proj/annotations/2026-03-02",
      predictions_dir: null,
    };
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
    useStore
      .getState()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .mergeSnapshot({ dataset: { ...newIdentity, current_image_index: 0 } } as any, 999);
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

  it("clears a prior dataset's reviewStatus when the focus switches identity", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "subject_a", date: "2026-01-01" });
    useStore.setState({
      reviewStatus: {
        byImage: { "old.jpg": "completed" },
        hasDetections: { "old.jpg": true },
        unreadable: [],
        activeFilter: "all",
      },
    });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: {
        project_root: "/ws/proj",
        dataset_root: "/ws/proj",
        subject: "subject_a",
        date: "2026-02-11",
        image_list: [],
        current_image_index: 0,
        annotations_dir: "/ws/proj/annotations/2026-02-11",
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "subject_a",
      date: "2026-02-11",
    });

    expect(useStore.getState().reviewStatus.byImage).toEqual({});
  });

  it("toasts a label_problem the selection carries", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "subject_a", date: "2026-01-01" });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: {
        project_root: "/ws/proj",
        dataset_root: "/ws/proj",
        subject: "subject_a",
        date: "2026-02-11",
        image_list: [],
        current_image_index: 0,
        annotations_dir: "/ws/proj/annotations/2026-02-11",
        predictions_dir: null,
      },
      label_problem: "/ws/proj/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    await applyAnnotateFocus({
      dataset_root: "/ws/proj",
      subject: "subject_a",
      date: "2026-02-11",
    });

    expect(pushToast).toHaveBeenCalledWith(
      "/ws/proj/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
    );
  });
});
