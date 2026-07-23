import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { applyReviewFocus } from "@/lib/reviewFocus";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: { dataset: { select: vi.fn(), nav: vi.fn(async () => ({ status: "ok" })) } },
}));

import { api } from "@/api/client";

function seedDataset(partial: Record<string, unknown>) {
  useStore.getState().patchGui({
    dataset: { ...useStore.getState().gui.dataset, ...partial },
  });
}

const SELECTION = {
  project_root: "/ws/proj",
  dataset_root: "/ws/proj",
  subject: "catkin",
  date: "2026-02-11",
  image_list: [],
  current_image_index: 0, // backend resets to 0
  annotations_dir: "/ws/proj/annotations/2026-02-11",
  predictions_dir: "/ws/proj/predictions/baseline/2026-02-11",
};

beforeEach(() => {
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
});
afterEach(() => vi.clearAllMocks());

describe("applyReviewFocus", () => {
  it("switches the dataset with the model, then applies filters + index + review tab locally", async () => {
    seedDataset({ dataset_root: "/ws/proj", subject: "catkin", date: "2026-01-01" });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    vi.mocked(api.dataset.select).mockResolvedValue({ status: "ok", selection: SELECTION } as any);

    await applyReviewFocus({
      project_root: "/ws/proj",
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      model_name: "baseline",
      image_index: 12,
      detection_idx: 3,
      filter_type: "fp",
      iou_threshold: 0.4,
      conf_threshold: 0.2,
    });

    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      model_name: "baseline",
    });
    const g = useStore.getState().gui;
    expect(g.dataset.current_image_index).toBe(12); // local view wins over backend index=0
    expect(g.review.filter_type).toBe("fp");
    expect(g.review.iou_threshold).toBe(0.4);
    expect(g.review.conf_threshold).toBe(0.2);
    expect(g.active_tab).toBe("review");
    // detection_idx is a one-shot the reload effect consumes (not written to gui.review directly).
    expect(useStore.getState().review.focusDetectionIdx).toBe(3);
  });

  it("does not re-select when identity and model already match, but still applies filters", async () => {
    seedDataset({
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      predictions_dir: "/ws/proj/predictions/baseline/2026-02-11",
    });

    await applyReviewFocus({
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      model_name: "baseline",
      image_index: 7,
      filter_type: "fn",
    });

    expect(api.dataset.select).not.toHaveBeenCalled();
    const g = useStore.getState().gui;
    expect(g.dataset.current_image_index).toBe(7);
    expect(g.review.filter_type).toBe("fn");
    expect(g.active_tab).toBe("review");
    // The focused frame is persisted like user navigation, so view_gui_state sees it.
    expect(api.dataset.nav).toHaveBeenCalledWith(7);
  });

  it("re-selects when the model differs even though the dataset identity matches", async () => {
    seedDataset({
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      predictions_dir: "/ws/proj/predictions/OTHER/2026-02-11",
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    vi.mocked(api.dataset.select).mockResolvedValue({ status: "ok", selection: SELECTION } as any);

    await applyReviewFocus({
      dataset_root: "/ws/proj",
      subject: "catkin",
      date: "2026-02-11",
      model_name: "baseline", // different model than the loaded predictions dir
    });

    expect(api.dataset.select).toHaveBeenCalledTimes(1);
  });
});
