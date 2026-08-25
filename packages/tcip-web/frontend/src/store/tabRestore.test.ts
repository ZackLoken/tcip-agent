import { beforeEach, describe, expect, it } from "vitest";

import { TAB_NAMES } from "@/api/types.generated";
import { datasetKey, loadLastTab, recordLastTab, saveDatasetUi } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { DatasetSelection, ReviewFilters, TabName } from "@/store/types";

const s = () => useStore.getState();

const ROOT = "/w/alpha";

function selection(over: Partial<DatasetSelection> = {}): DatasetSelection {
  return {
    project_root: ROOT,
    dataset_root: ROOT,
    subject: "subject_a",
    date: "2026-01-01",
    image_list: ["a.jpg"],
    current_image_index: 0,
    images_dir: null,
    annotations_dir: null,
    predictions_dir: null,
    ...over,
  };
}

describe("per-project last-used tab", () => {
  beforeEach(() => {
    localStorage.removeItem(`tcip.lasttab.${ROOT}`);
    useStore.setState((st) => ({
      gui: { ...st.gui, active_tab: "annotate", dataset: selection() },
    }));
  });

  it("records the tab per project as the user switches", () => {
    s().setActiveTab("results");
    expect(loadLastTab(ROOT)).toBe("results");
  });

  it("does not record a tab when no project is open", () => {
    useStore.setState((st) => ({
      gui: {
        ...st.gui,
        dataset: selection({ project_root: null, dataset_root: null }),
      },
    }));
    s().setActiveTab("results");
    expect(loadLastTab(ROOT)).toBeNull();
  });

  it("opening a project restores its recorded tab", () => {
    recordLastTab(ROOT, "inference");
    s().applyRestoredDataset(selection());
    expect(s().gui.active_tab).toBe("inference");
  });

  it("first-ever open of a project lands on Annotate", () => {
    useStore.setState((st) => ({ gui: { ...st.gui, active_tab: "review" } }));
    s().applyRestoredDataset(selection());
    expect(s().gui.active_tab).toBe("annotate");
  });

  it("loadLastTab rejects a value that is not a tab name", () => {
    localStorage.setItem(`tcip.lasttab.${ROOT}`, "garbage");
    expect(loadLastTab(ROOT)).toBeNull();
  });

  it("restores each of the app's tabs, not just some of them", () => {
    expect(TAB_NAMES).toHaveLength(7);
    for (const tab of TAB_NAMES) {
      const other: TabName = tab === "annotate" ? "review" : "annotate";
      recordLastTab(ROOT, tab);
      useStore.setState((st) => ({ gui: { ...st.gui, active_tab: other } }));
      s().applyRestoredDataset(selection());
      expect(s().gui.active_tab).toBe(tab);
    }
  });
});

describe("saved review filters on reopening a dataset", () => {
  const SAVED_FILTERS: ReviewFilters = {
    iou_threshold: 0.7,
    conf_threshold: 0.4,
    filter_type: "fp",
    filter_class: "subject_b",
    detection_idx: 3,
  };

  const LIVE_FILTERS: ReviewFilters = {
    iou_threshold: 0.5,
    conf_threshold: 0.25,
    filter_type: "all",
    filter_class: "all",
    detection_idx: 0,
  };

  beforeEach(() => {
    sessionStorage.clear();
    localStorage.removeItem(`tcip.lasttab.${ROOT}`);
    useStore.setState((st) => ({
      gui: { ...st.gui, review: LIVE_FILTERS, dataset: selection() },
      imageStatus: { ...st.imageStatus, activeFilter: "all" },
    }));
  });

  it("brings back the filters the reviewer left on this dataset", () => {
    const sel = selection({ image_list: ["a.jpg", "b.jpg", "c.jpg"] });
    const key = datasetKey(sel);
    expect(key).not.toBeNull();
    saveDatasetUi(key as string, {
      index: 2,
      review: SAVED_FILTERS,
      statusFilter: "negative",
    });

    s().applyRestoredDataset(sel);

    expect(s().gui.review.iou_threshold).toBe(0.7);
    expect(s().gui.review.conf_threshold).toBe(0.4);
    expect(s().gui.review.filter_type).toBe("fp");
    expect(s().gui.review.filter_class).toBe("subject_b");
    expect(s().gui.review.detection_idx).toBe(3);
    expect(s().gui.dataset.current_image_index).toBe(2);
    expect(s().imageStatus.activeFilter).toBe("negative");
  });

  it("keeps the live filters when this dataset has nothing saved", () => {
    s().applyRestoredDataset(selection());
    expect(s().gui.review).toEqual(LIVE_FILTERS);
    expect(s().imageStatus.activeFilter).toBe("all");
  });
});
