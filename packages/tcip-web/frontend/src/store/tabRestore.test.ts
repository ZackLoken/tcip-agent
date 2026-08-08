import { beforeEach, describe, expect, it } from "vitest";

import { loadLastTab, recordLastTab } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { DatasetSelection } from "@/store/types";

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
});
