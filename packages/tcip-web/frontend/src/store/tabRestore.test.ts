import { beforeEach, describe, expect, it } from "vitest";

import { TAB_NAMES } from "@/api/types.generated";
import { datasetKey, loadLastTab, recordLastTab, saveDatasetUi } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { TabName } from "@/store/types";
import { selected, type SelectionChange, SUBJECT_SELECT_ANSWER, TEST_PROJECT } from "@/test/store";

const s = () => useStore.getState();

const PROJECT = TEST_PROJECT;

const selection = (change: SelectionChange = {}) =>
  selected({ count: 1, ...change }, SUBJECT_SELECT_ANSWER.selection);

describe("per-project last-used tab", () => {
  beforeEach(() => {
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
    useStore.setState((st) => ({
      gui: { ...st.gui, active_tab: "annotate", dataset: selection() },
      openProject: PROJECT,
    }));
  });

  it("records the tab per project as the user switches", () => {
    s().setActiveTab("results");
    expect(loadLastTab(PROJECT.id)).toBe("results");
  });

  it("does not record a tab when no project is open", () => {
    useStore.setState({ openProject: null });
    s().setActiveTab("results");
    expect(loadLastTab(PROJECT.id)).toBeNull();
  });

  it("opening a project restores its recorded tab", () => {
    recordLastTab(PROJECT.id, "inference");
    s().applyRestoredDataset(selection(), PROJECT);
    expect(s().gui.active_tab).toBe("inference");
  });

  it("first-ever open of a project lands on Annotate", () => {
    useStore.setState((st) => ({ gui: { ...st.gui, active_tab: "results" } }));
    s().applyRestoredDataset(selection(), PROJECT);
    expect(s().gui.active_tab).toBe("annotate");
  });

  it("loadLastTab refuses a recorded value that is not a tab name", () => {
    localStorage.setItem(`tcip.lasttab.${PROJECT.id}`, "garbage");
    expect(() => loadLastTab(PROJECT.id)).toThrow(/names no tab: garbage/);
  });

  it("restores each of the app's tabs, not just some of them", () => {
    for (const tab of TAB_NAMES) {
      const other: TabName = tab === "annotate" ? "results" : "annotate";
      recordLastTab(PROJECT.id, tab);
      useStore.setState((st) => ({ gui: { ...st.gui, active_tab: other } }));
      s().applyRestoredDataset(selection(), PROJECT);
      expect(s().gui.active_tab).toBe(tab);
    }
  });
});

describe("saved position on reopening a dataset", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
    useStore.setState((st) => ({ gui: { ...st.gui, dataset: selection() }, openProject: PROJECT }));
  });

  it("brings back the image the person left this dataset on", () => {
    const sel = selection({ count: 3 });
    saveDatasetUi(datasetKey(PROJECT, sel), { index: 2 });

    s().applyRestoredDataset(sel, PROJECT);

    expect(s().gui.dataset?.current_image_index).toBe(2);
  });
});
