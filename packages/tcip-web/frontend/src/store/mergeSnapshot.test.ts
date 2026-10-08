import { beforeEach, describe, expect, it } from "vitest";

import { recordLastTab } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState, OpenProject } from "@/store/types";

const s = () => useStore.getState();

const PROJECT: OpenProject = { id: "a1b2c3d4e5f6", path: "/proj" };
const OTHER: OpenProject = { id: "ffffffffffff", path: "/other" };

function dataset(over: Partial<DatasetSelection> = {}): DatasetSelection {
  return {
    dataset_root: "/proj/ds",
    subject: "subject_a",
    date: "2-11-26",
    image_list: ["a.jpg", "b.jpg", "c.jpg"],
    current_image_index: 0,
    images_dir: "/proj/ds/images/2-11-26",
    bucket: null,
    ...over,
  };
}

const EMPTY = { dataset_root: null, date: null, image_list: [] };

function snapshot(over: Partial<GuiState> = {}): GuiState {
  return {
    active_tab: "annotate",
    dataset: dataset(),
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    mode: "box",
    active_subject: "subject_a",
    ...over,
  };
}

describe("mergeSnapshot ownership model", () => {
  beforeEach(() => {
    // Known populated local state: user is on Results, polygon mode, subject "bush", image 2.
    useStore.setState({
      gui: snapshot({
        active_tab: "results",
        mode: "polygon",
        active_subject: "bush",
        dataset: dataset({ current_image_index: 2 }),
      }),
      openProject: PROJECT,
      wsVersion: 5,
      wsEpoch: null,
      heldContributions: [],
    });
  });

  it("preserves client-owned fields on a same-dataset snapshot", () => {
    // Backend re-broadcasts with its stale active_tab / mode / index / subject.
    s().mergeSnapshot(
      snapshot({
        active_tab: "annotate",
        mode: "box",
        active_subject: "subject_a",
        dataset: dataset({ current_image_index: 0 }),
      }),
      6,
      PROJECT,
      null,
    );
    expect(s().gui.active_tab).toBe("results");
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset.current_image_index).toBe(2); // navigation kept
  });

  it("keeps a populated dataset when the same project's snapshot carries none", () => {
    s().mergeSnapshot(snapshot({ dataset: dataset(EMPTY) }), 7, PROJECT, null);
    expect(s().gui.dataset.dataset_root).toBe("/proj/ds");
    expect(s().gui.dataset.image_list).toHaveLength(3);
  });

  it("adopts the snapshot wholesale when it names another open project", () => {
    s().mergeSnapshot(
      snapshot({ dataset: dataset({ dataset_root: "/other", date: "3-2-26" }) }),
      7,
      OTHER,
      null,
    );
    expect(s().openProject).toEqual(OTHER);
    expect(s().gui.dataset.dataset_root).toBe("/other");
  });

  it("drops the dataset when the backend has no project open", () => {
    s().mergeSnapshot(snapshot({ dataset: dataset(EMPTY) }), 7, null, null);
    expect(s().openProject).toBeNull();
    expect(s().gui.dataset.dataset_root).toBeNull();
  });

  it("adopts a new dataset identity and resets the index", () => {
    s().mergeSnapshot(
      snapshot({ dataset: dataset({ date: "3-2-26", current_image_index: 0 }) }),
      8,
      PROJECT,
      null,
    );
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().gui.dataset.current_image_index).toBe(0);
  });

  it("drops a stale (older-version) replay", () => {
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "9-9-99" }) }), 3, PROJECT, null); // 3 < wsVersion 5
    expect(s().gui.dataset.date).toBe("2-11-26"); // unchanged
  });

  it("leaves an image visit open when a stale snapshot of another project is rejected", () => {
    s().setUser("grower");
    s().startImageSessionTracking("a.jpg");

    s().mergeSnapshot(snapshot(), 3, OTHER, null); // 3 < wsVersion 5

    expect(s().openProject).toEqual(PROJECT);
    expect(s().sessionTracking.currentImageName).toBe("a.jpg");
    expect(s().heldContributions).toEqual([]);
  });

  it("retires the departed project's visits in one notice naming every image when another project is adopted", () => {
    const images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"];
    s().setUser("grower");
    useStore.setState({ toasts: [] });
    images.forEach((image, i) => {
      s().startImageSessionTracking(image, Date.now() - 2000);
      if (i < images.length - 1) s().closeSessionInterval();
    });

    s().mergeSnapshot(snapshot(), 6, OTHER, null);

    expect(s().sessionTracking.currentImageName).toBeNull();
    expect(s().heldContributions).toEqual([]);
    expect(s().toasts.map((t) => t.message)).toEqual([
      `Not recorded, as their project was switched away from: ${images.join(", ")}`,
    ]);
  });

  it("applies a snapshot carrying the version already recorded", () => {
    // Only an older version is a stale replay; a re-broadcast at the current version is real state.
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "3-2-26" }) }), 5, PROJECT, null);
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().wsVersion).toBe(5);
  });

  it("accepts a lower version when the epoch moved (a restarted backend's own replay)", () => {
    useStore.setState({ wsEpoch: "epoch-1" });
    s().mergeSnapshot(
      snapshot({ dataset: dataset({ date: "3-2-26" }) }),
      1, // lower than wsVersion 5, and would ordinarily be dropped as stale
      PROJECT,
      "epoch-2",
    );
    expect(s().gui.dataset.date).toBe("3-2-26"); // accepted, not dropped
    expect(s().wsVersion).toBe(1);
    expect(s().wsEpoch).toBe("epoch-2");
  });

  it("keeps the local image-list array itself on a same-dataset snapshot", () => {
    // Effects keyed on the image_list reference must not re-fire, so the array object has to
    // survive, not just its contents.
    const localList = s().gui.dataset.image_list;
    const incoming = snapshot({
      dataset: dataset({ bucket: "m2/2-11-26" }),
    });
    expect(incoming.dataset.image_list).not.toBe(localList);
    expect(incoming.dataset.image_list).toEqual(localList);

    s().mergeSnapshot(incoming, 6, PROJECT, null);

    expect(s().gui.dataset.image_list).toBe(localList);
    expect(s().gui.dataset.bucket).toBe("m2/2-11-26");
  });

  it("adopts the persisted state on boot, with the tab from the project's own record", () => {
    // Boot adopts backend mode/position; the tab is the client's per-project record, since the
    // backend's active_tab only moves on agent focus events.
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
    useStore.setState({
      gui: snapshot({ active_subject: null, dataset: dataset({ ...EMPTY, subject: null }) }),
      openProject: null,
      wsVersion: 0,
    });
    s().mergeSnapshot(
      snapshot({
        active_tab: "results",
        mode: "polygon",
        active_subject: "bush",
        dataset: dataset({ current_image_index: 2 }),
      }),
      1,
      PROJECT,
      null,
    );
    expect(s().gui.active_tab).toBe("annotate"); // no record yet: first-open default
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset.current_image_index).toBe(2);
    expect(s().openProject).toEqual(PROJECT);
  });

  it("boot hydration lands on the project's recorded last-used tab when one exists", () => {
    recordLastTab(PROJECT.id, "training");
    useStore.setState({
      gui: snapshot({ active_subject: null, dataset: dataset({ ...EMPTY, subject: null }) }),
      openProject: null,
      wsVersion: 0,
    });
    s().mergeSnapshot(snapshot({ active_tab: "results" }), 1, PROJECT, null);
    expect(s().gui.active_tab).toBe("training");
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
  });
});

describe("applyRestoredDataset", () => {
  beforeEach(() => {
    useStore.setState({ gui: snapshot({ dataset: dataset() }), openProject: PROJECT });
  });

  it("adopts the open project in the same update as the dataset", () => {
    s().applyRestoredDataset(dataset({ dataset_root: "/other", date: "3-2-26" }), OTHER);
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().openProject).toEqual(OTHER);
  });
});

describe("clearDataset", () => {
  it("clears the dataset selection", () => {
    useStore.setState({ gui: snapshot({ dataset: dataset() }) });
    s().clearDataset();
    expect(s().gui.dataset.dataset_root).toBeNull();
  });
});
