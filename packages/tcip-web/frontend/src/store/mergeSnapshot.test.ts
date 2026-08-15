import { beforeEach, describe, expect, it } from "vitest";

import { recordLastTab } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";

const s = () => useStore.getState();

function dataset(over: Partial<DatasetSelection> = {}): DatasetSelection {
  return {
    project_root: "/proj",
    dataset_root: "/proj/ds",
    subject: "subject_a",
    date: "2-11-26",
    image_list: ["a.jpg", "b.jpg", "c.jpg"],
    current_image_index: 0,
    images_dir: "/proj/ds/images/2-11-26",
    annotations_dir: "/proj/ds/annotations/2-11-26",
    predictions_dir: null,
    ...over,
  };
}

function snapshot(over: Partial<GuiState> = {}): GuiState {
  return {
    active_tab: "annotate",
    dataset: dataset(),
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    mode: "box",
    active_subject: "subject_a",
    review: {
      iou_threshold: 0.5,
      conf_threshold: 0.25,
      filter_type: "all",
      filter_class: "all",
      detection_idx: 0,
    },
    pred_reference: null,
    ...over,
  };
}

describe("mergeSnapshot ownership model", () => {
  beforeEach(() => {
    // Known populated local state: user is on Review, polygon mode, subject "bush", image 2.
    useStore.setState({
      gui: snapshot({
        active_tab: "review",
        mode: "polygon",
        active_subject: "bush",
        dataset: dataset({ current_image_index: 2 }),
      }),
      wsVersion: 5,
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
    );
    expect(s().gui.active_tab).toBe("review");
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset.current_image_index).toBe(2); // navigation kept
  });

  it("does not clobber a populated dataset with an empty backend snapshot (restart)", () => {
    s().mergeSnapshot(
      snapshot({
        dataset: dataset({ project_root: null, dataset_root: null, date: null, image_list: [] }),
      }),
      7,
    );
    expect(s().gui.dataset.dataset_root).toBe("/proj/ds");
    expect(s().gui.dataset.image_list).toHaveLength(3);
  });

  it("adopts a new dataset identity and resets index + pred_reference", () => {
    useStore.setState((st) => ({
      gui: {
        ...st.gui,
        pred_reference: { type: "box", coords: [0, 0, 1, 1], confidence: null },
      },
    }));
    s().mergeSnapshot(
      snapshot({ dataset: dataset({ date: "3-2-26", current_image_index: 0 }) }),
      8,
    );
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().gui.dataset.current_image_index).toBe(0);
    expect(s().gui.pred_reference).toBeNull();
  });

  it("drops a stale (older-version) replay", () => {
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "9-9-99" }) }), 3); // 3 < wsVersion 5
    expect(s().gui.dataset.date).toBe("2-11-26"); // unchanged
  });

  it("applies a snapshot carrying the version already recorded", () => {
    // Only an older version is a stale replay; a re-broadcast at the current version is real state.
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "3-2-26" }) }), 5);
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().wsVersion).toBe(5);
  });

  it("keeps the local image-list array itself on a same-dataset snapshot", () => {
    // Effects keyed on the image_list reference must not re-fire, so the array object has to
    // survive, not just its contents.
    const localList = s().gui.dataset.image_list;
    const incoming = snapshot({
      dataset: dataset({ predictions_dir: "/proj/ds/predictions/m2/2-11-26" }),
    });
    expect(incoming.dataset.image_list).not.toBe(localList);
    expect(incoming.dataset.image_list).toEqual(localList);

    s().mergeSnapshot(incoming, 6);

    expect(s().gui.dataset.image_list).toBe(localList);
    expect(s().gui.dataset.predictions_dir).toBe("/proj/ds/predictions/m2/2-11-26");
  });

  it("adopts the persisted state on boot, with the tab from the project's own record", () => {
    // Boot adopts backend mode/filters/position; the tab is the client's per-project record,
    // since the backend's active_tab only moves on agent focus events (stale, often Review).
    localStorage.removeItem("tcip.lasttab./proj");
    useStore.setState({
      gui: snapshot({
        active_subject: null,
        dataset: dataset({
          project_root: null,
          dataset_root: null,
          subject: null,
          date: null,
          image_list: [],
          current_image_index: 0,
        }),
      }),
      wsVersion: 0,
    });
    s().mergeSnapshot(
      snapshot({
        active_tab: "review",
        mode: "polygon",
        active_subject: "bush",
        dataset: dataset({ current_image_index: 2 }),
      }),
      1,
    );
    expect(s().gui.active_tab).toBe("annotate"); // no record yet: first-open default
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset.current_image_index).toBe(2);
  });

  it("boot hydration lands on the project's recorded last-used tab when one exists", () => {
    recordLastTab("/proj", "training");
    useStore.setState({
      gui: snapshot({
        active_subject: null,
        dataset: dataset({
          project_root: null,
          dataset_root: null,
          subject: null,
          date: null,
          image_list: [],
          current_image_index: 0,
        }),
      }),
      wsVersion: 0,
    });
    s().mergeSnapshot(snapshot({ active_tab: "review" }), 1);
    expect(s().gui.active_tab).toBe("training");
    localStorage.removeItem("tcip.lasttab./proj");
  });
});
