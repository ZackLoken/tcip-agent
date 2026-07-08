import { beforeEach, describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";

const s = () => useStore.getState();

function dataset(over: Partial<DatasetSelection> = {}): DatasetSelection {
  return {
    project_root: "/proj",
    dataset_root: "/proj/ds",
    annotation_type: "catkin",
    date: "2-11-26",
    image_list: ["a.jpg", "b.jpg", "c.jpg"],
    current_image_index: 0,
    annotations_detect_dir: "/proj/ds/annotations/catkin/2-11-26/detect",
    annotations_segment_dir: null,
    predictions_detect_dir: null,
    predictions_segment_dir: null,
    ...over,
  };
}

function snapshot(over: Partial<GuiState> = {}): GuiState {
  return {
    active_tab: "annotate",
    dataset: dataset(),
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    mode: "box",
    active_class: 0,
    review: {
      iou_threshold: 0.5,
      conf_threshold: 0.25,
      filter_type: "all",
      filter_class: "all",
      status_filter: "all",
      detection_idx: 0,
    },
    pred_reference: null,
    ...over,
  };
}

describe("mergeSnapshot ownership model", () => {
  beforeEach(() => {
    // Known populated local state: user is on Review, polygon mode, class 3, image 2.
    useStore.setState({
      gui: snapshot({
        active_tab: "review",
        mode: "polygon",
        active_class: 3,
        dataset: dataset({ current_image_index: 2 }),
      }),
      wsVersion: 5,
    });
  });

  it("preserves client-owned fields on a same-dataset snapshot", () => {
    // Backend re-broadcasts with its stale active_tab / mode / index.
    s().mergeSnapshot(
      snapshot({ active_tab: "annotate", mode: "box", dataset: dataset({ current_image_index: 0 }) }),
      6,
    );
    expect(s().gui.active_tab).toBe("review");
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_class).toBe(3);
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
      gui: { ...st.gui, pred_reference: { type: "box", coords: [0, 0, 1, 1], class_id: 0, confidence: null } },
    }));
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "3-2-26", current_image_index: 0 }) }), 8);
    expect(s().gui.dataset.date).toBe("3-2-26");
    expect(s().gui.dataset.current_image_index).toBe(0);
    expect(s().gui.pred_reference).toBeNull();
  });

  it("drops a stale (older-version) replay", () => {
    s().mergeSnapshot(snapshot({ dataset: dataset({ date: "9-9-99" }) }), 3); // 3 < wsVersion 5
    expect(s().gui.dataset.date).toBe("2-11-26"); // unchanged
  });
});
