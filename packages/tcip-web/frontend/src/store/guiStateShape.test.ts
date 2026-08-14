import { describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { GuiState } from "@/store/types";

// Captured before any test runs, so the opening shape is read from the store itself and not
// from whatever an earlier test left behind.
const OPENING_GUI = useStore.getState().gui;

const s = () => useStore.getState();

describe("the GUI state the browser opens with", () => {
  it("carries every field of the backend state, at the values the backend also starts from", () => {
    expect(OPENING_GUI).toEqual({
      active_tab: "annotate",
      dataset: {
        project_root: null,
        dataset_root: null,
        subject: null,
        date: null,
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: null,
      review: {
        iou_threshold: 0.5,
        conf_threshold: 0.25,
        filter_type: "all",
        filter_class: "all",
        detection_idx: 0,
      },
      pred_reference: null,
    });
  });
});

describe("adopting a different dataset selection", () => {
  it("takes every field of the new selection, leaving none of the old one behind", () => {
    useStore.setState({
      gui: {
        active_tab: "review",
        dataset: {
          project_root: "/proj/alpha",
          dataset_root: "/proj/alpha/ds",
          subject: "leaf",
          date: "2026-03-01",
          image_list: ["l1.jpg", "l2.jpg"],
          current_image_index: 1,
          annotations_dir: "/proj/alpha/ds/annotations/2026-03-01",
          predictions_dir: "/proj/alpha/ds/predictions/m1/2026-03-01",
        },
        view: { scale: 2, offset_x: 30, offset_y: 70 },
        mode: "polygon",
        active_subject: "leaf",
        review: {
          iou_threshold: 0.5,
          conf_threshold: 0.25,
          filter_type: "all",
          filter_class: "all",
          detection_idx: 0,
        },
        pred_reference: null,
      },
      wsVersion: 4,
    });

    const incoming: GuiState = {
      active_tab: "annotate",
      dataset: {
        project_root: "/proj/beta",
        dataset_root: "/proj/beta/ds",
        subject: "bud",
        date: "2026-04-02",
        image_list: ["b1.jpg", "b2.jpg", "b3.jpg"],
        current_image_index: 2,
        annotations_dir: "/proj/beta/ds/annotations/2026-04-02",
        predictions_dir: "/proj/beta/ds/predictions/m2/2026-04-02",
      },
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: "bud",
      review: {
        iou_threshold: 0.5,
        conf_threshold: 0.25,
        filter_type: "all",
        filter_class: "all",
        detection_idx: 0,
      },
      pred_reference: null,
    };

    s().mergeSnapshot(incoming, 5);

    expect(s().gui.dataset).toEqual({
      project_root: "/proj/beta",
      dataset_root: "/proj/beta/ds",
      subject: "bud",
      date: "2026-04-02",
      image_list: ["b1.jpg", "b2.jpg", "b3.jpg"],
      current_image_index: 2,
      annotations_dir: "/proj/beta/ds/annotations/2026-04-02",
      predictions_dir: "/proj/beta/ds/predictions/m2/2026-04-02",
    });
  });
});
