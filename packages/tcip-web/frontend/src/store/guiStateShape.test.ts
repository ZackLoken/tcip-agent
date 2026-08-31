import { describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { GuiState } from "@/store/types";

// Captured before any test runs, so the opening shape is read from the store itself and not
// from whatever an earlier test left behind.
const OPENING_GUI = useStore.getState().gui;

const s = () => useStore.getState();

// tcip_web.state.GuiState's own field names, transcribed since this suite has no live backend
// to query; a field added there and not here fails the difference assertion below.
const BACKEND_GUI_STATE_FIELDS = [
  "active_tab",
  "dataset",
  "view",
  "mode",
  "active_subject",
  "review",
  "pred_reference",
];

describe("the GUI state the browser opens with", () => {
  it("carries every field of the backend state but pred_reference, at the values the backend also starts from", () => {
    expect(OPENING_GUI).toEqual({
      active_tab: "annotate",
      dataset: {
        project_root: null,
        dataset_root: null,
        subject: null,
        date: null,
        image_list: [],
        current_image_index: 0,
        images_dir: null,
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
    });

    // pred_reference is gui_snapshot's frozen v1 resident, kept on the backend with no browser
    // consumer; the difference between the two field sets must be exactly that one key.
    const missing = BACKEND_GUI_STATE_FIELDS.filter((field) => !(field in OPENING_GUI));
    const extra = Object.keys(OPENING_GUI).filter(
      (field) => !BACKEND_GUI_STATE_FIELDS.includes(field),
    );
    expect(missing).toEqual(["pred_reference"]);
    expect(extra).toEqual([]);
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
          images_dir: "/proj/alpha/ds/images/2026-03-01",
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
        images_dir: "/proj/beta/ds/images/2026-04-02",
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
    };

    s().mergeSnapshot(incoming, 5, null, null);

    expect(s().gui.dataset).toEqual({
      project_root: "/proj/beta",
      dataset_root: "/proj/beta/ds",
      subject: "bud",
      date: "2026-04-02",
      image_list: ["b1.jpg", "b2.jpg", "b3.jpg"],
      current_image_index: 2,
      images_dir: "/proj/beta/ds/images/2026-04-02",
      annotations_dir: "/proj/beta/ds/annotations/2026-04-02",
      predictions_dir: "/proj/beta/ds/predictions/m2/2026-04-02",
    });
  });
});

describe("booting with no local dataset yet", () => {
  it("does not carry the backend's pred_reference field into the typed store", () => {
    useStore.setState({
      gui: { ...OPENING_GUI, dataset: { ...OPENING_GUI.dataset, dataset_root: null } },
      wsVersion: 0,
    });

    // A real snapshot off the wire: state.py's GuiState always carries pred_reference, so the
    // browser's runtime object does too even though GuiState (the TS type) does not declare it.
    const incoming = {
      active_tab: "annotate",
      dataset: {
        project_root: "/proj/gamma",
        dataset_root: "/proj/gamma/ds",
        subject: "leaf",
        date: "2026-05-01",
        image_list: ["g1.jpg"],
        current_image_index: 0,
        images_dir: "/proj/gamma/ds/images/2026-05-01",
        annotations_dir: "/proj/gamma/ds/annotations/2026-05-01",
        predictions_dir: "/proj/gamma/ds/predictions/m1/2026-05-01",
      },
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: "leaf",
      review: {
        iou_threshold: 0.5,
        conf_threshold: 0.25,
        filter_type: "all",
        filter_class: "all",
        detection_idx: 0,
      },
      pred_reference: { type: "box", coords: [1, 2, 3, 4], subject: "leaf", confidence: 0.9 },
    } as unknown as GuiState;

    s().mergeSnapshot(incoming, 1, null, null);

    expect("pred_reference" in s().gui).toBe(false);
  });
});
