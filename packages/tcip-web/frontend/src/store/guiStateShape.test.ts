import { describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { GuiState } from "@/store/types";

// Captured before any test runs, so the opening shape is read from the store itself and not
// from whatever an earlier test left behind.
const OPENING_GUI = useStore.getState().gui;

const s = () => useStore.getState();

// tcip_web.state.GuiState's own field names, transcribed since this suite has no live backend
// to query; a field added there and not here fails the difference assertion below.
const BACKEND_GUI_STATE_FIELDS = ["active_tab", "dataset", "view", "mode", "active_subject"];

describe("the GUI state the browser opens with", () => {
  it("carries every field of the backend state, at the values the backend also starts from", () => {
    expect(OPENING_GUI).toEqual({
      active_tab: "annotate",
      dataset: {
        dataset_root: null,
        subject: null,
        date: null,
        image_list: [],
        current_image_index: 0,
        images_dir: null,
        annotations_dir: null,
        predictions_dir: null,
        label_paths: {},
        prediction_paths: {},
      },
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: null,
    });

    // The two field sets are one set: nothing the backend persists is dropped on the way in.
    const missing = BACKEND_GUI_STATE_FIELDS.filter((field) => !(field in OPENING_GUI));
    const extra = Object.keys(OPENING_GUI).filter(
      (field) => !BACKEND_GUI_STATE_FIELDS.includes(field),
    );
    expect(missing).toEqual([]);
    expect(extra).toEqual([]);
  });
});

describe("adopting a different dataset selection", () => {
  it("takes every field of the new selection, leaving none of the old one behind", () => {
    const project = { id: "a1b2c3d4e5f6", path: "/proj/alpha" };
    useStore.setState({
      gui: {
        active_tab: "results",
        dataset: {
          dataset_root: "/proj/alpha/ds",
          subject: "leaf",
          date: "2026-03-01",
          image_list: ["l1.jpg", "l2.jpg"],
          current_image_index: 1,
          images_dir: "/proj/alpha/ds/images/2026-03-01",
          annotations_dir: "/proj/alpha/ds/annotations/2026-03-01",
          predictions_dir: "/proj/alpha/ds/predictions/m1/2026-03-01",
          label_paths: { "l1.jpg": "/proj/alpha/ds/annotations/2026-03-01/l1.json" },
          prediction_paths: {},
        },
        view: { scale: 2, offset_x: 30, offset_y: 70 },
        mode: "polygon",
        active_subject: "leaf",
      },
      openProject: project,
      wsVersion: 4,
    });

    const incoming: GuiState = {
      active_tab: "annotate",
      dataset: {
        dataset_root: "/proj/alpha/ds2",
        subject: "bud",
        date: "2026-04-02",
        image_list: ["b1.jpg", "b2.jpg", "b3.jpg"],
        current_image_index: 2,
        images_dir: "/proj/alpha/ds2/images/2026-04-02",
        annotations_dir: "/proj/alpha/ds2/annotations/2026-04-02",
        predictions_dir: "/proj/alpha/ds2/predictions/m2/2026-04-02",
        label_paths: { "b1.jpg": "/proj/alpha/ds2/annotations/2026-04-02/b1.json" },
        prediction_paths: { "b1.jpg": "/proj/alpha/ds2/predictions/m2/2026-04-02/b1.json" },
      },
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: "bud",
    };

    s().mergeSnapshot(incoming, 5, project, null);

    expect(s().gui.dataset).toEqual(incoming.dataset);
  });
});
