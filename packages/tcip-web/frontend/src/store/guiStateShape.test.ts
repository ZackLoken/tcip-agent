import { describe, expect, it } from "vitest";

import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
import { useStore } from "@/store";
import type { GuiState } from "@/store/types";
import { PANEL_EVENTS } from "@/test/panelEvents";
import {
  DATASET_TREES,
  DS_DATE,
  GROUPED_SELECTION,
  selected,
  stateEnvelope,
  SUBJECT_SELECT_ANSWER,
  TEST_PROJECT,
} from "@/test/store";

// Captured before any test runs, so the opening shape is read from the store itself and not
// from whatever an earlier test left behind.
const OPENING_GUI = useStore.getState().gui;

const s = () => useStore.getState();

describe("the GUI state the browser opens with", () => {
  it("is the backend's default state, every field at the value the backend starts from", () => {
    expect(OPENING_GUI).toEqual(GUI_STATE_DEFAULTS);
  });
});

describe("adopting a different dataset selection", () => {
  it("takes every field of the new selection, leaving none of the old one behind", () => {
    const project = TEST_PROJECT;
    const [held, other] = DATASET_TREES.ds.buckets_by_date[DS_DATE];
    const { subject } = SUBJECT_SELECT_ANSWER.selection;
    const { subject: focused } = PANEL_EVENTS.annotate_focus.event;
    expect(focused).not.toBe(subject);
    useStore.setState({
      gui: {
        active_tab: "results",
        dataset: selected({ subject, count: 2, current_image_index: 1, bucket: held }),
        view: { scale: 2, offset_x: 30, offset_y: 70 },
        mode: "polygon",
        active_subject: subject,
      },
      openProject: project,
      wsVersion: 4,
    });

    const incoming: GuiState = {
      active_tab: "annotate",
      dataset: selected({ subject: focused, bucket: other }, GROUPED_SELECTION),
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      mode: "box",
      active_subject: focused,
    };

    s().mergeSnapshot(stateEnvelope({ state: incoming, version: 5, project }));

    expect(s().gui.dataset).toEqual(incoming.dataset);
  });
});
