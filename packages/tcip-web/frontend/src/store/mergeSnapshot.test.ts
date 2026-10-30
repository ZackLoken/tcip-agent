import { beforeEach, describe, expect, expectTypeOf, it } from "vitest";

import type { StateSnapshotMessage } from "@/api/types.generated";
import { recordLastTab } from "@/lib/datasetUiState";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";
import {
  DS_BUCKET,
  GROUPED_SELECTION,
  RAW_SELECTION,
  SELECTED_SUBJECT as SUBJECT,
  selected,
  type SelectionChange,
  stateEnvelope,
  TEST_EPOCH,
  TEST_PROJECT,
  testProject,
} from "@/test/store";

const s = () => useStore.getState();

const PROJECT = TEST_PROJECT;
const OTHER = testProject({ id: "ffffffffffff", path: "/other" });

function dataset(change: SelectionChange = {}): DatasetSelection {
  return selected({ subject: SUBJECT, count: 3, ...change });
}

/** Another capture: the produced grouped one. */
const OTHER_CAPTURE = selected({ subject: SUBJECT }, GROUPED_SELECTION);

function snapshot(over: Partial<GuiState> = {}): GuiState {
  return {
    active_tab: "annotate",
    dataset: dataset(),
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    mode: "box",
    active_subject: SUBJECT,
    ...over,
  };
}

describe("the envelope mergeSnapshot reads", () => {
  it("is the backend's envelope whole, every field it sends required", () => {
    expectTypeOf(s().mergeSnapshot).parameter(0).toEqualTypeOf<StateSnapshotMessage>();
    expectTypeOf<StateSnapshotMessage>().toEqualTypeOf<Required<StateSnapshotMessage>>();
  });
});

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
      wsEpoch: TEST_EPOCH,
      heldContributions: [],
    });
  });

  it("preserves client-owned fields on a same-dataset snapshot", () => {
    // Backend re-broadcasts with its stale active_tab / mode / index / subject.
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({
          active_tab: "annotate",
          mode: "box",
          active_subject: SUBJECT,
          dataset: dataset({ current_image_index: 0 }),
        }),
        version: 6,
        project: PROJECT,
      }),
    );
    expect(s().gui.active_tab).toBe("results");
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset?.current_image_index).toBe(2); // navigation kept
  });

  it("adopts the same project's snapshot carrying no selection", () => {
    s().mergeSnapshot(
      stateEnvelope({ state: snapshot({ dataset: null }), version: 7, project: PROJECT }),
    );
    expect(s().gui.dataset).toBeNull();
  });

  it("adopts the snapshot wholesale when it names another open project", () => {
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({ dataset: OTHER_CAPTURE }),
        version: 7,
        project: OTHER,
      }),
    );
    expect(s().openProject).toEqual(OTHER);
    expect(s().gui.dataset).toEqual(OTHER_CAPTURE);
  });

  it("drops the dataset when the backend has no project open", () => {
    s().mergeSnapshot(
      stateEnvelope({ state: snapshot({ dataset: null }), version: 7, project: null }),
    );
    expect(s().openProject).toBeNull();
    expect(s().gui.dataset).toBeNull();
  });

  it("adopts a new dataset identity and resets the index", () => {
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({ dataset: OTHER_CAPTURE }),
        version: 8,
        project: PROJECT,
      }),
    );
    expect(s().gui.dataset?.date).toBe(OTHER_CAPTURE.date);
    expect(s().gui.dataset?.current_image_index).toBe(0);
  });

  it("drops a stale (older-version) replay", () => {
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({ dataset: OTHER_CAPTURE }),
        version: 3, // < wsVersion 5
        project: PROJECT,
      }),
    );
    expect(s().gui.dataset?.date).toBe(RAW_SELECTION.date); // unchanged
  });

  it("leaves an image visit open when a stale snapshot of another project is rejected", () => {
    s().setUser("grower");
    s().startImageSessionTracking("a.jpg");

    s().mergeSnapshot(stateEnvelope({ state: snapshot(), version: 3, project: OTHER })); // 3 < wsVersion 5

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

    s().mergeSnapshot(stateEnvelope({ state: snapshot(), version: 6, project: OTHER }));

    expect(s().sessionTracking.currentImageName).toBeNull();
    expect(s().heldContributions).toEqual([]);
    expect(s().toasts.map((t) => t.message)).toEqual([
      `Not recorded, as their project was switched away from: ${images.join(", ")}`,
    ]);
  });

  it("applies a snapshot carrying the version already recorded", () => {
    // Only an older version is a stale replay; a re-broadcast at the current version is real state.
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({ dataset: OTHER_CAPTURE }),
        version: 5,
        project: PROJECT,
      }),
    );
    expect(s().gui.dataset?.date).toBe(OTHER_CAPTURE.date);
    expect(s().wsVersion).toBe(5);
  });

  it("accepts a lower version when the epoch moved (a restarted backend's own replay)", () => {
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({ dataset: OTHER_CAPTURE }),
        version: 1, // lower than wsVersion 5, and would ordinarily be dropped as stale
        project: PROJECT,
        epoch: "epoch-2",
      }),
    );
    expect(s().gui.dataset?.date).toBe(OTHER_CAPTURE.date); // accepted, not dropped
    expect(s().wsVersion).toBe(1);
    expect(s().wsEpoch).toBe("epoch-2");
  });

  it("adopts the backend's changed image list on a same-capture snapshot, at the local position", () => {
    const incoming = dataset({ bucket: DS_BUCKET, count: 4 });

    s().mergeSnapshot(
      stateEnvelope({ state: snapshot({ dataset: incoming }), version: 6, project: PROJECT }),
    );

    expect(s().gui.dataset?.images).toEqual(incoming.images);
    expect(s().gui.dataset?.bucket).toBe(DS_BUCKET);
    expect(s().gui.dataset?.current_image_index).toBe(2);
  });

  it("adopts the persisted state on boot, with the tab from the project's own record", () => {
    // Boot adopts backend mode/position; the tab is the client's per-project record, since the
    // backend's active_tab only moves on agent focus events.
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
    useStore.setState({
      gui: snapshot({ active_subject: null, dataset: null }),
      openProject: null,
      wsVersion: 0,
    });
    s().mergeSnapshot(
      stateEnvelope({
        state: snapshot({
          active_tab: "results",
          mode: "polygon",
          active_subject: "bush",
          dataset: dataset({ current_image_index: 2 }),
        }),
        version: 1,
        project: PROJECT,
      }),
    );
    expect(s().gui.active_tab).toBe("annotate"); // no record yet: first-open default
    expect(s().gui.mode).toBe("polygon");
    expect(s().gui.active_subject).toBe("bush");
    expect(s().gui.dataset?.current_image_index).toBe(2);
    expect(s().openProject).toEqual(PROJECT);
  });

  it("boot hydration lands on the project's recorded last-used tab when one exists", () => {
    recordLastTab(PROJECT.id, "training");
    useStore.setState({
      gui: snapshot({ active_subject: null, dataset: null }),
      openProject: null,
      wsVersion: 0,
    });
    s().mergeSnapshot(
      stateEnvelope({ state: snapshot({ active_tab: "results" }), version: 1, project: PROJECT }),
    );
    expect(s().gui.active_tab).toBe("training");
    localStorage.removeItem(`tcip.lasttab.${PROJECT.id}`);
  });
});

describe("applyRestoredDataset", () => {
  beforeEach(() => {
    useStore.setState({ gui: snapshot({ dataset: dataset() }), openProject: PROJECT });
  });

  it("adopts the open project in the same update as the dataset", () => {
    s().applyRestoredDataset(OTHER_CAPTURE, OTHER);
    expect(s().gui.dataset?.date).toBe(OTHER_CAPTURE.date);
    expect(s().openProject).toEqual(OTHER);
  });
});

describe("clearDataset", () => {
  it("clears the dataset selection", () => {
    useStore.setState({ gui: snapshot({ dataset: dataset() }) });
    s().clearDataset();
    expect(s().gui.dataset).toBeNull();
  });
});
