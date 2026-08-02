import { beforeEach, describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { ImageLabels } from "@/store/types";

const s = () => useStore.getState();

function loadPolygon(rings: [number, number][][]): void {
  const labels: ImageLabels = {
    image_path: "x",
    img_width: 100,
    img_height: 100,
    boxes: [],
    polygons: [{ rings, subject: "catkin", attributes: {} }],
    points: [],
    imageAnnotations: [],
  };
  s().loadLabelsIntoCanvas(labels);
}

const RING_A: [number, number][] = [
  [0, 0],
  [10, 0],
  [10, 10],
];
const RING_B: [number, number][] = [
  [40, 40],
  [60, 40],
  [60, 60],
];

function loadOnePolygon(): void {
  loadPolygon([RING_A]);
}

describe("canvas store", () => {
  beforeEach(() => {
    s().clearCanvas();
  });

  it("loadLabelsIntoCanvas resets dirty and undo/redo stacks", () => {
    loadOnePolygon();
    expect(s().canvas.polygons).toHaveLength(1);
    expect(s().canvas.dirty).toBe(false);
    expect(s().canvas.undoStack).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(0);
  });

  it("dragVertex moves a vertex without pushing an undo snapshot", () => {
    loadOnePolygon();
    const before = s().canvas.undoStack.length;
    s().dragVertex(0, 0, 1, [42, 7]);
    expect(s().canvas.polygons[0].rings[0][1]).toEqual([42, 7]);
    // The whole point of dragVertex: a live drag must not flood the undo stack.
    expect(s().canvas.undoStack.length).toBe(before);
    expect(s().canvas.dirty).toBe(true);
  });

  it("dragVertex edits the addressed ring and leaves the shape's other rings alone", () => {
    // A vertex belongs to one contour of one annotation; a multi-ring shape must stay whole through
    // an edit to one of its parts.
    loadPolygon([RING_A, RING_B]);
    s().dragVertex(0, 1, 2, [99, 98]);
    expect(s().canvas.polygons[0].rings[1][2]).toEqual([99, 98]);
    expect(s().canvas.polygons[0].rings[0]).toEqual(RING_A);
    expect(s().canvas.polygons[0].rings).toHaveLength(2);
  });

  it("addBox pushes an undo snapshot that undo restores", () => {
    expect(s().canvas.boxes).toHaveLength(0);
    s().addBox({ x1: 0, y1: 0, x2: 5, y2: 5, subject: "catkin", attributes: {} });
    expect(s().canvas.boxes).toHaveLength(1);
    expect(s().canvas.undoStack).toHaveLength(1);
    s().undo();
    expect(s().canvas.boxes).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(1);
  });

  it("dragBox moves a box without pushing an undo snapshot", () => {
    s().addBox({ x1: 0, y1: 0, x2: 5, y2: 5, subject: "catkin", attributes: {} });
    const before = s().canvas.undoStack.length;
    s().dragBox(0, { x1: 2, y1: 3, x2: 9, y2: 11, subject: "catkin", attributes: {} });
    expect(s().canvas.boxes[0]).toEqual({
      x1: 2,
      y1: 3,
      x2: 9,
      y2: 11,
      subject: "catkin",
      attributes: {},
    });
    // Like dragVertex: a live resize/move must not flood the undo stack.
    expect(s().canvas.undoStack.length).toBe(before);
    expect(s().canvas.dirty).toBe(true);
  });

  it("pushUndo caps the undo stack at 30 entries", () => {
    for (let i = 0; i < 40; i++) s().pushUndo();
    expect(s().canvas.undoStack).toHaveLength(30);
  });

  it("commitCurrentPolygon refuses with no subject, and tags the subject when one is set", () => {
    // Authoring is guarded: a subjectless shape has nowhere to attach (the backend save rejects it).
    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: null } }));
    s().setCurrentPolygon([
      [0, 0],
      [10, 0],
      [10, 10],
    ]);
    expect(s().commitCurrentPolygon()).toBe(false);
    expect(s().canvas.polygons).toHaveLength(0);

    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: "catkin" } }));
    s().setCurrentPolygon([
      [0, 0],
      [10, 0],
      [10, 10],
    ]);
    expect(s().commitCurrentPolygon()).toBe(true);
    expect(s().canvas.polygons[0].subject).toBe("catkin");
    // A hand-drawn shape is exactly one contour: the drawing tool never authors a second ring.
    expect(s().canvas.polygons[0].rings).toHaveLength(1);
  });
});

describe("canvas store points", () => {
  beforeEach(() => {
    s().clearCanvas();
  });

  const pt = (x: number, y: number, subject = "tip") => ({ x, y, subject, attributes: {} });

  it("addPoint pushes an undo snapshot that undo restores", () => {
    s().addPoint(pt(10, 20));
    expect(s().canvas.points).toEqual([pt(10, 20)]);
    expect(s().canvas.dirty).toBe(true);
    expect(s().canvas.undoStack).toHaveLength(1);
    s().undo();
    expect(s().canvas.points).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(1);
    s().redo();
    expect(s().canvas.points).toEqual([pt(10, 20)]);
  });

  it("dragPoint repositions without pushing an undo snapshot", () => {
    s().addPoint(pt(10, 20));
    const before = s().canvas.undoStack.length;
    s().dragPoint(0, 33, 44);
    expect(s().canvas.points[0]).toMatchObject({ x: 33, y: 44, subject: "tip" });
    // Like dragBox/dragVertex: a live drag must not flood the 30-entry undo stack.
    expect(s().canvas.undoStack.length).toBe(before);
    expect(s().canvas.dirty).toBe(true);
  });

  it("dragPoint on a missing index is a no-op (a stale drag can outlive its point)", () => {
    s().addPoint(pt(10, 20));
    s().dragPoint(5, 1, 1);
    expect(s().canvas.points).toEqual([pt(10, 20)]);
  });

  it("deletePoint removes it and keeps the selection pointing at the same annotation", () => {
    s().addPoint(pt(1, 1, "a"));
    s().addPoint(pt(2, 2, "b"));
    s().addPoint(pt(3, 3, "c"));
    s().selectPoint(2);
    s().deletePoint(0); // an earlier point goes: the selection shifts down with it
    expect(s().canvas.points.map((p) => p.subject)).toEqual(["b", "c"]);
    expect(s().canvas.selectedPointIdx).toBe(1);
    s().deletePoint(1); // the selected point itself goes
    expect(s().canvas.selectedPointIdx).toBeNull();
  });

  it("updatePoint edits attributes in place (undoable), leaving the position alone", () => {
    s().addPoint(pt(10, 20));
    s().updatePoint(0, { ...s().canvas.points[0], attributes: { stage: "open" } });
    expect(s().canvas.points[0]).toMatchObject({ x: 10, y: 20, attributes: { stage: "open" } });
    s().undo();
    expect(s().canvas.points[0].attributes).toEqual({});
  });

  it("loadLabelsIntoCanvas adopts loaded points and clears any point selection", () => {
    s().addPoint(pt(1, 1));
    s().selectPoint(0);
    s().loadLabelsIntoCanvas({
      image_path: "x",
      img_width: 100,
      img_height: 100,
      boxes: [],
      polygons: [],
      points: [pt(5, 6, "tip")],
      imageAnnotations: [],
    });
    expect(s().canvas.points).toEqual([pt(5, 6, "tip")]);
    expect(s().canvas.selectedPointIdx).toBeNull();
    expect(s().canvas.dirty).toBe(false);
  });
});

describe("toasts", () => {
  beforeEach(() => {
    useStore.setState({ toasts: [] });
  });

  it("pushToast adds a toast and dismissToast removes it", () => {
    s().pushToast("hi", "info");
    expect(s().toasts).toHaveLength(1);
    expect(s().toasts[0].message).toBe("hi");
    expect(s().toasts[0].level).toBe("info");
    s().dismissToast(s().toasts[0].id);
    expect(s().toasts).toHaveLength(0);
  });

  it("caps the toast stack at 4 (drops the oldest)", () => {
    for (let i = 0; i < 6; i++) s().pushToast(`t${i}`);
    expect(s().toasts).toHaveLength(4);
    expect(s().toasts[0].message).toBe("t2");
    expect(s().toasts[0].level).toBe("error"); // default level
  });
});

describe("agent activity", () => {
  it("pushAgentActivity records the event and increments seq", () => {
    s().pushAgentActivity("annotate", "labels_written", { stem: "IMG_1" });
    const first = s().agentActivity;
    expect(first?.panel).toBe("annotate");
    expect(first?.eventType).toBe("labels_written");
    expect(first?.data.stem).toBe("IMG_1");

    s().pushAgentActivity("annotate", "labels_written", { stem: "IMG_2" });
    expect(s().agentActivity?.seq).toBe((first?.seq ?? 0) + 1);
    expect(s().agentActivity?.data.stem).toBe("IMG_2");
  });
});
