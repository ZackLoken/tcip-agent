import { beforeEach, describe, expect, it } from "vitest";

import { useStore } from "@/store";
import type { ImageLabels } from "@/store/types";

const s = () => useStore.getState();

function loadOnePolygon(): void {
  const points: [number, number][] = [
    [0, 0],
    [10, 0],
    [10, 10],
  ];
  const labels: ImageLabels = {
    image_path: "x",
    img_width: 100,
    img_height: 100,
    boxes: [],
    polygons: [{ points, subject: "catkin", attributes: {} }],
    imageAnnotations: [],
  };
  s().loadLabelsIntoCanvas(labels);
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

  it("dragVertex moves a vertex WITHOUT pushing an undo snapshot", () => {
    loadOnePolygon();
    const before = s().canvas.undoStack.length;
    s().dragVertex(0, 1, [42, 7]);
    expect(s().canvas.polygons[0].points[1]).toEqual([42, 7]);
    // The whole point of dragVertex: a live drag must not flood the undo stack.
    expect(s().canvas.undoStack.length).toBe(before);
    expect(s().canvas.dirty).toBe(true);
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

  it("dragBox moves a box WITHOUT pushing an undo snapshot", () => {
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
