import { beforeEach, describe, expect, it } from "vitest";

import { serializeCanvas, type CanvasLabels } from "@/lib/labelSerde";
import { useStore } from "@/store";
import type { Box, ImageLabels } from "@/store/types";

const canvasToAnnotations = (labels: CanvasLabels) => serializeCanvas(labels).annotations;

const s = () => useStore.getState();

/** A save's answer holding `boxes` and nothing else. */
const answered = (boxes: Box[]): ImageLabels => ({
  image_path: "x",
  img_width: 100,
  img_height: 100,
  boxes,
  polygons: [],
  points: [],
  imageAnnotations: [],
  completion: {},
  flags: [],
});

function loadPolygon(rings: [number, number][][]): void {
  const labels: ImageLabels = {
    image_path: "x",
    img_width: 100,
    img_height: 100,
    boxes: [],
    polygons: [{ rings, subject: "subject_a", attributes: {} }],
    points: [],
    imageAnnotations: [],
    completion: {},
    flags: [],
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
    s().addBox({ x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} });
    expect(s().canvas.boxes).toHaveLength(1);
    expect(s().canvas.undoStack).toHaveLength(1);
    s().undo();
    expect(s().canvas.boxes).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(1);
  });

  it("dragBox moves a box without pushing an undo snapshot", () => {
    s().addBox({ x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} });
    const before = s().canvas.undoStack.length;
    s().dragBox(0, { x1: 2, y1: 3, x2: 9, y2: 11, subject: "subject_a", attributes: {} });
    expect(s().canvas.boxes[0]).toEqual({
      x1: 2,
      y1: 3,
      x2: 9,
      y2: 11,
      subject: "subject_a",
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

    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: "subject_a" } }));
    s().setCurrentPolygon([
      [0, 0],
      [10, 0],
      [10, 10],
    ]);
    expect(s().commitCurrentPolygon()).toBe(true);
    expect(s().canvas.polygons[0].subject).toBe("subject_a");
    // A hand-drawn shape is exactly one contour: the drawing tool never authors a second ring.
    expect(s().canvas.polygons[0].rings).toHaveLength(1);
  });

  it("commitCurrentPolygon clamps each vertex to its own axis on a non-square image", () => {
    s().loadLabelsIntoCanvas({
      image_path: "wide.jpg",
      img_width: 120,
      img_height: 80,
      boxes: [],
      polygons: [],
      points: [],
      imageAnnotations: [],
      completion: {},
      flags: [],
    });
    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: "subject_a" } }));
    s().setCurrentPolygon([
      [-10, 40],
      [200, 95],
      [30, 90],
      [50, 20],
    ]);

    expect(s().commitCurrentPolygon()).toBe(true);
    expect(s().canvas.polygons).toHaveLength(1);
    expect(s().canvas.polygons[0].rings[0]).toEqual([
      [0, 40],
      [120, 80],
      [30, 80],
      [50, 20],
    ]);
  });
});

describe("splitPolygon", () => {
  beforeEach(() => {
    s().clearCanvas();
  });

  const PIECE_A: [number, number][] = [
    [0, 0],
    [5, 0],
    [5, 5],
  ];
  const PIECE_B: [number, number][] = [
    [5, 0],
    [10, 0],
    [10, 10],
  ];

  it("replaces one polygon with two at the same index, keeping the parent's values", () => {
    s().loadLabelsIntoCanvas({
      image_path: "x",
      img_width: 100,
      img_height: 100,
      boxes: [],
      polygons: [
        {
          rings: [RING_A],
          subject: "subject_a",
          attributes: { health: "good" },
          authorship: "tool_accepted",
        },
      ],
      points: [],
      imageAnnotations: [],
      completion: {},
      flags: [],
    });
    // A second polygon after it, so the split's own effect on later indices is checked too.
    s().addPolygon({ rings: [RING_B], subject: "subject_a", attributes: {} });
    useStore.setState((st) => ({ annotateUi: { ...st.annotateUi, hoveredPolygonIdx: 1 } }));

    s().splitPolygon(0, [PIECE_A, PIECE_B]);

    expect(s().canvas.polygons).toHaveLength(3);
    expect(s().canvas.polygons[0].rings).toEqual([PIECE_A]);
    expect(s().canvas.polygons[1].rings).toEqual([PIECE_B]);
    expect(s().canvas.polygons[2].rings).toEqual([RING_B]); // the untouched polygon, shifted by one
    for (const piece of [s().canvas.polygons[0], s().canvas.polygons[1]]) {
      expect(piece.subject).toBe("subject_a");
      expect(piece.attributes).toEqual({ health: "good" });
      // Each piece is new content the save door stamps; the parent's authorship is not its own.
      expect(piece.authorship).toBeUndefined();
    }
    expect(s().canvas.focus).toEqual({ kind: "polygon", index: 0 }); // the first piece
    expect(s().annotateUi.hoveredPolygonIdx).toBeNull();
    expect(s().canvas.dirty).toBe(true);

    const payload = canvasToAnnotations({
      boxes: s().canvas.boxes,
      polygons: s().canvas.polygons.slice(0, 2),
      points: [],
      imageAnnotations: [],
    });
    expect(payload.map((p) => p.points)).toEqual([PIECE_A, PIECE_B]);
  });

  it("one undo restores the parent and its selection", () => {
    loadOnePolygon();
    s().addPoint({ x: 1, y: 1, subject: "tip", attributes: {} });
    s().setFocus({ kind: "point", index: 0 });
    s().splitPolygon(0, [PIECE_A, PIECE_B]);
    expect(s().canvas.polygons).toHaveLength(2);
    expect(s().canvas.focus).toEqual({ kind: "polygon", index: 0 });

    s().undo();
    expect(s().canvas.polygons).toHaveLength(1);
    expect(s().canvas.polygons[0].rings).toEqual([RING_A]);
    expect(s().canvas.focus).toEqual({ kind: "point", index: 0 });
  });

  it("undo restores an annotation focus and never a proposal's, whose bucket may have changed", () => {
    const point = { x: 1, y: 1, subject: "tip", attributes: {} };
    s().addPoint(point);
    s().setFocus({ kind: "point", index: 0 });
    s().addPoint(point);
    s().setFocus({ kind: "proposal", index: 4 });
    s().addPoint(point);

    s().undo();
    expect(s().canvas.focus).toBeNull();
    s().undo();
    expect(s().canvas.focus).toEqual({ kind: "point", index: 0 });
    s().redo();
    expect(s().canvas.focus).toBeNull();
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
    s().setFocus({ kind: "point", index: 2 });
    s().deletePoint(0); // an earlier point goes: the selection shifts down with it
    expect(s().canvas.points.map((p) => p.subject)).toEqual(["b", "c"]);
    expect(s().canvas.focus).toEqual({ kind: "point", index: 1 });
    s().deletePoint(1); // the selected point itself goes
    expect(s().canvas.focus).toBeNull();
  });

  it("deleting a shape of another kind leaves the focus alone", () => {
    s().addPoint(pt(1, 1, "a"));
    s().addBox({ x1: 0, y1: 0, x2: 5, y2: 5, subject: "a", attributes: {} });
    s().setFocus({ kind: "box", index: 0 });
    s().deletePoint(0);
    expect(s().canvas.focus).toEqual({ kind: "box", index: 0 });
    s().deleteBox(0);
    expect(s().canvas.focus).toBeNull();
  });

  it("setMode to another tool drops the focus; the same tool keeps it", () => {
    s().setMode("point");
    s().setFocus({ kind: "point", index: 0 });
    s().setMode("point");
    expect(s().canvas.focus).toEqual({ kind: "point", index: 0 });
    s().setMode("box");
    expect(s().canvas.focus).toBeNull();
  });

  it("updatePoint edits attributes in place (undoable), leaving the position alone", () => {
    s().addPoint(pt(10, 20));
    s().updatePoint(0, { ...s().canvas.points[0], attributes: { stage: "open" } });
    expect(s().canvas.points[0]).toMatchObject({ x: 10, y: 20, attributes: { stage: "open" } });
    s().undo();
    expect(s().canvas.points[0].attributes).toEqual({});
  });

  it("loadLabelsIntoCanvas adopts loaded points and leaves the focus to the context rule", () => {
    s().addPoint(pt(1, 1));
    s().setFocus({ kind: "point", index: 0 });
    s().loadLabelsIntoCanvas({
      image_path: "x",
      img_width: 100,
      img_height: 100,
      boxes: [],
      polygons: [],
      points: [pt(5, 6, "tip")],
      imageAnnotations: [],
      completion: {},
      flags: [],
    });
    expect(s().canvas.points).toEqual([pt(5, 6, "tip")]);
    expect(s().canvas.focus).toEqual({ kind: "point", index: 0 });
    expect(s().canvas.dirty).toBe(false);
  });
});

describe("focus follows the image and the bucket", () => {
  const select = (over: Partial<ReturnType<typeof s>["gui"]["dataset"]>) =>
    useStore.setState((st) => ({
      gui: { ...st.gui, dataset: { ...st.gui.dataset, ...over } },
    }));

  beforeEach(() => {
    s().clearCanvas();
    select({
      images_dir: "/d/a",
      image_list: ["1.jpg", "2.jpg"],
      current_image_index: 0,
      bucket: "b1",
    });
  });

  it("clears when the image changes, by index or by directory", () => {
    s().setFocus({ kind: "proposal", index: 2 });
    select({ current_image_index: 1 });
    expect(s().canvas.focus).toBeNull();

    s().setFocus({ kind: "box", index: 0 });
    select({ images_dir: "/d/b" });
    expect(s().canvas.focus).toBeNull();
  });

  it("clears when the bucket changes, so an index never names another bucket's proposal", () => {
    s().setFocus({ kind: "proposal", index: 2 });
    select({ bucket: "b2" });
    expect(s().canvas.focus).toBeNull();
  });

  it("survives a refresh of the same image and an unrelated change to the selection", () => {
    s().setFocus({ kind: "proposal", index: 2 });
    select({ subject: "other" });
    s().loadLabelsIntoCanvas({
      image_path: "/d/a/1.jpg",
      img_width: 10,
      img_height: 10,
      boxes: [],
      polygons: [],
      points: [],
      imageAnnotations: [],
      completion: {},
      flags: [],
    });
    expect(s().canvas.focus).toEqual({ kind: "proposal", index: 2 });
  });

  it("holds a focus set after the change that selected its image, as the agent's is", () => {
    select({ current_image_index: 1 });
    s().setFocus({ kind: "proposal", index: 4 });
    expect(s().canvas.focus).toEqual({ kind: "proposal", index: 4 });
  });

  describe("history restoration after the image changed", () => {
    const loaded = (name: string, boxes: Box[]): ImageLabels => ({
      image_path: `/d/a/${name}`,
      img_width: 100,
      img_height: 100,
      boxes,
      polygons: [],
      points: [],
      imageAnnotations: [],
      completion: {},
      flags: [],
    });
    const first = { x1: 10, y1: 10, x2: 50, y2: 50, subject: "subject_a", attributes: {} };

    function focusedBoxWithSnapshot() {
      s().loadLabelsIntoCanvas(loaded("1.jpg", [first]));
      s().setFocus({ kind: "box", index: 0 });
      s().pushUndo();
      s().dragBox(0, { ...first, x2: 11, y2: 11 });
    }

    it("a rollback leaves the focus the image change cleared, so the next image does not inherit it", () => {
      focusedBoxWithSnapshot();
      select({ current_image_index: 1 });
      s().rollbackLast();
      expect(s().canvas.boxes[0]).toEqual(first);
      s().loadLabelsIntoCanvas(loaded("2.jpg", [{ ...first, subject: "other" }]));
      expect(s().canvas.focus).toBeNull();
    });

    it("an undo does the same", () => {
      focusedBoxWithSnapshot();
      select({ current_image_index: 1 });
      s().undo();
      expect(s().canvas.focus).toBeNull();
    });

    it("a rollback within the same image restores the focus the snapshot held", () => {
      focusedBoxWithSnapshot();
      s().setFocus(null);
      s().rollbackLast();
      expect(s().canvas.focus).toEqual({ kind: "box", index: 0 });
    });
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

describe("content-based dirty tracking", () => {
  beforeEach(() => {
    s().clearCanvas();
  });

  const box = { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} };

  it("drawing a shape and deleting it again leaves the canvas clean", () => {
    loadOnePolygon();
    s().addBox(box);
    expect(s().canvas.dirty).toBe(true);
    s().deleteBox(0);
    expect(s().canvas.dirty).toBe(false);
  });

  it("undoing back to the loaded content leaves the canvas clean", () => {
    loadOnePolygon();
    s().addBox(box);
    s().undo();
    expect(s().canvas.dirty).toBe(false);
    s().redo();
    expect(s().canvas.dirty).toBe(true);
  });

  it("a loaded document re-baselines: deleting a loaded shape then undoing it is clean again", () => {
    s().addBox(box);
    s().loadLabelsIntoCanvas(answered([{ ...box, index: 0, authorship: "person" }]));
    expect(s().canvas.undoStack).toHaveLength(0);
    s().deleteBox(0);
    expect(s().canvas.dirty).toBe(true);
    s().undo();
    expect(s().canvas.dirty).toBe(false);
  });

  describe("while a save is in flight", () => {
    const withDocumentOneBox = () => {
      s().loadLabelsIntoCanvas(answered([{ ...box, index: 0, authorship: "person" }]));
      return s().holdForSave("/d/a/1.jpg");
    };

    it("refuses every content edit and leaves the canvas as it was", () => {
      withDocumentOneBox();
      const before = s().canvas;
      s().addBox({ ...box, x2: 9 });
      s().deleteBox(0);
      s().updateBox(0, { ...box, x2: 8 });
      s().undo();
      s().redo();
      expect(s().canvas.boxes).toBe(before.boxes);
      expect(s().canvas.undoStack).toHaveLength(0);
      expect(s().canvas.dirty).toBe(false);
    });

    const heldWithHistory = () => {
      s().loadLabelsIntoCanvas(answered([]));
      s().addBox(box);
      s().addBox({ ...box, x2: 9 });
      s().undo();
      s().holdForSave("/d/a/1.jpg");
      return s().canvas;
    };

    it("refuses undo over a history the canvas already has", () => {
      const held = heldWithHistory();
      s().undo();
      expect(s().canvas.boxes).toBe(held.boxes);
      expect(s().canvas.undoStack).toBe(held.undoStack);
    });

    it("refuses redo over a history the canvas already has", () => {
      const held = heldWithHistory();
      s().redo();
      expect(s().canvas.boxes).toBe(held.boxes);
      expect(s().canvas.redoStack).toBe(held.redoStack);
    });

    it("accepts edits again once the answer is adopted, which also ends the save", () => {
      withDocumentOneBox();
      s().loadLabelsIntoCanvas(answered([{ ...box, index: 3, authorship: "person" }]));
      expect(s().canvas.saving).toBeNull();
      s().addBox({ ...box, x2: 9 });
      expect(s().canvas.boxes).toHaveLength(2);
    });

    it("is released by the save that holds it and by no other", () => {
      const first = withDocumentOneBox();
      s().loadLabelsIntoCanvas(answered([])); // a load of another image drops the hold
      const second = s().holdForSave("/d/a/2.jpg");
      expect(second.id).not.toBe(first.id);
      s().releaseSave(first);
      expect(s().canvas.saving).toEqual(second);
      s().releaseSave({ ...second, image: "/d/a/1.jpg" });
      expect(s().canvas.saving).toEqual(second);
      s().releaseSave(second);
      expect(s().canvas.saving).toBeNull();
    });
  });

  it("a genuine change stays dirty", () => {
    loadOnePolygon();
    s().addBox(box);
    s().deleteBox(0);
    s().addBox({ ...box, x2: 6 });
    expect(s().canvas.dirty).toBe(true);
  });

  it("recomputeDirty settles a drag that returned a vertex to its origin", () => {
    loadOnePolygon();
    const [ox, oy] = RING_A[1];
    s().dragVertex(0, 0, 1, [42, 7]);
    expect(s().canvas.dirty).toBe(true); // drags flag without content compare; release settles it
    s().dragVertex(0, 0, 1, [ox, oy]);
    s().recomputeDirty();
    expect(s().canvas.dirty).toBe(false);
  });
});

describe("the hold on the canvas covers every content edit", () => {
  const polygon = { rings: [RING_A], subject: "subject_a", attributes: {} };
  const box = { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} };
  const rating = { subject: "subject_a", attributes: {}, iscrowd: false };

  /** A canvas on which every action changes state: content of each kind, a history to undo and
   *  to redo, a polygon being drawn, and a subject to commit it under. */
  function arrange() {
    s().clearCanvas();
    s().loadLabelsIntoCanvas({
      image_path: "x",
      img_width: 100,
      img_height: 100,
      boxes: [box],
      polygons: [polygon],
      points: [{ x: 1, y: 1, subject: "subject_a", attributes: {} }],
      imageAnnotations: [rating],
      completion: {},
      flags: [],
    });
    s().addBox({ ...box, x2: 9 });
    s().addBox({ ...box, x2: 8 });
    s().undo();
    s().setCurrentPolygon(RING_B);
    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: "subject_a" } }));
  }

  const ACTIONS: [string, () => unknown][] = [
    ["pushUndo", () => s().pushUndo()],
    ["undo", () => s().undo()],
    ["rollbackLast", () => s().rollbackLast()],
    ["redo", () => s().redo()],
    ["addBox", () => s().addBox({ ...box, x2: 7 })],
    ["updateBox", () => s().updateBox(0, { ...box, x2: 6 })],
    ["dragBox", () => s().dragBox(0, { ...box, x2: 6 })],
    ["deleteBox", () => s().deleteBox(0)],
    ["addPolygon", () => s().addPolygon({ ...polygon, rings: [RING_B] })],
    ["updatePolygon", () => s().updatePolygon(0, { ...polygon, rings: [RING_B] })],
    ["dragVertex", () => s().dragVertex(0, 0, 0, [3, 3])],
    ["deletePolygon", () => s().deletePolygon(0)],
    ["splitPolygon", () => s().splitPolygon(0, [RING_A, RING_B])],
    ["addPoint", () => s().addPoint({ x: 2, y: 2, subject: "subject_a", attributes: {} })],
    ["updatePoint", () => s().updatePoint(0, { x: 3, y: 3, subject: "subject_a", attributes: {} })],
    ["dragPoint", () => s().dragPoint(0, 9, 9)],
    ["deletePoint", () => s().deletePoint(0)],
    ["setCurrentPolygon", () => s().setCurrentPolygon(RING_A)],
    ["commitCurrentPolygon", () => s().commitCurrentPolygon()],
    ["addImageAnnotation", () => s().addImageAnnotation("subject_a")],
    ["updateImageAnnotation", () => s().updateImageAnnotation(0, { ...rating, iscrowd: true })],
    ["deleteImageAnnotation", () => s().deleteImageAnnotation(0)],
  ];

  it("lists every action it checks, so a new one is a decision", () => {
    expect(ACTIONS).toHaveLength(22);
  });

  it.each(ACTIONS)("%s changes the canvas unheld and does nothing held", (_name, run) => {
    arrange();
    const unheld = s().canvas;
    run();
    expect(s().canvas).not.toBe(unheld);

    arrange();
    s().holdForSave("x");
    const held = s().canvas;
    run();
    expect(s().canvas).toBe(held);
  });
});

describe("agent activity", () => {
  it("pushAgentActivity records the event, its client, and increments seq", () => {
    s().pushAgentActivity("annotate", "labels_written", { stem: "IMG_1" }, "claude-code 2.1.238");
    const first = s().agentActivity;
    expect(first?.panel).toBe("annotate");
    expect(first?.eventType).toBe("labels_written");
    expect(first?.data.stem).toBe("IMG_1");
    expect(first?.client).toBe("claude-code 2.1.238");

    s().pushAgentActivity("annotate", "labels_written", { stem: "IMG_2" }, null);
    expect(s().agentActivity?.seq).toBe((first?.seq ?? 0) + 1);
    expect(s().agentActivity?.data.stem).toBe("IMG_2");
    expect(s().agentActivity?.client).toBeNull();
  });
});
