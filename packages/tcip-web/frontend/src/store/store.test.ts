import { beforeEach, describe, expect, it } from "vitest";

import { serializeCanvas } from "@/lib/labelSerde";
import { useStore } from "@/store";
import { CONTENT_EDITS } from "@/store/slices/canvas";
import type { Box, CanvasContent, ImageLabels } from "@/store/types";

const canvasToAnnotations = (labels: CanvasContent) => serializeCanvas(labels).annotations;

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

  it("add pushes an undo snapshot that undo restores", () => {
    expect(s().canvas.boxes).toHaveLength(0);
    s().add("boxes", { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} });
    expect(s().canvas.boxes).toHaveLength(1);
    expect(s().canvas.undoStack).toHaveLength(1);
    s().undo();
    expect(s().canvas.boxes).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(1);
  });

  it("drag moves a box without pushing an undo snapshot", () => {
    s().add("boxes", { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} });
    const before = s().canvas.undoStack.length;
    s().drag("boxes", 0, { x1: 2, y1: 3, x2: 9, y2: 11, subject: "subject_a", attributes: {} });
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
    s().add("polygons", { rings: [RING_B], subject: "subject_a", attributes: {} });
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
    s().add("points", { x: 1, y: 1, subject: "tip", attributes: {} });
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
    s().add("points", point);
    s().setFocus({ kind: "point", index: 0 });
    s().add("points", point);
    s().setFocus({ kind: "proposal", index: 4 });
    s().add("points", point);

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

  it("adding a point pushes an undo snapshot that undo restores", () => {
    s().add("points", pt(10, 20));
    expect(s().canvas.points).toEqual([pt(10, 20)]);
    expect(s().canvas.dirty).toBe(true);
    expect(s().canvas.undoStack).toHaveLength(1);
    s().undo();
    expect(s().canvas.points).toHaveLength(0);
    expect(s().canvas.redoStack).toHaveLength(1);
    s().redo();
    expect(s().canvas.points).toEqual([pt(10, 20)]);
  });

  it("dragging a point repositions it without pushing an undo snapshot", () => {
    s().add("points", pt(10, 20));
    const before = s().canvas.undoStack.length;
    s().drag("points", 0, pt(33, 44));
    expect(s().canvas.points[0]).toMatchObject({ x: 33, y: 44, subject: "tip" });
    // Like a box or vertex drag: a live drag must not flood the 30-entry undo stack.
    expect(s().canvas.undoStack.length).toBe(before);
    expect(s().canvas.dirty).toBe(true);
  });

  it("a drag on a missing index is a no-op (a stale drag can outlive its point)", () => {
    s().add("points", pt(10, 20));
    s().drag("points", 5, pt(1, 1));
    expect(s().canvas.points).toEqual([pt(10, 20)]);
  });

  it("removing a point keeps the selection pointing at the same annotation", () => {
    s().add("points", pt(1, 1, "a"));
    s().add("points", pt(2, 2, "b"));
    s().add("points", pt(3, 3, "c"));
    s().setFocus({ kind: "point", index: 2 });
    s().remove("points", 0); // an earlier point goes: the selection shifts down with it
    expect(s().canvas.points.map((p) => p.subject)).toEqual(["b", "c"]);
    expect(s().canvas.focus).toEqual({ kind: "point", index: 1 });
    s().remove("points", 1); // the selected point itself goes
    expect(s().canvas.focus).toBeNull();
  });

  it("removing an item of another array leaves the focus alone", () => {
    s().add("points", pt(1, 1, "a"));
    s().add("boxes", { x1: 0, y1: 0, x2: 5, y2: 5, subject: "a", attributes: {} });
    s().add("imageAnnotations", { subject: "a", attributes: {}, iscrowd: false });
    s().setFocus({ kind: "box", index: 0 });
    s().remove("points", 0);
    s().remove("imageAnnotations", 0);
    expect(s().canvas.focus).toEqual({ kind: "box", index: 0 });
    s().remove("boxes", 0);
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

  it("update edits a point's attributes in place (undoable), leaving the position alone", () => {
    s().add("points", pt(10, 20));
    s().update("points", 0, { ...s().canvas.points[0], attributes: { stage: "open" } });
    expect(s().canvas.points[0]).toMatchObject({ x: 10, y: 20, attributes: { stage: "open" } });
    s().undo();
    expect(s().canvas.points[0].attributes).toEqual({});
  });

  it("loadLabelsIntoCanvas adopts loaded points and leaves the focus to the context rule", () => {
    s().add("points", pt(1, 1));
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
      s().drag("boxes", 0, { ...first, x2: 11, y2: 11 });
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
    s().add("boxes", box);
    expect(s().canvas.dirty).toBe(true);
    s().remove("boxes", 0);
    expect(s().canvas.dirty).toBe(false);
  });

  it("undoing back to the loaded content leaves the canvas clean", () => {
    loadOnePolygon();
    s().add("boxes", box);
    s().undo();
    expect(s().canvas.dirty).toBe(false);
    s().redo();
    expect(s().canvas.dirty).toBe(true);
  });

  it("a loaded document re-baselines: deleting a loaded shape then undoing it is clean again", () => {
    s().add("boxes", box);
    s().loadLabelsIntoCanvas(answered([{ ...box, index: 0, authorship: "person" }]));
    expect(s().canvas.undoStack).toHaveLength(0);
    s().remove("boxes", 0);
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
      s().add("boxes", { ...box, x2: 9 });
      s().remove("boxes", 0);
      s().update("boxes", 0, { ...box, x2: 8 });
      s().undo();
      s().redo();
      expect(s().canvas.boxes).toBe(before.boxes);
      expect(s().canvas.undoStack).toHaveLength(0);
      expect(s().canvas.dirty).toBe(false);
    });

    const heldWithHistory = () => {
      s().loadLabelsIntoCanvas(answered([]));
      s().add("boxes", box);
      s().add("boxes", { ...box, x2: 9 });
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
      s().add("boxes", { ...box, x2: 9 });
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
    s().add("boxes", box);
    s().remove("boxes", 0);
    s().add("boxes", { ...box, x2: 6 });
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
    s().add("boxes", { ...box, x2: 9 });
    s().add("boxes", { ...box, x2: 8 });
    s().undo();
    s().setCurrentPolygon(RING_B);
    useStore.setState((st) => ({ gui: { ...st.gui, active_subject: "subject_a" } }));
  }

  const point = { x: 2, y: 2, subject: "subject_a", attributes: {} };
  const RUNS: [(typeof CONTENT_EDITS)[number], string, () => unknown][] = [
    ["pushUndo", "", () => s().pushUndo()],
    ["undo", "", () => s().undo()],
    ["rollbackLast", "", () => s().rollbackLast()],
    ["redo", "", () => s().redo()],
    ["add", "boxes", () => s().add("boxes", { ...box, x2: 7 })],
    ["add", "polygons", () => s().add("polygons", { ...polygon, rings: [RING_B] })],
    ["add", "points", () => s().add("points", point)],
    ["add", "imageAnnotations", () => s().add("imageAnnotations", rating)],
    ["update", "boxes", () => s().update("boxes", 0, { ...box, x2: 6 })],
    ["update", "polygons", () => s().update("polygons", 0, { ...polygon, rings: [RING_B] })],
    ["update", "points", () => s().update("points", 0, { ...point, x: 3 })],
    [
      "update",
      "imageAnnotations",
      () => s().update("imageAnnotations", 0, { ...rating, iscrowd: true }),
    ],
    ["drag", "boxes", () => s().drag("boxes", 0, { ...box, x2: 6 })],
    ["drag", "polygons", () => s().drag("polygons", 0, { ...polygon, rings: [RING_B] })],
    ["drag", "points", () => s().drag("points", 0, { ...point, x: 9 })],
    [
      "drag",
      "imageAnnotations",
      () => s().drag("imageAnnotations", 0, { ...rating, iscrowd: true }),
    ],
    ["remove", "boxes", () => s().remove("boxes", 0)],
    ["remove", "polygons", () => s().remove("polygons", 0)],
    ["remove", "points", () => s().remove("points", 0)],
    ["remove", "imageAnnotations", () => s().remove("imageAnnotations", 0)],
    ["dragVertex", "", () => s().dragVertex(0, 0, 0, [3, 3])],
    ["splitPolygon", "", () => s().splitPolygon(0, [RING_A, RING_B])],
    ["setCurrentPolygon", "", () => s().setCurrentPolygon(RING_A)],
    ["commitCurrentPolygon", "", () => s().commitCurrentPolygon()],
  ];

  it("runs every action the hold covers, and only those", () => {
    expect(new Set(RUNS.map(([edit]) => edit))).toEqual(new Set(CONTENT_EDITS));
  });

  it.each(RUNS)("%s %s changes the canvas unheld and does nothing held", (_edit, _on, run) => {
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
