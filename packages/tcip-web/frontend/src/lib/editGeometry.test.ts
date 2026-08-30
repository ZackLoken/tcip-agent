import { describe, expect, it } from "vitest";

import {
  applyEditDrag,
  clampShapeToImage,
  hitTestEdit,
  seedEditShape,
  type EditShape,
} from "@/lib/editGeometry";
import type { ReviewGeom } from "@/lib/reviewGeometry";

const box = (b: [number, number, number, number]): EditShape => ({ kind: "box", box: b });
const poly = (points: [number, number][]): EditShape => ({ kind: "polygon", points });

describe("hitTestEdit", () => {
  it("grabs the nearest corner, not the first within tolerance", () => {
    // 12px box with tol 10: a click 2px from the top-right corner is also within
    // tolerance of the top-left: the nearest (top-right) must win, anchoring bottom-left.
    const drag = hitTestEdit(box([0, 0, 12, 12]), 10, 0, 10);
    expect(drag).toEqual({ mode: "corner", ax: 0, ay: 12 });
  });

  it("falls back to move inside the box and null outside", () => {
    expect(hitTestEdit(box([0, 0, 100, 100]), 50, 50, 5)).toEqual({
      mode: "move",
      lastX: 50,
      lastY: 50,
    });
    expect(hitTestEdit(box([0, 0, 100, 100]), 200, 200, 5)).toBeNull();
  });

  it("grabs the nearest polygon vertex, else the body, else nothing", () => {
    const p = poly([
      [0, 0],
      [20, 0],
      [20, 20],
      [0, 20],
    ]);
    expect(hitTestEdit(p, 19, 1, 10)).toEqual({ mode: "vertex", idx: 1 });
    expect(hitTestEdit(p, 10, 10, 3)).toEqual({ mode: "move", lastX: 10, lastY: 10 });
    expect(hitTestEdit(p, 100, 100, 3)).toBeNull();
  });
});

describe("applyEditDrag (box)", () => {
  it("corner drag re-anchors to the opposite corner and survives crossing over", () => {
    // Dragging the (100,100) corner (anchor 0,0) past the anchor to (-,-) still normalizes.
    const r = applyEditDrag(
      box([0, 0, 100, 100]),
      { mode: "corner", ax: 0, ay: 0 },
      40,
      60,
      500,
      500,
    );
    expect(r.shape).toEqual(box([0, 0, 40, 60]));
    const crossed = applyEditDrag(
      box([0, 0, 100, 100]),
      { mode: "corner", ax: 100, ay: 100 },
      150,
      160,
      500,
      500,
    );
    expect(crossed.shape).toEqual(box([100, 100, 150, 160]));
  });

  it("corner drag clamps to the image", () => {
    const r = applyEditDrag(
      box([0, 0, 100, 100]),
      { mode: "corner", ax: 0, ay: 0 },
      900,
      -50,
      500,
      500,
    );
    expect(r.shape).toEqual(box([0, 0, 500, 0]));
  });

  it("move clamps the shift so the box never leaves the image", () => {
    const r = applyEditDrag(
      box([300, 400, 400, 500]),
      { mode: "move", lastX: 350, lastY: 450 },
      550,
      450,
      500,
      500,
    );
    expect(r.shape).toEqual(box([400, 400, 500, 500])); // +200 requested, +100 available
    expect(r.drag).toEqual({ mode: "move", lastX: 550, lastY: 450 });
  });

  it("move returns the same shape reference when nothing moved", () => {
    const s = box([400, 400, 500, 500]);
    const r = applyEditDrag(s, { mode: "move", lastX: 450, lastY: 450 }, 700, 450, 500, 500);
    expect(r.shape).toBe(s);
  });
});

describe("applyEditDrag (polygon)", () => {
  const square: [number, number][] = [
    [10, 10],
    [30, 10],
    [30, 30],
    [10, 30],
  ];

  it("vertex drag moves only that vertex, clamped to the image", () => {
    const r = applyEditDrag(poly(square), { mode: "vertex", idx: 2 }, 600, -5, 500, 500);
    expect(r.shape).toEqual(
      poly([
        [10, 10],
        [30, 10],
        [500, 0],
        [10, 30],
      ]),
    );
  });

  it("move shifts all points, clamped by the polygon bbox", () => {
    const r = applyEditDrag(
      poly(square),
      { mode: "move", lastX: 20, lastY: 20 },
      20,
      -100,
      500,
      500,
    );
    expect(r.shape).toEqual(
      poly([
        [10, 0],
        [30, 0],
        [30, 20],
        [10, 20],
      ]),
    );
  });
});

describe("seedEditShape", () => {
  it("seeds a box shape from a box detection geometry", () => {
    const geom: ReviewGeom = { kind: "box", box: [10, 20, 100, 200] };
    expect(seedEditShape(geom)).toEqual(box([10, 20, 100, 200]));
  });

  it("seeds a polygon shape from the single ring, copying each point", () => {
    const ring: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
    ];
    const geom: ReviewGeom = { kind: "polygon", rings: [ring] };
    const seeded = seedEditShape(geom);
    expect(seeded).toEqual(poly(ring));
    if (seeded.kind === "polygon") {
      expect(seeded.points).not.toBe(ring); // a fresh copy, so dragging can't mutate matches
    }
  });
});

describe("clampShapeToImage", () => {
  it("pulls out-of-bounds seeds (tiled predictions) into the image", () => {
    expect(clampShapeToImage(box([-8, 5, 510, 490]), 500, 500)).toEqual(box([0, 5, 500, 490]));
    expect(
      clampShapeToImage(
        poly([
          [-3, 10],
          [505, 10],
          [250, 600],
        ]),
        500,
        500,
      ),
    ).toEqual(
      poly([
        [0, 10],
        [500, 10],
        [250, 500],
      ]),
    );
  });
});
