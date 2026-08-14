import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { CoverageMinimap } from "@/components/Canvas/CoverageMinimap";
import type { GridCell, GridGeometry } from "@/lib/coverage";

// A non-square rendered box: a vertical mapping taken against the width lands on the wrong cell.
const BOX = { left: 200, top: 100, width: 320, height: 180 };

const GRID: GridGeometry = {
  width: 1600,
  height: 900,
  tile_size: 400,
  overlap: 0,
  cols: 4,
  rows: 3,
};

const CELLS: GridCell[] = [
  { name: "A1", x0: 0, y0: 0, x1: 400, y1: 400 },
  { name: "B1", x0: 400, y0: 0, x1: 800, y1: 400 },
  { name: "C1", x0: 800, y0: 0, x1: 1200, y1: 400 },
  { name: "D1", x0: 1200, y0: 0, x1: 1600, y1: 400 },
  { name: "A2", x0: 0, y0: 400, x1: 400, y1: 800 },
  { name: "B2", x0: 400, y0: 400, x1: 800, y1: 800 },
  { name: "C2", x0: 800, y0: 400, x1: 1200, y1: 800 },
  { name: "D2", x0: 1200, y0: 400, x1: 1600, y1: 800 },
  { name: "A3", x0: 0, y0: 800, x1: 400, y1: 900 },
  { name: "B3", x0: 400, y0: 800, x1: 800, y1: 900 },
  { name: "C3", x0: 800, y0: 800, x1: 1200, y1: 900 },
  { name: "D3", x0: 1200, y0: 800, x1: 1600, y1: 900 },
];

function openMap(handlers: {
  onJump?: (cell: GridCell) => void;
  onToggleComplete?: (cell: GridCell) => void;
}): HTMLElement {
  render(
    <CoverageMinimap
      imagePath="C:/data/images/2026-01-01/orchard.tif"
      composite={{ bands: "1,2,3", stretch: "p2" }}
      grid={GRID}
      cells={CELLS}
      swept={new Set<string>()}
      sweptVersion={0}
      viewport={null}
      onJump={handlers.onJump ?? (() => {})}
      activeComplete={new Set<string>()}
      otherComplete={new Set<string>()}
      onToggleComplete={handlers.onToggleComplete ?? (() => {})}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Coverage" }));
  const box = screen.getByTitle(/double-click to toggle a cell complete/i);
  box.getBoundingClientRect = () =>
    ({
      ...BOX,
      right: BOX.left + BOX.width,
      bottom: BOX.top + BOX.height,
      x: BOX.left,
      y: BOX.top,
      toJSON: () => ({}),
    }) as DOMRect;
  box.setPointerCapture = () => {};
  return box;
}

// jsdom has no PointerEvent; a MouseEvent under the pointerdown type carries the same coords.
function pressAt(el: HTMLElement, clientX: number, clientY: number) {
  fireEvent(el, new MouseEvent("pointerdown", { bubbles: true, clientX, clientY }));
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
});

describe("CoverageMinimap cell targeting", () => {
  it("jumps to the cell under the pointer, scaling each axis by its own extent", () => {
    const onJump = vi.fn();
    const box = openMap({ onJump });
    pressAt(box, 280, 190);

    expect(onJump).toHaveBeenCalledTimes(1);
    // 80 of the 320px width is x=400 of 1600; 90 of the 180px height is y=450 of 900.
    expect(onJump.mock.calls[0][0].name).toBe("B2");
  });

  it("reaches the bottom row from a press near the bottom edge", () => {
    const onJump = vi.fn();
    const box = openMap({ onJump });
    pressAt(box, 230, 270);

    expect(onJump).toHaveBeenCalledTimes(1);
    // 30 of 320 is x=150; 170 of 180 is y=850, inside the short final row.
    expect(onJump.mock.calls[0][0].name).toBe("A3");
  });

  it("reports no cell for a press outside the mapped box", () => {
    const onJump = vi.fn();
    const box = openMap({ onJump });
    pressAt(box, 900, 190);

    expect(onJump).not.toHaveBeenCalled();
  });

  it("toggles completeness for the double-clicked cell", () => {
    const onToggleComplete = vi.fn();
    const box = openMap({ onToggleComplete });
    fireEvent.doubleClick(box, { clientX: 280, clientY: 190 });

    expect(onToggleComplete).toHaveBeenCalledTimes(1);
    expect(onToggleComplete.mock.calls[0][0].name).toBe("B2");
  });
});
