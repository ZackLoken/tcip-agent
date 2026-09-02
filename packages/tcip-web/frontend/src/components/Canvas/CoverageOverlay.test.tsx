import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";

import { CELL_LABEL_FLOOR_PX, CoverageOverlay } from "@/components/Canvas/CoverageOverlay";
import type { GridCell } from "@/lib/coverage";

// Konva needs a real 2D canvas; render each primitive as an inspectable div, the pattern
// HaloLabel.test.tsx / AnnotateTab.test.tsx / ReviewTab.test.tsx already use.
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => (
    <div data-testid="k-group">{props.children}</div>
  ),
  Rect: (props: { fill?: string; stroke?: string; width: number; height: number }) => (
    <div
      data-testid="k-rect"
      data-fill={props.fill}
      data-stroke={props.stroke}
      data-w={props.width}
      data-h={props.height}
    />
  ),
  Line: (props: { stroke?: string }) => <div data-testid="k-line" data-stroke={props.stroke} />,
  Text: (props: { text: string }) => <div data-testid="k-text" data-text={props.text} />,
}));

function cell(name: string, x0: number, y0: number, x1: number, y1: number): GridCell {
  return { name, x0, y0, x1, y1 };
}

describe("CoverageOverlay", () => {
  it("renders nothing without a measured viewport", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100)]}
        viewport={null}
        scale={1}
        swept={new Set()}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    expect(container.querySelectorAll("[data-testid=k-group]")).toHaveLength(0);
  });

  it("culls to the viewport: a cell entirely outside it draws nothing", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100), cell("Z9", 5000, 5000, 5100, 5100)]}
        viewport={{ x0: 0, y0: 0, x1: 200, y1: 200 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    expect(container.querySelectorAll("[data-testid=k-group]")).toHaveLength(1);
  });

  it("draws a cell's name only once its on-screen edge reaches the label floor", () => {
    const bigCell = cell("A1", 0, 0, CELL_LABEL_FLOOR_PX, CELL_LABEL_FLOOR_PX);
    const smallCell = cell("B1", 200, 0, 200 + CELL_LABEL_FLOOR_PX - 1, CELL_LABEL_FLOOR_PX - 1);
    const { container } = render(
      <CoverageOverlay
        cells={[bigCell, smallCell]}
        viewport={{ x0: 0, y0: 0, x1: 400, y1: 400 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    const labels = Array.from(container.querySelectorAll("[data-testid=k-text]")).map((el) =>
      el.getAttribute("data-text"),
    );
    expect(labels).toEqual(["A1"]);
  });

  it("shows the saved-annotation count beside a labeled cell's name", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, CELL_LABEL_FLOOR_PX, CELL_LABEL_FLOOR_PX)]}
        viewport={{ x0: 0, y0: 0, x1: 200, y1: 200 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{ A1: 3 }}
      />,
    );
    expect(container.querySelector("[data-testid=k-text]")).toHaveAttribute("data-text", "A1 (3)");
  });

  it("a swept cell fills, an unswept one does not", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100), cell("B1", 100, 0, 200, 100)]}
        viewport={{ x0: 0, y0: 0, x1: 200, y1: 100 }}
        scale={1}
        swept={new Set(["A1"])}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    const fills = Array.from(container.querySelectorAll("[data-testid=k-rect]")).filter((el) =>
      (el.getAttribute("data-fill") ?? "").includes("80, 119, 84"),
    );
    expect(fills).toHaveLength(1);
  });

  it("a stale attested cell draws the strike-through line the active-complete one does not", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100), cell("B1", 100, 0, 200, 100)]}
        viewport={{ x0: 0, y0: 0, x1: 200, y1: 100 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set(["B1"])}
        activeStale={new Set(["A1"])}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    expect(container.querySelectorAll("[data-testid=k-line]")).toHaveLength(1);
  });
});
