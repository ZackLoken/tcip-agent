import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";

import {
  CELL_LABEL_FLOOR_PX,
  CoverageOverlay,
  measureLabelWidth,
} from "@/components/Canvas/CoverageOverlay";
import type { GridCell } from "@/lib/coverage";

// Konva needs a real 2D canvas; render each primitive as an inspectable div, the pattern
// HaloLabel.test.tsx / AnnotateTab.test.tsx / ReviewTab.test.tsx already use.
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => (
    <div data-testid="k-group">{props.children}</div>
  ),
  Rect: (props: {
    fill?: string;
    stroke?: string;
    width: number;
    height: number;
    dash?: number[];
  }) => (
    <div
      data-testid="k-rect"
      data-fill={props.fill}
      data-stroke={props.stroke}
      data-w={props.width}
      data-h={props.height}
      data-dash={props.dash ? "true" : undefined}
    />
  ),
  Line: (props: { stroke?: string }) => <div data-testid="k-line" data-stroke={props.stroke} />,
  Text: (props: { text: string; fill?: string }) => (
    <div data-testid="k-text" data-text={props.text} data-fill={props.fill} />
  ),
}));

function cell(name: string, x0: number, y0: number, x1: number, y1: number): GridCell {
  return { name, x0, y0, x1, y1 };
}

describe("CELL_LABEL_FLOOR_PX", () => {
  it("fits the platform's own worst-case label: a 16-division lattice's widest cell name plus a three-digit count", () => {
    const worstCase = measureLabelWidth("P16 (137)", 10);
    expect(CELL_LABEL_FLOOR_PX).toBeGreaterThanOrEqual(worstCase + 4);
  });
});

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

  it("draws a cell's name only once its on-screen edge reaches the label floor, with a halo behind it", () => {
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
    // The halo pattern draws two Text nodes per label (a dark one behind, the light one on top).
    expect(labels).toEqual(["A1", "A1"]);
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
      (el.getAttribute("data-fill") ?? "").includes("201, 162, 75"),
    );
    expect(fills).toHaveLength(1);
  });

  it("a swept cell's dashed stroke is two-tone: a dark halo rect under the coloured one, never colour alone", () => {
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
    const dashed = Array.from(container.querySelectorAll("[data-testid=k-rect][data-dash=true]"));
    expect(dashed).toHaveLength(2);
    expect(
      dashed.some((el) => (el.getAttribute("data-stroke") ?? "").includes("201, 162, 75")),
    ).toBe(true);
    expect(dashed.some((el) => (el.getAttribute("data-stroke") ?? "").includes("30, 30, 30"))).toBe(
      true,
    );
  });

  it("every cell border is two-tone: a dark halo rect under the light one", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100)]}
        viewport={{ x0: 0, y0: 0, x1: 100, y1: 100 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set()}
        activeStale={new Set()}
        otherComplete={new Set()}
        annotationCounts={{}}
      />,
    );
    const borders = Array.from(container.querySelectorAll("[data-testid=k-rect]"));
    expect(
      borders.some((el) => (el.getAttribute("data-stroke") ?? "").includes("30, 30, 30")),
    ).toBe(true);
    expect(
      borders.some((el) => (el.getAttribute("data-stroke") ?? "").includes("231, 229, 220")),
    ).toBe(true);
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

  it("another subject's attestation carries a dotted (dashed) stroke the active subject's own does not", () => {
    const { container } = render(
      <CoverageOverlay
        cells={[cell("A1", 0, 0, 100, 100), cell("B1", 100, 0, 200, 100)]}
        viewport={{ x0: 0, y0: 0, x1: 200, y1: 100 }}
        scale={1}
        swept={new Set()}
        activeComplete={new Set(["A1"])}
        activeStale={new Set()}
        otherComplete={new Set(["B1"])}
        annotationCounts={{}}
      />,
    );
    const rects = Array.from(container.querySelectorAll("[data-testid=k-rect]"));
    const attestMarks = rects.filter((el) =>
      (el.getAttribute("data-stroke") ?? "").includes("230, 151, 107"),
    );
    // Both attested marks share the ATTEST_RGB stroke; only the other-subject one carries dash.
    expect(attestMarks).toHaveLength(2);
    expect(attestMarks.filter((el) => el.getAttribute("data-dash") === "true")).toHaveLength(1);
  });
});
