import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { ProposalShapes } from "@/components/annotate/ProposalShapes";
import type { Proposal } from "@/store/types";

// Konva needs a real 2D canvas; render its shapes as inspectable divs.
vi.mock("react-konva", () => ({
  Rect: (props: { opacity?: number; dash?: number[] }) => (
    <div
      data-testid="k-rect"
      data-opacity={props.opacity ?? ""}
      data-dash={props.dash ? "true" : ""}
    />
  ),
  Line: () => <div data-testid="k-line" />,
  Circle: () => <div data-testid="k-circle" />,
  Text: (props: { text?: string }) => <div data-testid="k-text" data-text={props.text} />,
}));

afterEach(() => {
  cleanup();
});

function proposal(index: number, admitted: boolean): Proposal {
  return {
    subject: "subject_a",
    bbox: [10, 10, 50, 50],
    attributes: {},
    iscrowd: false,
    score: 0.9,
    index,
    paired: null,
    decision: null,
    admitted,
  };
}

function drawn(admitted: boolean) {
  render(
    <ProposalShapes proposals={[proposal(0, admitted)]} selected={0} strokeW={1} scaleLineW={1} />,
  );
  const rect = screen.getByTestId("k-rect");
  const texts = screen.getAllByTestId("k-text").map((t) => t.getAttribute("data-text"));
  cleanup();
  return { opacity: rect.getAttribute("data-opacity"), dashed: rect.dataset.dash, texts };
}

describe("ProposalShapes", () => {
  it("dims a proposal below the operating point through its stroke alone, its label unchanged", () => {
    const above = drawn(true);
    const below = drawn(false);

    expect(above.opacity).toBe("");
    expect(Number(below.opacity)).toBeGreaterThan(0);
    expect(Number(below.opacity)).toBeLessThan(1);
    expect(below.dashed).toBe(above.dashed);
    expect(below.texts).toEqual(above.texts);
  });
});
