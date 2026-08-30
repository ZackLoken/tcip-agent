import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render } from "@testing-library/react";

import { HaloLabel } from "@/components/HaloLabel";

// Konva needs a real 2D canvas; render its Text as an inspectable div (the same pattern
// AnnotateTab.test.tsx / ReviewTab.test.tsx use for the tabs that mount this component).
vi.mock("react-konva", () => ({
  Text: (props: {
    text?: string;
    fill?: string;
    shadowColor?: string;
    shadowOffset?: { x: number; y: number };
  }) => (
    <div
      data-testid="k-text"
      data-text={props.text}
      data-fill={props.fill}
      data-shadow-color={props.shadowColor}
      data-shadow-offset={props.shadowOffset ? JSON.stringify(props.shadowOffset) : undefined}
    />
  ),
}));

afterEach(() => {
  cleanup();
});

describe("HaloLabel", () => {
  it("renders AnnotateTab's usage (a shape label) as a black halo behind the subject-colour fill", () => {
    const { getAllByTestId } = render(
      <HaloLabel x={10} y={20} text="tree" fill="#33cc33" size={11} />,
    );
    const nodes = getAllByTestId("k-text");
    expect(nodes).toHaveLength(2);
    expect(nodes[0]).toHaveAttribute("data-fill", "#000000");
    expect(nodes[0]).toHaveAttribute("data-shadow-color", "#000000");
    // The halo sits directly behind the text, not offset to one side (Konva's own default for
    // both shadowOffsetX/Y is already 0; this states the halo's intent explicitly).
    expect(nodes[0]).toHaveAttribute("data-shadow-offset", JSON.stringify({ x: 0, y: 0 }));
    expect(nodes[1]).toHaveAttribute("data-fill", "#33cc33");
    expect(nodes[0]).toHaveAttribute("data-text", "tree");
    expect(nodes[1]).toHaveAttribute("data-text", "tree");
  });

  it("renders ReviewTab's usage (a detection label with a confidence suffix)", () => {
    const { getAllByTestId } = render(
      <HaloLabel x={100} y={50} text="tree 0.87" fill="#3388ff" size={12.1} />,
    );
    const nodes = getAllByTestId("k-text");
    expect(nodes).toHaveLength(2);
    expect(nodes[0]).toHaveAttribute("data-text", "tree 0.87");
    expect(nodes[0]).toHaveAttribute("data-fill", "#000000");
    expect(nodes[1]).toHaveAttribute("data-fill", "#3388ff");
  });
});
