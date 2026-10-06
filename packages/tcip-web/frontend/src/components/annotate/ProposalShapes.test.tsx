import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { ProposalShapes } from "@/components/annotate/ProposalShapes";
import type { ReviewItem } from "@/lib/reviewItems";
import { FOCUS_HALO, STATUS_COLORS, strokeWidths } from "@/lib/symbology";
import type { Proposal } from "@/store/types";

// Konva needs a real 2D canvas; render its shapes as inspectable divs.
vi.mock("react-konva", () => ({
  Rect: (props: { stroke?: string; opacity?: number; dash?: number[] }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
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

function proposal(index: number, paired: number | null): Proposal {
  return {
    subject: "subject_a",
    bbox: [10, 10, 50, 50],
    attributes: {},
    iscrowd: false,
    score: 0.9,
    index,
    paired,
    decision: null,
  };
}

const focusedOn = (index: number): ReviewItem => ({
  kind: "proposal",
  shape: "box",
  ref: index,
  subject: "subject_a",
  bbox: [10, 10, 50, 50],
  status: "undecided",
  score: 0.9,
});

function drawn(proposals: Proposal[], focused: ReviewItem | null) {
  render(<ProposalShapes proposals={proposals} focused={focused} widths={strokeWidths(1)} />);
  const rects = screen.getAllByTestId("k-rect");
  const texts = screen.queryAllByTestId("k-text").map((t) => t.getAttribute("data-text"));
  cleanup();
  return { rects, texts };
}

describe("ProposalShapes", () => {
  it("draws every proposal dotted in the undecided status color", () => {
    const { rects } = drawn([proposal(0, null), proposal(1, 3)], null);
    expect(rects).toHaveLength(2);
    expect(rects.every((r) => r.getAttribute("data-stroke") === STATUS_COLORS.undecided)).toBe(
      true,
    );
    expect(rects.every((r) => r.dataset.dash === "true")).toBe(true);
  });

  it("labels an unpaired proposal at rest and a paired one only while focused", () => {
    const atRest = drawn([proposal(0, null), proposal(1, 3)], null);
    expect(atRest.texts.filter((t) => t?.startsWith("subject_a proposal 0.90"))).toHaveLength(2);
    expect(atRest.texts.some((t) => t?.includes("pairs with"))).toBe(false);

    const pairedFocused = drawn([proposal(0, null), proposal(1, 3)], focusedOn(1));
    expect(pairedFocused.texts.some((t) => t?.includes("pairs with an annotation"))).toBe(true);
  });

  it("the focused proposal draws a halo under its own stroke and keeps its color", () => {
    const { rects } = drawn([proposal(0, null)], focusedOn(0));
    expect(rects).toHaveLength(2);
    expect(rects[0]).toHaveAttribute("data-stroke", FOCUS_HALO.color);
    expect(Number(rects[0].getAttribute("data-opacity"))).toBe(FOCUS_HALO.opacity);
    expect(rects[1]).toHaveAttribute("data-stroke", STATUS_COLORS.undecided);
  });
});
