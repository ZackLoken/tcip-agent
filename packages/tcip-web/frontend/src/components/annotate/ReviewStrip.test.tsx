import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { ReviewStrip } from "@/components/annotate/ReviewStrip";
import { itemName, type ReviewItem } from "@/lib/reviewItems";

afterEach(cleanup);

const focusedProposal: ReviewItem = {
  kind: "proposal",
  shape: "box",
  ref: 3,
  subject: "fruit",
  bbox: [0, 0, 10, 10],
  status: "undecided",
  score: 0.9,
};

function renderStrip(over: Partial<Parameters<typeof ReviewStrip>[0]> = {}) {
  const handlers = {
    onProposalsShown: vi.fn(),
    onFilters: vi.fn(),
    onScope: vi.fn(),
    onStep: vi.fn(),
    onJump: vi.fn(),
    onAccept: vi.fn(),
    onEdit: vi.fn(),
    onReject: vi.fn(),
  };
  render(
    <ReviewStrip
      bucket="m1/2026-01-01"
      proposalsShown
      reviewing={over.reviewing ?? true}
      operatingPoint={{ conf: 0.42, reason: "" }}
      filters={{ confidence: 0.42, shape: "all" }}
      scope="all"
      counts={{ items: 7, undecided: 3 }}
      focused={focusedProposal}
      position={2}
      total={7}
      {...handlers}
      {...over}
    />,
  );
  return handlers;
}

describe("ReviewStrip", () => {
  it("names the focused item for the stepper", () => {
    expect(itemName(focusedProposal)).toBe("fruit proposal 0.90");
    expect(itemName({ ...focusedProposal, kind: "annotation", score: null })).toBe("fruit box");
    expect(itemName(null)).toBeNull();
  });

  it("carries the filters, the toggle and the scope to the tab", () => {
    const h = renderStrip();
    fireEvent.click(screen.getByRole("checkbox", { name: "Proposals" }));
    expect(h.onProposalsShown).toHaveBeenCalledWith(false);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Confidence floor" }), {
      target: { value: "0.6" },
    });
    expect(h.onFilters).toHaveBeenCalledWith({ confidence: 0.6, shape: "all" });
    fireEvent.change(screen.getByRole("combobox", { name: "Item type" }), {
      target: { value: "point" },
    });
    expect(h.onFilters).toHaveBeenCalledWith({ confidence: 0.42, shape: "point" });
    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "undecided" },
    });
    expect(h.onScope).toHaveBeenCalledWith("undecided");
  });

  it("shows the counts the filters leave and the position in the review path", () => {
    renderStrip();
    expect(screen.getByText("7 items, 3 awaiting a decision")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Item position" })).toHaveValue("2");
    expect(screen.getByText("/ 7")).toBeInTheDocument();
  });

  it("accept, edit and reject reach the tab, each naming its key", () => {
    const h = renderStrip();
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(h.onAccept).toHaveBeenCalledTimes(1);
    expect(h.onEdit).toHaveBeenCalledTimes(1);
    expect(h.onReject).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Accept" })).toHaveAttribute(
      "title",
      expect.stringMatching(/\(a\)$/),
    );
  });

  it("with no bucket the proposal controls are disabled and say why, the stepper still works", () => {
    const h = renderStrip({ bucket: null, reviewing: false, operatingPoint: null });
    expect(screen.getByRole("checkbox", { name: "Proposals" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Proposals" }).closest("label")).toHaveAttribute(
      "title",
      expect.stringMatching(/prediction bucket/),
    );
    expect(screen.getByRole("spinbutton", { name: "Confidence floor" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Accept" })).toBeDisabled();
    expect(screen.getByText("7 items")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next item" }));
    expect(h.onStep).toHaveBeenCalledWith(1);
  });

  it("says why no operating point seeds the confidence floor", () => {
    renderStrip({
      operatingPoint: { conf: null, reason: "bucket 'm1' holds no detector's predictions" },
      filters: { confidence: null, shape: "all" },
    });
    expect(
      screen.getByRole("spinbutton", { name: "Confidence floor" }).closest("label"),
    ).toHaveAttribute("title", expect.stringMatching(/No validated operating point: bucket/));
  });
});
