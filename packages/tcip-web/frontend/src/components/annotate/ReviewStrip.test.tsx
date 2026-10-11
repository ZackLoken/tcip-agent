import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewStrip } from "@/components/annotate/ReviewStrip";
import { itemName, type ReviewItem } from "@/lib/reviewItems";
import type { Flag } from "@/store/types";

afterEach(cleanup);

const focusedProposal: ReviewItem = {
  kind: "proposal",
  shape: "box",
  ref: 3,
  subject: "fruit",
  bbox: [0, 0, 10, 10],
  match: "proposal_only",
  reviewed: false,
  score: 0.9,
  at: [5, 5],
  flags: [],
};

const openFlag: Flag = {
  id: "f1",
  text: "fruit or gall?",
  by: "user:first",
  at: "2026-01-01T00:00:00+00:00",
  point: null,
  subject: null,
  proposal: ["m1/2026-01-01", 3],
  resolved_by: null,
  resolved_at: null,
  reply: "",
  removed: false,
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
    onFlagsOpen: vi.fn(),
    onFlag: vi.fn(),
    onResolve: vi.fn(),
  };
  render(
    <ReviewStrip
      bucket="m1/2026-01-01"
      proposalsShown
      reviewing
      operatingPoint={{ conf: 0.42, reason: "" }}
      filters={{ confidence: 0.42, match: "all" }}
      scope="all"
      canAccept
      canReject
      counts={{ items: 7, unreviewed: 3, flags: 0 }}
      focused={focusedProposal}
      position={2}
      total={7}
      flags={[]}
      flagsOpen={false}
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
    expect(h.onFilters).toHaveBeenCalledWith({ confidence: 0.6, match: "all" });
    fireEvent.change(screen.getByRole("combobox", { name: "Match type" }), {
      target: { value: "proposal_only" },
    });
    expect(h.onFilters).toHaveBeenCalledWith({ confidence: 0.42, match: "proposal_only" });
    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "flagged" },
    });
    expect(h.onScope).toHaveBeenCalledWith("flagged");
  });

  it("offers the three match types by their plain names and no geometry filter", () => {
    renderStrip();
    const options = screen.getByRole("combobox", { name: "Match type" }).querySelectorAll("option");
    expect(Array.from(options, (o) => o.textContent)).toEqual([
      "all",
      "Matched",
      "Proposal only",
      "Annotation only",
    ]);
    expect(screen.queryByRole("combobox", { name: "Item type" })).not.toBeInTheDocument();
  });

  it("shows the counts the filters leave and the position in the review path", () => {
    renderStrip();
    expect(screen.getByText("7 items, 3 unreviewed")).toBeInTheDocument();
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

  it("accept and reject are each enabled by their own target", () => {
    renderStrip({ canAccept: true, canReject: false });
    expect(screen.getByRole("button", { name: "Accept" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled();
    cleanup();
    renderStrip({ canAccept: false, canReject: true });
    expect(screen.getByRole("button", { name: "Accept" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeEnabled();
  });

  it("offers Edit for a focused proposal only", () => {
    renderStrip({ focused: { ...focusedProposal, kind: "annotation", score: null } });
    expect(screen.getByRole("button", { name: "Edit" })).toBeDisabled();
  });

  it("with no bucket the proposal controls are disabled and say why, the stepper still works", () => {
    const h = renderStrip({
      bucket: null,
      reviewing: false,
      operatingPoint: null,
      canAccept: false,
      canReject: false,
    });
    expect(screen.getByRole("checkbox", { name: "Proposals" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Proposals" }).closest("label")).toHaveAttribute(
      "title",
      expect.stringMatching(/prediction bucket/),
    );
    expect(screen.getByRole("spinbutton", { name: "Confidence floor" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Match type" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Accept" })).toBeDisabled();
    expect(screen.getByText("7 items")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next item" }));
    expect(h.onStep).toHaveBeenCalledWith(1);
  });

  it("says why no operating point seeds the confidence floor", () => {
    renderStrip({
      operatingPoint: { conf: null, reason: "bucket 'm1' holds no detector's predictions" },
      filters: { confidence: null, match: "all" },
    });
    expect(
      screen.getByRole("spinbutton", { name: "Confidence floor" }).closest("label"),
    ).toHaveAttribute("title", expect.stringMatching(/No validated operating point: bucket/));
  });
});

describe("ReviewStrip flags", () => {
  it("counts the image's open flags on the button and opens the panel", () => {
    const h = renderStrip({ counts: { items: 7, unreviewed: 3, flags: 2 } });
    const button = screen.getByRole("button", { name: /Flag \(2\)/ });
    fireEvent.click(button);
    expect(h.onFlagsOpen).toHaveBeenCalledWith(true);
  });

  it("raises a flag on the focused item with the typed comment, never a blank one", () => {
    const h = renderStrip({ flagsOpen: true });
    expect(
      screen.getByRole("dialog", { name: "Flags on fruit proposal 0.90" }),
    ).toBeInTheDocument();
    const comment = screen.getByRole("textbox", { name: "New flag comment" });
    fireEvent.keyDown(comment, { key: "Enter" });
    expect(h.onFlag).not.toHaveBeenCalled();
    fireEvent.change(comment, { target: { value: "  fruit or gall?  " } });
    fireEvent.keyDown(comment, { key: "Enter" });
    expect(h.onFlag).toHaveBeenCalledWith("fruit or gall?");
  });

  it("names the image as the target when nothing is focused", () => {
    renderStrip({ flagsOpen: true, focused: null });
    expect(screen.getByRole("dialog", { name: "Flags on this image" })).toBeInTheDocument();
  });

  it("lists each open flag with who raised it and resolves it with the reply", () => {
    const h = renderStrip({ flagsOpen: true, flags: [openFlag] });
    expect(screen.getByText("fruit or gall?")).toBeInTheDocument();
    expect(screen.getByText("user:first, 2026-01-01")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: 'Reply to "fruit or gall?"' }), {
      target: { value: "gall" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Resolve" }));
    expect(h.onResolve).toHaveBeenCalledWith("f1", "gall");
  });
});
