import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { api } from "@/api/client";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

const modeButton = (name: "Box" | "Polygon" | "Point") =>
  screen.getByRole("button", { name: new RegExp(`^${name}$`) });

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  // The toolbar's nav hook persists the settled index; nothing here should reach the backend.
  vi.spyOn(api.dataset, "nav").mockResolvedValue({ status: "ok" } as never);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderToolbar() {
  render(<AnnotateToolbar onSave={() => {}} saveDisabled={false} dirty={false} />);
}

describe("AnnotateToolbar draw mode", () => {
  it("offers all three geometry kinds, with the active one pressed", () => {
    renderToolbar();
    expect(modeButton("Box")).toHaveAttribute("aria-pressed", "true"); // the default mode
    expect(modeButton("Polygon")).toHaveAttribute("aria-pressed", "false");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "false");
  });

  it("switching to Point sets the store's mode and moves the pressed state", () => {
    renderToolbar();
    fireEvent.click(modeButton("Point"));
    expect(useStore.getState().gui.mode).toBe("point");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "true");
    expect(modeButton("Box")).toHaveAttribute("aria-pressed", "false");

    // ...and back out again, so Point is a mode like the others rather than a trap.
    fireEvent.click(modeButton("Box"));
    expect(useStore.getState().gui.mode).toBe("box");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "false");
  });

  it("counts points in the subject pill alongside boxes, polygons and ratings", () => {
    renderToolbar();
    fireEvent.click(modeButton("Point"));
    act(() => {
      const s = useStore.getState();
      s.setActiveSubject("tip");
      s.addPoint({ x: 1, y: 2, subject: "tip", attributes: {} });
      s.addPoint({ x: 3, y: 4, subject: "tip", attributes: {} });
      s.addPoint({ x: 5, y: 6, subject: "other", attributes: {} });
    });
    // The pill shows the active subject's own count — a placed point is an annotation like any other.
    expect(screen.getByText("(2)")).toBeInTheDocument();
  });

  it("Snap and Stream stay polygon-only in point mode (they have no meaning for a point)", () => {
    renderToolbar();
    fireEvent.click(screen.getByRole("button", { name: /Editor/ })); // drop the tools shelf
    fireEvent.click(modeButton("Point"));
    expect(screen.getByRole("button", { name: /Snap/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Stream/ })).toBeDisabled();
  });

  it("renders Point immediately to the left of Box (Point, Box, Polygon)", () => {
    renderToolbar();
    const group = screen.getByRole("group", { name: "Draw mode" });
    const labels = Array.from(group.querySelectorAll("button")).map((b) => b.textContent);
    expect(labels).toEqual(["Point", "Box", "Polygon"]);
  });
});

describe("AnnotateToolbar status filter", () => {
  it("lists the start state (Unannotated) before the terminal states, matching Review's convention", () => {
    renderToolbar();
    const select = screen.getByTitle("Status filter");
    const options = Array.from(select.querySelectorAll("option")).map((o) => o.textContent);
    expect(options).toEqual(["All", "Unannotated", "Partial", "Complete", "Negative"]);
  });
});
