import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import type { ImageBandsResponse } from "@/api/client";
import { api } from "@/api/client";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

const modeButton = (name: "Box" | "Polygon" | "Point") =>
  screen.getByRole("button", { name: new RegExp(`^${name}$`) });

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  // The Editor shelf's open/closed state persists to localStorage, which some environments keep
  // across every test in this file (one jsdom instance per file, not per test): an earlier test
  // leaving it open would make a later test's fresh render start open too. Guarded because
  // `localStorage` itself is unavailable in some Node/environment combinations (the component's
  // own read/write of this key is guarded the same way, for the same reason).
  try {
    localStorage.removeItem("tcip.annotate.editorOpen");
  } catch {
    /* not available in this environment, nothing to clear */
  }
  // The toolbar's nav hook persists the settled index; nothing here should reach the backend.
  vi.spyOn(api.dataset, "nav").mockResolvedValue({ status: "ok" } as never);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderToolbar(
  bandsInfo?: ImageBandsResponse | null,
  bandSelection?: BandSelection | null,
) {
  render(
    <AnnotateToolbar
      onSave={() => {}}
      saveDisabled={false}
      dirty={false}
      bandsInfo={bandsInfo}
      bandSelection={bandSelection}
      onBandSelectionChange={() => {}}
    />,
  );
}

const FOUR_BANDS: ImageBandsResponse = {
  band_count: 4,
  bands: [
    { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
    { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
    { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
    { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
  ],
  sampled: false,
  pixel_fraction: 1.0,
  seed: 0,
};

const THREE_BANDS: ImageBandsResponse = {
  band_count: 3,
  bands: [
    { name: "Red", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
    { name: "Green", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
    { name: "Blue", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
  ],
};

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
    // The pill shows the active subject's own count: a placed point is an annotation like any other.
    expect(screen.getByText("(2)")).toBeInTheDocument();
  });

  it("Snap and Stream stay polygon-only in point mode (they have no meaning for a point)", () => {
    renderToolbar();
    // The shelf's open/closed state persists to localStorage, which vitest keeps across every
    // test in this file (one jsdom environment per file, not per test): a blind toggle click can
    // close a shelf an earlier test left open instead of opening it.
    const editorBtn = screen.getByRole("button", { name: /Editor/ });
    if (editorBtn.getAttribute("aria-expanded") !== "true") fireEvent.click(editorBtn);
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

describe("AnnotateToolbar band picker (progressive disclosure)", () => {
  function openEditor() {
    const btn = screen.getByRole("button", { name: /Editor/ });
    if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
  }

  it("is absent with no bandsInfo at all (a project with no channel-count fact yet)", () => {
    renderToolbar(null, null);
    openEditor();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
  });

  it("is hidden for a standard 3-band RGB dataset", () => {
    renderToolbar(THREE_BANDS, defaultBandSelection(THREE_BANDS.bands));
    openEditor();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Band")).not.toBeInTheDocument();
  });

  it("is shown for a >3-band (multispectral) dataset", () => {
    renderToolbar(FOUR_BANDS, defaultBandSelection(FOUR_BANDS.bands));
    openEditor();
    expect(screen.getByLabelText("R band")).toBeInTheDocument();
    expect(screen.getByLabelText("G band")).toBeInTheDocument();
    expect(screen.getByLabelText("B band")).toBeInTheDocument();
  });

  it("stays hidden while the Editor shelf itself is collapsed, even for a multispectral dataset", () => {
    renderToolbar(FOUR_BANDS, defaultBandSelection(FOUR_BANDS.bands));
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
  });
});
