import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import type { ImageBandsResponse } from "@/api/client";
import { api } from "@/api/client";
import { subjectsApi } from "@/api/subjects";
import { StructuredRefusalError } from "@/api/http";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";
import { useStore } from "@/store";
import type { SubjectState } from "@/store/types";

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
  extra?: {
    subjectState?: SubjectState | null;
    onComplete?: (next: boolean) => void;
    onCompleteView?: () => void;
    hideProposals?: boolean;
    onHideProposals?: (next: boolean) => void;
  },
) {
  render(
    <AnnotateToolbar
      onSave={() => {}}
      saveDisabled={false}
      dirty={false}
      bandsInfo={bandsInfo}
      bandSelection={bandSelection}
      onBandSelectionChange={() => {}}
      subjectState={extra?.subjectState ?? null}
      onComplete={extra?.onComplete ?? (() => {})}
      onCompleteView={extra?.onCompleteView}
      hideProposals={extra?.hideProposals ?? false}
      onHideProposals={extra?.onHideProposals ?? (() => {})}
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
    const group = screen.getByRole("group", { name: "Tool" });
    const labels = Array.from(group.querySelectorAll("button")).map((b) => b.textContent);
    expect(labels).toEqual(["Point", "Box", "Polygon"]);
  });

  it("Box and Polygon describe themselves, the same as Point does", () => {
    renderToolbar();
    expect(screen.getByRole("button", { name: "Box" })).toHaveAttribute(
      "title",
      expect.stringMatching(/box/i),
    );
    expect(screen.getByRole("button", { name: "Polygon" })).toHaveAttribute(
      "title",
      expect.stringMatching(/polygon/i),
    );
  });

  it("the image-position textbox has an accessible name, not just a visible counter beside it", () => {
    renderToolbar();
    expect(screen.getByRole("textbox", { name: "Image position" })).toBeInTheDocument();
  });
});

describe("AnnotateToolbar Cut button", () => {
  function openEditor() {
    const btn = screen.getByRole("button", { name: /Editor/ });
    if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
  }

  it("states the disabled reason outside polygon mode", () => {
    renderToolbar();
    openEditor();
    const cutButton = screen.getByRole("button", { name: "Cut" });
    expect(cutButton).toBeDisabled();
    expect(cutButton).toHaveAttribute("title", "Cut: in polygon mode only");
  });

  it("states the how-to once enabled, in polygon mode", () => {
    renderToolbar();
    openEditor();
    fireEvent.click(modeButton("Polygon"));
    const cutButton = screen.getByRole("button", { name: "Cut" });
    expect(cutButton).not.toBeDisabled();
    expect(cutButton).toHaveAttribute(
      "title",
      "Click two points on either side of the selected polygon to split it (x)",
    );
  });
});

describe("AnnotateToolbar subject authoring", () => {
  function seedDataset() {
    useStore.setState((s) => ({
      gui: {
        ...s.gui,
        dataset: {
          ...s.gui.dataset,
          dataset_root: "C:/data",
          date: "2026-01-01",
        },
      },
      openProject: { id: "a1b2c3d4e5f6", path: "C:/proj" },
    }));
  }

  function openSubjectMenu() {
    fireEvent.click(screen.getByRole("button", { name: /select subject/ }));
  }

  function answerPrompt(answer: string | null) {
    return vi.spyOn(window, "prompt").mockReturnValue(answer);
  }

  it("registers a typed name with its surrounding whitespace stripped", async () => {
    // An untrimmed registry key would open a second subject that reads as the first.
    seedDataset();
    act(() => useStore.getState().setRegistry({ leaf: {} }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      subjects_path: "C:/data/subjects.json",
      version: "v1",
    });
    answerPrompt("  husk  ");
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    expect(Object.keys(useStore.getState().registry.subjects)).toEqual(["leaf", "husk"]);
    expect(useStore.getState().gui.active_subject).toBe("husk");
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0]).toEqual({ leaf: {}, husk: {} });
  });

  it("selects an existing subject rather than resetting its attribute definitions", async () => {
    seedDataset();
    const leafDef = {
      attributes: { stage: { type: "ordinal" as const, values: ["early", "late"] } },
    };
    act(() => useStore.getState().setRegistry({ leaf: leafDef, husk: {} }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      subjects_path: "C:/data/subjects.json",
      version: "v1",
    });
    answerPrompt("leaf");
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    expect(useStore.getState().registry.subjects).toEqual({ leaf: leafDef, husk: {} });
    expect(useStore.getState().gui.active_subject).toBe("leaf");
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("adds nothing when the prompt is dismissed or answered with only whitespace", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry({ leaf: {} }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      subjects_path: "C:/data/subjects.json",
      version: "v1",
    });
    const promptSpy = answerPrompt(null);
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });
    expect(Object.keys(useStore.getState().registry.subjects)).toEqual(["leaf"]);

    promptSpy.mockReturnValue("   ");
    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    expect(Object.keys(useStore.getState().registry.subjects)).toEqual(["leaf"]);
    expect(useStore.getState().gui.active_subject).toBeNull();
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("posts the loaded registry version and stores the version the save returns", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry({ leaf: {} }, "v1"));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      subjects_path: "C:/data/subjects.json",
      version: "v2",
    });
    answerPrompt("husk");
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    expect(saveSpy.mock.calls[0][2]).toBe("v1");
    expect(useStore.getState().registry.version).toBe("v2");
  });

  it("reloads the registry from the server when the save is refused", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry({ leaf: {} }, "v1"));
    vi.spyOn(subjectsApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(subjectsApi, "load").mockResolvedValue({
      subjects: { leaf: {} },
      discovered: [],
      version: "v3",
      unreadable: [],
    });
    answerPrompt("husk");
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    // The refused optimistic add is discarded in favor of what the server actually holds.
    expect(useStore.getState().registry.subjects).toEqual({ leaf: {} });
    expect(useStore.getState().registry.version).toBe("v3");
  });

  it("reverts the optimistically set active subject when the save is refused", async () => {
    seedDataset();
    act(() => {
      useStore.getState().setRegistry({ leaf: {} }, "v1");
      useStore.getState().setActiveSubject("leaf");
    });
    vi.spyOn(subjectsApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(subjectsApi, "load").mockResolvedValue({
      subjects: { leaf: {} },
      discovered: [],
      version: "v3",
      unreadable: [],
    });
    answerPrompt("husk");
    renderToolbar();

    // "leaf" is already active, so the pill reads its name rather than the default placeholder.
    fireEvent.click(screen.getByRole("button", { name: /leaf|select subject/ }));
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    // "husk" was set optimistically as active; the refusal must not leave it active.
    expect(useStore.getState().gui.active_subject).toBe("leaf");
  });

  it("adopts the committed registry and keeps the active subject when the save's audit line is lost", async () => {
    seedDataset();
    act(() => {
      useStore.getState().setRegistry({ leaf: {} }, "v1");
      useStore.getState().setActiveSubject("leaf");
    });
    const committed = {
      status: "ok",
      n_subjects: 2,
      subjects_path: "C:/data/subjects.json",
      version: "v2",
    };
    const message = "replace_registry completed and its audit entry could not be written";
    vi.spyOn(subjectsApi, "save").mockRejectedValue(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message, committed },
        409,
        message,
      ),
    );
    answerPrompt("husk");
    renderToolbar();

    // "leaf" is already active, so the pill reads its name rather than the default placeholder.
    fireEvent.click(screen.getByRole("button", { name: /leaf|select subject/ }));
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    // The committed body is adopted as though the save had answered 200, and the optimistic
    // active subject is kept rather than reverted (unlike an ordinary refusal above).
    expect(useStore.getState().registry.version).toBe("v2");
    expect(useStore.getState().gui.active_subject).toBe("husk");
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

function seedSubject(subject: string) {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: { ...s.gui.dataset, subject, image_list: ["img1.jpg"], current_image_index: 0 },
    },
  }));
}

describe("AnnotateToolbar image stepping", () => {
  it("leaves the focused proposal and the selected polygon behind on the image it left", () => {
    useStore.setState((s) => ({
      gui: {
        ...s.gui,
        dataset: { ...s.gui.dataset, image_list: ["img1.jpg", "img2.jpg"], current_image_index: 0 },
      },
      annotateUi: { ...s.annotateUi, focusedProposal: 3 },
      canvas: { ...s.canvas, selectedPolygonIdx: 0 },
    }));
    renderToolbar();

    fireEvent.click(screen.getByRole("button", { name: "Next image" }));

    expect(useStore.getState().gui.dataset.current_image_index).toBe(1);
    expect(useStore.getState().annotateUi.focusedProposal).toBeNull();
    expect(useStore.getState().canvas.selectedPolygonIdx).toBeNull();
  });
});

describe("AnnotateToolbar Complete toggle", () => {
  it("reads checked from the served state, with or without annotations", () => {
    seedSubject("subject_a");
    renderToolbar(null, null, { subjectState: "negative" });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).toBeChecked();
    cleanup();
    renderToolbar(null, null, { subjectState: "partial" });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).not.toBeChecked();
  });

  it("asks the tab to mark or withdraw, and is disabled until the document loads", () => {
    seedSubject("subject_a");
    const onComplete = vi.fn();
    renderToolbar(null, null, { subjectState: "partial", onComplete });
    fireEvent.click(screen.getByRole("checkbox", { name: /Complete/ }));
    expect(onComplete).toHaveBeenCalledWith(true);
    cleanup();
    renderToolbar(null, null, { subjectState: null, onComplete });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).toBeDisabled();
  });

  it("offers the region mark only when the tab supplies one", () => {
    seedSubject("subject_a");
    const onCompleteView = vi.fn();
    renderToolbar(null, null, { subjectState: "partial", onCompleteView });
    fireEvent.click(screen.getByRole("button", { name: "Complete view" }));
    expect(onCompleteView).toHaveBeenCalledTimes(1);
    cleanup();
    renderToolbar(null, null, { subjectState: "partial" });
    expect(screen.queryByRole("button", { name: "Complete view" })).not.toBeInTheDocument();
  });

  it("toggles hiding proposals through the tab", () => {
    const onHideProposals = vi.fn();
    renderToolbar(null, null, { onHideProposals });
    fireEvent.click(screen.getByRole("checkbox", { name: /Hide proposals/ }));
    expect(onHideProposals).toHaveBeenCalledWith(true);
  });
});
