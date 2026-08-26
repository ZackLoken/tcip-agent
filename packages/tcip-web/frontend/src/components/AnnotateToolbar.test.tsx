import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import type { ImageBandsResponse } from "@/api/client";
import { api } from "@/api/client";
import { classesApi, type ImageStatus } from "@/api/classes";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";
import { imagePath } from "@/lib/paths";
import { useStore } from "@/store";
import type { DatasetSelection } from "@/store/types";

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

describe("AnnotateToolbar subject authoring", () => {
  function seedDataset() {
    useStore.setState((s) => ({
      gui: {
        ...s.gui,
        dataset: {
          ...s.gui.dataset,
          project_root: "C:/proj",
          dataset_root: "C:/data",
          date: "2026-01-01",
          annotations_dir: "C:/data/annotations/2026-01-01",
        },
      },
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
    const saveSpy = vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      classes_path: "C:/data/classes.json",
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
    expect(saveSpy.mock.calls[0][1]).toEqual({ leaf: {}, husk: {} });
  });

  it("selects an existing subject rather than resetting its attribute definitions", async () => {
    seedDataset();
    const leafDef = {
      attributes: { stage: { type: "ordinal" as const, values: ["early", "late"] } },
    };
    act(() => useStore.getState().setRegistry({ leaf: leafDef, husk: {} }));
    const saveSpy = vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      classes_path: "C:/data/classes.json",
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
    const saveSpy = vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      classes_path: "C:/data/classes.json",
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
    const saveSpy = vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 2,
      classes_path: "C:/data/classes.json",
      version: "v2",
    });
    answerPrompt("husk");
    renderToolbar();

    openSubjectMenu();
    await act(async () => {
      fireEvent.click(screen.getByText("+ New subject"));
    });

    expect(saveSpy.mock.calls[0][4]).toBe("v1");
    expect(useStore.getState().registry.version).toBe("v2");
  });

  it("reloads the registry from the server when the save is refused", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry({ leaf: {} }, "v1"));
    vi.spyOn(classesApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(classesApi, "load").mockResolvedValue({
      subjects: { leaf: {} },
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
    vi.spyOn(classesApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(classesApi, "load").mockResolvedValue({
      subjects: { leaf: {} },
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

function seedImageDataset(opts: { subject: string; currentStatus?: ImageStatus; stale?: boolean }) {
  const dataset: DatasetSelection = {
    project_root: "C:/proj",
    dataset_root: "C:/data",
    subject: opts.subject,
    date: "2026-01-01",
    image_list: ["img1.jpg"],
    current_image_index: 0,
    images_dir: "C:/data/images/2026-01-01",
    annotations_dir: "C:/data/annotations/2026-01-01",
    predictions_dir: null,
  };
  const byImage: Record<string, ImageStatus> = {};
  if (opts.currentStatus) byImage["img1.jpg"] = opts.currentStatus;
  useStore.setState((s) => ({
    gui: { ...s.gui, dataset },
    canvas: { ...s.canvas, loadedImagePath: imagePath(dataset, "img1.jpg") },
    imageStatus: {
      ...s.imageStatus,
      byImage,
      staleMarks: opts.stale ? ["img1.jpg"] : [],
    },
  }));
}

function setCanvasBoxSubjects(subjects: string[]) {
  useStore.setState((s) => ({
    canvas: {
      ...s.canvas,
      boxes: subjects.map((subject) => ({ x1: 0, y1: 0, x2: 10, y2: 10, subject, attributes: {} })),
    },
  }));
}

describe("AnnotateToolbar Complete toggle, subject-scoped", () => {
  it("writes negative for the dataset's subject when the canvas holds only another subject's shapes", async () => {
    seedImageDataset({ subject: "subject_a" });
    setCanvasBoxSubjects(["subject_b"]);
    const setStatus = vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
    renderToolbar();

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Complete"));
    });

    expect(setStatus).toHaveBeenCalledWith(
      "C:/proj",
      "img1.jpg",
      "negative",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );
  });

  it("writes complete for the dataset's subject when the canvas holds this subject's shapes", async () => {
    seedImageDataset({ subject: "subject_a" });
    setCanvasBoxSubjects(["subject_a"]);
    const setStatus = vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
    renderToolbar();

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Complete"));
    });

    expect(setStatus).toHaveBeenCalledWith(
      "C:/proj",
      "img1.jpg",
      "complete",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );
  });
});

describe("AnnotateToolbar stale re-confirm", () => {
  it("writes negative for the dataset's subject and clears the mark when the canvas holds only another subject's shapes", async () => {
    seedImageDataset({ subject: "subject_a", currentStatus: "complete", stale: true });
    setCanvasBoxSubjects(["subject_b"]);
    const setStatus = vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
    renderToolbar();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Re-confirm" }));
    });

    expect(setStatus).toHaveBeenCalledWith(
      "C:/proj",
      "img1.jpg",
      "negative",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );
    expect(useStore.getState().imageStatus.staleMarks).toEqual([]);
  });

  it("writes complete and clears the mark when the canvas holds this subject's shapes", async () => {
    seedImageDataset({ subject: "subject_a", currentStatus: "complete", stale: true });
    setCanvasBoxSubjects(["subject_a"]);
    const setStatus = vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
    renderToolbar();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Re-confirm" }));
    });

    expect(setStatus).toHaveBeenCalledWith(
      "C:/proj",
      "img1.jpg",
      "complete",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );
    expect(useStore.getState().imageStatus.staleMarks).toEqual([]);
  });

  it("does nothing over a stale image whose labels have not loaded yet (loadedImagePath mismatch)", async () => {
    seedImageDataset({ subject: "subject_a", currentStatus: "complete", stale: true });
    useStore.setState((s) => ({ canvas: { ...s.canvas, loadedImagePath: null } }));
    setCanvasBoxSubjects(["subject_a"]);
    const setStatus = vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
    renderToolbar();

    const button = screen.getByRole("button", { name: "Re-confirm" });
    expect(button).toBeDisabled();
    fireEvent.click(button);

    expect(setStatus).not.toHaveBeenCalled();
    expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]);
  });

  it("restores the stale mark when the re-confirm write fails to persist", async () => {
    seedImageDataset({ subject: "subject_a", currentStatus: "complete", stale: true });
    setCanvasBoxSubjects(["subject_a"]);
    vi.spyOn(classesApi, "setImageStatus").mockRejectedValue(new Error("network error"));
    renderToolbar();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Re-confirm" }));
    });

    expect(useStore.getState().imageStatus.staleMarks).toEqual(["img1.jpg"]);
  });
});
