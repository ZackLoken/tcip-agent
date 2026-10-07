import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { api } from "@/api/client";
import type { LoadedLabels, SaveResult } from "@/api/client";
import { FLAG_MARK, FOCUS_HALO, MATCH_COLORS, MATCH_TYPES, MATCH_WORDS } from "@/lib/symbology";
import type { Flag, ServedProposals } from "@/store/types";
import { subjectsApi, subjectColor } from "@/api/subjects";
import * as canvasSync from "@/lib/canvasSync";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { CUT_MISSES_REFUSAL } from "@/lib/polygonGeometry";
import { StructuredRefusalError } from "@/api/http";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { openTestProject } from "@/test/store";

// Konva needs a real 2D canvas; these tests exercise label I/O ordering and
// store->canvas prop flow, not drawing. Render Konva shapes as inspectable divs
// and CanvasStage as a passthrough so AnnotationShapes' memo behavior is intact.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  Rect: (props: { stroke?: string; dash?: number[]; fill?: string; opacity?: number }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
      data-dash={props.dash ? "true" : undefined}
      data-fill={props.fill}
      data-opacity={props.opacity}
    />
  ),
  Line: (props: { stroke?: string; points?: number[]; opacity?: number }) => (
    <div
      data-testid="k-line"
      data-stroke={props.stroke}
      data-points={props.points?.join(",")}
      data-opacity={props.opacity}
    />
  ),
  Circle: (props: { x?: number; y?: number; fill?: string; radius?: number }) => (
    <div
      data-testid="k-circle"
      data-x={props.x}
      data-y={props.y}
      data-fill={props.fill}
      data-radius={props.radius}
    />
  ),
  Text: (props: { text?: string; fill?: string }) => (
    <div data-testid="k-text" data-text={props.text} data-fill={props.fill} />
  ),
}));
// Pixel handlers are forwarded as plain mouse events (clientX/Y stand in for image-pixel coords;
// the real screen<->image conversion is CanvasStage's own concern) so tests can drive the drawing
// tools without a real Konva stage.
vi.mock("@/components/Canvas/CanvasStage", () => {
  return {
    CanvasStage: (props: {
      children?: React.ReactNode;
      overlay?: React.ReactNode;
      imageUrl?: string | null;
      onPixelDown?: (x: number, y: number, ev: unknown) => void;
      onPixelMove?: (x: number, y: number, ev: unknown) => void;
      onPixelUp?: (x: number, y: number, ev: unknown) => void;
      onPixelClick?: (x: number, y: number, ev: unknown) => void;
      onPixelDoubleClick?: (x: number, y: number, ev: unknown) => void;
      onPixelContextMenu?: (x: number, y: number, ev: unknown) => void;
    }) => {
      return (
        <div
          data-testid="canvas-stage"
          data-canvas-host
          data-image-url={props.imageUrl ?? ""}
          onMouseDown={(e) =>
            props.onPixelDown?.(e.clientX, e.clientY, { evt: { button: e.button } })
          }
          onMouseMove={(e) => props.onPixelMove?.(e.clientX, e.clientY, { evt: { buttons: 1 } })}
          onMouseUp={(e) => props.onPixelUp?.(e.clientX, e.clientY, { evt: {} })}
          onClick={(e) => props.onPixelClick?.(e.clientX, e.clientY, { evt: { button: e.button } })}
          onDoubleClick={(e) => props.onPixelDoubleClick?.(e.clientX, e.clientY, { evt: {} })}
          onContextMenu={(e) =>
            props.onPixelContextMenu?.(e.clientX, e.clientY, {
              evt: { button: 2, preventDefault: () => {} },
            })
          }
        >
          {props.children}
          {props.overlay}
        </div>
      );
    },
  };
});
vi.mock("@/components/AnnotateToolbar", () => ({
  AnnotateToolbar: (props: {
    bandsInfo?: { band_count: number } | null;
    subjectState: string | null;
    onComplete: (next: boolean) => void;
    onCancelSave: () => void;
  }) => (
    <div
      data-testid="toolbar"
      data-band-count={props.bandsInfo?.band_count ?? ""}
      data-subject-state={props.subjectState ?? ""}
    >
      <button onClick={() => props.onComplete(true)}>toolbar-complete</button>
      <button onClick={props.onCancelSave}>toolbar-cancel-save</button>
    </div>
  ),
}));

// The review strip's proposals toggle, the one place proposals are shown or hidden.
const toggleProposals = () => fireEvent.click(screen.getByRole("checkbox", { name: "Proposals" }));

const initialStoreState = useStore.getState();

// Distinct mtime tokens per image so a save's echoed token identifies which image's load it
// came from. Strings, since the ns value exceeds JS's exact-integer range.
const LOAD_MTIME: Record<string, number> = { "img1.jpg": 100, "img2.jpg": 200 };

function labelsFor(imagePath: string) {
  const name = imagePath.split("/").pop() ?? "";
  return {
    image_path: imagePath,
    img_width: 1000,
    img_height: 800,
    boxes: [],
    polygons: [],
    points: [],
    imageAnnotations: [],
    completion: {},
    flags: [] as Flag[],
    base_mtime: String(LOAD_MTIME[name] ?? 1),
  };
}

/** A landed save: the document it answers and the token that names it. */
const saved = (
  base_mtime: string,
  over: Partial<LoadedLabels> = {},
  accepted: Record<string, number> = {},
): SaveResult => ({
  status: "ok",
  labels: { ...labelsFor("C:/data/images/2026-01-01/img1.jpg"), ...over, base_mtime },
  accepted,
});

function setupDataset() {
  openTestProject(
    {
      dataset_root: "C:/data",
      subject: "subject_a",
      date: "2026-01-01",
      image_list: ["img1.jpg", "img2.jpg"],
      current_image_index: 0,
      images_dir: "C:/data/images/2026-01-01",
      bucket: null,
    },
    { mode: "box", active_subject: "subject_a" },
  );
}

function addBox() {
  useStore
    .getState()
    .addBox({ x1: 10, y1: 10, x2: 50, y2: 50, subject: "subject_a", attributes: {} });
}

const flush = () => act(async () => {});

// Save now lives in the (mocked-out) toolbar's Editor shelf; drive it via its Ctrl+S shortcut.
const pressSave = () => fireEvent.keyDown(window, { key: "s", ctrlKey: true });

let loadSpy: MockInstance<typeof api.annotate.load>;
let saveSpy: MockInstance<typeof api.annotate.save>;

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
  setupDataset();
  loadSpy = vi
    .spyOn(api.annotate, "load")
    .mockImplementation((imagePath) => Promise.resolve(labelsFor(imagePath)));
  saveSpy = vi.spyOn(api.annotate, "save").mockResolvedValue(saved("1"));
  vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue({});
  // Default: a standard 3-band RGB image; the band picker's own describe block overrides this
  // per-case to exercise the >3-band path.
  vi.spyOn(api.images, "bands").mockResolvedValue({
    band_count: 3,
    bands: [
      { name: "Red", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
      { name: "Green", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
      { name: "Blue", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
    ],
  });
  // jsdom's own getBoundingClientRect is always zero; the viewport math needs a real host.
  vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
  vi.spyOn(api.images, "viewReads").mockResolvedValue({ reads: [] });
  // One jsdom instance per file, not per test: a recolor left by an earlier test must not leak
  // into a later one's derived-color assertions.
  try {
    localStorage.removeItem("tcip.annotate.subjectColors");
  } catch {
    /* not available in this environment, nothing to clear */
  }
});

afterEach(cleanup);

describe("AnnotateTab session contributions", () => {
  async function leaveFirstImage() {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
  }

  it("posts a named visit with its person and activity, and retires it once accepted", async () => {
    useStore.setState({ user: "jordan" });
    await leaveFirstImage();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[0][0]).toMatchObject({
      image_name: "img1.jpg",
      activity: "review",
      user: "jordan",
    });
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("holds a visit the backend did not accept", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(new Error("backend down"));
    await leaveFirstImage();

    expect(useStore.getState().heldContributions).toHaveLength(1);
  });

  it("never posts a held visit into a project other than its own", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(new Error("backend down"));
    await leaveFirstImage();

    act(() => useStore.setState({ openProject: { id: "other0000000", path: "C:/other" } }));
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(useStore.getState().heldContributions).toHaveLength(1);
  });

  it("retires a visit the backend committed without its line, and never sends it again", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message: "unrecorded", committed: { status: "ok" } },
        409,
        "unrecorded",
      ),
    );
    await leaveFirstImage();
    act(() => useStore.setState({ openProject: { ...useStore.getState().openProject! } }));
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(useStore.getState().heldContributions).toEqual([]);
  });
});

describe("AnnotateTab save/load race", () => {
  it("saves to the loaded path and re-echoes the returned mtime on the next save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    saveSpy.mockResolvedValueOnce(saved("101"));
    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(saveSpy.mock.calls[0][0].base_mtime).toBe("100");
    expect(useStore.getState().canvas.dirty).toBe(false);

    // Second save on the same image must echo the mtime the first save returned.
    act(addBox);
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe("101");
  });

  it("a flush save resolving after navigation does not rewind the loaded path or wipe dirty", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Dirty img1, then navigate. flushLeaving() fires the save without awaiting
    // it; hold its response so the img2 load resolves first (slow label write vs
    // cached read): this interleaving must never corrupt cross-image GT.
    act(addBox);
    let resolveFlushSave!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(
      () => new Promise<SaveResult>((res) => (resolveFlushSave = res)),
    );
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(saveSpy.mock.calls[0][0].base_mtime).toBe("100");

    // Edit img2 while the img1 save is still in flight, then let it resolve late.
    act(addBox);
    await act(async () => {
      resolveFlushSave(saved("150"));
    });

    // The stale result must not markClean() the img2 edits...
    expect(useStore.getState().canvas.dirty).toBe(true);

    // ...and the next save must target img2 with img2's loaded mtime, not
    // img1's file with the stale save's echoed mtime.
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].image_path).toBe("C:/data/images/2026-01-01/img2.jpg");
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe("200");
  });

  it("an earlier image's save settling does not release the hold of the later image's save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    const pending: ((r: SaveResult) => void)[] = [];
    saveSpy.mockImplementation(() => new Promise<SaveResult>((res) => pending.push(res)));
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    act(addBox);
    pressSave();
    await flush();
    expect(pending).toHaveLength(2);
    expect(useStore.getState().canvas.saving?.image).toBe("C:/data/images/2026-01-01/img2.jpg");

    await act(async () => pending[0](saved("150")));
    expect(useStore.getState().canvas.saving?.image).toBe("C:/data/images/2026-01-01/img2.jpg");
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(1);

    await act(async () => pending[1](saved("9", { boxes: [] })));
    expect(useStore.getState().canvas.saving).toBeNull();
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
  });

  describe("a save in flight when the editor unmounts and a new one is still loading", () => {
    async function departing() {
      const first = render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox);
      let answer!: (r: SaveResult) => void;
      let signal!: AbortSignal;
      saveSpy.mockImplementationOnce((_body, s) => {
        signal = s as AbortSignal;
        return new Promise<SaveResult>((res) => (answer = res));
      });
      pressSave();
      await flush();
      first.unmount();
      loadSpy.mockImplementation(() => new Promise(() => {}));
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
      await flush();
      return { answer, signal };
    }

    it("is canceled by the toolbar of the new editor", async () => {
      const { signal } = await departing();
      expect(useStore.getState().canvas.saving).not.toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
      expect(signal.aborted).toBe(true);
      expect(useStore.getState().canvas.saving).toBeNull();
    });

    it("lands without adopting anything into the canvas the new editor is loading", async () => {
      const { answer } = await departing();
      const held = useStore.getState().canvas.boxes;
      await act(async () => answer(saved("150", { boxes: [] })));
      expect(useStore.getState().canvas.boxes).toBe(held);
      expect(useStore.getState().canvas.saving).toBeNull();
    });
  });

  it("canceling reports nothing over an image loaded after the save began", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    let reject!: (e: unknown) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((_res, rej) => (reject = rej)));
    pressSave();
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
    act(() => {
      const st = useStore.getState();
      st.patchGui({ dataset: { ...st.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    await act(async () => reject(new DOMException("aborted", "AbortError")));
    expect(screen.queryByText(/Save canceled/)).not.toBeInTheDocument();
  });

  describe("a save from an editor that has since unmounted", () => {
    async function remountedWithEdit() {
      const first = render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox);
      const pending: ((r: SaveResult) => void)[] = [];
      saveSpy.mockImplementation(() => new Promise<SaveResult>((res) => pending.push(res)));
      pressSave();
      await flush();
      first.unmount();
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
      await flush();
      act(addBox);
      return pending;
    }

    it("does not overwrite what the remounted editor has since changed", async () => {
      const pending = await remountedWithEdit();
      const edited = useStore.getState().canvas.boxes;
      await act(async () => pending[0](saved("150", { boxes: [] })));
      expect(useStore.getState().canvas.boxes).toBe(edited);
      expect(useStore.getState().canvas.dirty).toBe(true);
    });

    it("does not release the hold of the remounted editor's own save", async () => {
      const pending = await remountedWithEdit();
      pressSave();
      await flush();
      expect(pending).toHaveLength(2);
      await act(async () => pending[0](saved("150")));
      expect(useStore.getState().canvas.saving).not.toBeNull();
      act(addBox);
      expect(useStore.getState().canvas.boxes).toHaveLength(1);
    });
  });

  it("canceling a save in flight releases the canvas, keeps the edits and drops a late answer", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    let late!: (r: SaveResult) => void;
    let signal: AbortSignal | undefined;
    saveSpy.mockImplementationOnce((_body, s) => {
      signal = s;
      return new Promise<SaveResult>((res) => (late = res));
    });
    pressSave();
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
    await flush();

    expect(signal?.aborted).toBe(true);
    expect(useStore.getState().canvas.saving).toBeNull();
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
    await act(async () => late(saved("150", { boxes: [] })));
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
    expect(useStore.getState().canvas.dirty).toBe(true);
  });

  it("a save whose request fails releases the canvas", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    saveSpy.mockRejectedValueOnce(new Error("network down"));
    pressSave();
    await flush();
    expect(useStore.getState().canvas.saving).toBeNull();
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
  });

  it("a stale conflict for a since-left image does not show the Reload banner over the new image", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Interactive Ctrl+S save held in flight, then the user navigates away.
    act(addBox);
    const pending: ((r: SaveResult) => void)[] = [];
    saveSpy.mockImplementation(() => new Promise<SaveResult>((res) => pending.push(res)));
    pressSave();
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();

    // Both the interactive save and the navigation flush save 409 late.
    await act(async () => {
      for (const res of pending) res({ status: "conflict" });
    });
    expect(screen.queryByText("Reload")).not.toBeInTheDocument();
    expect(screen.queryByText(/changed elsewhere/)).not.toBeInTheDocument();
  });

  it("names the person the app is set to on the save, and sends no provenance of its own", async () => {
    act(() => useStore.getState().setUser("breeder"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    const body = saveSpy.mock.calls[0][0];
    expect(body.user).toBe("breeder");
    for (const a of body.annotations) {
      expect(Object.keys(a).filter((k) => /_(by|at)$/.test(k))).toEqual([]);
    }
  });
});

describe("AnnotateTab subject rendering", () => {
  it("renders a box with the subject-derived color, named on selection", async () => {
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({ gui: { ...s.gui, active_subject: "tip" } }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() =>
      useStore
        .getState()
        .addBox({ x1: 10, y1: 10, x2: 50, y2: 50, subject: "tip", attributes: {} }),
    );
    // Color is GUI-local (name-derived); the label is the subject name, no integer id, and
    // appears on selection (labels are hover/selection-only; the legend is the standing key).
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("tip"));
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "tip");
  });

  it("box mode draws an active-subject polygon's read-only derived box (no handles), never a stored box", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    const poly = {
      rings: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      ] as [number, number][][],
      subject: "subject_a",
      attributes: {},
    };
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), polygons: [poly] }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Box mode (setupDataset). The polygon shows only its derived box: a single Rect with no corner
    // handles (handles are extra Rects), never in canvas.boxes, so unsaveable; read-only is structural.
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).not.toHaveAttribute("data-dash");
    expect(rects[0]).toHaveAttribute("data-stroke", subjectColor("subject_a"));
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });

  it("a tool-authored polygon's derived box draws in the polygon's own line style, unlabeled", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    const poly = {
      rings: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      ] as [number, number][][],
      subject: "subject_a",
      attributes: {},
      authorship: "tool",
    };
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), polygons: [poly] }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Line style says whose shape it is: the derived box is the tool's polygon's, so it draws
    // dotted like the polygon; it is no item of its own, so nothing labels it.
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).toHaveAttribute("data-dash", "true");
    expect(screen.queryAllByTestId("k-text")).toHaveLength(0);
  });

  it("box mode draws a person's editable box solid and a person's polygon's derived box solid too", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).not.toHaveAttribute("data-dash");
  });

  it("point mode draws a placed point as its reticle mark, never a box or a closed outline", async () => {
    // A point asserts a location and no extent: rendering it as a box (or letting it render as a
    // degenerate polygon) would show the annotator an extent the annotation does not claim.
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "point" as const, active_subject: "tip" },
    }));
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        points: [{ x: 100, y: 200, subject: "tip", attributes: {} }],
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // The core sits exactly on the coordinate, in the subject's color...
    const core = screen.getByTestId("k-circle");
    expect(core).toHaveAttribute("data-x", "100");
    expect(core).toHaveAttribute("data-y", "200");
    expect(core).toHaveAttribute("data-fill", subjectColor("tip"));
    // ...with four radial ticks (the mark that reads as a location, not a tiny shape)...
    expect(screen.getAllByTestId("k-line")).toHaveLength(4);
    // ...and no box of any kind. (Naming on selection/hover is covered by the label tests.)
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);
  });

  it("polygon mode draws every ring of an occlusion-split shape, labeled once", async () => {
    // An organ behind a branch loads as one annotation with two disjoint regions. Drawing only
    // the first would show the breeder part of the object and let them confirm it as the whole.
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "polygon" as const, active_subject: "tip" },
    }));
    const rings: [number, number][][] = [
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      [
        [40, 40],
        [60, 40],
        [60, 60],
      ],
    ];
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        polygons: [{ rings, subject: "tip", attributes: {} }],
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const lines = screen.getAllByTestId("k-line");
    expect(lines).toHaveLength(2);
    expect(lines.map((l) => l.getAttribute("data-points"))).toEqual([
      "0,0,10,0,10,10",
      "40,40,60,40,60,60",
    ]);
    // Both parts wear the annotation's own color...
    expect(lines.every((l) => l.getAttribute("data-stroke") === subjectColor("tip"))).toBe(true);
    // ...and selecting it names the annotation once, not once per ring (HaloLabel = halo + fill).
    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 8, clientY: 5 });
    expect(
      screen.getAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === "tip"),
    ).toHaveLength(2);
  });
});

describe("AnnotateTab authorship symbology", () => {
  it("a tool's box and an unattributed one draw dotted, a person's and an accepted tool's solid; focus names the tool's", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        boxes: [
          {
            x1: 10,
            y1: 10,
            x2: 50,
            y2: 50,
            subject: "subject_a",
            attributes: {},
            authorship: "tool",
          },
          {
            x1: 60,
            y1: 10,
            x2: 90,
            y2: 50,
            subject: "subject_a",
            attributes: {},
            authorship: "person",
          },
          {
            x1: 10,
            y1: 60,
            x2: 50,
            y2: 90,
            subject: "subject_a",
            attributes: {},
            authorship: "tool_accepted",
          },
          {
            x1: 60,
            y1: 60,
            x2: 90,
            y2: 90,
            subject: "subject_a",
            attributes: {},
            authorship: "unattributed",
          },
        ],
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(4);
    expect(rects[0]).toHaveAttribute("data-dash", "true"); // tool: dotted
    expect(rects[1]).not.toHaveAttribute("data-dash"); // person: solid
    expect(rects[2]).not.toHaveAttribute("data-dash"); // tool_accepted: solid
    expect(rects[3]).toHaveAttribute("data-dash", "true"); // unattributed: no person behind it

    // Focusing the tool box (a press inside it) names it with the authorship it draws with.
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 30,
      clientY: 30,
      button: 0,
    });
    await flush();
    expect(
      screen
        .getAllByTestId("k-text")
        .some((t) => t.getAttribute("data-text") === "subject_a, tool"),
    ).toBe(true);
  });

  it("the legend states the dotted stroke means no person has stood behind it yet", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.getByText("Dotted: no person has stood behind it yet")).toBeInTheDocument();
  });
});

describe("AnnotateTab point tool", () => {
  const stage = () => screen.getByTestId("canvas-stage");
  // The move handler is rAF-throttled; jsdom fires rAF off a timer, so let one frame land.
  const frame = () => act(async () => void (await new Promise((r) => setTimeout(r, 25))));

  async function mountPointMode(points: { x: number; y: number }[] = []) {
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "point" as const, active_subject: "tip" },
    }));
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        points: points.map((p) => ({ ...p, subject: "tip", attributes: {} })),
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
  }

  it("a single click commits one point at the clicked coordinate", async () => {
    await mountPointMode();
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    await flush();

    // One click is the whole gesture: no drag-out, no second click to close.
    expect(useStore.getState().canvas.points).toEqual([
      { x: 120, y: 340, subject: "tip", attributes: {} },
    ]);
    expect(useStore.getState().canvas.dirty).toBe(true);
  });

  it("refuses to place a point with no subject selected, and says so once", async () => {
    await mountPointMode();
    act(() => useStore.getState().setActiveSubject(null));
    fireEvent.click(stage(), { clientX: 10, clientY: 10 });
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(0);
    expect(useStore.getState().toasts.at(-1)?.message).toMatch(/Select a subject before drawing/);
  });

  it("a press-drag repositions a placed point, with one undo snapshot for the whole drag", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 102, clientY: 101, button: 0 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    fireEvent.mouseMove(stage(), { clientX: 300, clientY: 250 });
    await frame();
    fireEvent.mouseMove(stage(), { clientX: 310, clientY: 260 });
    await frame();
    fireEvent.mouseUp(stage(), { clientX: 310, clientY: 260 });
    await flush();

    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 310, y: 260 });
    // One snapshot for the gesture: a per-move push would evict the whole 30-entry history.
    expect(useStore.getState().canvas.undoStack).toHaveLength(1);
    act(() => useStore.getState().undo());
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 100, y: 100 });
  });

  it("a drag in progress when a save starts ends there and does not resume on the adopted document", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: 200, clientY: 200 });
    await frame();
    let answer!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((res) => (answer = res)));
    pressSave();
    await flush();
    await act(async () =>
      answer(
        saved("5", { points: [{ x: 200, y: 200, subject: "tip", attributes: {}, index: 0 }] }),
      ),
    );

    fireEvent.mouseMove(stage(), { clientX: 400, clientY: 400 });
    await frame();
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 200, y: 200 });
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);
  });

  describe("with the canvas held by a save", () => {
    const hold = () => act(() => void useStore.getState().holdForSave("x"));
    const unhold = () =>
      act(() => {
        const held = useStore.getState().canvas.saving;
        if (held) useStore.getState().releaseSave(held);
      });

    it("a click leaves the focus where it was", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      act(() => useStore.getState().setFocus({ kind: "point", index: 0 }));
      hold();
      fireEvent.click(stage(), { clientX: 500, clientY: 500 });
      expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    });

    it("a right-click leaves the focus where it was", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      act(() => useStore.getState().setFocus({ kind: "point", index: 0 }));
      hold();
      fireEvent.contextMenu(stage(), { clientX: 500, clientY: 500 });
      expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    });

    it("a move queued before the hold does nothing when its frame fires", async () => {
      useStore.getState().setRegistry({ tip: {} });
      useStore.setState((s) => ({
        gui: { ...s.gui, mode: "polygon" as const, active_subject: "tip" },
      }));
      loadSpy.mockImplementation((imagePath) =>
        Promise.resolve({
          ...labelsFor(imagePath),
          polygons: [
            {
              rings: [
                [
                  [100, 100],
                  [200, 100],
                  [200, 200],
                ],
              ] as [number, number][][],
              subject: "tip",
              attributes: {},
            },
          ],
        }),
      );
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      fireEvent.mouseMove(stage(), { clientX: 180, clientY: 150 });
      hold();
      await frame();
      expect(useStore.getState().annotateUi.hoveredPolygonIdx).toBeNull();
    });

    it("Escape ends a point drag in progress, which later movement does not resume", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
      fireEvent.keyDown(window, { key: "Escape" });
      fireEvent.mouseMove(stage(), { clientX: 300, clientY: 300 });
      await frame();
      expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 100, y: 100 });
    });

    it("a double-click leaves a freehand stream running", async () => {
      useStore.getState().setRegistry({ tip: {} });
      useStore.setState((s) => ({
        gui: { ...s.gui, mode: "polygon" as const, active_subject: "tip" },
      }));
      useStore.getState().setStream(true);
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      fireEvent.click(stage(), { clientX: 300, clientY: 300 });
      expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
      hold();
      fireEvent.doubleClick(stage(), { clientX: 300, clientY: 300 });
      unhold();
      fireEvent.mouseMove(stage(), { clientX: 500, clientY: 500 });
      await frame();
      expect(useStore.getState().canvas.currentPolygon).toHaveLength(2);
    });

    it("a release leaves the drag in progress for the save that ends it", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
      hold();
      fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
      unhold();
      fireEvent.mouseMove(stage(), { clientX: 300, clientY: 300 });
      await frame();
      expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 300, y: 300 });
    });
  });

  it("the click that ends a drag does not place a second point on top of the moved one", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: 150, clientY: 150 });
    await frame();
    fireEvent.mouseUp(stage(), { clientX: 150, clientY: 150 });
    fireEvent.click(stage(), { clientX: 150, clientY: 150 }); // the release's trailing click
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(1);
  });

  it("right-click removes the point under the cursor and leaves a neighbor alone", async () => {
    await mountPointMode([
      { x: 100, y: 100 },
      { x: 400, y: 400 },
    ]);
    fireEvent.contextMenu(stage(), { clientX: 103, clientY: 100 });
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(1);
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 400, y: 400 });
  });

  it("Delete removes the selected point", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });

    fireEvent.keyDown(window, { key: "Delete" });
    await flush();
    expect(useStore.getState().canvas.points).toHaveLength(0);
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("saves a placed point as a `point` payload the save route can author", async () => {
    await mountPointMode();
    fireEvent.click(stage(), { clientX: 12, clientY: 34 });
    await flush();
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      { subject: "tip", point: [12, 34], attributes: {} },
    ]);
  });

  it("m cycles Box -> Polygon -> Point -> Box", async () => {
    await mountPointMode();
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("box");
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("polygon");
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("point");
  });
});

describe("AnnotateTab AttributePanel", () => {
  it("collapses to a pill with no active subject, nothing selected, no image-level ratings", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    act(() => useStore.getState().setActiveSubject(null)); // beforeEach's setupDataset sets one
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.queryByText("Select a shape to set its attributes.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Attributes" })).toBeInTheDocument();
  });

  it("an active subject opens the panel even with nothing selected, showing the subject block", async () => {
    useStore.getState().setRegistry({ subject_a: {} }); // beforeEach's setupDataset already made it active
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.getByText("Select a shape to set its attributes.")).toBeInTheDocument();
    expect(screen.getByText("Attributes for subject_a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "+ Attribute" })).toBeInTheDocument();
  });

  it("reopens on its own when a shape gets selected, and can be closed manually", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    // The subject registry also has a "subject_a" entry in the (always-mounted, hover-revealed)
    // legend, so assert on the panel's own close button rather than ambiguous shared text.
    expect(screen.getByRole("button", { name: "Close attributes panel" })).toBeInTheDocument();
    expect(useStore.getState().canvas.focus?.kind).toBe("box"); // sanity: a box, not a polygon, is focused

    fireEvent.click(screen.getByRole("button", { name: "Close attributes panel" }));
    expect(
      screen.queryByRole("button", { name: "Close attributes panel" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Attributes" })).toBeInTheDocument();
  });
});

describe("AnnotateTab AttributePanel authoring", () => {
  async function openPanelOnASelectedBox() {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
  }

  async function declareAttribute() {
    fireEvent.click(screen.getByRole("button", { name: "+ Attribute" }));
    fireEvent.change(screen.getByPlaceholderText("attribute name"), {
      target: { value: "size" },
    });
    fireEvent.change(screen.getByPlaceholderText(/one value per line/), {
      target: { value: "small\nlarge" },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Add" }));
    });
  }

  it("posts the grown registry through subjectsApi.save with the loaded version and installs it", async () => {
    await openPanelOnASelectedBox();
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      subjects_path: "C:/data/subjects.json",
      version: "v2",
    });

    await declareAttribute();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0]).toEqual({
      subject_a: { attributes: { size: { type: "categorical", values: ["small", "large"] } } },
    });
    expect(saveSpy.mock.calls[0][2]).toBeNull(); // no version was ever loaded in this test
    expect(useStore.getState().registry.version).toBe("v2");
    expect(useStore.getState().registry.subjects.subject_a.attributes?.size.values).toEqual([
      "small",
      "large",
    ]);
  });

  it("reloads the registry from the server when the save is refused, same as the subject add", async () => {
    await openPanelOnASelectedBox();
    vi.spyOn(subjectsApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(subjectsApi, "load").mockResolvedValue({
      subjects: { subject_a: {} },
      discovered: [],
      version: "v3",
      unreadable: [],
    });

    await declareAttribute();

    expect(useStore.getState().registry.subjects).toEqual({ subject_a: {} });
    expect(useStore.getState().registry.version).toBe("v3");
  });

  it("counts the active subject's shapes carrying no value for each declared attribute", async () => {
    await openPanelOnASelectedBox();
    vi.spyOn(subjectsApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      subjects_path: "C:/data/subjects.json",
      version: "v2",
    });

    await declareAttribute();

    // The one drawn box carries no value for the attribute just declared.
    expect(
      screen.getByText("1 of 1 subject_a shapes on this image carry no size value."),
    ).toBeInTheDocument();
  });

  it("refuses a name the subject already declares, inline, and posts nothing", async () => {
    await openPanelOnASelectedBox();
    const saveSpy = vi.spyOn(subjectsApi, "save");
    act(() => {
      useStore
        .getState()
        .setRegistry(
          { subject_a: { attributes: { size: { type: "categorical", values: ["small"] } } } },
          "v1",
        );
    });

    fireEvent.click(screen.getByRole("button", { name: "+ Attribute" }));
    fireEvent.change(screen.getByPlaceholderText("attribute name"), {
      target: { value: "size" },
    });
    fireEvent.change(screen.getByPlaceholderText(/one value per line/), {
      target: { value: "large" },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Add" }));
    });

    expect(
      screen.getByText("size is already declared; add values to it with + value."),
    ).toBeInTheDocument();
    expect(saveSpy).not.toHaveBeenCalled();
  });
});

describe("AnnotateTab AttributePanel accessible names", () => {
  it("names the value select by the attribute alone, the + value button named and outside it", async () => {
    useStore.getState().setRegistry({
      subject_a: { attributes: { ripeness: { type: "ordinal", values: ["green", "ripe"] } } },
    });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });

    expect(screen.getByRole("combobox", { name: "ripeness" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add a ripeness value" })).toBeInTheDocument();
  });

  it("names the new-attribute type select", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "+ Attribute" }));
    expect(screen.getByRole("combobox", { name: "attribute type" })).toBeInTheDocument();
  });
});

describe("AnnotateTab legend keyboard access", () => {
  it("opens the legend by keyboard and reaches a subject row on the following Tab", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const legendButton = screen.getByRole("button", { name: "Legend" });
    expect(legendButton).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(legendButton);
    expect(legendButton).toHaveAttribute("aria-expanded", "true");

    const subjectRow = screen.getByTitle("Change subject_a's color (this browser only)");
    expect(subjectRow).toBeInTheDocument();
    // DOM order: the panel follows the button, so a forward Tab from it reaches the row.
    expect(
      legendButton.compareDocumentPosition(subjectRow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("opens the color picker as a labeled dialog with a named hex input", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Legend" }));
    fireEvent.click(screen.getByTitle("Change subject_a's color (this browser only)"));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName(
      "subject_a's color (this browser only; derives from the name elsewhere)",
    );
    expect(screen.getByRole("textbox", { name: "hex color" })).toBeInTheDocument();
  });
});

describe("AnnotateTab ioError banner", () => {
  it("can be dismissed manually, independent of the conditional Reload button", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    saveSpy.mockResolvedValueOnce({ status: "conflict" } as SaveResult);
    pressSave();
    await flush();

    expect(screen.getByText(/changed elsewhere/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(/changed elsewhere/)).not.toBeInTheDocument();
    expect(screen.queryByText("Reload")).not.toBeInTheDocument();
  });

  it("sits in the row below the Overview pill, never on top of it", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    saveSpy.mockResolvedValueOnce({ status: "conflict" } as SaveResult);
    pressSave();
    await flush();

    const banner = screen.getByText(/changed elsewhere/).closest("div");
    const overview = screen.getByRole("button", { name: "Overview" });
    expect(banner).toHaveClass("top-12");
    expect(overview).toHaveClass("top-3");
  });
});

describe("AnnotateTab labels-written conflict sentence", () => {
  it("names the declared harness that wrote the current image's labels", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    act(() => {
      useStore
        .getState()
        .pushAgentActivity(
          "annotate",
          "labels_written",
          { image_path: "C:/data/images/2026-01-01/img1.jpg" },
          "claude-code 2.1.238",
        );
    });

    expect(
      screen.getByText(
        "claude-code 2.1.238 just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.",
      ),
    ).toBeInTheDocument();
  });

  it("names a process when the write declared no actor", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    act(() => {
      useStore
        .getState()
        .pushAgentActivity(
          "annotate",
          "labels_written",
          { image_path: "C:/data/images/2026-01-01/img1.jpg" },
          null,
        );
    });

    expect(
      screen.getByText(
        "A process just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.",
      ),
    ).toBeInTheDocument();
  });
});

describe("AnnotateTab legend", () => {
  it("lists the subjects until proposals are shown, then the match types", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    const proposalsSpy = withBucket();
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    await waitFor(() => expect(proposalsSpy).toHaveBeenCalled());
    await flush();

    // The legend button is hover-revealed, so query its content directly.
    const legend = within(document.getElementById("annotate-legend-panel")!);
    for (const match of MATCH_TYPES) {
      expect(legend.getByText(MATCH_WORDS[match])).toBeInTheDocument();
    }
    expect(legend.queryByRole("button", { name: "subject_a" })).not.toBeInTheDocument();

    toggleProposals();
    await flush();
    expect(legend.getByRole("button", { name: "subject_a" })).toBeInTheDocument();
    expect(legend.queryByText(MATCH_WORDS.matched)).not.toBeInTheDocument();
  });

  it("a recolored subject's box stroke and the pushed canvas_meta swatch both follow", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("subject_a"));

    fireEvent.click(screen.getByRole("button", { name: "subject_a" }));
    const hexInput = screen.getByRole("textbox", { name: "hex color" });
    fireEvent.change(hexInput, { target: { value: "#123456" } });
    fireEvent.keyDown(hexInput, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "OK" }));
    await flush();

    expect(subjectColor("subject_a")).toBe("#123456");
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", "#123456");

    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    act(() => notifyCanvasStateRequest());
    await flush();
    const pushed = pushSpy.mock.calls.at(-1)?.[0];
    expect(pushed?.classes).toEqual(
      expect.arrayContaining([{ name: "subject_a", color: "#123456" }]),
    );
  });
});

describe("AnnotateTab band-composite wiring", () => {
  it("passes the standard 3-band dataset's own bandsInfo down to the toolbar untouched", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-count", "3");
    // No bands/stretch param for a plain RGB dataset: the canvas URL is unaffected.
    const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
    expect(url).not.toContain("bands=");
    expect(url).not.toContain("stretch=");
  });

  it("carries a >3-band dataset's picked bands/stretch into the canvas image URL", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue({
      band_count: 4,
      bands: [
        { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
        { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
        { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
        { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
      ],
    });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    await waitFor(() =>
      expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-count", "4"),
    );
    const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
    // Defaulted to the first three reported bands (Blue, Green, Red) and the Min-Max stretch.
    expect(url).toContain(`bands=${encodeURIComponent("Blue,Green,Red")}`);
    expect(url).toContain("stretch=minmax");
  });
});

// The CanvasStage mock forwards clientX/Y as image-pixel coords, so these drive the real
// pointer state machine against a 1000x800 image.
const POLY_A = {
  rings: [
    [
      [10, 10],
      [200, 10],
      [200, 200],
      [10, 200],
    ] as [number, number][],
  ],
  subject: "subject_a",
  attributes: {},
};
const POLY_B = {
  rings: [
    [
      [300, 300],
      [400, 300],
      [400, 400],
      [300, 400],
    ] as [number, number][],
  ],
  subject: "subject_a",
  attributes: {},
};

function seedPolygons(polygons: (typeof POLY_A)[]) {
  useStore.getState().loadLabelsIntoCanvas({
    image_path: "C:/data/images/2026-01-01/img1.jpg",
    img_width: 1000,
    img_height: 800,
    boxes: [],
    polygons,
    points: [],
    imageAnnotations: [],
    completion: {},
    flags: [],
  });
}

const nextFrame = () =>
  act(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));

async function renderPolygonCanvas(polygons: (typeof POLY_A)[] = [POLY_A, POLY_B]) {
  render(<AnnotateTab />);
  await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
  await flush();
  act(() => {
    useStore.getState().setMode("polygon");
    seedPolygons(polygons);
  });
  return screen.getByTestId("canvas-stage");
}

describe("click-selection parity across the Snap/Stream toggles", () => {
  it("a click on a polygon selects it with Stream on, never starts a new one", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setStream(true));
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("a click on another polygon switches the selection with Stream on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
    });
    fireEvent.click(stage, { clientX: 350, clientY: 350 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 1 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("with Stream on, empty space deselects first and a later click still streams", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
    });
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.focus).toBeNull();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toEqual([[600, 600]]);
  });

  it("a click on a polygon selects it with Snap on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setSnap(true));
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });
});

describe("Cut tool arming", () => {
  it("x arms and disarms the cut flag in polygon mode", async () => {
    await renderPolygonCanvas();
    expect(useStore.getState().annotateUi.cut).toBe(false);
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(true);
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  it("x does nothing outside polygon mode", async () => {
    await renderPolygonCanvas();
    act(() => useStore.getState().setMode("box"));
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  it("arming and disarming each schedule a push carrying the current cut_armed value", async () => {
    await renderPolygonCanvas();
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    pushSpy.mockClear();

    vi.useFakeTimers();
    try {
      act(() => useStore.getState().setCut(true));
      act(() => vi.advanceTimersByTime(1600));
      expect(pushSpy.mock.calls.at(-1)?.[0].cut_armed).toBe(true);

      pushSpy.mockClear();
      act(() => useStore.getState().setCut(false));
      act(() => vi.advanceTimersByTime(1600));
      expect(pushSpy.mock.calls.at(-1)?.[0].cut_armed).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Cut gesture", () => {
  function armOnPolyA(): void {
    act(() => {
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
      useStore.getState().setCut(true);
    });
  }

  it("with a polygon selected, the two clicks produce two polygons and the flag stays set", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2); // only the start is pending so far
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3);
    expect(useStore.getState().annotateUi.cut).toBe(true);
  });

  it("a first click near the selected outline places the start and inserts no vertex", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    // 2px outside POLY_A's top edge: within the edge-insert threshold, so without the cut guard
    // onDown would splice a new vertex into the ring here instead of arming the start.
    fireEvent.mouseDown(stage, { clientX: 100, clientY: 8, button: 0 });
    fireEvent.click(stage, { clientX: 100, clientY: 8, button: 0 });
    expect(useStore.getState().canvas.polygons[0].rings[0]).toHaveLength(4);
    fireEvent.click(stage, { clientX: 100, clientY: 250, button: 0 }); // the start was real
    expect(useStore.getState().canvas.polygons).toHaveLength(3);
  });

  it("with none selected, the first click toasts", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 500, clientY: 500, button: 0 });
    expect(useStore.getState().toasts.at(-1)?.message).toBe(
      "Select a polygon to cut, then click two points on either side of it.",
    );
  });

  it("Escape clears the start and the flag", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
    // and the start really cleared: re-arming and clicking once is a fresh first click, not a cut
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("right-click clears the start alone and deletes nothing", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.contextMenu(stage, { clientX: 105, clientY: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().annotateUi.cut).toBe(true);
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("an image change clears the start and the flag", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => {
      useStore
        .getState()
        .patchGui({ dataset: { ...useStore.getState().gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  const POLYGON_CHANGED_SENTENCE =
    "The polygon changed since the first click; the cut was canceled. Select it and place both " +
    "points again.";

  it("a selection change between the clicks refuses with the polygon-changed sentence", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setFocus({ kind: "polygon", index: 1 }));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().toasts.at(-1)?.message).toBe(POLYGON_CHANGED_SENTENCE);
  });

  it("an attribute edit between the clicks does not cancel the cut", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => {
      const poly = useStore.getState().canvas.polygons[0];
      useStore.getState().updatePolygon(0, { ...poly, attributes: { health: "good" } });
    });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // the cut still landed
  });

  it("a vertex edit between the clicks refuses with the polygon-changed sentence", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().dragVertex(0, 0, 0, [11, 11]));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2); // unchanged: the cut was canceled
    expect(useStore.getState().toasts.at(-1)?.message).toBe(POLYGON_CHANGED_SENTENCE);
  });

  it("a refused cut leaves the polygons unchanged and the flag set", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.click(stage, { clientX: 105, clientY: 5, button: 0 }); // never reaches the outline
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().canvas.polygons[0].rings[0]).toHaveLength(4);
    expect(useStore.getState().annotateUi.cut).toBe(true);
    expect(useStore.getState().toasts.at(-1)?.message).toBe(CUT_MISSES_REFUSAL);
  });

  it("after cutting one polygon, a click inside another selects it with no start placed", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // POLY_A split, POLY_B untouched
    expect(useStore.getState().annotateUi.cut).toBe(true);

    // POLY_B now sits at index 2 (POLY_A's two pieces occupy 0 and 1).
    fireEvent.click(stage, { clientX: 350, clientY: 350, button: 0 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 2 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // the click authored nothing

    fireEvent.click(stage, { clientX: 305, clientY: 250, button: 0 });
    fireEvent.click(stage, { clientX: 305, clientY: 450, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(4); // POLY_B cut too
  });

  it("disarming via setCut clears a pending start (the toolbar button's own action)", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setCut(false));
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("disarming via x clears a pending start", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.keyDown(window, { key: "x" });
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("a mode change away from polygon clears the flag and the pending start", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setMode("box"));
    expect(useStore.getState().annotateUi.cut).toBe(false);
    act(() => {
      useStore.getState().setMode("polygon");
      useStore.getState().setCut(true);
    });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });
});

describe("clicks outside the image extent are inert", () => {
  it("box mode: a press-drag from outside authors no box", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const stage = screen.getByTestId("canvas-stage");
    fireEvent.mouseDown(stage, { clientX: 1200, clientY: -50 });
    fireEvent.mouseMove(stage, { clientX: -100, clientY: 900 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: -100, clientY: 900 });
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
  });

  it("point mode: an outside click places no point", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => useStore.getState().setMode("point"));
    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.points).toHaveLength(0);
  });

  it("polygon mode: an outside click starts no polygon", async () => {
    const stage = await renderPolygonCanvas([]);
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("polygon mode: an outside click adds no vertex to a polygon in progress", async () => {
    const stage = await renderPolygonCanvas([]);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
  });

  it("Escape drops the focus", async () => {
    const stage = await renderPolygonCanvas();
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });

    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("an outside click does not drop an existing selection", async () => {
    const stage = await renderPolygonCanvas();
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    fireEvent.click(stage, { clientX: 1050, clientY: 400 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
  });

  it("with Stream on, an outside click starts no stream and later moves lay nothing", async () => {
    const stage = await renderPolygonCanvas([]);
    act(() => useStore.getState().setStream(true));
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
    fireEvent.mouseMove(stage, { clientX: 500, clientY: 400 });
    await nextFrame();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });
});

describe("AnnotateTab authoring writes what the annotator meant", () => {
  it("bounds a new point by the image's own height on a taller-than-wide frame", async () => {
    // Only a portrait frame separates the bounds: a y past the width but inside the height.
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "point" as const, active_subject: "tip" },
    }));
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), img_width: 600, img_height: 900 }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 420, clientY: 730 });
    await flush();

    expect(useStore.getState().canvas.points).toEqual([
      { x: 420, y: 730, subject: "tip", attributes: {} },
    ]);
  });

  it("addresses the save by the image alone, naming no place its labels are kept", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(Object.keys(saveSpy.mock.calls[0][0])).not.toContain("label_path");
  });

  it("commits a drawn box under the subject the drag started on, not the one active at release", async () => {
    useStore.getState().setRegistry({ subject_a: {}, subject_b: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => useStore.getState().setActiveSubject("subject_a"));
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 120, clientY: 60, button: 0 });
    act(() => useStore.getState().setActiveSubject("subject_b"));
    fireEvent.mouseMove(stage, { clientX: 470, clientY: 330 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 470, clientY: 330 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 120, y1: 60, x2: 470, y2: 330, subject: "subject_a", attributes: {} },
    ]);
  });

  it("keeps a freshly drawn box exactly at the minimum side", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 103, clientY: 103 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 103, clientY: 103 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 100, y1: 100, x2: 103, y2: 103, subject: "subject_a", attributes: {} },
    ]);
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("refuses a freshly drawn box smaller than the minimum, with a toast", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 102, clientY: 102 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 102, clientY: 102 });
    await flush();

    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().toasts.at(-1)?.message).toMatch(/too small/i);
  });

  it("undoes a resize that shrinks a box below the minimum, with a toast", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox); // {x1:10, y1:10, x2:50, y2:50}
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 30, clientY: 30, button: 0 }); // press inside selects
    fireEvent.mouseUp(stage, { clientX: 30, clientY: 30 });
    await flush();

    fireEvent.mouseDown(stage, { clientX: 50, clientY: 50, button: 0 }); // bottom-right corner
    fireEvent.mouseMove(stage, { clientX: 11, clientY: 11 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 11, clientY: 11 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 10, y1: 10, x2: 50, y2: 50, subject: "subject_a", attributes: {} },
    ]);
    expect(useStore.getState().toasts.at(-1)?.message).toMatch(/too small/i);
  });

  describe("a save pressed while a box resize is still being dragged", () => {
    async function dragCornerTo(x: number, y: number) {
      useStore.getState().setRegistry({ subject_a: {} });
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox); // {x1:10, y1:10, x2:50, y2:50}
      const stage = screen.getByTestId("canvas-stage");
      fireEvent.mouseDown(stage, { clientX: 30, clientY: 30, button: 0 });
      fireEvent.mouseUp(stage, { clientX: 30, clientY: 30 });
      await flush();
      fireEvent.mouseDown(stage, { clientX: 50, clientY: 50, button: 0 });
      fireEvent.mouseMove(stage, { clientX: x, clientY: y });
      await nextFrame();
    }

    it("undoes one below the minimum side and saves the box as it was", async () => {
      await dragCornerTo(11, 11);
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      expect(useStore.getState().toasts.at(-1)?.message).toMatch(/too small/i);
    });

    it("reverts the resize, not a vertex of a polygon being drawn, and leaves the draft whole", async () => {
      await dragCornerTo(11, 11);
      const draft: [number, number][] = [
        [200, 200],
        [260, 200],
        [260, 260],
      ];
      act(() => useStore.getState().setCurrentPolygon(draft));
      let draftWhenSent: unknown;
      saveSpy.mockImplementationOnce(() => {
        draftWhenSent = useStore.getState().canvas.currentPolygon;
        return Promise.resolve(saved("1"));
      });
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      expect(draftWhenSent).toEqual(draft);
    });

    it("is settled before the image changes, and the next image inherits no focus from it", async () => {
      await dragCornerTo(11, 11);
      expect(useStore.getState().canvas.focus).toEqual({ kind: "box", index: 0 });
      loadSpy.mockImplementation((imagePath) =>
        Promise.resolve({
          ...labelsFor(imagePath),
          boxes: [
            { x1: 300, y1: 300, x2: 340, y2: 340, subject: "other", attributes: {}, index: 0 },
          ],
        }),
      );
      act(() => {
        const st = useStore.getState();
        st.patchGui({ dataset: { ...st.gui.dataset, current_image_index: 1 } });
      });
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      await waitFor(() => expect(useStore.getState().canvas.boxes[0]?.subject).toBe("other"));
      expect(useStore.getState().canvas.focus).toBeNull();
    });

    it("saves one at or above it as dragged", async () => {
      await dragCornerTo(30, 30);
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 30, 30] });
    });
  });

  it("carries a geometry-less image rating into the save payload", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => {
      const s = useStore.getState();
      s.addImageAnnotation("subject_a");
      s.updateImageAnnotation(0, {
        subject: "subject_a",
        attributes: { canopy_cover: "sparse" },
        iscrowd: false,
      });
    });
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      { subject: "subject_a", attributes: { canopy_cover: "sparse" }, iscrowd: false },
    ]);
  });
});

describe("AnnotateTab labels show on the focused item only", () => {
  // The legend is the standing symbology reference; a committed shape is named on the canvas
  // only while it is the focused item, never on hover, for every shape kind.
  const stage = () => screen.getByTestId("canvas-stage");
  const frame = () => act(async () => void (await new Promise((r) => setTimeout(r, 25))));
  const labelsNamed = (name: string) =>
    screen.queryAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === name);

  function setupSubject(mode: "box" | "polygon" | "point") {
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({ gui: { ...s.gui, mode, active_subject: "tip" } }));
  }

  it("a box is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("box");
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() =>
      useStore
        .getState()
        .addBox({ x1: 10, y1: 10, x2: 50, y2: 50, subject: "tip", attributes: {} }),
    );

    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 30, clientY: 30 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 30, clientY: 30, button: 0 }); // press inside focuses
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });

  it("a polygon is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("polygon");
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        polygons: [
          {
            rings: [
              [
                [100, 100],
                [300, 100],
                [300, 300],
                [100, 300],
              ],
            ] as [number, number][][],
            subject: "tip",
            attributes: {},
          },
        ],
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 200, clientY: 200 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.click(stage(), { clientX: 200, clientY: 200 }); // click inside focuses
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });

  it("a point is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("point");
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        points: [{ x: 100, y: 100, subject: "tip", attributes: {} }],
      }),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 102, clientY: 101 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 }); // press focuses
    fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });
});

describe("AnnotateTab canvas push names its project", () => {
  it("pushes nothing while the backend has no project open", async () => {
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    useStore.setState({ openProject: null });

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).not.toHaveBeenCalled();
  });

  it("pushes carrying the open project's id", async () => {
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).toHaveBeenCalled();
    expect(pushSpy.mock.calls[0][0].project_id).toBe("a1b2c3d4e5f6");
  });
});

const BUCKET = "m1/2026-01-01";
const PROPOSALS: ServedProposals = {
  bucket: BUCKET,
  operating_point: { conf: 0.5, reason: "" },
  proposals: [
    {
      subject: "subject_a",
      bbox: [10, 10, 50, 50] as [number, number, number, number],
      attributes: {},
      iscrowd: false,
      score: 0.9,
      index: 0,
      paired: null,
      decision: null,
    },
  ],
};

function withBucket() {
  useStore.setState((s) => ({
    gui: { ...s.gui, dataset: { ...s.gui.dataset, bucket: BUCKET } },
  }));
  return vi.spyOn(api.annotate, "proposals").mockResolvedValue(PROPOSALS);
}

async function mountTab() {
  render(<AnnotateTab />);
  await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
  await flush();
}

const dottedRects = () =>
  screen.queryAllByTestId("k-rect").filter((r) => r.getAttribute("data-dash") === "true");

describe("AnnotateTab completion marks", () => {
  it("marks the subject through the save door and shows the state the answered document derives", async () => {
    await mountTab();
    saveSpy.mockResolvedValueOnce(saved("1", { completion: { subject_a: "negative" } }));

    fireEvent.click(screen.getByText("toolbar-complete"));
    await flush();

    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      complete: { subject_a: true },
      rect: null,
      proposals_hidden: false,
    });
    await waitFor(() =>
      expect(screen.getByTestId("toolbar")).toHaveAttribute("data-subject-state", "negative"),
    );
  });

  it("records on the mark that proposals were hidden while it was made", async () => {
    withBucket();
    await mountTab();

    toggleProposals();
    await flush();
    fireEvent.click(screen.getByText("toolbar-complete"));
    await flush();

    expect(saveSpy.mock.calls[0][0].proposals_hidden).toBe(true);
  });
});

describe("AnnotateTab proposals", () => {
  it("shows the bucket's undecided proposals and requests none while they are hidden", async () => {
    const proposalsSpy = withBucket();
    await mountTab();
    await waitFor(() => expect(dottedRects()).toHaveLength(1));
    expect(proposalsSpy).toHaveBeenCalledWith("C:/data/images/2026-01-01/img1.jpg", BUCKET);

    toggleProposals();
    await flush();
    expect(dottedRects()).toHaveLength(0);
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset } });
    });
    await flush();
    expect(proposalsSpy).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["a", "accept"],
    ["r", "reject"],
  ] as const)(
    "%s adjudicates the selected proposal through the save door and adopts its answer",
    async (key, field) => {
      withBucket();
      await mountTab();
      await waitFor(() => expect(dottedRects()).toHaveLength(1));

      fireEvent.keyDown(window, { key });
      await flush();

      expect(saveSpy.mock.calls[0][0]).toMatchObject({ bucket: BUCKET, [field]: [0] });
      expect(loadSpy).toHaveBeenCalledTimes(1);
    },
  );
});

describe("AnnotateTab review symbology", () => {
  // Box 0 pairs with an accepted proposal, box 1 with an undecided one, box 2 with none.
  const boxes = () => [
    { x1: 10, y1: 10, x2: 50, y2: 50, subject: "subject_a", attributes: {}, index: 0 },
    { x1: 300, y1: 300, x2: 340, y2: 340, subject: "subject_a", attributes: {}, index: 1 },
    { x1: 500, y1: 50, x2: 540, y2: 90, subject: "subject_a", attributes: {}, index: 2 },
  ];
  const polygon = () => ({
    rings: [
      [
        [800, 600],
        [900, 600],
        [900, 700],
      ],
    ] as [number, number][][],
    subject: "subject_a",
    attributes: {},
    index: 3,
  });
  const at = (x: number, y: number, score: number) => ({
    ...PROPOSALS.proposals[0],
    bbox: [x, y, x + 40, y + 40] as [number, number, number, number],
    score,
  });
  const served = (): ServedProposals => ({
    bucket: BUCKET,
    operating_point: { conf: 0.5, reason: "" },
    proposals: [
      { ...at(10, 10, 0.9), index: 0, paired: 0, decision: "accepted" },
      { ...at(600, 600, 0.9), index: 1 },
      { ...at(700, 100, 0.2), index: 2 },
      { ...at(300, 300, 0.8), index: 3, paired: 1 },
    ],
  });
  const onBoxZero: Flag = {
    id: "f1",
    text: "open?",
    by: "user:first",
    at: "2026-01-01T00:00:00+00:00",
    point: [30, 30],
    subject: "subject_a",
    proposal: null,
    resolved_by: null,
    resolved_at: null,
    reply: "",
    removed: false,
  };
  const strokes = () => screen.getAllByTestId("k-rect").map((r) => r.getAttribute("data-stroke"));
  const position = () => screen.getByRole("textbox", { name: "Item position" });
  const down = async () => {
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
  };

  async function mountReviewing(flags: Flag[] = []) {
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), boxes: boxes(), polygons: [polygon()], flags }),
    );
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, bucket: BUCKET } },
    }));
    const proposalsSpy = vi.spyOn(api.annotate, "proposals").mockResolvedValue(served());
    await mountTab();
    await waitFor(() => expect(proposalsSpy).toHaveBeenCalled());
    await flush();
    return proposalsSpy;
  }

  it("outline color says match type while proposals are shown and subject otherwise", async () => {
    await mountReviewing();
    // The two undecided proposals above the floor draw first, then the three boxes, then the
    // polygon's derived box, which nothing proposes.
    expect(strokes()).toEqual([
      MATCH_COLORS.proposal_only,
      MATCH_COLORS.matched,
      MATCH_COLORS.matched,
      MATCH_COLORS.matched,
      MATCH_COLORS.annotation_only,
      MATCH_COLORS.annotation_only,
    ]);

    toggleProposals();
    await flush();
    expect(strokes()).toEqual(Array(4).fill(subjectColor("subject_a")));
  });

  it("the confidence floor starts at the operating point and the person moves it", async () => {
    await mountReviewing();
    const floor = screen.getByRole("spinbutton", { name: "Confidence floor" });
    expect(floor).toHaveValue(0.5);
    // Three boxes and one unpaired proposal; box 1, with its undecided paired proposal, and that
    // unpaired proposal await a decision.
    expect(screen.getByText("4 items, 2 unreviewed")).toBeInTheDocument();

    fireEvent.change(floor, { target: { value: "0.1" } });
    await flush();
    expect(screen.getByText("5 items, 3 unreviewed")).toBeInTheDocument();
    expect(dottedRects()).toHaveLength(3);
  });

  it("a floor the person moved belongs to its bucket; another bucket starts at its own", async () => {
    await mountReviewing();
    const floor = () => screen.getByRole("spinbutton", { name: "Confidence floor" });
    fireEvent.change(floor(), { target: { value: "0.1" } });
    await flush();
    expect(floor()).toHaveValue(0.1);

    act(() =>
      useStore.setState((s) => ({
        gui: { ...s.gui, dataset: { ...s.gui.dataset, bucket: "other/2026-01-01" } },
      })),
    );
    await flush();
    expect(floor()).toHaveValue(0.5);
  });

  it("the match filter keeps one match type on the canvas and in the path", async () => {
    await mountReviewing();
    const filter = screen.getByRole("combobox", { name: "Match type" });

    fireEvent.change(filter, { target: { value: "annotation_only" } });
    await flush();
    expect(screen.getByText("1 item, 0 unreviewed")).toBeInTheDocument();
    expect(dottedRects()).toHaveLength(0);

    fireEvent.change(filter, { target: { value: "proposal_only" } });
    await flush();
    expect(screen.getByText("1 item, 1 unreviewed")).toBeInTheDocument();
    expect(dottedRects()).toHaveLength(1);
  });

  it("a match filter applies only while proposals are shown", async () => {
    await mountReviewing();
    fireEvent.change(screen.getByRole("combobox", { name: "Match type" }), {
      target: { value: "annotation_only" },
    });
    await flush();
    expect(screen.getByText("1 item, 0 unreviewed")).toBeInTheDocument();

    toggleProposals();
    await flush();
    expect(screen.getByText("3 items")).toBeInTheDocument();
  });

  it("the items are the selected tool's geometry, and switching tools drops the focus", async () => {
    await mountReviewing();
    await down();
    expect(position()).toHaveValue("1");
    expect(strokes()).toContain(FOCUS_HALO.color);

    act(() => useStore.getState().setMode("polygon"));
    await flush();
    // One polygon and no polygon proposal: the boxes and the box proposals left the path.
    expect(screen.getByText("1 item, 0 unreviewed")).toBeInTheDocument();
    expect(position()).toHaveValue("");
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);

    await down();
    expect(screen.getByText("subject_a polygon")).toBeInTheDocument();
    expect(useStore.getState().gui.mode).toBe("polygon");
  });

  it("focus is a halo under the item's own stroke, never a recolor", async () => {
    await mountReviewing();
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 30,
      clientY: 30,
      button: 0,
    });
    await flush();
    const rects = screen.getAllByTestId("k-rect");
    const halo = rects.find((r) => r.getAttribute("data-stroke") === FOCUS_HALO.color)!;
    expect(halo).toBeDefined();
    expect(Number(halo.getAttribute("data-opacity"))).toBe(FOCUS_HALO.opacity);
    // The focused box keeps its match color on its own stroke and its corner handles.
    expect(strokes()).toContain(MATCH_COLORS.matched);
    expect(strokes()).not.toContain("#00BFFF");
  });

  it("the down and up arrows step the nearest-neighbor path, set the subject and zoom", async () => {
    await mountReviewing();
    useStore.getState().setActiveSubject(null);
    const before = useStore.getState().gui.view;

    // The path starts nearest the top-left: box 0. Its subject becomes active and the view lands
    // on it (zoomed in past the fitted view).
    await down();
    expect(position()).toHaveValue("1");
    expect(screen.getByText("subject_a box")).toBeInTheDocument();
    expect(useStore.getState().gui.active_subject).toBe("subject_a");
    expect(useStore.getState().gui.view.scale).toBeGreaterThan(before.scale);

    await down();
    await down();
    expect(position()).toHaveValue("3");
    await down();
    expect(position()).toHaveValue("4");
    expect(screen.getByText("subject_a proposal 0.90")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "ArrowUp" });
    await flush();
    expect(position()).toHaveValue("3");
    // The left and right arrows still step images, never items.
    expect(useStore.getState().gui.dataset.current_image_index).toBe(0);

    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "unreviewed" },
    });
    await flush();
    expect(screen.getByText("/ 2")).toBeInTheDocument();
  });

  it("a decides the focused matched annotation's own proposal", async () => {
    await mountReviewing();
    await down();
    await down();
    expect(position()).toHaveValue("2");

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({ bucket: BUCKET, accept: [3] });
  });

  it("e accepts the focused unpaired proposal so it can be corrected as an annotation", async () => {
    await mountReviewing();
    fireEvent.keyDown(window, { key: "ArrowUp" });
    await flush();
    expect(screen.getByText("subject_a proposal 0.90")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({ bucket: BUCKET, accept: [1] });
  });

  it("a confirms a focused annotation no proposal pairs with when a tool left it unconfirmed", async () => {
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        boxes: boxes().map((b) => (b.index === 2 ? { ...b, authorship: "tool" } : b)),
        polygons: [polygon()],
        flags: [],
      }),
    );
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, bucket: BUCKET } },
    }));
    vi.spyOn(api.annotate, "proposals").mockResolvedValue(served());
    await mountTab();
    await flush();
    expect(screen.getByText("4 items, 3 unreviewed")).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 520,
      clientY: 70,
      button: 0,
    });
    await flush();
    fireEvent.keyDown(window, { key: "a" });
    await flush();

    expect(saveSpy.mock.calls[0][0]).toMatchObject({ confirm: [2] });
    expect(saveSpy.mock.calls[0][0].accept).toBeUndefined();
  });

  const toolBoxTwo = () => boxes().map((b) => (b.index === 2 ? { ...b, authorship: "tool" } : b));
  const pressBoxTwo = async () => {
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 520,
      clientY: 70,
      button: 0,
    });
    await flush();
  };

  it("a confirms an unconfirmed annotation with no bucket selected", async () => {
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), boxes: toolBoxTwo() }),
    );
    await mountTab();
    await flush();
    await pressBoxTwo();
    expect(screen.getByRole("button", { name: "Accept" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled();

    fireEvent.keyDown(window, { key: "r" });
    await flush();
    expect(saveSpy).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({ confirm: [2] });
  });

  it("a reject on a focused annotation with no proposal never reaches another item's proposal", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "box", index: 2 }));
    await flush();
    fireEvent.keyDown(window, { key: "r" });
    await flush();
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("the pointer handlers act on the item the canvas draws handles on, a paired proposal's annotation", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 3 }));
    await flush();
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);

    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 300,
      clientY: 300,
      button: 0,
    });
    // A press on a handle of the focused annotation picks it up, one undo snapshot per drag.
    expect(useStore.getState().canvas.undoStack).toHaveLength(1);
  });

  it("a gesture's answer is adopted whole: its document and the token that names it together", async () => {
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), boxes: toolBoxTwo() }),
    );
    saveSpy.mockResolvedValueOnce(
      saved("500", { boxes: boxes().map((b) => ({ ...b, authorship: "tool_accepted" })) }),
    );
    await mountTab();
    await flush();
    await pressBoxTwo();
    fireEvent.keyDown(window, { key: "a" });
    await flush();

    expect(useStore.getState().canvas.boxes[2].authorship).toBe("tool_accepted");
    expect(loadSpy).toHaveBeenCalledTimes(1);
    addBox();
    pressSave();
    await flush();
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe("500");
  });

  it("e on a proposal whose save the canvas did not adopt focuses no annotation", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    saveSpy.mockResolvedValueOnce({ status: "conflict" });
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({ kind: "proposal", index: 1 });
  });

  it("e whose answer arrives after another image was selected focuses nothing there", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    let answer!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((resolve) => (answer = resolve)));
    fireEvent.keyDown(window, { key: "e" });
    await flush();

    loadSpy.mockImplementation(() => new Promise(() => {}));
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await flush();
    await act(async () => answer(saved("9", { boxes: boxes() })));

    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("e focuses the annotation the answer says the proposal resolved to, one already drawn included", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    saveSpy.mockResolvedValueOnce(
      saved("8", { boxes: boxes(), polygons: [polygon()] }, { "1": 0 }),
    );
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({ kind: "box", index: 0 });
  });

  it("e's focus is cleared by a navigation that follows its adoption, and the next image keeps none", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    let answer!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((res) => (answer = res)));
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    let armed = true;
    const stop = useStore.subscribe((now, prev) => {
      if (!armed || now.canvas.boxes === prev.canvas.boxes) return;
      armed = false;
      queueMicrotask(() => {
        const st = useStore.getState();
        st.patchGui({ dataset: { ...st.gui.dataset, current_image_index: 1 } });
      });
    });
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({
        ...labelsFor(imagePath),
        boxes: [{ ...boxes()[0], subject: "other", index: 0 }],
      }),
    );
    await act(async () =>
      answer(saved("8", { boxes: boxes(), polygons: [polygon()] }, { "1": 0 })),
    );
    stop();
    await waitFor(() => expect(useStore.getState().canvas.boxes[0]?.subject).toBe("other"));
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("e sets the tool from the tool in use when the answer lands, not when e was pressed", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    let answer!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((res) => (answer = res)));
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    act(() => useStore.getState().setMode("polygon"));
    await act(async () =>
      answer(saved("8", { boxes: boxes(), polygons: [polygon()] }, { "1": 0 })),
    );
    expect(useStore.getState().gui.mode).toBe("box");
    expect(useStore.getState().canvas.focus).toEqual({ kind: "box", index: 0 });
  });

  it("e focuses a polygon the proposal resolved to, switching to its tool", async () => {
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: 1 }));
    saveSpy.mockResolvedValueOnce(
      saved("8", { boxes: boxes(), polygons: [polygon()] }, { "1": 3 }),
    );
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().gui.mode).toBe("polygon");
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
  });

  describe("an adopted save, ordinary or gesture", () => {
    const editedTwo = async () => {
      loadSpy.mockImplementation((imagePath) =>
        Promise.resolve({ ...labelsFor(imagePath), boxes: toolBoxTwo() }),
      );
      await mountTab();
      await flush();
      act(() => useStore.getState().updateBox(2, { ...toolBoxTwo()[2], x2: 541 }));
      expect(useStore.getState().canvas.undoStack).toHaveLength(1);
    };
    const personBoxes = () => boxes().map((b) => ({ ...b, authorship: "person" }));

    it("gives the canvas the answer's authorship and drops the undo history", async () => {
      await editedTwo();
      saveSpy.mockResolvedValueOnce(saved("7", { boxes: personBoxes() }));
      pressSave();
      await flush();
      const { canvas } = useStore.getState();
      expect(canvas.boxes[2].authorship).toBe("person");
      expect([canvas.dirty, canvas.undoStack.length]).toEqual([false, 0]);
    });

    it("does the same for a gesture's answer", async () => {
      await editedTwo();
      await pressBoxTwo();
      saveSpy.mockResolvedValueOnce(saved("7", { boxes: personBoxes() }));
      fireEvent.keyDown(window, { key: "a" });
      await flush();
      const { canvas } = useStore.getState();
      expect(canvas.boxes[2].authorship).toBe("person");
      expect([canvas.dirty, canvas.undoStack.length]).toEqual([false, 0]);
    });

    it("serves the bucket's proposals again, dropping the pairings the old document gave", async () => {
      const proposalsSpy = await mountReviewing();
      act(() => useStore.getState().updateBox(0, { ...boxes()[0], x2: 51 }));
      saveSpy.mockResolvedValueOnce(saved("7", { boxes: boxes(), polygons: [polygon()] }));
      pressSave();
      await flush();
      expect(proposalsSpy).toHaveBeenCalledTimes(2);
    });

    it("refuses an edit made while it is in flight, then adopts the answer whole", async () => {
      await editedTwo();
      let answer!: (r: SaveResult) => void;
      saveSpy.mockImplementationOnce(
        () => new Promise<SaveResult>((resolve) => (answer = resolve)),
      );
      pressSave();
      await flush();
      expect(useStore.getState().canvas.saving).not.toBeNull();

      act(() => useStore.getState().addBox({ ...boxes()[0], index: undefined }));
      await pressBoxTwo();
      const rectsBefore = screen.getAllByTestId("k-rect").length;
      fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
        clientX: 100,
        clientY: 100,
        button: 0,
      });
      await flush();
      expect(screen.getAllByTestId("k-rect")).toHaveLength(rectsBefore);
      expect(useStore.getState().canvas.boxes).toHaveLength(3);
      expect(useStore.getState().canvas.focus).toBeNull();

      await act(async () => answer(saved("7", { boxes: personBoxes() })));
      const { canvas } = useStore.getState();
      expect(canvas.saving).toBeNull();
      expect(canvas.boxes[2].authorship).toBe("person");
      expect(canvas.dirty).toBe(false);
    });

    it("cancels a box being drawn when it starts, and counts no box it did not take", async () => {
      await editedTwo();
      const baseline = () => useStore.getState().sessionTracking.annotationsAddedDelta;
      const added = baseline();
      const rectsBefore = screen.getAllByTestId("k-rect").length;
      const stage = screen.getByTestId("canvas-stage");
      fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
      fireEvent.mouseMove(stage, { clientX: 200, clientY: 200 });
      await act(async () => void (await new Promise((r) => setTimeout(r, 25))));
      expect(screen.getAllByTestId("k-rect").length).toBe(rectsBefore + 1);

      saveSpy.mockImplementationOnce(() => new Promise<SaveResult>(() => {}));
      pressSave();
      await flush();
      expect(screen.getAllByTestId("k-rect")).toHaveLength(rectsBefore);
      fireEvent.mouseUp(stage, { clientX: 200, clientY: 200 });
      await flush();
      expect(useStore.getState().canvas.boxes).toHaveLength(3);
      expect(baseline()).toBe(added);
    });

    it("proposals are dropped when the answer is adopted, before they are served again", async () => {
      const proposalsSpy = await mountReviewing();
      expect(strokes()).toContain(MATCH_COLORS.proposal_only);
      act(() => useStore.getState().updateBox(0, { ...boxes()[0], x2: 51 }));
      proposalsSpy.mockImplementation(() => new Promise(() => {}));
      saveSpy.mockResolvedValueOnce(saved("7", { boxes: boxes(), polygons: [polygon()] }));
      pressSave();
      await flush();
      expect(strokes()).not.toContain(MATCH_COLORS.proposal_only);
    });
  });

  it("the agent's mirror and the canvas state the same colors and the same focus", async () => {
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    await mountReviewing([onBoxZero]);
    await down();
    act(() => notifyCanvasStateRequest());
    await flush();

    const distinct = (values: (string | null | undefined)[]) => [...new Set(values)].sort();
    const pushed = pushSpy.mock.calls.at(-1)?.[0].shapes ?? [];
    // The canvas also draws the halo and the focused box's handles; the colors it uses for items
    // are the colors the mirror states, and so are the labels.
    expect(distinct(pushed.filter((s) => s.tag !== "flag").map((s) => s.color))).toEqual(
      distinct(strokes().filter((s) => s !== FOCUS_HALO.color)),
    );
    expect(pushed.filter((s) => s.halo)).toHaveLength(1);
    // The flag is one mark on each side: the glyph on the canvas, the glyph and comment mirrored.
    const flagged = pushed.filter((s) => s.tag === "flag");
    expect(flagged).toHaveLength(1);
    expect(flagged[0]).toMatchObject({ color: FLAG_MARK.color, points: [[30, 30]] });
    const canvasLabels = screen.getAllByTestId("k-text").map((t) => t.getAttribute("data-text"));
    expect(canvasLabels).toContain(FLAG_MARK.glyph);
    expect(distinct(pushed.filter((s) => s.label && s.tag !== "flag").map((s) => s.label))).toEqual(
      distinct(canvasLabels.filter((t) => t !== FLAG_MARK.glyph)),
    );
  });
});

describe("AnnotateTab flags", () => {
  const box = { x1: 10, y1: 10, x2: 50, y2: 50, subject: "subject_a", attributes: {}, index: 0 };
  const flag = (over: Partial<Flag>): Flag => ({
    id: "f1",
    text: "open?",
    by: "user:first",
    at: "2026-01-01T00:00:00+00:00",
    point: [30, 30],
    subject: "subject_a",
    proposal: null,
    resolved_by: null,
    resolved_at: null,
    reply: "",
    removed: false,
    ...over,
  });

  async function mountWith(flags: Flag[]) {
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve({ ...labelsFor(imagePath), boxes: [box], flags }),
    );
    await mountTab();
  }
  const comment = () => screen.getByRole("textbox", { name: "New flag comment" });

  it("f flags the image as a whole when nothing is focused", async () => {
    await mountWith([]);
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    expect(screen.getByRole("dialog", { name: "Flags on this image" })).toBeInTheDocument();

    fireEvent.change(comment(), { target: { value: "glare" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(saveSpy.mock.calls[0][0].flag).toEqual([{ text: "glare" }]);
    expect(comment()).toHaveValue("");
  });

  it("keeps what was typed after a flag was submitted, when that submission lands", async () => {
    await mountWith([]);
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    let answer!: (r: SaveResult) => void;
    saveSpy.mockImplementationOnce(() => new Promise<SaveResult>((res) => (answer = res)));
    fireEvent.change(comment(), { target: { value: "first" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    fireEvent.change(comment(), { target: { value: "second, not submitted" } });
    await act(async () => answer(saved("5")));
    expect(comment()).toHaveValue("second, not submitted");
  });

  it("keeps the typed comment when its save was not taken, and clears it once one is", async () => {
    await mountWith([]);
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    fireEvent.change(comment(), { target: { value: "glare" } });
    saveSpy.mockResolvedValueOnce({ status: "conflict" });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(comment()).toHaveValue("glare");

    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(comment()).toHaveValue("");
  });

  it("flags the focused annotation at a place inside it, through the save door", async () => {
    await mountWith([]);
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
    fireEvent.click(screen.getByRole("button", { name: /Flag/ }));
    expect(screen.getByRole("dialog", { name: "Flags on subject_a box" })).toBeInTheDocument();

    fireEvent.change(comment(), { target: { value: "open or closed?" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      image_path: "C:/data/images/2026-01-01/img1.jpg",
      flag: [{ text: "open or closed?", point: [30, 30], subject: "subject_a" }],
    });
    expect(loadSpy).toHaveBeenCalledTimes(1);
  });

  it("counts open flags, marks them on the canvas and steps to the flagged items", async () => {
    await mountWith([flag({}), flag({ id: "f2", resolved_by: "user:second" })]);
    expect(screen.getByRole("button", { name: /Flag \(1\)/ })).toBeInTheDocument();
    expect(
      screen
        .getAllByTestId("k-text")
        .filter((t) => t.getAttribute("data-text") === FLAG_MARK.glyph),
    ).not.toHaveLength(0);

    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "flagged" },
    });
    await flush();
    expect(screen.getByText("/ 1")).toBeInTheDocument();
  });

  it("resolves the focused item's flag with a reply, through the save door", async () => {
    await mountWith([flag({})]);
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    expect(screen.getByText("open?")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: 'Reply to "open?"' }), {
      target: { value: "open" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Resolve" }));
    await flush();
    expect(saveSpy.mock.calls[0][0].resolve).toEqual({ f1: "open" });
  });

  it("adopts the flags a plain save answers, so a removal's resolution shows at once", async () => {
    await mountWith([flag({})]);
    saveSpy.mockResolvedValue(
      saved("2", { flags: [flag({ resolved_by: "user:second", removed: true })] }),
    );
    act(() => useStore.getState().deleteBox(0));
    pressSave();
    await flush();
    expect(screen.getByRole("button", { name: /^.?Flag$/ })).toBeInTheDocument();
  });
});

describe("AnnotateTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Annotate");
  });
});
