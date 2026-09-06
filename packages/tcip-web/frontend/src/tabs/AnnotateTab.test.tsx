import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import type { SaveResult } from "@/api/client";
import { classesApi, subjectColor } from "@/api/classes";
import * as CanvasStageMock from "@/components/Canvas/CanvasStage";
import * as canvasSync from "@/lib/canvasSync";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import type { CompletenessRecord } from "@/lib/coverage";
import { CUT_MISSES_REFUSAL } from "@/lib/polygonGeometry";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";

// Konva needs a real 2D canvas; these tests exercise label I/O ordering and
// store->canvas prop flow, not drawing. Render Konva shapes as inspectable divs
// and CanvasStage as a passthrough so AnnotationShapes' memo behavior is intact.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  Rect: (props: { stroke?: string; dash?: number[]; fill?: string }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
      data-dash={props.dash ? "true" : undefined}
      data-fill={props.fill}
    />
  ),
  Line: (props: { stroke?: string; points?: number[] }) => (
    <div data-testid="k-line" data-stroke={props.stroke} data-points={props.points?.join(",")} />
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
  let capturedOnBaseFacts: ((facts: unknown) => void) | undefined;
  return {
    CanvasStage: (props: {
      children?: React.ReactNode;
      overlay?: React.ReactNode;
      imageUrl?: string | null;
      onBaseFacts?: (facts: unknown) => void;
      onPixelDown?: (x: number, y: number, ev: unknown) => void;
      onPixelMove?: (x: number, y: number, ev: unknown) => void;
      onPixelUp?: (x: number, y: number, ev: unknown) => void;
      onPixelClick?: (x: number, y: number, ev: unknown) => void;
      onPixelContextMenu?: (x: number, y: number, ev: unknown) => void;
    }) => {
      capturedOnBaseFacts = props.onBaseFacts;
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
    __triggerBaseFacts: (facts: unknown) => capturedOnBaseFacts?.(facts),
  };
});
vi.mock("@/components/AnnotateToolbar", () => ({
  AnnotateToolbar: (props: { bandsInfo?: { band_count: number } | null }) => (
    <div data-testid="toolbar" data-band-count={props.bandsInfo?.band_count ?? ""} />
  ),
}));

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
    base_mtime: String(LOAD_MTIME[name] ?? 1),
  };
}

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      mode: "box" as const,
      active_subject: "subject_a",
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
        subject: "subject_a",
        date: "2026-01-01",
        image_list: ["img1.jpg", "img2.jpg"],
        current_image_index: 0,
        images_dir: "C:/data/images/2026-01-01",
        annotations_dir: "C:/data/annotations/2026-01-01",
        predictions_dir: null,
      },
    },
  }));
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
  setupDataset();
  loadSpy = vi
    .spyOn(api.annotate, "load")
    .mockImplementation((imagePath) => Promise.resolve(labelsFor(imagePath)));
  saveSpy = vi.spyOn(api.annotate, "save").mockResolvedValue({ status: "ok", base_mtime: "1" });
  vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({ status: "ok", digest_stamped: true });
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
  // The coverage-grid fetch is gated on a real canvas-host measurement (useCoverageGrid); jsdom's
  // own getBoundingClientRect is always zero, so every test needing that fetch stubs one here.
  vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 1000, h: 800 });
  // One jsdom instance per file, not per test: a recolour left by an earlier test must not leak
  // into a later one's derived-colour assertions.
  try {
    localStorage.removeItem("tcip.annotate.subjectColors");
  } catch {
    /* not available in this environment, nothing to clear */
  }
});

afterEach(cleanup);

describe("AnnotateTab save/load race", () => {
  it("saves to the loaded path and re-echoes the returned mtime on the next save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    saveSpy.mockResolvedValueOnce({ status: "ok", base_mtime: "101" });
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
    // cached read): the exact interleaving that used to corrupt cross-image GT.
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
      resolveFlushSave({ status: "ok", base_mtime: "150" });
    });

    // The stale result must not markClean() the img2 edits...
    expect(useStore.getState().canvas.dirty).toBe(true);
    // ...but the per-image status for the image actually saved is still recorded, scoped to the
    // selected subject, so it cannot mark the image negative under another subject.
    expect(classesApi.setImageStatus).toHaveBeenCalledWith(
      "C:/proj",
      "img1.jpg",
      "partial",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );

    // ...and the next save must target img2 with img2's loaded mtime, not
    // img1's file with the stale save's echoed mtime.
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].image_path).toBe("C:/data/images/2026-01-01/img2.jpg");
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe("200");
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

  it("records the status under the name the app is set to, not the backend's own identity", async () => {
    act(() => useStore.getState().setUser("breeder"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(useStore.getState().user).toBe("breeder");
    expect(vi.mocked(classesApi.setImageStatus).mock.calls[0][7]).toBe("breeder");
  });

  it("never rewrites a confirmed negative to partial, even when the save adds content", async () => {
    useStore.setState((s) => ({
      imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "negative" } },
    }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(classesApi.setImageStatus).not.toHaveBeenCalled();
    expect(useStore.getState().imageStatus.byImage["img1.jpg"]).toBe("negative");
  });
});

describe("AnnotateTab subject rendering", () => {
  it("renders a box with the subject-derived colour, named on selection", async () => {
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
    // Colour is GUI-local (name-derived); the label is the subject name, no integer id, and
    // appears on selection (labels are hover/selection-only; the legend is the standing key).
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("tip"));
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "tip");
  });

  it("box mode draws an active-subject polygon's read-only derived box (dashed, no handles), never a stored box", async () => {
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
    // handles (handles are extra Rects), and it never entered canvas.boxes, so unsaveable. Dashed
    // distinguishes it from a real editable box (solid), the same convention in-progress/
    // under-review shapes already use, not read-only enforcement (that's structural).
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).toHaveAttribute("data-dash", "true");
    expect(rects[0]).toHaveAttribute("data-stroke", subjectColor("subject_a"));
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });

  it("a tool-authored polygon's derived box in box mode still names itself by authorship", async () => {
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

    // Still the derived dash, never the tool's own pattern (one channel can't carry both), but
    // the hover label still names the polygon's own authorship, like every other shape's does.
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).toHaveAttribute("data-dash", "true");
    fireEvent.mouseMove(screen.getByTestId("canvas-stage"), { clientX: 3, clientY: 3 });
    await act(async () => void (await new Promise((r) => setTimeout(r, 25))));
    expect(
      screen
        .getAllByTestId("k-text")
        .some((t) => t.getAttribute("data-text") === "subject_a, tool"),
    ).toBe(true);
  });

  it("box mode still draws a real editable box solid, distinct from a derived one", async () => {
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

    // The core sits exactly on the coordinate, in the subject's colour...
    const core = screen.getByTestId("k-circle");
    expect(core).toHaveAttribute("data-x", "100");
    expect(core).toHaveAttribute("data-y", "200");
    expect(core).toHaveAttribute("data-fill", subjectColor("tip"));
    // ...with four radial ticks (the mark that reads as a location, not a tiny shape)...
    expect(screen.getAllByTestId("k-line")).toHaveLength(4);
    // ...and no box of any kind. (Naming on selection/hover is covered by the label tests.)
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);
  });

  it("polygon mode draws every ring of an occlusion-split shape, labelled once", async () => {
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
    // Both parts wear the annotation's own colour...
    expect(lines.every((l) => l.getAttribute("data-stroke") === subjectColor("tip"))).toBe(true);
    // ...and selecting it names the annotation once, not once per ring (HaloLabel = halo + fill).
    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 8, clientY: 5 });
    expect(
      screen.getAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === "tip"),
    ).toHaveLength(2);
  });
});

describe("AnnotateTab authorship symbology", () => {
  it("a tool's own box draws dotted and names itself on hover; the other three stay solid", async () => {
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
    expect(rects[3]).not.toHaveAttribute("data-dash"); // unattributed: solid

    // Hovering the tool box names it with the authorship it draws with (the move handler is
    // rAF-throttled, so a real timer tick must land before the hover state updates).
    fireEvent.mouseMove(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    await act(async () => void (await new Promise((r) => setTimeout(r, 25))));
    expect(
      screen
        .getAllByTestId("k-text")
        .some((t) => t.getAttribute("data-text") === "subject_a, tool"),
    ).toBe(true);
  });

  it("the legend states the dotted stroke means a tool drew it, unaccepted", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.getByText("Dotted = drawn by a tool, not yet accepted")).toBeInTheDocument();
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
    expect(useStore.getState().canvas.selectedPointIdx).toBe(0);
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

  it("right-click removes the point under the cursor and leaves a neighbour alone", async () => {
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
    expect(useStore.getState().canvas.selectedPointIdx).toBe(0);

    fireEvent.keyDown(window, { key: "Delete" });
    await flush();
    expect(useStore.getState().canvas.points).toHaveLength(0);
    expect(useStore.getState().canvas.selectedPointIdx).toBeNull();
  });

  it("saves a placed point as a `point` payload the save route can author", async () => {
    await mountPointMode();
    fireEvent.click(stage(), { clientX: 12, clientY: 34 });
    await flush();
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      {
        subject: "tip",
        point: [12, 34],
        attributes: {},
        created_by: null,
        created_at: null,
        accepted_by: null,
        accepted_at: null,
      },
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

  it("a locked image mounts the panel with the subject block only", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    useStore.setState((s) => ({
      imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "complete" } },
    }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.getByText("Attributes for subject_a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "+ Attribute" })).toBeInTheDocument();
    expect(screen.queryByText("Select a shape to set its attributes.")).not.toBeInTheDocument();
    expect(screen.queryByText("Ratings for this whole image")).not.toBeInTheDocument();
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
    expect(useStore.getState().canvas.selectedPolygonIdx).toBeNull(); // sanity: a box, not a polygon, is selected

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

  it("posts the grown registry through classesApi.save with the loaded version and installs it", async () => {
    await openPanelOnASelectedBox();
    const saveSpy = vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      classes_path: "C:/data/classes.json",
      version: "v2",
      schema_change_sweep: { newly_stamped: {}, predating_vocabulary: {}, warning: null },
    });

    await declareAttribute();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][1]).toEqual({
      subject_a: { attributes: { size: { type: "categorical", values: ["small", "large"] } } },
    });
    expect(saveSpy.mock.calls[0][4]).toBeNull(); // no version was ever loaded in this test
    expect(useStore.getState().registry.version).toBe("v2");
    expect(useStore.getState().registry.subjects.subject_a.attributes?.size.values).toEqual([
      "small",
      "large",
    ]);
  });

  it("reloads the registry from the server when the save is refused, same as the subject add", async () => {
    await openPanelOnASelectedBox();
    vi.spyOn(classesApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(classesApi, "load").mockResolvedValue({
      subjects: { subject_a: {} },
      version: "v3",
      unreadable: [],
    });

    await declareAttribute();

    expect(useStore.getState().registry.subjects).toEqual({ subject_a: {} });
    expect(useStore.getState().registry.version).toBe("v3");
  });

  it("toasts the schema_change_sweep the save response carries, naming the subject and count", async () => {
    await openPanelOnASelectedBox();
    vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      classes_path: "C:/data/classes.json",
      version: "v2",
      schema_change_sweep: {
        newly_stamped: { subject_a: 4 },
        predating_vocabulary: { subject_a: 4 },
        warning: null,
      },
    });

    await declareAttribute();

    expect(useStore.getState().toasts.at(-1)?.message).toMatch(
      /4 confirmed image\(s\) of subject_a/,
    );
  });

  it("counts the active subject's shapes carrying no value for each declared attribute", async () => {
    await openPanelOnASelectedBox();
    vi.spyOn(classesApi, "save").mockResolvedValue({
      status: "ok",
      n_subjects: 1,
      classes_path: "C:/data/classes.json",
      version: "v2",
      schema_change_sweep: { newly_stamped: {}, predating_vocabulary: {}, warning: null },
    });

    await declareAttribute();

    // The one drawn box carries no value for the attribute just declared.
    expect(
      screen.getByText("1 of 1 subject_a shapes on this image carry no size value."),
    ).toBeInTheDocument();
  });

  it("refuses a name the subject already declares, inline, and posts nothing", async () => {
    await openPanelOnASelectedBox();
    const saveSpy = vi.spyOn(classesApi, "save");
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

    const subjectRow = screen.getByTitle("Change subject_a's colour (this browser only)");
    expect(subjectRow).toBeInTheDocument();
    // DOM order: the panel follows the button, so a forward Tab from it reaches the row.
    expect(
      legendButton.compareDocumentPosition(subjectRow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("opens the colour picker as a labelled dialog with a named hex input", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Legend" }));
    fireEvent.click(screen.getByTitle("Change subject_a's colour (this browser only)"));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName(
      "subject_a's colour (this browser only; derives from the name elsewhere)",
    );
    expect(screen.getByRole("textbox", { name: "hex colour" })).toBeInTheDocument();
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

describe("AnnotateTab legend", () => {
  it("explains the dashed derived box only in box mode", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Box mode (setupDataset default): the legend button is hover-revealed, so query its content
    // directly rather than simulating hover.
    expect(screen.getByText("Dashed = polygon's box (read-only)")).toBeInTheDocument();

    act(() => useStore.getState().setMode("polygon"));
    expect(screen.queryByText("Dashed = polygon's box (read-only)")).not.toBeInTheDocument();
  });

  it("a recoloured subject's box stroke and the pushed canvas_meta swatch both follow", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("subject_a"));

    fireEvent.click(screen.getByRole("button", { name: "subject_a" }));
    const hexInput = screen.getByRole("textbox");
    fireEvent.change(hexInput, { target: { value: "#123456" } });
    fireEvent.keyDown(hexInput, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "OK" }));
    await flush();

    expect(subjectColor("subject_a")).toBe("#123456");
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", "#123456");

    useStore.setState({ bindingGeneration: 1 });
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

// A concave outline (the same U shape polygonGeometry.test.ts cuts, offset into the image): a
// centred axis cut crosses it more than twice, so both keyboard axes refuse it.
const CONCAVE_POLY = {
  rings: [
    [
      [400, 400],
      [430, 400],
      [430, 430],
      [420, 430],
      [420, 410],
      [410, 410],
      [410, 430],
      [400, 430],
    ] as [number, number][],
  ],
  subject: "subject_a",
  attributes: {},
};

const MULTI_RING_POLY = {
  rings: [
    [
      [500, 500],
      [520, 500],
      [520, 520],
      [500, 520],
    ] as [number, number][],
    [
      [540, 500],
      [560, 500],
      [560, 520],
      [540, 520],
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
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(0);
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("a click on another polygon switches the selection with Stream on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().selectPolygon(0);
    });
    fireEvent.click(stage, { clientX: 350, clientY: 350 });
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(1);
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("with Stream on, empty space deselects first and a later click still streams", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().selectPolygon(0);
    });
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.selectedPolygonIdx).toBeNull();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toEqual([[600, 600]]);
  });

  it("a click on a polygon selects it with Snap on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setSnap(true));
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(0);
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

  it("x does nothing on a locked image", async () => {
    await renderPolygonCanvas();
    act(() => {
      useStore.setState((s) => ({
        imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "complete" } },
      }));
    });
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  it("arming and disarming each schedule a push carrying the current cut_armed value", async () => {
    await renderPolygonCanvas();
    useStore.setState({ bindingGeneration: 1 });
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
      useStore.getState().selectPolygon(0);
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
    "The polygon changed since the first click; the cut was cancelled. Select it and place both " +
    "points again.";

  it("a selection change between the clicks refuses with the polygon-changed sentence", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().selectPolygon(1));
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
    expect(useStore.getState().canvas.polygons).toHaveLength(2); // unchanged: the cut was cancelled
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
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(2);
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

  it("confirming the image (locking it) clears the flag and the pending start", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => {
      useStore.setState((s) => ({
        imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "complete" } },
      }));
    });
    expect(useStore.getState().annotateUi.cut).toBe(false);
    act(() => {
      useStore.setState((s) => ({ imageStatus: { ...s.imageStatus, byImage: {} } }));
      useStore.getState().setCut(true);
    });
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

describe("Cut keyboard path (Shift+H / Shift+V)", () => {
  function xs(ring: [number, number][]): number[] {
    return ring.map(([x]) => x);
  }
  function ys(ring: [number, number][]): number[] {
    return ring.map(([, y]) => y);
  }
  function armAndSelectPolyA(): void {
    act(() => {
      useStore.getState().selectPolygon(0);
      useStore.getState().setCut(true);
    });
  }

  it("Shift+V splits the selected convex polygon through the vertical centre of its bounding box", async () => {
    await renderPolygonCanvas();
    armAndSelectPolyA();
    fireEvent.keyDown(window, { key: "V", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(3);
    const pieces = useStore.getState().canvas.polygons.slice(0, 2);
    const byMinX = [...pieces].sort(
      (p, q) => Math.min(...xs(p.rings[0])) - Math.min(...xs(q.rings[0])),
    );
    expect(Math.max(...xs(byMinX[0].rings[0]))).toBeLessThanOrEqual(105);
    expect(Math.min(...xs(byMinX[1].rings[0]))).toBeGreaterThanOrEqual(105);
  });

  it("Shift+H splits the selected convex polygon through the horizontal centre of its bounding box", async () => {
    await renderPolygonCanvas();
    armAndSelectPolyA();
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(3);
    const pieces = useStore.getState().canvas.polygons.slice(0, 2);
    const byMinY = [...pieces].sort(
      (p, q) => Math.min(...ys(p.rings[0])) - Math.min(...ys(q.rings[0])),
    );
    expect(Math.max(...ys(byMinY[0].rings[0]))).toBeLessThanOrEqual(105);
    expect(Math.min(...ys(byMinY[1].rings[0]))).toBeGreaterThanOrEqual(105);
  });

  it("undoes an axis cut in one step", async () => {
    await renderPolygonCanvas();
    armAndSelectPolyA();
    fireEvent.keyDown(window, { key: "H", shiftKey: true });
    expect(useStore.getState().canvas.polygons).toHaveLength(3);

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("refuses a concave shape with the keyboard's own sentence, one toast", async () => {
    await renderPolygonCanvas([POLY_A, CONCAVE_POLY]);
    act(() => {
      useStore.getState().selectPolygon(1);
      useStore.getState().setCut(true);
    });
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().toasts).toHaveLength(1);
    expect(useStore.getState().toasts[0].message).toBe(
      "A centred straight cut does not divide this shape into two pieces; cut it through one " +
        "part with the pointer, or redraw it as two shapes.",
    );
  });

  it("refuses a multi-ring shape with the tab's own multi-ring sentence", async () => {
    await renderPolygonCanvas([MULTI_RING_POLY]);
    act(() => {
      useStore.getState().selectPolygon(0);
      useStore.getState().setCut(true);
    });
    fireEvent.keyDown(window, { key: "V", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(1);
    expect(useStore.getState().toasts.at(-1)?.message).toContain("separate parts of one object");
  });

  it("leaves a pending click-cut start alone", async () => {
    const stage = await renderPolygonCanvas();
    armAndSelectPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("pushes the selection-required notice, not silence, with nothing selected", async () => {
    await renderPolygonCanvas();
    act(() => useStore.getState().setCut(true));
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().toasts.at(-1)?.message).toBe(
      "Select a polygon to cut, then click two points on either side of it.",
    );
  });

  it("does nothing on a locked image", async () => {
    await renderPolygonCanvas();
    armAndSelectPolyA();
    act(() => {
      useStore.setState((s) => ({
        imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "complete" } },
      }));
    });
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("does nothing while the cut tool is disarmed", async () => {
    await renderPolygonCanvas();
    act(() => useStore.getState().selectPolygon(0));
    fireEvent.keyDown(window, { key: "H", shiftKey: true });

    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("fires while focus sits on a control marked data-keyboard-passthrough", async () => {
    await renderPolygonCanvas();
    armAndSelectPolyA();
    const cutButtonStandIn = document.createElement("button");
    cutButtonStandIn.setAttribute("data-keyboard-passthrough", "");
    document.body.appendChild(cutButtonStandIn);
    try {
      fireEvent.keyDown(cutButtonStandIn, { key: "H", shiftKey: true });
      expect(useStore.getState().canvas.polygons).toHaveLength(3);
    } finally {
      cutButtonStandIn.remove();
    }
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

  it("an outside click does not drop an existing selection", async () => {
    const stage = await renderPolygonCanvas();
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(0);
    fireEvent.click(stage, { clientX: 1050, clientY: 400 });
    expect(useStore.getState().canvas.selectedPolygonIdx).toBe(0);
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

  it("addresses the save at the label file the annotations directory names for this image", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].label_path).toBe("C:/data/annotations/2026-01-01/img1.json");
  });

  it("refuses to save locally when the dataset has no annotations directory", async () => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, annotations_dir: null } },
    }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).not.toHaveBeenCalled();
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

  it("carries a geometry-less image rating into the save payload", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => {
      const s = useStore.getState();
      s.addImageAnnotation("subject_a");
      s.updateImageAnnotation(0, { subject: "subject_a", attributes: { canopy_cover: "sparse" } });
    });
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      {
        subject: "subject_a",
        attributes: { canopy_cover: "sparse" },
        created_by: null,
        created_at: null,
        accepted_by: null,
        accepted_at: null,
      },
    ]);
  });
});

describe("AnnotateTab labels show on selection or hover only", () => {
  // The legend is the standing symbology reference; a committed shape is named on the canvas
  // only while it is selected or hovered, for every shape kind.
  const stage = () => screen.getByTestId("canvas-stage");
  const frame = () => act(async () => void (await new Promise((r) => setTimeout(r, 25))));
  const labelsNamed = (name: string) =>
    screen.queryAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === name);

  function setupSubject(mode: "box" | "polygon" | "point") {
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({ gui: { ...s.gui, mode, active_subject: "tip" } }));
  }

  it("a box is unlabelled at rest, labelled while hovered, labelled while selected", async () => {
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
    expect(labelsNamed("tip").length).toBeGreaterThan(0);

    fireEvent.mouseMove(stage(), { clientX: 500, clientY: 500 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 30, clientY: 30, button: 0 }); // press inside selects
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });

  it("a polygon is unlabelled at rest, labelled while hovered, labelled while selected", async () => {
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
    expect(labelsNamed("tip").length).toBeGreaterThan(0);

    fireEvent.mouseMove(stage(), { clientX: 500, clientY: 500 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.click(stage(), { clientX: 200, clientY: 200 }); // click inside selects
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });

  it("a point is unlabelled at rest, labelled while hovered, labelled while selected", async () => {
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
    expect(labelsNamed("tip").length).toBeGreaterThan(0);

    fireEvent.mouseMove(stage(), { clientX: 500, clientY: 500 });
    await frame();
    expect(labelsNamed("tip")).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 }); // press selects
    fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
    await flush();
    expect(labelsNamed("tip").length).toBeGreaterThan(0);
  });
});

describe("AnnotateTab canvas-push binding-presence gate", () => {
  it("blocks the push and sets canvasBindingMissing when no generation is adopted", async () => {
    useStore.setState({ bindingGeneration: null });
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).not.toHaveBeenCalled();
    expect(useStore.getState().canvasBindingMissing).toBe(true);
  });

  it("pushes with the adopted generation and clears canvasBindingMissing", async () => {
    useStore.setState({ bindingGeneration: 3, canvasBindingMissing: true });
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).toHaveBeenCalled();
    expect(pushSpy.mock.calls[0][0].binding_generation).toBe(3);
    expect(useStore.getState().canvasBindingMissing).toBe(false);
  });
});

const MULTI_CELL_GRID = {
  width: 1000,
  height: 800,
  tile_size: 500,
  overlap: 0,
  cols: 2,
  rows: 2,
};
const MULTI_CELL_CELLS = [
  { name: "A1", x0: 0, y0: 0, x1: 500, y1: 400 },
  { name: "B1", x0: 500, y0: 0, x1: 1000, y1: 400 },
  { name: "A2", x0: 0, y0: 400, x1: 500, y1: 800 },
  { name: "B2", x0: 500, y0: 400, x1: 1000, y1: 800 },
];
const BELOW_NATIVE_BASE_FACTS = {
  ok: true,
  servedSize: { w: 500, h: 400 },
  servedSizeRaw: "500x400",
  statsSource: null,
  displayBounds: null,
  imageError: null,
  image: null,
  aborted: false,
  headerParseError: null,
};

// A rendered geometry block used for both `grid` and `serving` (get_grid's own nested shape):
// these tests exercise neither the set-zoom lookup nor the serving grid's own derivation.
function gridResponse(geometry: typeof MULTI_CELL_GRID, cells: typeof MULTI_CELL_CELLS) {
  const rendered = {
    ...geometry,
    derivation: "cells sized to one full-resolution screenful",
    cells,
  };
  return { grid: rendered, reason: null, fresh_derivation_differs: null, serving: rendered };
}

function mockMultiCellGrid() {
  // The Map tests below also mount the coverage-tracking hook, which reads and pushes the
  // session sweep record for the same raster; mocked here too so no real fetch is attempted.
  vi.spyOn(api.coverage, "get").mockResolvedValue(null);
  vi.spyOn(api.coverage, "push").mockResolvedValue({
    record: { cells_seen_at_scale: {} },
  });
  return vi
    .spyOn(api.coverage, "grid")
    .mockResolvedValue(gridResponse(MULTI_CELL_GRID, MULTI_CELL_CELLS));
}

// The mock module's own extra export, absent from the real CanvasStage's type: cast once here
// rather than at every call site.
const triggerBaseFacts = (
  CanvasStageMock as unknown as { __triggerBaseFacts: (facts: unknown) => void }
).__triggerBaseFacts;

// Drives the coverage grid's fetch (useCoverageGrid needs a served-below-native base serve).
function triggerBelowNativeBaseFacts() {
  act(() => triggerBaseFacts(BELOW_NATIVE_BASE_FACTS));
}

describe("AnnotateTab Map tool", () => {
  const stage = () => screen.getByTestId("canvas-stage");

  async function mountMapMode() {
    mockMultiCellGrid();
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {},
      annotation_counts: {},
      counts_grid: null,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "map" as const, active_subject: "tip" },
    }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    triggerBelowNativeBaseFacts();
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    await flush();
    expect(useStore.getState().gui.mode).toBe("map");
  }

  it("a press-and-click in Map mode authors no point, box or polygon vertex", async () => {
    await mountMapMode();
    fireEvent.mouseDown(stage(), { clientX: 50, clientY: 50, button: 0 });
    fireEvent.click(stage(), { clientX: 50, clientY: 50, button: 0 });
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(0);
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(0);
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("Map mode is inert even over a lockedImage, since navigation is not an edit", async () => {
    // The same click a Point-mode press would use to author a point.
    await mountMapMode();
    useStore.setState((s) => ({
      imageStatus: { ...s.imageStatus, byImage: { "img1.jpg": "complete" } },
    }));
    fireEvent.click(stage(), { clientX: 50, clientY: 50, button: 0 });
    await flush();
    expect(useStore.getState().canvas.points).toHaveLength(0);
  });

  it("falls back to a drawing tool when the Map tool is withdrawn (no multi-cell grid)", async () => {
    vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    // An ordinary raster's own lattice is one cell: settled, never pending, and offers no Map.
    vi.spyOn(api.coverage, "grid").mockResolvedValue(
      gridResponse({ width: 1000, height: 800, tile_size: 1000, overlap: 0, cols: 1, rows: 1 }, [
        { name: "A1", x0: 0, y0: 0, x1: 1000, y1: 800 },
      ]),
    );
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {},
      annotation_counts: {},
      counts_grid: null,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    useStore.getState().setRegistry({ tip: {} });
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "map" as const, active_subject: "tip" },
    }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => triggerBaseFacts({ ...BELOW_NATIVE_BASE_FACTS, servedSize: { w: 1000, h: 800 } }));
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    await flush();

    expect(useStore.getState().gui.mode).toBe("box");
  });
});

describe("AnnotateTab completeness refresh and attestation control", () => {
  it("a save refetches completeness, so counts and staleness never go stale on the open image", async () => {
    const completenessSpy = vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {},
      annotation_counts: {},
      counts_grid: null,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const callsBeforeSave = completenessSpy.mock.calls.length;

    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(completenessSpy.mock.calls.length).toBeGreaterThan(callsBeforeSave);
  });

  it("a completeness read failure shows in the chrome on an ordinary single-cell raster", async () => {
    vi.spyOn(api.coverage, "completeness").mockRejectedValue(new Error("network down"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(
      screen.getByText(/labels could not be read, so nothing can be attested.*network down/),
    ).toBeInTheDocument();
  });

  async function mountWithCoverageChrome(bySubject: Record<string, CompletenessRecord>) {
    localStorage.removeItem("tcip.annotate.coverageGridOverlayOpen");
    mockMultiCellGrid();
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: bySubject,
      annotation_counts: {},
      counts_grid: null,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 200,
      height: 200,
      top: 0,
      left: 0,
      right: 200,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => "",
    } as DOMRect);
    // Matches this test's own 200x200 geometry above, so computeViewport's center is the same.
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 200, h: 200 });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    triggerBelowNativeBaseFacts();
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    await flush();
    // The attest control is offered only while the overlay is on (ruling: no writing about an
    // unseen cell); these tests exercise the control itself, so switch it on first.
    fireEvent.click(screen.getByRole("button", { name: /Overlay off/ }));
  }

  it("labels the control Attest, naming the active subject, when the raw stored set never held the cell", async () => {
    await mountWithCoverageChrome({});
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for subject_a" }),
    ).toBeInTheDocument();
  });

  it("labels the control Unattest, naming the active subject, when the raw stored set holds the cell and it is fresh", async () => {
    await mountWithCoverageChrome({
      subject_a: {
        grid: MULTI_CELL_GRID,
        cells_complete: ["A1"],
        attested_by: "user:z",
        attested_at: "t",
        stem: "img1",
        date: "2026-01-01",
        subject: "subject_a",
        stale_cells: [],
        cells_attested_view: {},
      },
    });
    expect(screen.getByRole("button", { name: "Unattest A1 for subject_a" })).toBeInTheDocument();
  });

  it("labels the control Re-attest when the raw stored set holds the cell and it is stale", async () => {
    await mountWithCoverageChrome({
      subject_a: {
        grid: MULTI_CELL_GRID,
        cells_complete: ["A1"],
        attested_by: "user:z",
        attested_at: "t",
        stem: "img1",
        date: "2026-01-01",
        subject: "subject_a",
        stale_cells: ["A1"],
        cells_attested_view: {},
      },
    });
    expect(
      screen.getByRole("button", {
        name: "Re-attest A1 for subject_a (changed since attested)",
      }),
    ).toBeInTheDocument();
  });

  it("switching the subject picker moves the attest control and the counts to that subject", async () => {
    localStorage.removeItem("tcip.annotate.coverageGridOverlayOpen");
    mockMultiCellGrid();
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {
        subject_a: {
          grid: MULTI_CELL_GRID,
          cells_complete: ["A1"],
          attested_by: "user:z",
          attested_at: "t",
          stem: "img1",
          date: "2026-01-01",
          subject: "subject_a",
          stale_cells: [],
          cells_attested_view: {},
        },
      },
      annotation_counts: { subject_a: { A1: 2 }, subject_b: { A1: 5 } },
      counts_grid: MULTI_CELL_GRID,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 200,
      height: 200,
      top: 0,
      left: 0,
      right: 200,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => "",
    } as DOMRect);
    // Matches this test's own 200x200 geometry above, so computeViewport's center is the same.
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 200, h: 200 });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    triggerBelowNativeBaseFacts();
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    await flush();
    fireEvent.click(screen.getByRole("button", { name: /Overlay off/ }));

    expect(screen.getByText("Coverage for subject_a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unattest A1 for subject_a" })).toBeInTheDocument();
    expect(screen.getByText("saved for subject_a: A1 (2)")).toBeInTheDocument();

    act(() => useStore.getState().setActiveSubject("subject_b"));
    await flush();

    expect(screen.getByText("Coverage for subject_b")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for subject_b" }),
    ).toBeInTheDocument();
    expect(screen.getByText("saved for subject_b: A1 (5)")).toBeInTheDocument();
  });

  it("a single-cell raster renders the chrome with its attest control", async () => {
    localStorage.removeItem("tcip.annotate.coverageGridOverlayOpen");
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 200,
      height: 200,
      top: 0,
      left: 0,
      right: 200,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => "",
    } as DOMRect);
    // Matches this test's own 200x200 geometry above, so computeViewport's center is the same.
    vi.spyOn(canvasSync, "measureCanvasHost").mockReturnValue({ w: 200, h: 200 });
    vi.spyOn(api.coverage, "get").mockResolvedValue(null);
    vi.spyOn(api.coverage, "push").mockResolvedValue({
      record: { cells_seen_at_scale: {} },
    });
    vi.spyOn(api.coverage, "grid").mockResolvedValue(
      gridResponse({ width: 800, height: 600, tile_size: 800, overlap: 0, cols: 1, rows: 1 }, [
        { name: "A1", x0: 0, y0: 0, x1: 800, y1: 600 },
      ]),
    );
    vi.spyOn(api.coverage, "completeness").mockResolvedValue({
      by_subject: {},
      annotation_counts: {},
      counts_grid: null,
      counts_error: null,
      working_scale: {},
      working_scale_error: null,
      working_scale_reason: {},
    });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    await waitFor(() => expect(api.coverage.grid).toHaveBeenCalled());
    await flush();

    // A single cell has no overlay to draw, so the toggle is withdrawn and the control is
    // offered directly, with nothing to hide the cell behind.
    expect(screen.queryByRole("button", { name: /Overlay/ })).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for subject_a" }),
    ).toBeInTheDocument();
    // The Map tool stays withdrawn: a single-cell raster has nowhere else to jump to.
    expect(screen.queryByRole("button", { name: /^Map$/ })).not.toBeInTheDocument();
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
