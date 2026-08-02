import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import type { SaveResult } from "@/api/client";
import { classesApi, subjectColor } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";

// Konva needs a real 2D canvas; these tests exercise label I/O ordering and
// store->canvas prop flow, not drawing. Render Konva shapes as inspectable divs
// and CanvasStage as a passthrough so AnnotationShapes' memo behavior is intact.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
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
vi.mock("@/components/Canvas/CanvasStage", () => ({
  CanvasStage: (props: {
    children?: React.ReactNode;
    overlay?: React.ReactNode;
    imageUrl?: string | null;
    onPixelDown?: (x: number, y: number, ev: unknown) => void;
    onPixelMove?: (x: number, y: number, ev: unknown) => void;
    onPixelUp?: (x: number, y: number, ev: unknown) => void;
    onPixelClick?: (x: number, y: number, ev: unknown) => void;
    onPixelContextMenu?: (x: number, y: number, ev: unknown) => void;
  }) => (
    <div
      data-testid="canvas-stage"
      data-image-url={props.imageUrl ?? ""}
      onMouseDown={(e) => props.onPixelDown?.(e.clientX, e.clientY, { evt: { button: e.button } })}
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
  ),
}));
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
      active_subject: "catkin",
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
        subject: "catkin",
        date: "2026-01-01",
        image_list: ["img1.jpg", "img2.jpg"],
        current_image_index: 0,
        annotations_dir: "C:/data/annotations/2026-01-01",
        predictions_dir: null,
      },
    },
  }));
}

function addBox() {
  useStore.getState().addBox({ x1: 10, y1: 10, x2: 50, y2: 50, subject: "catkin", attributes: {} });
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
  vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
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
      "catkin",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
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
});

describe("AnnotateTab subject rendering", () => {
  it("renders a box with the subject-derived colour and the subject-name label", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    // Colour is GUI-local (name-derived), and the label is the subject name, no integer id.
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("catkin"));
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "catkin");
  });

  it("box mode draws an active-subject polygon's read-only derived box (dashed, no handles), never a stored box", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    const poly = {
      rings: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      ] as [number, number][][],
      subject: "catkin",
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
    expect(rects[0]).toHaveAttribute("data-stroke", subjectColor("catkin"));
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });

  it("box mode still draws a real editable box solid, distinct from a derived one", async () => {
    useStore.getState().setRegistry({ catkin: {} });
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
    // ...and no box of any kind.
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);
    expect(screen.getAllByTestId("k-text").some((t) => t.getAttribute("data-text") === "tip")).toBe(
      true,
    );
  });

  it("polygon mode draws every ring of an occlusion-split shape, labelled once", async () => {
    // A catkin behind a branch loads as one annotation with two disjoint regions. Drawing only the
    // first would show the breeder part of the object and let them confirm it as the whole.
    useStore.getState().setRegistry({ catkin: {} });
    useStore.setState((s) => ({ gui: { ...s.gui, mode: "polygon" as const } }));
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
        polygons: [{ rings, subject: "catkin", attributes: {} }],
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
    // Both parts wear the annotation's own colour, and it is named once (HaloLabel = halo + fill).
    expect(lines.every((l) => l.getAttribute("data-stroke") === subjectColor("catkin"))).toBe(true);
    expect(
      screen.getAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === "catkin"),
    ).toHaveLength(2);
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
  it("collapses to a pill when nothing is selected and there are no image-level ratings", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.queryByText("Select a shape to set its attributes.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Attributes" })).toBeInTheDocument();
  });

  it("reopens on its own when a shape gets selected, and can be closed manually", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    // The subject registry also has a "catkin" entry in the (always-mounted, hover-revealed)
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
});

describe("AnnotateTab legend", () => {
  it("explains the dashed derived box only in box mode", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Box mode (setupDataset default): the legend button is hover-revealed, so query its content
    // directly rather than simulating hover.
    expect(screen.getByText("Dashed = polygon's box (read-only)")).toBeInTheDocument();

    act(() => useStore.getState().setMode("polygon"));
    expect(screen.queryByText("Dashed = polygon's box (read-only)")).not.toBeInTheDocument();
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
