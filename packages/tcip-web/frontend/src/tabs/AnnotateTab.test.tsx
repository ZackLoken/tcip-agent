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
  Line: (props: { stroke?: string }) => <div data-testid="k-line" data-stroke={props.stroke} />,
  Circle: () => <div data-testid="k-circle" />,
  Text: (props: { text?: string; fill?: string }) => (
    <div data-testid="k-text" data-text={props.text} data-fill={props.fill} />
  ),
}));
vi.mock("@/components/Canvas/CanvasStage", () => ({
  CanvasStage: (props: { children?: React.ReactNode; overlay?: React.ReactNode }) => (
    <div data-testid="canvas-stage">
      {props.children}
      {props.overlay}
    </div>
  ),
}));
vi.mock("@/components/AnnotateToolbar", () => ({
  AnnotateToolbar: () => <div data-testid="toolbar" />,
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
    // cached read) — the exact interleaving that used to corrupt cross-image GT.
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
    // ...but the per-image status for the image actually saved is still recorded — scoped to the
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

    // ...and the next save must target img2 with img2's loaded mtime — not
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
    // Colour is GUI-local (name-derived), and the label is the subject name — no integer id.
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor("catkin"));
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "catkin");
  });

  it("box mode draws an active-subject polygon's read-only derived box (solid, no handles), never a stored box", async () => {
    useStore.getState().setRegistry({ catkin: {} });
    const poly = {
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ] as [number, number][],
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
    // handles (handles are extra Rects), and it never entered canvas.boxes — so unsaveable. Solid
    // like every committed shape (read-only is enforced structurally, not by line style).
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).not.toHaveAttribute("data-dash"); // solid — dashed is reserved for transient shapes
    expect(rects[0]).toHaveAttribute("data-stroke", subjectColor("catkin"));
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });
});
