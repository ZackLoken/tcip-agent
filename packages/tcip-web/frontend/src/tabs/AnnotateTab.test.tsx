import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import type { Mtimes, SaveResult } from "@/api/client";
import { classesApi } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";

// Konva needs a real 2D canvas; these tests exercise label I/O ordering and
// store->canvas prop flow, not drawing. Render Konva shapes as inspectable divs
// and CanvasStage as a passthrough so AnnotationShapes' memo behavior is intact.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  Rect: (props: { stroke?: string }) => <div data-testid="k-rect" data-stroke={props.stroke} />,
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

const mt = (detect: number): Mtimes => ({ detect: String(detect), segment: null });

// Distinct mtimes per image so a save's echoed base_mtimes identify which
// image's load they came from.
const LOAD_MTIME: Record<string, number> = { "img1.jpg": 100, "img2.jpg": 200 };

function labelsFor(imagePath: string) {
  const name = imagePath.split("/").pop() ?? "";
  return {
    image_path: imagePath,
    img_width: 1000,
    img_height: 800,
    boxes: [],
    polygons: [],
    base_mtimes: mt(LOAD_MTIME[name] ?? 1),
  };
}

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      mode: "box" as const,
      active_class: 0,
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
        annotation_type: "annotations",
        date: "2026-01-01",
        image_list: ["img1.jpg", "img2.jpg"],
        current_image_index: 0,
        annotations_detect_dir: "C:/data/annotations/detect/2026-01-01",
        annotations_segment_dir: null,
      },
    },
  }));
}

function addBox() {
  useStore.getState().addBox({ x1: 10, y1: 10, x2: 50, y2: 50, class_id: 0 });
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
  saveSpy = vi.spyOn(api.annotate, "save").mockResolvedValue({ status: "ok", base_mtimes: mt(1) });
  vi.spyOn(classesApi, "setImageStatus").mockResolvedValue({});
  vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue({});
});

afterEach(cleanup);

describe("AnnotateTab save/load race", () => {
  it("saves to the loaded paths and re-echoes the returned mtimes on the next save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    saveSpy.mockResolvedValueOnce({ status: "ok", base_mtimes: mt(101) });
    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(saveSpy.mock.calls[0][0].base_mtimes).toEqual(mt(100));
    expect(useStore.getState().canvas.dirty).toBe(false);

    // Second save on the same image must echo the mtimes the first save returned.
    act(addBox);
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].base_mtimes).toEqual(mt(101));
  });

  it("a flush save resolving after navigation does not rewind the loaded paths or wipe dirty", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Dirty img1, then navigate. flushLeaving() fires the save WITHOUT awaiting
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
    expect(saveSpy.mock.calls[0][0].base_mtimes).toEqual(mt(100));

    // Edit img2 while the img1 save is still in flight, then let it resolve late.
    act(addBox);
    await act(async () => {
      resolveFlushSave({ status: "ok", base_mtimes: mt(150) });
    });

    // The stale result must not markClean() the img2 edits...
    expect(useStore.getState().canvas.dirty).toBe(true);
    // ...but the per-image status for the image actually saved is still recorded.
    expect(classesApi.setImageStatus).toHaveBeenCalledWith("C:/proj", "img1.jpg", "partial");

    // ...and the next save must target img2 with img2's loaded mtimes — NOT
    // img1's file with the stale save's echoed mtimes.
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].image_path).toBe("C:/data/images/2026-01-01/img2.jpg");
    expect(saveSpy.mock.calls[1][0].base_mtimes).toEqual(mt(200));
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

describe("AnnotateTab class registry propagation", () => {
  it("re-renders canvas shapes when a class color/name is edited", async () => {
    useStore.getState().setClasses([{ id: 0, name: "catkin", color: "#ff0000" }]);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", "#ff0000");
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "0: catkin");

    // A ColorPickerModal/agent class edit must reach the memoized canvas layer
    // immediately — not wait for an unrelated zoom/edit/navigation re-render.
    act(() => {
      useStore.getState().upsertClass({ id: 0, name: "catkin_open", color: "#00ff00" });
    });
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", "#00ff00");
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", "0: catkin_open");
  });
});
