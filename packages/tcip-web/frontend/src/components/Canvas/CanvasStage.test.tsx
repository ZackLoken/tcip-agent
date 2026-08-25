import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { CanvasStage, type CanvasRegion } from "@/components/Canvas/CanvasStage";
import type { LoadedImage } from "@/lib/imageLoader";
import { useStore } from "@/store";

// Stage/Layer/Image render as divs but keep Konva's event surface: `evt` is the native event.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", async () => {
  const React = await import("react");
  type StageEvent = { evt: MouseEvent };
  type StageHandler = (e: StageEvent) => void;
  interface StageProps {
    children?: React.ReactNode;
    onWheel?: StageHandler;
    onMouseDown?: StageHandler;
    onMouseMove?: StageHandler;
    onMouseUp?: StageHandler;
    onClick?: StageHandler;
    onDblClick?: StageHandler;
    onContextMenu?: StageHandler;
  }
  interface StageHandle {
    getPointerPosition: () => { x: number; y: number } | null;
  }
  return {
    Stage: React.forwardRef<StageHandle, StageProps>(function Stage(props, ref) {
      const pointer = React.useRef<{ x: number; y: number } | null>(null);
      React.useImperativeHandle(ref, () => ({ getPointerPosition: () => pointer.current }));
      const forward = (handler?: StageHandler) => (e: React.SyntheticEvent) => {
        const evt = e.nativeEvent as MouseEvent;
        pointer.current = { x: evt.clientX, y: evt.clientY };
        handler?.({ evt });
      };
      return (
        <div
          data-testid="k-stage"
          onWheel={forward(props.onWheel)}
          onMouseDown={forward(props.onMouseDown)}
          onMouseMove={forward(props.onMouseMove)}
          onMouseUp={forward(props.onMouseUp)}
          onClick={forward(props.onClick)}
          onDoubleClick={forward(props.onDblClick)}
          onContextMenu={forward(props.onContextMenu)}
        >
          {props.children}
        </div>
      );
    }),
    Layer: (props: { children?: React.ReactNode }) => (
      <div data-testid="k-layer">{props.children}</div>
    ),
    Image: (props: { x?: number; y?: number; width?: number; height?: number }) => (
      <div
        data-testid="k-image"
        data-x={props.x}
        data-y={props.y}
        data-width={props.width}
        data-height={props.height}
      />
    ),
  };
});

// Controllable stand-in for the shared loader: each call parks until a test resolves it.
const loader = vi.hoisted(() => {
  const pending: { url: string; resolve: (r: unknown) => void }[] = [];
  return { pending };
});
vi.mock("@/lib/imageLoader", () => ({
  loadImage: (url: string) =>
    new Promise((resolve) => {
      loader.pending.push({ url, resolve });
    }),
}));

function makeLoaded(over: Partial<LoadedImage> = {}): LoadedImage {
  return {
    image: {} as HTMLImageElement,
    ok: true,
    aborted: false,
    servedSize: null,
    servedSizeRaw: null,
    statsSource: null,
    displayBounds: null,
    imageError: null,
    headerParseError: null,
    ...over,
  };
}

async function resolveLoad(url: string, result: LoadedImage) {
  const entry = loader.pending.find((p) => p.url === url);
  expect(entry, `no pending load for ${url}`).toBeDefined();
  await act(async () => {
    entry!.resolve(result);
  });
}

const REGION: CanvasRegion = {
  key: "B3",
  url: "/api/images?path=m.tif&x0=100&y0=200&x1=150&y1=240&max_width=50",
  x: 100,
  y: 200,
  width: 50,
  height: 40,
};

const BASE_URL = "/api/images?path=orchard.tif&max_width=800";

function stageProps(regions: CanvasRegion[]) {
  return { imageUrl: null, imgWidth: 1000, imgHeight: 800, regions };
}

function putView(view: { scale: number; offset_x: number; offset_y: number }) {
  act(() => {
    useStore.getState().setView(view);
  });
}

afterEach(() => {
  cleanup();
  loader.pending.length = 0;
});

describe("CanvasStage regions", () => {
  it("renders a region at its image-coord rect once its serve loads, and reports the facts", async () => {
    const onLoaded = vi.fn();
    render(<CanvasStage {...stageProps([{ ...REGION, onLoaded }])} />);
    expect(screen.queryAllByTestId("k-image")).toHaveLength(0);

    const facts = makeLoaded({ servedSize: { w: 50, h: 40 } });
    await resolveLoad(REGION.url, facts);
    const node = screen.getByTestId("k-image");
    expect(node).toHaveAttribute("data-x", "100");
    expect(node).toHaveAttribute("data-y", "200");
    expect(node).toHaveAttribute("data-width", "50");
    expect(node).toHaveAttribute("data-height", "40");
    expect(onLoaded).toHaveBeenCalledWith(facts);
  });

  it("unmounts a region dropped from the prop (no longer intersecting the viewport)", async () => {
    const { rerender } = render(<CanvasStage {...stageProps([REGION])} />);
    await resolveLoad(REGION.url, makeLoaded());
    expect(screen.queryAllByTestId("k-image")).toHaveLength(1);

    rerender(<CanvasStage {...stageProps([])} />);
    expect(screen.queryAllByTestId("k-image")).toHaveLength(0);
  });

  it("keeps the last-loaded bitmap on screen while the same cell refetches at a new tier", async () => {
    const { rerender } = render(<CanvasStage {...stageProps([REGION])} />);
    await resolveLoad(REGION.url, makeLoaded());
    expect(screen.queryAllByTestId("k-image")).toHaveLength(1);

    const sharper = { ...REGION, url: `${REGION.url}&tier=native` };
    rerender(<CanvasStage {...stageProps([sharper])} />);
    // The replacement is still in flight: the previous bitmap stays, never a blank cell.
    expect(screen.queryAllByTestId("k-image")).toHaveLength(1);

    await resolveLoad(sharper.url, makeLoaded());
    expect(screen.queryAllByTestId("k-image")).toHaveLength(1);
  });

  it("ignores a failed region serve (no bitmap installed, facts still reported)", async () => {
    const onLoaded = vi.fn();
    render(<CanvasStage {...stageProps([{ ...REGION, onLoaded }])} />);
    await resolveLoad(REGION.url, makeLoaded({ ok: false, image: null }));
    expect(screen.queryAllByTestId("k-image")).toHaveLength(0);
    expect(onLoaded).toHaveBeenCalled();
  });
});

describe("CanvasStage pointer geometry", () => {
  it("reports a press, a move and a release in image space, not in screen space", () => {
    putView({ scale: 2, offset_x: 30, offset_y: 70 });
    const onPixelDown = vi.fn();
    const onPixelMove = vi.fn();
    const onPixelUp = vi.fn();
    render(
      <CanvasStage
        imageUrl={null}
        imgWidth={1000}
        imgHeight={800}
        onPixelDown={onPixelDown}
        onPixelMove={onPixelMove}
        onPixelUp={onPixelUp}
      />,
    );
    const stage = screen.getByTestId("k-stage");
    fireEvent.mouseDown(stage, { button: 0, clientX: 210, clientY: 470 });
    fireEvent.mouseMove(stage, { clientX: 330, clientY: 250 });
    fireEvent.mouseUp(stage, { button: 0, clientX: 90, clientY: 610 });

    expect(onPixelDown).toHaveBeenCalledTimes(1);
    expect(onPixelMove).toHaveBeenCalledTimes(1);
    expect(onPixelUp).toHaveBeenCalledTimes(1);
    // Each screen point runs back through its own axis of the view (offset 30, 70 at scale 2).
    expect(onPixelDown.mock.calls[0].slice(0, 2)).toEqual([90, 200]);
    expect(onPixelMove.mock.calls[0].slice(0, 2)).toEqual([150, 90]);
    expect(onPixelUp.mock.calls[0].slice(0, 2)).toEqual([30, 270]);
  });

  it("reports clicks, double-clicks and context menus through the same transform", () => {
    putView({ scale: 0.5, offset_x: -40, offset_y: 25 });
    const onPixelClick = vi.fn();
    const onPixelDoubleClick = vi.fn();
    const onPixelContextMenu = vi.fn();
    render(
      <CanvasStage
        imageUrl={null}
        imgWidth={1000}
        imgHeight={800}
        onPixelClick={onPixelClick}
        onPixelDoubleClick={onPixelDoubleClick}
        onPixelContextMenu={onPixelContextMenu}
      />,
    );
    const stage = screen.getByTestId("k-stage");
    fireEvent.click(stage, { button: 0, clientX: 260, clientY: 145 });
    fireEvent.doubleClick(stage, { clientX: 110, clientY: 305 });
    fireEvent.contextMenu(stage, { clientX: 60, clientY: 85 });

    expect(onPixelClick).toHaveBeenCalledTimes(1);
    expect(onPixelDoubleClick).toHaveBeenCalledTimes(1);
    expect(onPixelContextMenu).toHaveBeenCalledTimes(1);
    expect(onPixelClick.mock.calls[0].slice(0, 2)).toEqual([600, 240]);
    expect(onPixelDoubleClick.mock.calls[0].slice(0, 2)).toEqual([300, 560]);
    expect(onPixelContextMenu.mock.calls[0].slice(0, 2)).toEqual([200, 120]);
  });
});

describe("CanvasStage wheel zoom", () => {
  const START = { scale: 1, offset_x: -400, offset_y: -300 };

  async function pinchWheel(deltaY: number) {
    await act(async () => {
      fireEvent.wheel(screen.getByTestId("k-stage"), {
        deltaY,
        ctrlKey: true,
        clientX: 300,
        clientY: 120,
      });
      // The view write is batched into one animation frame.
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
  }

  it("zooms in about the pointer when the wheel scrolls up", async () => {
    putView(START);
    render(<CanvasStage imageUrl={null} imgWidth={1000} imgHeight={800} />);
    await pinchWheel(-100);

    const view = useStore.getState().gui.view;
    // A 100-unit notch at wheel gain 0.002 scales by e^0.2 about the cursor's image point.
    expect(view.scale).toBeCloseTo(1.2214028, 6);
    expect(view.offset_x).toBeCloseTo(-554.98193, 4);
    expect(view.offset_y).toBeCloseTo(-392.98916, 4);
  });

  it("zooms out about the pointer when the wheel scrolls down", async () => {
    putView(START);
    render(<CanvasStage imageUrl={null} imgWidth={1000} imgHeight={800} />);
    await pinchWheel(100);

    const view = useStore.getState().gui.view;
    expect(view.scale).toBeCloseTo(0.8187308, 6);
    expect(view.offset_x).toBeCloseTo(-273.11153, 4);
    expect(view.offset_y).toBeCloseTo(-223.86692, 4);
  });
});

describe("CanvasStage auto-fit", () => {
  it("fits the whole image inside the canvas when the image first loads", async () => {
    putView({ scale: 1, offset_x: 0, offset_y: 0 });
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 300,
      top: 0,
      left: 0,
      right: 800,
      bottom: 300,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    render(<CanvasStage imageUrl={BASE_URL} imgWidth={1000} imgHeight={1200} />);
    await resolveLoad(BASE_URL, makeLoaded());

    const view = useStore.getState().gui.view;
    // An 800x300 canvas holds a 1000x1200 image only at its height ratio, 300/1200.
    expect(view.scale).toBe(0.25);
    expect(view.offset_x).toBe(275);
    expect(view.offset_y).toBe(0);
  });
});
