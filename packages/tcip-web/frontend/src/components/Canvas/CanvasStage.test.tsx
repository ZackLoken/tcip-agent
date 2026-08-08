import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

import { CanvasStage, type CanvasRegion } from "@/components/Canvas/CanvasStage";
import type { LoadedImage } from "@/lib/imageLoader";

// Konva needs a real 2D canvas; these tests exercise the regions contract (mount, staleness,
// keep-last-while-refetching), not drawing. Stage/Layer/Image render as inspectable divs.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", async () => {
  const React = await import("react");
  return {
    Stage: React.forwardRef<HTMLDivElement, { children?: React.ReactNode }>(
      function Stage(props, ref) {
        return (
          <div data-testid="k-stage" ref={ref}>
            {props.children}
          </div>
        );
      },
    ),
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
    statsSource: null,
    displayBounds: null,
    imageError: null,
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

function stageProps(regions: CanvasRegion[]) {
  return { imageUrl: null, imgWidth: 1000, imgHeight: 800, regions };
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
