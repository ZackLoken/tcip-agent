import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { ServingCell } from "@/api/types.generated";
import { useDisplayPixels } from "@/hooks/useDisplayPixels";
import { useRegionServes } from "@/hooks/useRegionServes";
import type { LoadedImage } from "@/lib/imageLoader";

const BASE_FACTS: LoadedImage = {
  ok: true,
  servedSize: { w: 500, h: 300 },
  imageError: null,
  image: null,
  aborted: false,
};

function baseArgs() {
  return {
    imagePath: "C:/data/images/2026-01-01/mosaic.tif",
    imgW: 1000,
    imgH: 600,
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    displayPixels: renderHook(() => useDisplayPixels()).result.current,
    baseFacts: BASE_FACTS,
    composite: {},
  };
}

beforeEach(() => {
  const host = document.createElement("div");
  host.setAttribute("data-canvas-host", "");
  document.body.appendChild(host);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
    width: 1200,
    height: 800,
    top: 0,
    left: 0,
    right: 1200,
    bottom: 800,
    x: 0,
    y: 0,
    toJSON: () => "",
  });
});

afterEach(() => {
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

/** A native read the server names: its rect in the native grid is also where it lies. */
function native(name: string, x0: number, y0: number, x1: number, y1: number): ServingCell {
  return { name, level: 0, x0, y0, x1, y1, nx0: x0, ny0: y0, nx1: x1, ny1: y1 };
}

describe("useRegionServes", () => {
  it("asks the server for the visible native rect and serves each read it answers", async () => {
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue({
      reads: [native("S1", 0, 0, 500, 600), native("S2", 500, 0, 1000, 600)],
    });
    const args = baseArgs();
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() => expect(result.current.map((r) => r.key).sort()).toEqual(["0:S1", "0:S2"]));
    expect(reads).toHaveBeenCalledWith(args.imagePath, args.displayPixels, {
      x0: 0,
      y0: 0,
      x1: 1000,
      y1: 600,
    });
    expect(result.current.find((r) => r.key === "0:S2")).toMatchObject({ x: 500, width: 500 });
    expect(result.current[0].url).toBe(
      api.images.url(args.imagePath, args.displayPixels, { x0: 0, y0: 0, x1: 500, y1: 600 }),
    );
  });

  it("serves a level tile at the native place the server names, by its own level grid", async () => {
    vi.spyOn(api.images, "viewReads").mockResolvedValue({
      reads: [
        {
          name: "B1",
          level: 2,
          x0: 125,
          y0: 0,
          x1: 250,
          y1: 150,
          nx0: 500,
          ny0: 0,
          nx1: 1000,
          ny1: 600,
        },
      ],
    });
    const args = baseArgs();
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() => expect(result.current.map((r) => r.key)).toEqual(["2:B1"]));
    expect(result.current[0]).toMatchObject({ x: 500, y: 0, width: 500, height: 600 });
    const params = new URL(result.current[0].url, "http://host").searchParams;
    expect(
      Object.fromEntries(
        ["path", "display_pixels", "x0", "y0", "x1", "y1", "level"].map((k) => [k, params.get(k)]),
      ),
    ).toEqual({
      path: args.imagePath,
      display_pixels: String(args.displayPixels),
      x0: "125",
      y0: "0",
      x1: "250",
      y1: "150",
      level: "2",
    });
  });

  it("never lets an answer for a view it has left replace the current one", async () => {
    let releaseFirst: (value: { reads: ServingCell[] }) => void = () => {};
    const first = new Promise<{ reads: ServingCell[] }>((resolve) => {
      releaseFirst = resolve;
    });
    const reads = vi
      .spyOn(api.images, "viewReads")
      .mockReturnValueOnce(first)
      .mockResolvedValueOnce({ reads: [native("NEW", 0, 0, 500, 300)] });
    const args = baseArgs();
    const { result, rerender } = renderHook((props) => useRegionServes(props), {
      initialProps: args,
    });
    rerender({ ...args, view: { scale: 2, offset_x: 0, offset_y: 0 } });
    await waitFor(() => expect(result.current.map((r) => r.key)).toEqual(["0:NEW"]));
    await act(async () => releaseFirst({ reads: [native("OLD", 0, 0, 1000, 600)] }));
    expect(result.current.map((r) => r.key)).toEqual(["0:NEW"]);
    expect(reads).toHaveBeenCalledTimes(2);
  });

  it("asks again for the same view when the display changes", async () => {
    const reads = vi
      .spyOn(api.images, "viewReads")
      .mockResolvedValue({ reads: [native("S1", 0, 0, 500, 600)] });
    const args = baseArgs();
    const { rerender } = renderHook((props) => useRegionServes(props), { initialProps: args });
    await waitFor(() => expect(reads).toHaveBeenCalledTimes(1));
    rerender({ ...args, displayPixels: 2 * args.displayPixels });
    await waitFor(() => expect(reads).toHaveBeenCalledTimes(2));
    expect(reads.mock.calls[1][1]).toBe(2 * args.displayPixels);
  });

  it("asks for a one-pixel-wide frame whose base bitmap was reduced only in height", async () => {
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue({
      reads: [native("A1", 0, 0, 1, 1600)],
    });
    const args = {
      ...baseArgs(),
      imgW: 1,
      imgH: 100_000,
      view: { scale: 0.5, offset_x: 0, offset_y: 0 },
      baseFacts: { ...BASE_FACTS, servedSize: { w: 1, h: 20_000 } },
    };
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() => expect(result.current.map((r) => r.key)).toEqual(["0:A1"]));
    expect(reads).toHaveBeenCalledWith(args.imagePath, args.displayPixels, {
      x0: 0,
      y0: 0,
      x1: 1,
      y1: 1600,
    });
  });

  it("asks nothing for a band composite or while the base bitmap carries the zoom", () => {
    const reads = vi.spyOn(api.images, "viewReads");
    const args = baseArgs();
    const composite = renderHook(() => useRegionServes({ ...args, composite: { bands: "1,2,3" } }));
    expect(composite.result.current).toEqual([]);
    const zoomedOut = renderHook(() =>
      useRegionServes({ ...args, view: { scale: 0.4, offset_x: 0, offset_y: 0 } }),
    );
    expect(zoomedOut.result.current).toEqual([]);
    expect(reads).not.toHaveBeenCalled();
  });
});
