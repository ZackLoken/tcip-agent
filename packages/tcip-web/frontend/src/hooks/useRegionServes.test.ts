import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { ServingCell, ViewReads } from "@/api/types.generated";
import { useDisplayPixels } from "@/hooks/useDisplayPixels";
import { useRegionServes } from "@/hooks/useRegionServes";
import { type LoadedImage, readFacts } from "@/lib/imageLoader";
import { extentOf, IMAGE_ANSWERS } from "@/test/images";

/** The whole view's serve of the native raster, its facts read off the produced image answer. */
const BASE_FACTS: LoadedImage = {
  ...readFacts(new Headers(IMAGE_ANSWERS.image.headers)),
  ok: true,
  image: null,
  aborted: false,
};

/** The hook's arguments over the raster whose whole frame `answer`'s reads cover. */
function baseArgs(answer: ViewReads = IMAGE_ANSWERS.native) {
  const { w, h } = extentOf(answer.reads);
  return {
    imagePath: "C:/data/images/2026-01-01/mosaic.tif",
    imgW: w,
    imgH: h,
    view: { scale: 1, offset_x: 0, offset_y: 0 },
    displayPixels: renderHook(() => useDisplayPixels()).result.current,
    baseFacts: BASE_FACTS,
    composite: {},
  };
}

/** The key the hook serves `cell` under. */
const keyOf = (cell: ServingCell) => `${cell.level}:${cell.name}`;

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
});

describe("useRegionServes", () => {
  it("asks the server for the visible native rect and serves each read it answers", async () => {
    const answer = IMAGE_ANSWERS.native;
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue(answer);
    const args = baseArgs();
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() =>
      expect(result.current.map((r) => r.key).sort()).toEqual(answer.reads.map(keyOf).sort()),
    );
    expect(reads).toHaveBeenCalledWith(args.imagePath, args.displayPixels, {
      x0: 0,
      y0: 0,
      x1: args.imgW,
      y1: args.imgH,
    });
    for (const cell of answer.reads) {
      const served = result.current.find((r) => r.key === keyOf(cell));
      expect(served).toMatchObject({ x: cell.nx0, width: cell.nx1 - cell.nx0 });
      expect(served?.url).toBe(
        api.images.url(args.imagePath, args.displayPixels, {
          region: [cell.x0, cell.y0, cell.x1, cell.y1],
        }),
      );
    }
  });

  it("serves a level tile at the native place the server names, by its own level grid", async () => {
    const answer = IMAGE_ANSWERS.level;
    vi.spyOn(api.images, "viewReads").mockResolvedValue(answer);
    const args = baseArgs(answer);
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() =>
      expect(result.current.map((r) => r.key).sort()).toEqual(answer.reads.map(keyOf).sort()),
    );
    for (const cell of answer.reads) {
      const served = result.current.find((r) => r.key === keyOf(cell));
      expect(served).toMatchObject({
        x: cell.nx0,
        y: cell.ny0,
        width: cell.nx1 - cell.nx0,
        height: cell.ny1 - cell.ny0,
      });
      const params = new URL(served?.url ?? "", "http://host").searchParams;
      expect(
        Object.fromEntries(["path", "display_pixels", "level"].map((k) => [k, params.get(k)])),
      ).toEqual({
        path: args.imagePath,
        display_pixels: String(args.displayPixels),
        level: String(cell.level),
      });
      expect(params.getAll("region")).toEqual([cell.x0, cell.y0, cell.x1, cell.y1].map(String));
    }
  });

  it("never lets an answer for a view it has left replace the current one", async () => {
    const [left, current] = [IMAGE_ANSWERS.level, IMAGE_ANSWERS.native];
    const first = Promise.withResolvers<ViewReads>();
    const reads = vi
      .spyOn(api.images, "viewReads")
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(current);
    const args = baseArgs();
    const { result, rerender } = renderHook((props) => useRegionServes(props), {
      initialProps: args,
    });
    rerender({ ...args, view: { scale: 2, offset_x: 0, offset_y: 0 } });
    const currentKeys = current.reads.map(keyOf).sort();
    await waitFor(() => expect(result.current.map((r) => r.key).sort()).toEqual(currentKeys));
    await act(() => Promise.resolve(first.resolve(left)));
    expect(result.current.map((r) => r.key).sort()).toEqual(currentKeys);
    expect(reads).toHaveBeenCalledTimes(2);
  });

  it("asks again for the same view when the display changes", async () => {
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue(IMAGE_ANSWERS.native);
    const args = baseArgs();
    const { rerender } = renderHook((props) => useRegionServes(props), { initialProps: args });
    await waitFor(() => expect(reads).toHaveBeenCalledTimes(1));
    rerender({ ...args, displayPixels: 2 * args.displayPixels });
    await waitFor(() => expect(reads).toHaveBeenCalledTimes(2));
    expect(reads.mock.calls[1][1]).toBe(2 * args.displayPixels);
  });

  it("asks for a one-pixel-wide frame whose base bitmap was reduced only in height", async () => {
    const answer = IMAGE_ANSWERS.skinny;
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue(answer);
    const args = {
      ...baseArgs(answer),
      view: { scale: 1, offset_x: 0, offset_y: 0 },
      baseFacts: {
        ...BASE_FACTS,
        ...readFacts(new Headers(IMAGE_ANSWERS.image_skinny.headers)),
      },
    };
    const { result } = renderHook(() => useRegionServes(args));
    await waitFor(() => expect(result.current.map((r) => r.key)).toEqual(answer.reads.map(keyOf)));
    expect(reads).toHaveBeenCalledWith(args.imagePath, args.displayPixels, {
      x0: 0,
      y0: 0,
      x1: 1,
      y1: 800,
    });
  });

  it("asks nothing for a band composite or while the base bitmap carries the zoom", () => {
    const reads = vi.spyOn(api.images, "viewReads");
    const args = baseArgs();
    const composite = renderHook(() =>
      useRegionServes({ ...args, composite: { bands: ["1", "2", "3"] } }),
    );
    expect(composite.result.current).toEqual([]);
    const zoomedOut = renderHook(() =>
      useRegionServes({ ...args, view: { scale: 0.4, offset_x: 0, offset_y: 0 } }),
    );
    expect(zoomedOut.result.current).toEqual([]);
    expect(reads).not.toHaveBeenCalled();
  });
});
