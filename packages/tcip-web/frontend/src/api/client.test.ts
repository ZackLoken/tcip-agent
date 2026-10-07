import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RENDER_CACHE_VERSION } from "@/api/types.generated";
import { api } from "@/api/client";
import { stateSocket } from "@/api/ws";
import { useDisplayPixels } from "@/hooks/useDisplayPixels";

/** The display pixel count the client's own producer reads off the test screen. */
function display(): number {
  return renderHook(() => useDisplayPixels()).result.current;
}

function stubFetch(status: number, body: unknown = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response),
  );
}

describe("annotate.save lost-update handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns ok + a fresh mtime token on a 200, preserved exactly at real ns magnitude", async () => {
    // A 2026 st_mtime_ns (~1.78e18) exceeds 2**53: as a JSON number it would be rounded
    // by JSON.parse and every echo would 409. String tokens must survive byte-for-byte.
    stubFetch(200, {
      status: "ok",
      image_path: "x",
      img_width: 10,
      img_height: 10,
      annotations: [{ subject: "tip", iscrowd: false, point: [1, 2], attributes: {}, index: 0 }],
      completion: {},
      flags: [],
      base_mtime: "1783702599549301100",
      accepted: { "2": 0 },
    });
    const res = await api.annotate.save({
      image_path: "x",
      annotations: [],
      user: "breeder",
    });
    expect(res.status).toBe("ok");
    if (res.status === "ok") {
      expect(res.labels.base_mtime).toBe("1783702599549301100");
      expect(res.labels.points).toHaveLength(1);
      expect(res.accepted).toEqual({ "2": 0 });
    }
  });

  it("returns a conflict (not a thrown error) on a 409", async () => {
    stubFetch(409, { detail: { error: "label document changed since it was loaded" } });
    const res = await api.annotate.save({
      image_path: "x",
      annotations: [],
      base_mtime: "1",
      user: "breeder",
    });
    expect(res.status).toBe("conflict");
  });

  it("throws on a 409 whose body carries no detail, as every JSON call does", async () => {
    stubFetch(409, { error: "label document changed since it was loaded" });
    const save = api.annotate.save({
      image_path: "x",
      annotations: [],
      base_mtime: "1",
      user: "breeder",
    });
    await expect(save).rejects.toThrow("409");
  });
});

describe("canvas.pushState 409 recovery", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const body = () => ({
    project_id: "a1b2c3d4e5f6",
    tab: "annotate" as const,
    image_path: "/p/img.jpg",
    image: "img.jpg",
    img_width: 100,
    img_height: 80,
    viewport: null,
    user: "grower",
    classes: [],
    shapes: null,
  });

  it("returns ok + shapes_written on a 200", async () => {
    stubFetch(200, { status: "ok", shapes_written: true });
    const res = await api.canvas.pushState(body());
    expect(res).toEqual({ status: "ok", shapes_written: true });
  });

  it("returns a conflict and triggers a resync on a 409", async () => {
    const resync = vi.spyOn(stateSocket, "resync").mockImplementation(() => {});
    // The route's real 409 body, as HTTPException(409, {...}) serializes it: nested under
    // "detail", never the flat shape a naive stub would guess.
    stubFetch(409, {
      detail: {
        error: "this push was built for a project the backend does not have open",
        open_project_id: "ffffffffffff",
      },
    });
    const res = await api.canvas.pushState(body());
    expect(res).toEqual({ status: "conflict" });
    expect(resync).toHaveBeenCalledTimes(1);
  });
});

describe("annotate.load", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports the raster's own width and height, each in its own field", async () => {
    // Non-square on purpose: equal dimensions would read the same whichever field they land in.
    stubFetch(200, {
      image_path: "C:/data/images/2026-01-01/img1.jpg",
      img_width: 4032,
      img_height: 3024,
      annotations: [{ subject: "bush", bbox: [10, 20, 110, 220] }],
      base_mtime: "1783702599549301100",
    });
    const labels = await api.annotate.load("C:/data/images/2026-01-01/img1.jpg");
    expect(labels.img_width).toBe(4032);
    expect(labels.img_height).toBe(3024);
    expect(labels.base_mtime).toBe("1783702599549301100");
  });

  it("splits the served annotation list into the canvas buckets", async () => {
    stubFetch(200, {
      image_path: "C:/data/images/2026-01-01/img1.jpg",
      img_width: 4032,
      img_height: 3024,
      annotations: [{ subject: "bush", bbox: [10, 20, 110, 220] }],
      base_mtime: null,
    });
    const labels = await api.annotate.load("C:/data/images/2026-01-01/img1.jpg");
    expect(labels.boxes).toHaveLength(1);
    expect(labels.boxes[0].x1).toBe(10);
    expect(labels.boxes[0].y1).toBe(20);
    expect(labels.boxes[0].x2).toBe(110);
    expect(labels.boxes[0].y2).toBe(220);
    expect(labels.polygons).toHaveLength(0);
    expect(labels.points).toHaveLength(0);
  });
});

describe("query string assembly", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("carries each named parameter, a bucket name with its separator encoded", async () => {
    stubFetch(200, { bucket: "m/2026-01-01", proposals: [] });
    await api.annotate.proposals("img.jpg", "m/2026-01-01");
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
      "/api/annotate/proposals?image_path=img.jpg&bucket=m%2F2026-01-01",
    );
  });

  it("drops an omitted parameter instead of sending the text undefined", async () => {
    stubFetch(200, { path: "", parent: null, entries: [] });
    await api.fs.list();
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/fs/list?");
  });

  it("keeps a parameter whose value is zero", async () => {
    stubFetch(200, {});
    await api.images.bands("mosaic.tif");
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/images/bands?path=mosaic.tif");
    expect(api.images.url("mosaic.tif", display(), { x0: 0, y0: 4067 })).toBe(
      `/api/images?path=mosaic.tif&x0=0&y0=4067&display_pixels=${display()}&v=${RENDER_CACHE_VERSION}`,
    );
  });
});

describe("request defaults", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends the JSON content type on a request that names no headers of its own", async () => {
    stubFetch(200, { status: "ok", current_image_index: 7 });
    await api.dataset.nav(7);
    const init = vi.mocked(fetch).mock.calls[0][1];
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ current_image_index: 7 }));
  });
});

function parseQuery(url: string): URLSearchParams {
  return new URLSearchParams(url.split("?")[1] ?? "");
}

describe("images.url", () => {
  it("omits bands/stretch when not given, so a plain RGB request is unaffected", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", display()));
    expect(params.get("path")).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(params.has("bands")).toBe(false);
    expect(params.has("stretch")).toBe(false);
  });

  it("names this display's pixel count and no size or encoding, whole or region", () => {
    vi.stubGlobal("devicePixelRatio", 2);
    try {
      const pixels = display();
      expect(pixels).toBe(3840 * 2160);
      for (const opts of [{}, { x0: 0, y0: 0, x1: 512, y1: 512 }]) {
        const params = parseQuery(
          api.images.url("C:/data/images/2026-01-01/img1.jpg", pixels, opts),
        );
        expect(params.get("display_pixels")).toBe(String(pixels));
        for (const key of ["max_width", "width", "quality"]) expect(params.has(key)).toBe(false);
      }
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("carries bands/stretch through to the query string when given", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/img1.bandgroup", display(), {
        bands: "Red,Green,Blue",
        stretch: "minmax",
      }),
    );
    expect(params.get("bands")).toBe("Red,Green,Blue");
    expect(params.get("stretch")).toBe("minmax");
  });
});

describe("images.url region params", () => {
  it("carries the native-pixel rect corners through to the query string", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/mosaic.tif", display(), {
        x0: 0,
        y0: 4067,
        x1: 4067,
        y1: 8134,
      }),
    );
    expect(params.get("x0")).toBe("0");
    expect(params.get("y0")).toBe("4067");
    expect(params.get("x1")).toBe("4067");
    expect(params.get("y1")).toBe("8134");
  });

  it("omits every rect param when no region is requested", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", display()));
    for (const key of ["x0", "y0", "x1", "y1"]) expect(params.has(key)).toBe(false);
  });
});

describe("images.bands", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits GET /api/images/bands with the path and returns the reported contract", async () => {
    stubFetch(200, {
      band_count: 4,
      bands: [
        { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
        { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
        { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
        { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
      ],
    });
    const res = await api.images.bands("C:/data/images/2026-01-01/DJI_0001.bandgroup");
    expect(res.band_count).toBe(4);
    expect(res.bands.map((b) => b.name)).toEqual(["Blue", "Green", "Red", "NIR"]);
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls[0][0]).toContain("/api/images/bands?path=");
  });
});
