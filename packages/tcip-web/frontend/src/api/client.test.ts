import { afterEach, describe, expect, it, vi } from "vitest";

import { RENDER_CACHE_VERSION } from "@/api/types.generated";
import { api } from "@/api/client";
import { stateSocket } from "@/api/ws";

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
    stubFetch(200, { base_mtime: "1783702599549301100" });
    const res = await api.annotate.save({ image_path: "x", label_path: "x.json", annotations: [] });
    expect(res.status).toBe("ok");
    if (res.status === "ok") expect(res.base_mtime).toBe("1783702599549301100");
  });

  it("returns a conflict (not a thrown error) on a 409", async () => {
    stubFetch(409, { error: "label file changed since it was loaded" });
    const res = await api.annotate.save({
      image_path: "x",
      label_path: "x.json",
      annotations: [],
      base_mtime: "1",
    });
    expect(res.status).toBe("conflict");
  });
});

describe("canvas.pushState 409 recovery", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const body = () => ({
    binding_generation: 3,
    tab: "annotate" as const,
    image_path: "/p/img.jpg",
    image: "img.jpg",
    img_width: 100,
    img_height: 80,
    viewport: null,
    classes: [],
    shapes: null,
  });

  it("returns ok + shapes_written on a 200", async () => {
    stubFetch(200, { status: "ok", shapes_written: true });
    const res = await api.canvas.pushState(body());
    expect(res).toEqual({ status: "ok", shapes_written: true });
  });

  it("returns a conflict and triggers a resync on a 409, never adopting its generation", async () => {
    const resync = vi.spyOn(stateSocket, "resync").mockImplementation(() => {});
    stubFetch(409, { error: "the GUI's open project has changed", generation: 7 });
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

  it("drops a null parameter instead of sending the text null", async () => {
    stubFetch(200, { coverage: null });
    await api.coverage.get("mosaic.tif", "bush", null);
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/coverage?path=mosaic.tif&subject=bush");
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
    expect(api.images.url("mosaic.tif", { x0: 0, y0: 4067 })).toBe(
      `/api/images?path=mosaic.tif&x0=0&y0=4067&v=${RENDER_CACHE_VERSION}`,
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
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg"));
    expect(params.get("path")).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(params.has("bands")).toBe(false);
    expect(params.has("stretch")).toBe(false);
  });

  it("names no width, leaving the server's own display bound to apply", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg"));
    expect(params.has("max_width")).toBe(false);
  });

  it("carries bands/stretch through to the query string when given", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/img1.bandgroup", {
        bands: "Red,Green,Blue",
        stretch: "minmax",
      }),
    );
    expect(params.get("bands")).toBe("Red,Green,Blue");
    expect(params.get("stretch")).toBe("minmax");
  });
});

describe("images.url region params", () => {
  it("carries the native-pixel rect corners and max_width through to the query string", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/mosaic.tif", {
        x0: 0,
        y0: 4067,
        x1: 4067,
        y1: 8134,
        max_width: 2034,
      }),
    );
    expect(params.get("x0")).toBe("0");
    expect(params.get("y0")).toBe("4067");
    expect(params.get("x1")).toBe("4067");
    expect(params.get("y1")).toBe("8134");
    expect(params.get("max_width")).toBe("2034");
  });

  it("omits every rect param when no region is requested", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg"));
    for (const key of ["x0", "y0", "x1", "y1"]) expect(params.has(key)).toBe(false);
  });
});

describe("review request scoping", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("names the dataset root in the label-backup body, the root that opens the review engine", async () => {
    stubFetch(200, { status: "ok", files_backed_up: 3 });
    await api.review.backupLabels("C:/data", ["C:/data/annotations/2026-01-01"]);
    const init = vi.mocked(fetch).mock.calls[0][1];
    expect(init?.body).toBe(
      JSON.stringify({ dataset_root: "C:/data", label_dirs: ["C:/data/annotations/2026-01-01"] }),
    );
  });

  it("names the dataset root in the batch image-status query string", async () => {
    stubFetch(200, { statuses: {}, detection_stems: [] });
    await api.review.imageStatuses({
      dataset_root: "C:/data",
      gt_dir: "C:/data/annotations/2026-01-01",
      pred_dir: null,
    });
    const params = parseQuery(String(vi.mocked(fetch).mock.calls[0][0]));
    expect(params.get("dataset_root")).toBe("C:/data");
    expect(params.has("project_root")).toBe(false);
    expect(params.get("gt_dir")).toBe("C:/data/annotations/2026-01-01");
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
