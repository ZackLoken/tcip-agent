import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";

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
    const res = await api.annotate.save({ image_path: "x", annotations: [] });
    expect(res.status).toBe("ok");
    if (res.status === "ok") expect(res.base_mtime).toBe("1783702599549301100");
  });

  it("returns a conflict (not a thrown error) on a 409", async () => {
    stubFetch(409, { error: "label file changed since it was loaded" });
    const res = await api.annotate.save({
      image_path: "x",
      annotations: [],
      base_mtime: "1",
    });
    expect(res.status).toBe("conflict");
  });
});

function parseQuery(url: string): URLSearchParams {
  return new URLSearchParams(url.split("?")[1] ?? "");
}

describe("images.url", () => {
  it("omits bands/stretch when not given, so a plain RGB request is unaffected", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", 4096));
    expect(params.get("path")).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(params.get("max_width")).toBe("4096");
    expect(params.has("bands")).toBe(false);
    expect(params.has("stretch")).toBe(false);
  });

  it("carries bands/stretch through to the query string when given", () => {
    const params = parseQuery(
      api.images.url(
        "C:/data/images/2026-01-01/img1.bandgroup",
        4096,
        undefined,
        "Red,Green,Blue",
        "minmax",
      ),
    );
    expect(params.get("bands")).toBe("Red,Green,Blue");
    expect(params.get("stretch")).toBe("minmax");
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
