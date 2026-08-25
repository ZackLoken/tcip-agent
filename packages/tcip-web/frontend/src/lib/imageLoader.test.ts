import { afterEach, describe, expect, it, vi } from "vitest";

import { loadImage } from "@/lib/imageLoader";

function stubFetch(status: number, headers: Record<string, string>) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      headers: new Headers(headers),
      blob: async () => new Blob([]),
    } as unknown as Response),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadImage header parsing", () => {
  it("parses X-TCIP-Stats-Source as the structured StatsSource it now carries", async () => {
    stubFetch(404, {
      "X-TCIP-Stats-Source": JSON.stringify({
        read: "window_sample",
        seed: 0,
        pixel_fraction: 0.5,
        overview_scale: null,
      }),
    });
    const result = await loadImage("/api/images?path=x");
    expect(result.statsSource).toEqual({
      read: "window_sample",
      seed: 0,
      pixel_fraction: 0.5,
      overview_scale: null,
    });
    expect(result.headerParseError).toBeNull();
  });

  it("parses X-TCIP-Display-Bounds as the JSON list of pairs it now carries", async () => {
    stubFetch(404, {
      "X-TCIP-Display-Bounds": JSON.stringify([
        [0, 1000],
        [5, 20],
      ]),
    });
    const result = await loadImage("/api/images?path=x");
    expect(result.displayBounds).toEqual([
      [0, 1000],
      [5, 20],
    ]);
    expect(result.headerParseError).toBeNull();
  });

  it("carries the raw X-TCIP-Served-Size string beside the parsed pair", async () => {
    stubFetch(404, { "X-TCIP-Served-Size": "800x600" });
    const result = await loadImage("/api/images?path=x");
    expect(result.servedSizeRaw).toBe("800x600");
    expect(result.servedSize).toEqual({ w: 800, h: 600 });
  });

  it("a response with no stats-source header at all parses as null, not a refusal", async () => {
    stubFetch(404, {});
    const result = await loadImage("/api/images?path=x");
    expect(result.statsSource).toBeNull();
    expect(result.displayBounds).toBeNull();
    expect(result.headerParseError).toBeNull();
  });

  it("refuses, naming the header, when X-TCIP-Stats-Source will not parse", async () => {
    stubFetch(200, { "X-TCIP-Stats-Source": "not-json" });
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(false);
    expect(result.image).toBeNull();
    expect(result.headerParseError).toContain("X-TCIP-Stats-Source");
  });

  it("refuses, naming the header, when X-TCIP-Display-Bounds will not parse", async () => {
    stubFetch(200, { "X-TCIP-Display-Bounds": "0,1000;5,20" });
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(false);
    expect(result.image).toBeNull();
    expect(result.headerParseError).toContain("X-TCIP-Display-Bounds");
  });
});
