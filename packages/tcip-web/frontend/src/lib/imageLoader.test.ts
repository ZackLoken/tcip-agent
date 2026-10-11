import { afterEach, describe, expect, it, vi } from "vitest";

import { loadImage } from "@/lib/imageLoader";

function stubFetch(status: number, headers: Record<string, string>) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      headers: new Headers(headers),
      blob: () => Promise.resolve(new Blob([])),
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadImage header parsing", () => {
  it("parses X-TCIP-Served-Size into its pair", async () => {
    stubFetch(404, { "X-TCIP-Served-Size": "800x600" });
    const result = await loadImage("/api/images?path=x");
    expect(result.servedSize).toEqual({ w: 800, h: 600 });
  });

  it("names the refusal condition the server sent", async () => {
    stubFetch(409, { "X-TCIP-Image-Error": "overviews_required" });
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(false);
    expect(result.imageError).toBe("overviews_required");
    expect(result.servedSize).toBeNull();
  });
});
