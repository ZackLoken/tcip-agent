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

  it("returns ok + fresh mtime tokens on a 200, preserved exactly at real ns magnitude", async () => {
    // A 2026 st_mtime_ns (~1.78e18) exceeds 2**53 — as a JSON number it would be rounded
    // by JSON.parse and every echo would 409. String tokens must survive byte-for-byte.
    stubFetch(200, { base_mtimes: { detect: "1783702599549301100", segment: null } });
    const res = await api.annotate.save({ image_path: "x", boxes: [], polygons: [] });
    expect(res.status).toBe("ok");
    if (res.status === "ok") expect(res.base_mtimes.detect).toBe("1783702599549301100");
  });

  it("returns a conflict (not a thrown error) on a 409", async () => {
    stubFetch(409, { error: "label file changed since it was loaded" });
    const res = await api.annotate.save({
      image_path: "x",
      boxes: [],
      polygons: [],
      base_mtimes: { detect: "1", segment: null },
    });
    expect(res.status).toBe("conflict");
  });
});
